#pragma once

#include "common.hh"
#include "gpu_bits.hh"

#include <numeric>
#include <stdexcept>
#include <vector>

#include <ndzip/gpu_encoder.hh>


namespace ndzip::detail::gpu {

template<typename T>
using local_accessor
        = sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local>;

template<unsigned Dims, typename U, typename T>
U extent_cast(const T &e) {
    U v;
    for (unsigned i = 0; i < Dims; ++i) {
        v[i] = e[i];
    }
    return v;
}

template<typename U, unsigned Dims>
U extent_cast(const extent<Dims> &e) {
    return extent_cast<Dims, U>(e);
}

template<typename T, int Dims>
T extent_cast(const sycl::range<Dims> &r) {
    return extent_cast<static_cast<unsigned>(Dims), T>(r);
}

template<typename T, int Dims>
T extent_cast(const sycl::id<Dims> &r) {
    return extent_cast<static_cast<unsigned>(Dims), T>(r);
}

template<typename U, typename T>
[[gnu::always_inline]] U bit_cast(T v) {
    static_assert(std::is_pod_v<T> && std::is_pod_v<U> && sizeof(U) == sizeof(T));
    U cast;
    __builtin_memcpy(&cast, &v, sizeof cast);
    return cast;
}


// SYCL and CUDA have opposite indexing directions for vector components, so the last component of
// any SYCL id is component 0 of the CUDA id. CUDA limits the global size along all components > 0,
// so the largest extent (in our case: the HC index) should always be in the last component.

class work_item : public sycl::nd_item<1> {
  public:
    work_item(const sycl::nd_item<1> &nd_item)  // NOLINT(google-explicit-constructor)
        : sycl::nd_item<1>(nd_item) {}

    size_t num_threads() const { return warp_size; }
    size_t thread_id() const { return this->get_local_id(0) % warp_size; }
    size_t global_warp_id() const { return this->get_global_id(0) / warp_size; }
    size_t local_warp_id() const { return this->get_local_id(0) / warp_size; }
    void local_memory_barrier() const { this->barrier(sycl::access::fence_space::local_space); }
};

class work_range : public sycl::nd_range<1> {
  public:
    explicit work_range(size_t global_num_warps, size_t local_num_warps = 1)
        : sycl::nd_range<1>{global_num_warps * warp_size, local_num_warps * warp_size} {}
};

using hypercube_item = work_item;

class hypercube_range {
  public:
    explicit hypercube_range(
            size_t num_hypercubes, size_t max_hcs_per_work_group, size_t first_hc_index = 0)
        : _num_hypercubes(num_hypercubes)
        , _max_hcs_per_work_group(max_hcs_per_work_group)
        , _first_hc_index(first_hc_index) {}

    work_range item_space() const {
        return work_range{(_num_hypercubes + _max_hcs_per_work_group - 1) / _max_hcs_per_work_group
                        * _max_hcs_per_work_group,
                _max_hcs_per_work_group};
    }

    size_t global_index_of(hypercube_item item) const {
        return _first_hc_index + item.global_warp_id();
    }

    size_t local_index_of(hypercube_item item) const { return item.local_warp_id(); }

    bool contains(hypercube_item item) const { return item.global_warp_id() < _num_hypercubes; }

    size_t num_hypercubes() const { return _num_hypercubes; }

  private:
    size_t _num_hypercubes;
    size_t _max_hcs_per_work_group;
    size_t _first_hc_index;
};


template<typename CGF>
auto submit_and_profile(sycl::queue &q, const char *label, CGF &&cgf) {
    if (auto env = getenv("NDZIP_VERBOSE"); env && *env) {
        // no kernel profiling in hipSYCL yet!
        q.wait_and_throw();
        auto before = std::chrono::system_clock::now();
        auto evt = q.submit(std::forward<CGF>(cgf));
        q.wait_and_throw();
        auto after = std::chrono::system_clock::now();
        auto seconds = std::chrono::duration_cast<std::chrono::duration<double>>(after - before);
        printf("[profile] %s: %.3fms\n", label, seconds.count() * 1e3);
        return evt;
    } else {
        return q.submit(std::forward<CGF>(cgf));
    }
}


template<typename DestAccessor, typename SourceAccessor>
void nd_memcpy(
        const DestAccessor &dest, const SourceAccessor &source, size_t count, work_item item) {
    for (size_t i = item.thread_id(); i < count; i += item.num_threads()) {
        dest[i] = source[i];
    }
}


// TODO do a sub-group inclusive scan, as implemented in gpu_bits.hh
template<typename Scalar>
void local_inclusive_prefix_sum(
        /* local */ Scalar *scratch, size_t n_elements, work_item item) {
    // Hillis-Steele (short-span) prefix sum
    size_t pout = 0;
    size_t pin;
    for (size_t offset = 1; offset < n_elements || pout > 0; offset <<= 1u) {
        pout = 1 - pout;
        pin = 1 - pout;
        for (size_t i = item.thread_id(); i < n_elements; i += item.num_threads()) {
            if (i >= offset) {
                scratch[pout * n_elements + i]
                        = scratch[pin * n_elements + i] + scratch[pin * n_elements + i - offset];
            } else {
                scratch[pout * n_elements + i] = scratch[pin * n_elements + i];
            }
        }
        item.local_memory_barrier();
    }
}


template<typename Scalar>
void global_inclusive_prefix_sum_reduce(
        /* global */ Scalar *__restrict big, /* global */ Scalar *__restrict small,
        /* local */ Scalar *__restrict scratch, size_t count, work_item item) {
    const size_t global_id = item.get_global_id(0);
    const size_t local_id = item.get_local_id(0);
    const size_t local_size = item.get_local_range(0);
    const size_t group_id = item.get_group(0);

    scratch[local_id] = global_id < count ? big[global_id] : 0;
    item.local_memory_barrier();

    local_inclusive_prefix_sum(scratch, local_size, work_item{item});

    const Scalar local_result = scratch[local_id];
    if (global_id < count) { big[global_id] = local_result; }
    if (local_id == local_size - 1) { small[group_id] = local_result; }
}


template<typename Scalar>
void global_inclusive_prefix_sum_expand(/* global */ Scalar *__restrict small,
        /* global */ Scalar *__restrict big, size_t count, work_item item) {
    // TODO range check (necessary? or can it be eliminated by increasing buffer size?)
    // `+ get_local_size`: We skip the first WG since `small` orginates from an inclusive, not an
    // exclusive scan
    const size_t global_id = item.get_global_id(0) + item.get_local_range(0);
    if (global_id < count) { big[global_id] += small[item.get_group(0)]; }
}


template<typename Scalar>
class hierarchical_inclusive_prefix_sum {
  public:
    hierarchical_inclusive_prefix_sum(size_t n_elems, size_t local_size)
        : _n_elems(n_elems), _local_size(local_size) {
        while (n_elems > 1) {
            n_elems = (n_elems + local_size - 1) / local_size;
            _intermediate_buffers.emplace_back(n_elems);
        }
    }

    void operator()(sycl::queue &queue, sycl::buffer<Scalar> &in_out_buffer) const {
        using sam = sycl::access::mode;

        for (size_t i = 0; i < _intermediate_buffers.size(); ++i) {
            auto &big_buffer = i > 0 ? _intermediate_buffers[i - 1] : in_out_buffer;
            auto count = i > 0 ? big_buffer.get_count() : _n_elems;
            auto &small_buffer = _intermediate_buffers[i];
            const auto global_range = sycl::range<1>{
                    (big_buffer.get_count() + _local_size - 1) / _local_size * _local_size};
            const auto local_range = sycl::range<1>{_local_size};
            const auto scratch_range = sycl::range<1>{_local_size * 2};

            char label[50];
            sprintf(label, "hierarchical_inclusive_prefix_sum reduce %zu", i);
            submit_and_profile(queue, label, [&](sycl::handler &cgh) {
                auto big_acc = big_buffer.template get_access<sam::read_write>(cgh);
                auto small_acc = small_buffer.template get_access<sam::discard_write>(cgh);
                auto scratch_acc = local_accessor<Scalar>{scratch_range, cgh};
                cgh.parallel_for<reduction_kernel>(sycl::nd_range<1>(global_range, local_range),
                        [big_acc, small_acc, scratch_acc, count](sycl::nd_item<1> item) {
                            global_inclusive_prefix_sum_reduce<Scalar>(big_acc.get_pointer(),
                                    small_acc.get_pointer(), scratch_acc.get_pointer(), count,
                                    item);
                        });
            });
        }

        for (size_t i = 1; i < _intermediate_buffers.size(); ++i) {
            auto ii = _intermediate_buffers.size() - 1 - i;
            auto &small_buffer = _intermediate_buffers[ii];
            auto &big_buffer = ii > 0 ? _intermediate_buffers[ii - 1] : in_out_buffer;
            auto count = ii > 0 ? big_buffer.get_count() : _n_elems;
            const auto global_range = sycl::range<1>{
                    (big_buffer.get_count() + _local_size - 1) / _local_size * _local_size};
            const auto local_range = sycl::range<1>{_local_size};

            char label[50];
            sprintf(label, "hierarchical_inclusive_prefix_sum expand %zu", ii);
            submit_and_profile(queue, label, [&](sycl::handler &cgh) {
                auto small_acc = small_buffer.template get_access<sam::read_write>(cgh);
                auto big_acc = big_buffer.template get_access<sam::discard_write>(cgh);
                cgh.parallel_for<expansion_kernel>(sycl::nd_range<1>(global_range, local_range),
                        [small_acc, big_acc, count](sycl::nd_item<1> item) {
                            global_inclusive_prefix_sum_expand<Scalar>(
                                    small_acc.get_pointer(), big_acc.get_pointer(), count, item);
                        });
            });
        }
    }

  private:
    class reduction_kernel;
    class expansion_kernel;

    size_t _n_elems;
    size_t _local_size;
    mutable std::vector<sycl::buffer<Scalar>> _intermediate_buffers;
};


template<typename Profile>
size_t global_offset(size_t local_offset, extent<Profile::dimensions> global_size) {
    size_t global_offset = 0;
    size_t global_stride = 1;
    for (unsigned d = 0; d < Profile::dimensions; ++d) {
        global_offset += global_stride * (local_offset % Profile::hypercube_side_length);
        local_offset /= Profile::hypercube_side_length;
        global_stride *= global_size[Profile::dimensions - 1 - d];
    }
    return global_offset;
}


template<typename Profile, typename F>
void distribute_for_hypercube_indices(extent<Profile::dimensions> data_size,
        hypercube_range hc_range, hypercube_item item, F &&f) {
    auto side_length = Profile::hypercube_side_length;
    auto hc_size = ipow(side_length, Profile::dimensions);
    auto hc_offset
            = detail::extent_from_linear_id(hc_range.global_index_of(item), data_size / side_length)
            * side_length;

    size_t initial_offset = linear_offset(hc_offset, data_size);
    for (size_t local_idx = item.thread_id(); local_idx < hc_size;
            local_idx += item.num_threads()) {
        // TODO re-calculating the GO every iteration is probably painfully slow
        size_t global_idx = initial_offset + global_offset<Profile>(local_idx, data_size);
        f(global_idx, local_idx);
    }
}


template<typename Profile>
void store_hypercube(/* local */ const typename Profile::bits_type *__restrict cube,
        /* global */ typename Profile::data_type *__restrict data,
        extent<Profile::dimensions> data_size, hypercube_range hc_range, hypercube_item item) {
    distribute_for_hypercube_indices<Profile>(
            data_size, hc_range, item, [&](size_t global_idx, size_t local_idx) {
                data[global_idx] = bit_cast<typename Profile::data_type>(cube[local_idx]);
            });
}


template<typename Profile>
void inverse_block_transform(
        /* local */ typename Profile::bits_type *x, work_item item) {
    constexpr auto n = Profile::hypercube_side_length;
    constexpr auto dims = Profile::dimensions;
    constexpr auto hc_size = ipow(n, dims);

    for (size_t i = item.thread_id(); i < hc_size; i += item.num_threads()) {
        x[i] = complement_negative(x[i]);
    }

    item.local_memory_barrier();

    if (dims == 1) {
        if (item.thread_id() == 0) { inverse_block_transform_step(x, n, 1); }
    } else if (dims == 2) {
        for (size_t i = item.thread_id(); i < n; i += item.num_threads()) {
            inverse_block_transform_step(x + i, n, n);
        }
        item.local_memory_barrier();
        for (size_t i = item.thread_id(); i < n; i += item.num_threads()) {
            auto ii = i * n;
            inverse_block_transform_step(x + ii, n, 1);
        }
    } else if (dims == 3) {
        for (size_t i = item.thread_id(); i < n * n; i += item.num_threads()) {
            inverse_block_transform_step(x + i, n, n * n);
        }
        item.local_memory_barrier();
        for (size_t i = item.thread_id(); i < n * n; i += item.num_threads()) {
            auto ii = i * n;
            inverse_block_transform_step(x + ii, n, 1);
        }
        item.local_memory_barrier();
        for (size_t i = item.thread_id(); i < n; i += item.num_threads()) {
            auto ii = i * n * n;
            for (size_t j = 0; j < n; ++j) {
                inverse_block_transform_step(x + ii + j, n, n);
            }
        }
    }

    item.local_memory_barrier();

    for (size_t i = item.thread_id(); i < hc_size; i += item.num_threads()) {
        x[i] = rotate_right_1(x[i]);
    }
}


template<typename Bits>
void transpose_bits(/* local */ Bits *cube, work_item item) {
    constexpr auto cols_per_thread = bitsof<Bits> / warp_size;

    Bits columns[cols_per_thread] = {0};
    for (size_t k = 0; k < bitsof<Bits>; ++k) {
        auto row = cube[k];
        for (size_t c = 0; c < cols_per_thread; ++c) {
            size_t i = c * item.num_threads() + item.thread_id();
            columns[c] |= ((row >> (bitsof<Bits> - 1 - i)) & Bits{1}) << (bitsof<Bits> - 1 - k);
        }
    }

    item.local_memory_barrier();

    for (size_t c = 0; c < cols_per_thread; ++c) {
        size_t i = c * item.num_threads() + item.thread_id();
        cube[i] = columns[c];
    }
}


template<typename Bits>
size_t expand_zero_words(/* local */ Bits *__restrict out, /* global */ const Bits *__restrict in,
        /* local */ Bits *__restrict scratch, work_item item) {
    constexpr auto n_columns = bitsof<Bits>;

    auto head = in[0];
    for (size_t i = 0; i < n_columns / warp_size; ++i) {
        // Indexing works around LLVM bug
        size_t ii = i * warp_size + item.thread_id();
        scratch[ii] = (head >> (n_columns - 1 - ii)) & Bits{1};
    }

    item.local_memory_barrier();

    local_inclusive_prefix_sum(scratch, n_columns, item);

    for (size_t i = 0; i < n_columns / warp_size; ++i) {
        // Indexing works around LLVM bug
        size_t ii = i * warp_size + item.thread_id();
        if (((head >> (n_columns - 1 - ii)) & Bits{1}) != 0) {
            out[ii] = in[scratch[ii]];  // +1 from header offset and -1 from inclusive scan cancel
        } else {
            out[ii] = 0;
        }
    }

    return 1 + scratch[n_columns - 1];
}


template<typename Bits>
void zero_bit_decode(/* global */ const Bits *__restrict stream, /* local */ Bits *__restrict cube,
        /* local */ Bits *__restrict scratch, size_t hc_size, work_item item) {
    auto in = stream;
    for (size_t offset = 0; offset < hc_size; offset += bitsof<Bits>) {
        in += expand_zero_words(cube + offset, in, scratch, item);
        item.local_memory_barrier();
    }

    for (size_t offset = 0; offset < hc_size; offset += bitsof<Bits>) {
        transpose_bits(cube + offset, item);
    }
}


template<typename Profile>
constexpr size_t local_memory_words_per_hypercube
        = detail::ipow(Profile::hypercube_side_length, Profile::dimensions)
        + 3 * detail::bitsof<typename Profile::bits_type>;

inline size_t max_work_group_size(size_t local_memory_per_wg, const sycl::device &device) {
    auto available_local_memory
            = static_cast<size_t>(device.get_info<sycl::info::device::local_mem_size>());
    auto max_local_size
            = static_cast<size_t>(device.get_info<sycl::info::device::max_work_group_size>());
    auto max_wg_size = available_local_memory / local_memory_per_wg;
    if (max_wg_size == 0) {
        throw std::runtime_error("Not enough local memory on this device (has "
                + std::to_string(available_local_memory) + " bytes, needs at least "
                + std::to_string(local_memory_per_wg) + " bytes)");
    }
    max_wg_size = std::min(max_wg_size, max_local_size / warp_size);
    if (auto env = getenv("NDZIP_VERBOSE"); env && *env) {
        printf("Allowing %zu work groups per SM (allocates %zu of %zu bytes)\n", max_wg_size,
                max_wg_size * local_memory_per_wg, available_local_memory);
    }
    return max_wg_size;
}


template<typename Profile>
struct compressed_chunks {
    using bits_type = typename Profile::bits_type;

    const static auto stride
            = (Profile::compressed_block_size_bound + sizeof(bits_type) - 1) / sizeof(bits_type);

    index_type num_hypercubes;
    bits_type *buffer;

    bits_type *hypercube(index_type hc_index) { return buffer + hc_index * stride; }
};

using stream_align_t = uint64_t;

template<typename Profile>
struct stream {
    using bits_type = typename Profile::bits_type;

    index_type num_hypercubes;
    stream_align_t *buffer;

    file_offset_type *header() { return static_cast<file_offset_type *>(buffer); }

    index_type offset_after(index_type hc_index) {
        return static_cast<index_type>(
                (header()[hc_index] - num_hypercubes * sizeof(file_offset_type))
                / sizeof(bits_type));
    }

    void set_offset_after(index_type hc_index, index_type position) {
        header()[hc_index]
                = num_hypercubes * sizeof(file_offset_type) + position * sizeof(bits_type);
    }

    // requires header() to be initialized
    bits_type *hypercube(index_type hc_index) {
        bits_type *base = reinterpret_cast<bits_type *>(
                static_cast<file_offset_type *>(buffer) + num_hypercubes);
        if (hc_index == 0) {
            return base;
        } else {
            return base + offset_after(hc_index - 1);
        }
    }

    index_type hypercube_size(index_type hc_index) {
        return hc_index == 0 ? offset_after(0)
                             : offset_after(hc_index) - offset_after(hc_index - 1);
    }

    // requires header() to be initialized
    bits_type *border() { return offset_after(num_hypercubes); }
};


template<typename Profile>
struct grid {
    using data_type = std::conditional_t<std::is_const_v<Profile>,
            const typename Profile::data_type, typename Profile::data_type>;
    constexpr static unsigned dimensions = Profile::dimensions;

    slice<data_type, dimensions> data;
};


// Fine tuning block size. For block transform:
//    -  double precision, 256 >> 128
//    - single precision 1D, 512 >> 128 > 256.
//    - single precision forward 1D 2D, 512 >> 256.
// At least for sm_61 profile<double, 3d> exceeds maximum register usage with 512
inline constexpr index_type hypercube_group_size = 256;
using hypercube_group = known_size_group<hypercube_group_size>;


struct forward_transform_tag;
struct inverse_transform_tag;

template<unsigned Dims, typename Transform>
struct hypercube_layout;

template<unsigned Direction, unsigned Dims, typename Transform>
struct directional_accessor;

// std::optional is not allowed in kernels
inline constexpr index_type no_such_lane = ~index_type{};

template<>
struct hypercube_layout<1, forward_transform_tag> {
    constexpr static index_type side_length = 4096;
    constexpr static index_type hc_size = 4096;
    constexpr static index_type num_lanes = hypercube_group_size;
    constexpr static index_type lane_length = hc_size / num_lanes;

    constexpr static index_type pad(index_type i, index_type width) {
        return width * i + width * i / warp_size;
    }
};

template<>
struct hypercube_layout<1, inverse_transform_tag> {
    constexpr static index_type side_length = 4096;
    constexpr static index_type hc_size = 4096;
    constexpr static index_type num_lanes = hypercube_group_size;
    constexpr static index_type lane_length = hc_size / num_lanes;

    // Special case: 1D inverse transform uses prefix sum, which is optimal without padding.
};

// TODO directional access is not really useful for 1D, and the "lanes" logic requires padding.
//  Is it faster to load the entire 1D hc into registers instead? It might not, since the lanes
//  approach only requires one read and one write per element whereas the register variant needs
//  one read and one read-write op.
template<typename Transform>
struct directional_accessor<0, 1, Transform> {
    using layout = hypercube_layout<1, Transform>;

    constexpr static index_type prev_lane_in_row(index_type lane) {
        return lane > 0 ? lane - 1 : no_such_lane;
    }

    constexpr static index_type offset(index_type lane) {
        return lane * layout::hc_size / layout::num_lanes;
    }

    static inline const index_type stride = 1;
};

template<>
struct hypercube_layout<2, forward_transform_tag> {
    constexpr static index_type num_lanes = hypercube_group_size;
    constexpr static index_type side_length = 64;
    constexpr static index_type hc_size = 64 * 64;
    constexpr static index_type lane_length = hc_size / num_lanes;

    constexpr static index_type pad(index_type i, index_type width) {
        return width * i + width * i / warp_size;
    }
};

template<>
struct directional_accessor<0, 2, forward_transform_tag> {
    using layout = hypercube_layout<2, forward_transform_tag>;

    constexpr static index_type prev_lane_in_row(index_type lane) {
        if (lane % (layout::side_length / layout::lane_length) > 0) {
            return lane - 1;
        } else {
            return no_such_lane;
        }
    }
    constexpr static index_type offset(index_type lane) {
        return lane * (layout::hc_size / layout::num_lanes);
    }
    constexpr static index_type stride = 1;
};

template<>
struct directional_accessor<1, 2, forward_transform_tag> {
    using layout = hypercube_layout<2, forward_transform_tag>;

    constexpr static index_type prev_lane_in_row(index_type lane) {
        if (lane >= layout::side_length) {
            return lane - layout::side_length;
        } else {
            return no_such_lane;
        }
    }
    constexpr static index_type offset(index_type lane) {
        return (lane / layout::side_length)
                * (layout::hc_size / layout::num_lanes * layout::side_length)
                + lane % layout::side_length;
    }
    constexpr static index_type stride = layout::side_length;
};

template<>
struct hypercube_layout<2, inverse_transform_tag> {
    constexpr static index_type side_length = 64;
    constexpr static index_type hc_size = 64 * 64;
    constexpr static index_type num_lanes = side_length;
    constexpr static index_type lane_length = hc_size / num_lanes;

    constexpr static index_type pad(index_type i, index_type width) {
        if (width == 1) {
            return i + i / side_length;
        } else {
            return width * i + i / (warp_size / width) - i / side_length;
        }
    }
};

template<>
struct directional_accessor<0, 2, inverse_transform_tag> {
    using layout = hypercube_layout<2, inverse_transform_tag>;
    constexpr static index_type prev_lane_in_row(index_type) { return no_such_lane; }
    constexpr static index_type offset(index_type lane) { return lane * layout::side_length; }
    constexpr static index_type stride = 1;
};

template<>
struct directional_accessor<1, 2, inverse_transform_tag> {
    using layout = hypercube_layout<2, inverse_transform_tag>;
    constexpr static index_type prev_lane_in_row(index_type) { return no_such_lane; }
    constexpr static index_type offset(index_type lane) { return lane % layout::side_length; }
    constexpr static index_type stride = layout::side_length;
};

template<typename Transform>
struct hypercube_layout<3, Transform> {
    constexpr static index_type side_length = 16;
    constexpr static index_type hc_size = ipow(side_length, 3);
    // TODO implement support for forward_transform with > 256 lanes
    constexpr static index_type num_lanes = ipow(side_length, 2);
    constexpr static index_type lane_length = hc_size / num_lanes;

    constexpr static index_type pad(index_type i, index_type width) {
        auto padded = width * i + width * i / warp_size;
        if (width == 2) { padded -= i / (width * num_lanes); }
        return padded;
    }
};

template<typename Transform>
struct directional_accessor<0, 3, Transform> {
    using layout = hypercube_layout<3, Transform>;
    constexpr static index_type prev_lane_in_row(index_type) { return no_such_lane; }
    constexpr static index_type offset(index_type lane) { return lane * layout::side_length; }
    constexpr static index_type stride = 1;
};

template<typename Transform>
struct directional_accessor<1, 3, Transform> {
    using layout = hypercube_layout<3, Transform>;

    constexpr static index_type prev_lane_in_row(index_type) { return no_such_lane; }
    constexpr static index_type offset(index_type lane) {
        return (lane / layout::side_length) * 2 * layout::num_lanes
                - (lane / (layout::num_lanes / 2))
                * (layout::hc_size - ipow(layout::side_length, 2))
                + lane % layout::side_length;
    }
    constexpr static index_type stride = layout::side_length;
};

template<typename Transform>
struct directional_accessor<2, 3, Transform> {
    using layout = hypercube_layout<3, Transform>;
    constexpr static index_type prev_lane_in_row(index_type) { return no_such_lane; }
    constexpr static index_type offset(index_type lane) { return lane; }
    constexpr static index_type stride = layout::side_length * layout::side_length;
};


template<typename Bits, typename Layout>
struct hypercube_allocation {
    using backing_type = uint_bank_t;
    constexpr static index_type size
            = ceil(Layout::pad(Layout::hc_size, sizeof(Bits) / sizeof(uint_bank_t)),
                    sizeof(Bits) / sizeof(uint_bank_t));
};

template<typename Bits>
struct hypercube_allocation<Bits, hypercube_layout<1, inverse_transform_tag>> {
    using backing_type = Bits;
    constexpr static index_type size = hypercube_layout<1, inverse_transform_tag>::hc_size;
};

template<typename Bits, typename Layout>
using hypercube_memory = sycl::local_memory<typename hypercube_allocation<Bits,
        Layout>::backing_type[hypercube_allocation<Bits, Layout>::size]>;


template<typename Profile, typename F>
void for_hypercube_indices(
        hypercube_group grp, index_type hc_index, extent<Profile::dimensions> data_size, F &&f) {
    const auto side_length = Profile::hypercube_side_length;
    const auto hc_size = ipow(side_length, Profile::dimensions);
    const auto hc_offset
            = detail::extent_from_linear_id(hc_index, data_size / side_length) * side_length;

    size_t initial_offset = linear_offset(hc_offset, data_size);
    grp.distribute_for(hc_size, [&](index_type local_idx) {
        size_t global_idx = initial_offset + global_offset<Profile>(local_idx, data_size);
        f(global_idx, local_idx);
    });
}

template<typename Profile, typename Transform>
struct hypercube_ptr {
    using bits_type = typename Profile::bits_type;
    using layout = hypercube_layout<Profile::dimensions, Transform>;

    uint_bank_t *memory;

    bits_type load(index_type i) const {
        return load_aligned<alignof(uint_bank_t), bits_type>(
                memory + layout::pad(i, sizeof(bits_type) / sizeof(uint_bank_t)));
    }

    void store(index_type i, bits_type bits) {
        store_aligned<alignof(uint_bank_t), bits_type>(
                memory + layout::pad(i, sizeof(bits_type) / sizeof(uint_bank_t)), bits);
    }
};

// We guarantee that memory is laid out sequentially for 1D inverse transform, which is implemented
// using gpu_bits prefix_sum
template<typename Data>
struct hypercube_ptr<profile<Data, 1>, inverse_transform_tag> {
    using bits_type = typename profile<Data, 1>::bits_type;
    using layout = hypercube_layout<1, inverse_transform_tag>;

    bits_type *memory;

    bits_type load(index_type i) const { return memory[i]; }

    void store(index_type i, bits_type bits) { memory[i] = bits; }
};

template<typename Profile>
void load_hypercube(sycl::group<1> grp, index_type hc_index,
        slice<const typename Profile::data_type, Profile::dimensions> data,
        hypercube_ptr<Profile, forward_transform_tag> hc) {
    using bits_type = typename Profile::bits_type;

    for_hypercube_indices<Profile>(
            grp, hc_index, data.size(), [&](index_type global_idx, index_type local_idx) {
                hc.store(local_idx, rotate_left_1(bit_cast<bits_type>(data.data()[global_idx])));
                // TODO merge with block_transform to avoid LM round-trip?
                //  we could assign directly to private_memory
            });
}

template<typename Profile>
void store_hypercube(sycl::group<1> grp, index_type hc_index,
        slice<typename Profile::data_type, Profile::dimensions> data,
        hypercube_ptr<Profile, inverse_transform_tag> hc) {
    using data_type = typename Profile::data_type;
    for_hypercube_indices<Profile>(
            grp, hc_index, data.size(), [&](index_type global_idx, index_type local_idx) {
                data.data()[global_idx] = bit_cast<data_type>(rotate_right_1(hc.load(local_idx)));
            });
}


template<unsigned Direction, typename Profile>
void forward_transform_lanes(
        hypercube_group grp, hypercube_ptr<Profile, forward_transform_tag> hc) {
    using accessor = directional_accessor<Direction, Profile::dimensions, forward_transform_tag>;
    using layout = typename accessor::layout;
    using bits_type = typename Profile::bits_type;

    // 1D and 2D transforms have multiple lanes per row, so a barrier is required to synchronize
    // the read of the last element from (lane-1) with the write to the last element of (lane)
    constexpr bool needs_carry = layout::side_length > layout::lane_length;
    // std::max: size might be zero if !needs_carry, but that is not a valid type
    sycl::private_memory<
            bits_type[std::max(index_type{1}, layout::num_lanes / hypercube_group_size)]>
            carry{grp};

    if constexpr (needs_carry) {
        grp.distribute_for<layout::num_lanes>([&](index_type lane, index_type iteration,
                                                      sycl::logical_item<1> idx) {
            if (auto prev_lane = accessor::prev_lane_in_row(lane); prev_lane != no_such_lane) {
                carry(idx)[iteration] = hc.load(
                        accessor::offset(prev_lane) + (layout::lane_length - 1) * accessor::stride);
            } else {
                carry(idx)[iteration] = 0;
            }
        });
    }

    grp.distribute_for<layout::num_lanes>(
            [&](index_type lane, index_type iteration, sycl::logical_item<1> idx) {
                bits_type a = needs_carry ? carry(idx)[iteration] : 0;
                index_type index = accessor::offset(lane);
                for (index_type i = 0; i < layout::lane_length; ++i) {
                    auto b = hc.load(index);
                    hc.store(index, b - a);
                    a = b;
                    index += accessor::stride;
                }
            });
}


template<typename Profile>
void forward_block_transform(
        hypercube_group grp, hypercube_ptr<Profile, forward_transform_tag> hc) {
    constexpr auto dims = Profile::dimensions;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, dims);

    // Why is there no constexpr for?
    forward_transform_lanes<0>(grp, hc);
    if constexpr (dims >= 2) { forward_transform_lanes<1>(grp, hc); }
    if constexpr (dims >= 3) { forward_transform_lanes<2>(grp, hc); }

    // TODO move complement operation elsewhere to avoid local memory round-trip
    grp.distribute_for(
            hc_size, [&](index_type item) { hc.store(item, complement_negative(hc.load(item))); });
}


template<unsigned Direction, typename Profile>
void inverse_transform_lanes(
        hypercube_group grp, hypercube_ptr<Profile, inverse_transform_tag> hc) {
    using accessor = directional_accessor<Direction, Profile::dimensions, inverse_transform_tag>;
    using layout = typename accessor::layout;
    using bits_type = typename Profile::bits_type;

    grp.distribute_for<layout::num_lanes>([&](index_type lane) {
        index_type index = accessor::offset(lane);
        bits_type a = hc.load(index);
        for (index_type i = 1; i < layout::lane_length; ++i) {
            index += accessor::stride;
            a += hc.load(index);
            hc.store(index, a);
        }
    });
}


unsigned popcount(unsigned int x) {
    return __builtin_popcount(x);
}

unsigned popcount(unsigned long x) {
    return __builtin_popcountl(x);
}

unsigned popcount(unsigned long long x) {
    return __builtin_popcountll(x);
}


template<typename Profile>
void inverse_block_transform(
        hypercube_group grp, hypercube_ptr<Profile, inverse_transform_tag> hc) {
    using bits_type = typename Profile::bits_type;
    constexpr auto dims = Profile::dimensions;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, dims);

    // TODO move complement operation elsewhere to avoid local memory round-trip
    grp.distribute_for(
            hc_size, [&](index_type item) { hc.store(item, complement_negative(hc.load(item))); });

    // TODO how to do 2D?
    //   For 2D we have 64 parallel work items but we _probably_ want at least 256 threads per SM
    //   (= per HC for double) to hide latencies. Maybe hybrid approach - do (64/32)*64 sub-group
    //   prefix sums and optimize inclusive_prefix_sum to skip the recursion since the second
    //   level does not actually need a reduction. Benchmark against leaving 256-64 = 192 threads
    //   idle and going with a sequential-per-lane transform.

    if constexpr (dims == 1) {
        // 1D inverse hypercube_ptr guarantees linear layout of hc.memory
        inclusive_scan<Profile::hypercube_side_length>(grp, hc.memory, sycl::plus<bits_type>{});
    }
    if constexpr (dims == 2) {
        // TODO inefficient, see above
        inverse_transform_lanes<0>(grp, hc);
        inverse_transform_lanes<1>(grp, hc);
    }
    if constexpr (dims == 3) {
        inverse_transform_lanes<0>(grp, hc);
        inverse_transform_lanes<1>(grp, hc);
        inverse_transform_lanes<2>(grp, hc);
    }
}

template<typename Profile>
void write_transposed_chunks(hypercube_group grp, hypercube_ptr<Profile, forward_transform_tag> hc,
        typename Profile::bits_type *out_heads, typename Profile::bits_type *out_columns,
        index_type *out_lengths) {
    using bits_type = typename Profile::bits_type;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    static_assert(hc_size % warp_size == 0);

    // One group per warp (for subgroup reductions)
    constexpr index_type chunk_size = bitsof<bits_type>;

    grp.distribute_for(
            hc_size, [&](index_type item, index_type, sycl::logical_item<1>, sycl::sub_group sg) {
                auto warp_index = item / warp_size;
                // TODO 32 bit for double!
                auto mask = ~bits_type{};
                if constexpr (sizeof(bits_type) == 8) {
                    if (warp_index % 2 == 0) {
                        mask <<= 32;
                    } else {
                        mask >>= 32;
                    }
                }

                // TODO this is weird. Can we not "transpose" this for the 64-bit case so we don't
                //  have to mask and only need a single iteration?
                bits_type head = 0;
                for (index_type j = 0; j < chunk_size / warp_size; ++j) {
                    auto col = floor(item, chunk_size) + item % warp_size + j * warp_size;
                    head |= sycl::group_reduce(sg, hc.load(col) & mask, sycl::bit_or<bits_type>{});
                }

                index_type this_warp_size = 0;
                bits_type column = 0;
                // TODO this shortcut does not improve performance - but why? Shortcut' warps are
                // stalled on the final barrier, but given a large enough group_size, this should
                // still result in much fewer wasted cycles
                if (head != 0) {
                    const auto chunk_base = floor(item, chunk_size);
                    const auto cell = item - chunk_base;
                    for (index_type i = 0; i < chunk_size; ++i) {
                        // TODO for double, can we still operate on 32 bit words? e.g split into
                        //  low / high loop
                        column |= (hc.load(chunk_base + i) >> (chunk_size - 1 - cell)
                                          & bits_type{1})
                                << (chunk_size - 1 - i);
                    }
                    this_warp_size = popcount(head);
                    auto base = floor(item, warp_size);
                    auto relative_pos = sycl::group_exclusive_scan(
                            sg, index_type{column != 0}, sycl::plus<index_type>{});
                    if (column != 0) { out_columns[base + relative_pos] = column; }
                }
                if (warp_index == 0) {
                    this_warp_size += hc_size / chunk_size;  // heads
                }
                if (sg.leader()) {
                    // TODO collect in local memory, write coalesced - otherwise 3 full GM
                    //  transaction per HC instead of 1!
                    out_heads[warp_index] = head;
                    out_lengths[warp_index] = this_warp_size;
                }
            });
}


template<typename Profile>
void read_transposed_chunks(hypercube_group grp, hypercube_ptr<Profile, inverse_transform_tag> hc,
        const typename Profile::bits_type *stream) {
    using bits_type = typename Profile::bits_type;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr index_type chunk_size = bitsof<bits_type>;
    constexpr index_type num_chunks = hc_size / chunk_size;

    sycl::local_memory<index_type[1 + num_chunks]> chunk_offsets{grp};
    grp.distribute_for(
            num_chunks, [&](index_type item) { chunk_offsets[1 + item] = popcount(stream[item]); });
    grp.single_item([&] { chunk_offsets[0] = num_chunks; });
    inclusive_scan<num_chunks>(grp, chunk_offsets(), sycl::plus<index_type>());

    grp.distribute_for(
            hc_size, [&](index_type item, index_type, sycl::logical_item<1>, sycl::sub_group sg) {
                auto chunk_index = item / chunk_size;
                auto head = stream[chunk_index];

                bits_type row = 0;
                if (head != 0) {
                    auto offset = chunk_offsets[chunk_index];
                    const auto chunk_base = floor(item, chunk_size);
                    const auto cell = item - chunk_base;
                    for (index_type i = 0; i < chunk_size; ++i) {
                        // TODO for double, can we still operate on 32 bit words? e.g split into
                        //  low / high loop
                        if ((head >> (chunk_size - 1 - i)) & bits_type{1}) {
                            row |= (stream[offset] >> (chunk_size - 1 - cell) & bits_type{1})
                                    << (chunk_size - 1 - i);
                            offset += 1;
                        }
                    }
                }
                hc.store(item, row);
            });
}


template<typename Profile>
void compact_chunks(sycl::group<1> grp, const typename Profile::bits_type *heads,
        const typename Profile::bits_type *columns, const index_type *offsets,
        typename Profile::bits_type *stream) {
    // One group per warp (for subgroup reductions)
    using bits_type = typename Profile::bits_type;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr index_type chunk_size = bitsof<bits_type>;
    constexpr index_type warps_per_hc = hc_size / warp_size;
    constexpr index_type warps_per_chunk = chunk_size / warp_size;

    grp.distribute_for([&](sycl::sub_group sg, sycl::logical_item<1> idx) {
        auto item = idx.get_global_id(0);
        auto warp_index = item / warp_size;

        auto body_offset = offsets[warp_index];
        if (warp_index % warps_per_chunk == 0) {
            bits_type head = 0;
            for (index_type i = 0; i < (chunk_size / warp_size); ++i) {
                head |= heads[warp_index + i];
            }
            if (sg.leader()) {
                auto hc_first_warp_index = floor(warp_index, warps_per_hc);
                auto head_offset = offsets[hc_first_warp_index]
                        + (warp_index - hc_first_warp_index) / warps_per_chunk;
                stream[head_offset] = head;
            }
        }
        if (warp_index % warps_per_hc == 0) { body_offset += hc_size / chunk_size; }
        index_type tid = sg.get_local_id()[0];
        if (body_offset + tid < offsets[warp_index + 1]) {
            stream[body_offset + tid] = columns[item];
        }
    });
}


// TODO we might be able to avoid this kernel altogether by writing the reduction results directly
//  to the stream header. Requires the stream format to be fixed first.
// => Probably not after the fine-grained compaction refactor
template<typename Profile>
void fill_stream_header(index_type num_hypercubes, global_write<stream_align_t> stream_acc,
        global_read<file_offset_type> offset_acc, sycl::handler &cgh) {
    constexpr detail::gpu::index_type hc_size
            = detail::ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr detail::gpu::index_type warps_per_hc = hc_size / detail::gpu::warp_size;
    constexpr index_type group_size = 256;
    const index_type num_groups = div_ceil(num_hypercubes, group_size);
    cgh.parallel(sycl::range<1>{num_groups}, sycl::range<1>{group_size},
            [=](known_size_group<group_size> grp, sycl::physical_item<1>) {
                stream<Profile> stream{num_hypercubes, stream_acc.get_pointer()};
                const index_type base = static_cast<index_type>(grp.get_id(0)) * group_size;
                const index_type num_elements = std::min(group_size, num_hypercubes - base);
                grp.distribute_for(num_elements, [&](index_type i) {
                    stream.set_offset_after(base + i, offset_acc[(base + i + 1) * warps_per_hc]);
                });
            });
}


template<typename Profile>
void compact_hypercubes(index_type num_hypercubes, global_write<stream_align_t> stream_acc,
        global_read<typename Profile::bits_type> chunks_acc, sycl::handler &cgh) {
    constexpr index_type num_threads_per_hc = 64;
    cgh.parallel(sycl::range<1>{num_hypercubes}, sycl::range<1>{num_threads_per_hc},
            [=](known_size_group<num_threads_per_hc> grp, sycl::physical_item<1>) {
                auto hc_index = static_cast<index_type>(grp.get_id(0));
                compressed_chunks<Profile> chunks{num_hypercubes, chunks_acc.get_pointer()};
                auto in = chunks.hypercube(hc_index);
                stream<Profile> stream{num_hypercubes, stream_acc.get_pointer()};
                auto out = stream.hypercube(hc_index);
                grp.distribute_for(
                        stream.hypercube_size(hc_index), [&](index_type i) { out[i] = in[i]; });
            });
}


template<typename Profile>
void decompress_hypercubes(/* global */ const typename Profile::bits_type *__restrict stream,
        /* global */ typename Profile::data_type *__restrict data,
        /* local */ typename Profile::bits_type *__restrict local_memory,
        extent<Profile::dimensions> data_size, hypercube_range hc_range, hypercube_item item) {
    using bits_type = typename Profile::bits_type;

    const auto side_length = Profile::hypercube_side_length;
    const auto hc_size = detail::ipow(side_length, Profile::dimensions);
    const auto this_hc_local_memory = local_memory
            + hc_range.local_index_of(item) * local_memory_words_per_hypercube<Profile>;
    const auto cube = this_hc_local_memory;
    const auto scratch = this_hc_local_memory + hc_size;

    // TODO casting / byte-offsetting is ugly. Use separate buffers like in compress()
    const auto offset_table = reinterpret_cast<const file_offset_type *>(stream);
    const auto chunk_offset_bytes = hc_range.global_index_of(item)
            ? offset_table[hc_range.global_index_of(item) - 1]
            : hc_range.num_hypercubes() * sizeof(file_offset_type);
    const auto chunk_offset_words = chunk_offset_bytes / sizeof(bits_type);

    zero_bit_decode<bits_type>(stream + chunk_offset_words, cube, scratch, hc_size, item);
    item.local_memory_barrier();
    inverse_block_transform<Profile>(cube, item);
    item.local_memory_barrier();
    store_hypercube<Profile>(cube, data, data_size, hc_range, item);
}


// SYCL kernel names
template<typename, unsigned>
class block_compression_kernel;

template<typename, unsigned>
class header_encoding_kernel;

template<typename, unsigned>
class stream_compaction_kernel;

template<typename, unsigned>
class stream_decompression_kernel;

}  // namespace ndzip::detail::gpu


template<typename T, unsigned Dims>
struct ndzip::gpu_encoder<T, Dims>::impl {
    sycl::queue q;

    impl() : q{sycl::gpu_selector{}} {
        if (auto env = getenv("NDZIP_VERBOSE"); env && *env) {
            auto device = q.get_device();
            printf("Using %s on %s %s (%lu bytes of local memory)\n",
                    device.get_platform().get_info<sycl::info::platform::name>().c_str(),
                    device.get_info<sycl::info::device::vendor>().c_str(),
                    device.get_info<sycl::info::device::name>().c_str(),
                    (unsigned long) device.get_info<sycl::info::device::local_mem_size>());
        }
    }
};

template<typename T, unsigned Dims>
ndzip::gpu_encoder<T, Dims>::gpu_encoder() : _pimpl(std::make_unique<impl>()) {
}

template<typename T, unsigned Dims>
ndzip::gpu_encoder<T, Dims>::~gpu_encoder() = default;


template<typename T, unsigned Dims>
size_t ndzip::gpu_encoder<T, Dims>::compress(
        const slice<const data_type, dimensions> &data, void *stream) const {
    using namespace detail::gpu;

    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;
    using sam = sycl::access::mode;
    using hc_layout = hypercube_layout<profile::dimensions, forward_transform_tag>;

    constexpr index_type hc_size
            = detail::ipow(profile::hypercube_side_length, profile::dimensions);
    constexpr index_type warps_per_hc = hc_size / warp_size;

    // TODO edge case w/ 0 hypercubes

    detail::file<profile> file(data.size());
    auto num_hypercubes = file.num_hypercubes();
    if (auto env = getenv("NDZIP_VERBOSE"); env && *env) {
        printf("Have %zu hypercubes\n", num_hypercubes);
    }

    sycl::buffer<data_type, dimensions> data_buffer{
            extent_cast<sycl::range<dimensions>>(data.size())};

    submit_and_profile(_pimpl->q, "copy input to device", [&](sycl::handler &cgh) {
        cgh.copy(data.data(), data_buffer.template get_access<sam::discard_write>(cgh));
    });

    sycl::buffer<bits_type> columns_buf(num_hypercubes * hc_size);
    sycl::buffer<bits_type> heads_buf(num_hypercubes * warps_per_hc);
    sycl::buffer<index_type> chunk_lengths_buf(1 + num_hypercubes * warps_per_hc);

    submit_and_profile(_pimpl->q, "transform + chunk encode", [&](sycl::handler &cgh) {
        auto data_acc = data_buffer.template get_access<sam::read>(cgh);
        auto heads_acc = heads_buf.template get_access<sam::discard_write>(cgh);
        auto columns_acc = columns_buf.template get_access<sam::discard_write>(cgh);
        auto chunk_lengths_acc = chunk_lengths_buf.get_access<sam::discard_write>(cgh);
        auto data_size = data.size();
        cgh.parallel<block_compression_kernel<T, Dims>>(sycl::range<1>{file.num_hypercubes()},
                sycl::range<1>{hypercube_group_size},
                [=](hypercube_group grp, sycl::physical_item<1> phys_idx) {
                    slice<const data_type, dimensions> data{data_acc.get_pointer(), data_size};
                    hypercube_memory<bits_type, hc_layout> lm{grp};
                    hypercube_ptr<profile, forward_transform_tag> hc{&lm[0]};

                    index_type hc_index = grp.get_id(0);
                    load_hypercube(grp, hc_index, {data}, hc);
                    forward_block_transform(grp, hc);
                    write_transposed_chunks(grp, hc, &heads_acc[hc_index * warps_per_hc],
                            &columns_acc[hc_index * hc_size],
                            &chunk_lengths_acc[1 + hc_index * warps_per_hc]);
                    // hack
                    if (phys_idx.get_global_linear_id() == 0) {
                        grp.single_item([&] { chunk_lengths_acc[0] = 0; });
                    }
                });
    });

    std::vector<index_type> dbg_lengths(chunk_lengths_buf.get_range()[0]);
    _pimpl->q
            .submit([&](sycl::handler &cgh) {
                cgh.copy(chunk_lengths_buf.get_access<sam::read>(cgh), dbg_lengths.data());
            })
            .wait();
    hierarchical_inclusive_prefix_sum<index_type> prefix_sum(
            1 + num_hypercubes * warps_per_hc, 256 /* local size */);
    prefix_sum(_pimpl->q, chunk_lengths_buf);
    std::vector<index_type> dbg_offsets(chunk_lengths_buf.get_range()[0]);
    _pimpl->q
            .submit([&](sycl::handler &cgh) {
                cgh.copy(chunk_lengths_buf.get_access<sam::read>(cgh), dbg_offsets.data());
            })
            .wait();

    index_type num_compressed_words;
    auto num_compressed_words_available = _pimpl->q.submit([&](sycl::handler &cgh) {
        cgh.copy(chunk_lengths_buf.template get_access<sam::read>(
                         cgh, sycl::range<1>{1}, sycl::id<1>{num_hypercubes * warps_per_hc}),
                &num_compressed_words);
    });

    sycl::buffer<stream_align_t> stream_buf(
            (compressed_size_bound<data_type, dimensions>(data.size()) + sizeof(stream_align_t) - 1)
            / sizeof(stream_align_t));

    submit_and_profile(_pimpl->q, "fill header", [&](sycl::handler &cgh) {
        fill_stream_header<profile>(num_hypercubes,
                stream_buf.get_access<sam::write>(cgh),  // TODO limit access range
                chunk_lengths_buf.get_access<sam::read>(cgh), cgh);
    });

    submit_and_profile(_pimpl->q, "compact chunks", [&](sycl::handler &cgh) {
        auto columns_acc = columns_buf.template get_access<sam::read>(cgh);
        auto heads_acc = heads_buf.template get_access<sam::read>(cgh);
        auto offsets_acc = chunk_lengths_buf.template get_access<sam::read>(cgh);
        auto stream_acc = stream_buf.template get_access<sam::discard_write>(cgh);
        constexpr size_t group_size = 1024;
        const size_t header_offset = file.file_header_length() / sizeof(stream_align_t);
        cgh.parallel<stream_compaction_kernel<T, Dims>>(
                sycl::range<1>{hc_size / group_size * num_hypercubes}, sycl::range<1>{group_size},
                [=](sycl::group<1> grp, sycl::physical_item<1>) {
                    compact_chunks<profile>(grp,
                            static_cast<const bits_type *>(heads_acc.get_pointer()),
                            static_cast<const bits_type *>(columns_acc.get_pointer()),
                            static_cast<const index_type *>(offsets_acc.get_pointer()),
                            reinterpret_cast<bits_type *>(
                                    static_cast<stream_align_t *>(stream_acc.get_pointer())
                                    + header_offset));
                });
    });

    num_compressed_words_available.wait();
    auto stream_pos = file.file_header_length() + num_compressed_words * sizeof(bits_type);

    auto n_aligned_words = (stream_pos + sizeof(stream_align_t) - 1) / sizeof(stream_align_t);
    auto stream_transferred
            = submit_and_profile(_pimpl->q, "copy stream to host", [&](sycl::handler &cgh) {
                  cgh.copy(stream_buf.get_access<sam::read>(cgh, n_aligned_words),
                          static_cast<stream_align_t *>(stream));
              });


    stream_transferred.wait();  // TODO I need to wait since I'm potentially overwriting the
    // border
    //  in the aligned copy
    stream_pos += detail::pack_border(
            static_cast<char *>(stream) + stream_pos, data, profile::hypercube_side_length);

    _pimpl->q.wait();
    return stream_pos;
}


template<typename T, unsigned Dims>
size_t ndzip::gpu_encoder<T, Dims>::decompress(
        const void *stream, size_t bytes, const slice<data_type, dimensions> &data) const {
    using namespace detail;
    using namespace detail::gpu;

    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;
    using sam = sycl::access::mode;
    using hc_layout = hypercube_layout<profile::dimensions, inverse_transform_tag>;

    detail::file<profile> file(data.size());

    // TODO the range computation here is questionable at best
    sycl::buffer<stream_align_t> stream_buffer{
            sycl::range<1>{div_ceil(bytes, sizeof(stream_align_t))}};
    sycl::buffer<data_type, dimensions> data_buffer{
            detail::gpu::extent_cast<sycl::range<dimensions>>(data.size())};

    submit_and_profile(_pimpl->q, "copy stream to device", [&](sycl::handler &cgh) {
        cgh.copy(static_cast<const stream_align_t *>(stream),
                stream_buffer.template get_access<sam::discard_write>(cgh));
    });

    submit_and_profile(_pimpl->q, "decompress blocks", [&](sycl::handler &cgh) {
        auto stream_acc = stream_buffer.template get_access<sam::read>(cgh);
        auto data_acc = data_buffer.template get_access<sam::discard_write>(cgh);
        auto data_size = data.size();
        auto num_hypercubes = file.num_hypercubes();
        cgh.parallel<stream_decompression_kernel<T, Dims>>(sycl::range<1>{num_hypercubes},
                sycl::range<1>{hypercube_group_size},
                [=](hypercube_group grp, sycl::physical_item<1>) {
                    slice<data_type, dimensions> data{data_acc.get_pointer(), data_size};
                    hypercube_memory<bits_type, hc_layout> lm{grp};
                    hypercube_ptr<profile, inverse_transform_tag> hc{lm()};

                    index_type hc_index = grp.get_id(0);
                    detail::gpu::stream<profile> stream{num_hypercubes, stream_acc.get_pointer()};
                    read_transposed_chunks<profile>(grp, hc, stream.hypercube(hc_index));
                    inverse_block_transform<profile>(grp, hc);
                    store_hypercube(grp, hc_index, {data}, hc);
                });
    });

    auto data_copy_event = detail::gpu::submit_and_profile(
            _pimpl->q, "copy output to host", [&](sycl::handler &cgh) {
                cgh.copy(data_buffer.template get_access<sam::read>(cgh), data.data());
            });

    auto stream_pos
            = detail::load_aligned<detail::file_offset_type>(static_cast<const std::byte *>(stream)
                    + ((file.num_hypercubes() - 1) * sizeof(detail::file_offset_type)));

    data_copy_event.wait();

    // TODO GPU border handling!
    stream_pos += detail::unpack_border(data, static_cast<const std::byte *>(stream) + stream_pos,
            profile::hypercube_side_length);

    return stream_pos;
}


namespace ndzip {

extern template class gpu_encoder<float, 1>;
extern template class gpu_encoder<float, 2>;
extern template class gpu_encoder<float, 3>;
extern template class gpu_encoder<double, 1>;
extern template class gpu_encoder<double, 2>;
extern template class gpu_encoder<double, 3>;

}  // namespace ndzip
