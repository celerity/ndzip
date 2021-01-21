#pragma once

#include "common.hh"

#include <numeric>
#include <stdexcept>
#include <vector>

#include <SYCL/sycl.hpp>


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

template<typename Integer>
constexpr Integer div_ceil(Integer p, Integer q) {
    return (p + q - 1) / q;
}

template<typename Integer>
constexpr Integer ceil(Integer x, Integer multiple) {
    return div_ceil(x, multiple) * multiple;
}


inline const size_t warp_size = 32;

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
            auto &big_buffer = i ? _intermediate_buffers[i - 1] : in_out_buffer;
            auto count = i ? big_buffer.get_count() : _n_elems;
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
            auto count = i ? big_buffer.get_count() : _n_elems;
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
void load_hypercube(/* global */ const typename Profile::data_type *__restrict data,
        /* local */ typename Profile::bits_type *__restrict cube,
        extent<Profile::dimensions> data_size, hypercube_range hc_range, hypercube_item item) {
    distribute_for_hypercube_indices<Profile>(
            data_size, hc_range, item, [&](size_t global_idx, size_t local_idx) {
                cube[local_idx] = bit_cast<typename Profile::bits_type>(data[global_idx]);
            });
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
void block_transform(/* local */ typename Profile::bits_type *x, work_item item) {
    constexpr auto n = Profile::hypercube_side_length;
    constexpr auto dims = Profile::dimensions;
    constexpr auto hc_size = ipow(n, dims);

    for (size_t i = item.thread_id(); i < hc_size; i += item.num_threads()) {
        x[i] = rotate_left_1(x[i]);
    }

    item.local_memory_barrier();

    if constexpr (dims == 1) {
        if (item.thread_id() == 0) { block_transform_step(x, n, 1); }
    } else if constexpr (dims == 2) {
        for (size_t i = item.thread_id(); i < n; i += item.num_threads()) {
            const auto ii = n * i;
            block_transform_step(x + ii, n, 1);
        }
        item.local_memory_barrier();
        for (size_t i = item.thread_id(); i < n; i += item.num_threads()) {
            block_transform_step(x + i, n, n);
        }
    } else if constexpr (dims == 3) {
        for (size_t i = item.thread_id(); i < n; i += item.num_threads()) {
            const auto ii = n * n * i;
            for (size_t j = 0; j < n; ++j) {
                block_transform_step(x + ii + j, n, n);
            }
        }
        item.local_memory_barrier();
        for (size_t i = item.thread_id(); i < n * n; i += item.num_threads()) {
            const auto ii = n * i;
            block_transform_step(x + ii, n, 1);
        }
        item.local_memory_barrier();
        for (size_t i = item.thread_id(); i < n * n; i += item.num_threads()) {
            block_transform_step(x + i, n, n * n);
        }
    }

    item.local_memory_barrier();

    for (size_t i = item.thread_id(); i < hc_size; i += item.num_threads()) {
        x[i] = complement_negative(x[i]);
    }
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


// Header zero-bitmap is in scratch[0] afterwards
template<typename Bits>
Bits generate_zero_map(/* local */ const Bits *__restrict in,
        /* local */ Bits *__restrict scratch, work_item item) {
    constexpr auto n_columns = bitsof<Bits>;

    for (size_t i = 0; i < n_columns / warp_size; ++i) {
        // Indexing works around LLVM bug
        size_t ii = i * warp_size + item.thread_id();
        // TODO unroll reduction once instead of copying here
        scratch[ii] = in[ii];
    }

    item.local_memory_barrier();

    for (size_t offset = n_columns / 2; offset > 0; offset /= 2) {
        for (size_t i = item.thread_id(); i < n_columns; i += item.num_threads()) {
            if (i < offset) { scratch[i] |= scratch[i + offset]; }
        }
        item.local_memory_barrier();
    }

    return scratch[0];
}


template<typename Bits>
size_t compact_zero_words(/* global */ Bits *__restrict out, /* local */ const Bits *__restrict in,
        /* local */ Bits *__restrict scratch, work_item item) {
    constexpr auto n_columns = bitsof<Bits>;

    // printf("compact_zero_word gs=%lu ls=%lu lid=%lu nt=%lu tid=%lu\n", item.get_global_range(0),
    //         item.get_local_range(0), item.get_local_id(0), item.num_threads(), item.thread_id());

    item.local_memory_barrier();

    // for (size_t i = item.thread_id(); i < n_columns; i += item.num_threads()) {
    //     printf("compact_zero_word scratch_before [%lu] %d %d\n", i, int(in[i] != 0),
    //     (int)scratch[i]);
    // }
    // item.local_memory_barrier();

    // for (size_t i = item.thread_id(); i < n_columns; i += item.num_threads()) {
    for (size_t i = 0; i < n_columns / warp_size; ++i) {
        // Indexing works around LLVM bug
        size_t ii = i * warp_size + item.thread_id();
        scratch[ii] = in[ii] != 0;
    }

    item.local_memory_barrier();

    local_inclusive_prefix_sum(scratch, n_columns, item);

    // item.local_memory_barrier();

    for (size_t i = 0; i < n_columns / warp_size; ++i) {
        size_t ii = i * warp_size + item.thread_id();
        if (in[ii] != 0) {
            // Indexing works around LLVM bug
            size_t offset = ii ? scratch[ii - 1] : 0;
            out[offset] = in[ii];
        }
    }

    return scratch[n_columns - 1];
}


template<typename Bits>
size_t zero_bit_encode(/* local */ Bits *__restrict cube, /* global */ Bits *__restrict stream,
        /* local */ Bits *__restrict scratch, size_t hc_size, work_item item) {
    constexpr auto n_columns = bitsof<Bits>;

    auto out = stream;
    for (size_t offset = 0; offset < hc_size; offset += n_columns) {
        auto in = cube + offset;
        auto zero_map = generate_zero_map(in, scratch, item);
        if (item.thread_id() == 0) { out[0] = zero_map; }
        ++out;
        // TODO we *want* to do a `if (zero_map != 0)` check here, but that would cause divergence
        //  around barriers (UB) if there are multiple blocks per thread
        transpose_bits(in, item);
        item.local_memory_barrier();
        out += compact_zero_words(out, in, scratch, item);
    }

    return out - stream;
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
void compress_hypercubes(/* global */ const typename Profile::data_type *__restrict data,
        /* global */ typename Profile::bits_type *__restrict stream_chunks,
        /* global */ detail::file_offset_type *__restrict stream_chunk_lengths,
        /* local */ typename Profile::bits_type *__restrict local_memory,
        extent<Profile::dimensions> data_size, hypercube_range hc_range, hypercube_item item) {
    using bits_type = typename Profile::bits_type;

    const auto side_length = Profile::hypercube_side_length;
    const auto hc_size = detail::ipow(side_length, Profile::dimensions);
    const auto this_hc_local_memory = local_memory
            + hc_range.local_index_of(item) * local_memory_words_per_hypercube<Profile>;
    const auto cube = this_hc_local_memory;
    const auto scratch = this_hc_local_memory + hc_size;
    const auto max_chunk_size
            = (Profile::compressed_block_size_bound + sizeof(bits_type) - 1) / sizeof(bits_type);

    load_hypercube<Profile>(data, cube, data_size, hc_range, item);
    item.local_memory_barrier();
    block_transform<Profile>(cube, item);
    item.local_memory_barrier();
    stream_chunk_lengths[hc_range.global_index_of(item)] = zero_bit_encode<bits_type>(cube,
            stream_chunks + hc_range.global_index_of(item) * max_chunk_size, scratch, hc_size,
            item);
}


template<typename Profile>
void compact_stream(/* global */ typename Profile::bits_type *stream,
        /* global */ typename Profile::bits_type *stream_chunks,
        /* global */ const file_offset_type *stream_chunk_offsets,
        /* global */ const file_offset_type *stream_chunk_lengths, hypercube_range hc_range,
        hypercube_item item) {
    using bits_type = typename Profile::bits_type;

    const auto max_chunk_size
            = (Profile::compressed_block_size_bound + sizeof(bits_type) - 1) / sizeof(bits_type);
    const auto this_chunk_size = stream_chunk_lengths[hc_range.global_index_of(item)];
    const auto this_chunk_offset = hc_range.global_index_of(item)
            ? stream_chunk_offsets[hc_range.global_index_of(item) - 1]
            : 0;
    const auto source = stream_chunks + hc_range.global_index_of(item) * max_chunk_size;
    const auto dest = stream + this_chunk_offset;
    detail::gpu::nd_memcpy(dest, source, this_chunk_size, item);
}


using index_type = file_offset_type;

template<typename T>
using global_read = sycl::accessor<T, 1, sycl::access::mode::read>;
template<typename T>
using global_write = sycl::accessor<T, 1, sycl::access::mode::write>;
template<typename T>
using global_read_write = sycl::accessor<T, 1, sycl::access::mode::read_write>;


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


template<typename Profile>
struct hypercube {
    constexpr static unsigned dimensions = Profile::dimensions;
    constexpr static unsigned side_length = Profile::hypercube_side_length;
    constexpr static unsigned padding_every = 32;
    constexpr static index_type allocation_size
            = div_ceil(ipow(side_length, dimensions), padding_every) * (padding_every + 1);

    using bits_type = typename Profile::bits_type;
    using extent = ndzip::extent<dimensions>;

    bits_type *bits;

    bits_type &operator[](index_type linear_idx) {
        return bits[(linear_idx / padding_every) * (padding_every + 1)
                + linear_idx % padding_every];
    }

    bits_type &operator[](extent position) {
        return at(linear_offset(position, extent::broadcast(side_length)));
    }
};


template<typename F>
void for_range(sycl::group<1> grp, index_type known_group_size, index_type range, F &&f) {
    auto invoke_f = [&](index_type item, index_type iteration, sycl::logical_item<1> idx) {
        if constexpr (std::is_invocable_v<F, index_type, index_type, sycl::logical_item<1>>) {
            f(item, iteration, idx);
        } else if constexpr (std::is_invocable_v<F, index_type, index_type>) {
            f(item, iteration);
        } else {
            f(item);
        }
    };
    grp.distribute_for([&](sycl::sub_group, sycl::logical_item<1> idx) {
        const index_type num_full_iterations = range / known_group_size;
        const index_type partial_iteration_length = range % known_group_size;
        const auto tid = static_cast<index_type>(idx.get_local_id(0));
        for (index_type iteration = 0; iteration < num_full_iterations; ++iteration) {
            auto item = iteration * known_group_size + tid;
            invoke_f(item, iteration, idx);
        }
        if (tid < partial_iteration_length) {
            auto iteration = num_full_iterations;
            auto item = iteration * known_group_size + tid;
            invoke_f(item, iteration, idx);
        }
    });
}

template<typename F>
void for_range(sycl::group<1> grp, index_type range, F &&f) {
    for_range(grp, static_cast<index_type>(grp.get_local_range(0)), range, std::forward<F>(f));
}

inline const index_type hypercube_group_size = 64;

template<typename Profile, typename F>
void for_hypercube_indices(
        sycl::group<1> grp, index_type hc_index, extent<Profile::dimensions> data_size, F &&f) {
    const auto side_length = Profile::hypercube_side_length;
    const auto hc_size = ipow(side_length, Profile::dimensions);
    const auto hc_offset
            = detail::extent_from_linear_id(hc_index, data_size / side_length) * side_length;

    size_t initial_offset = linear_offset(hc_offset, data_size);
    for_range(grp, hypercube_group_size, hc_size, [&](index_type local_idx) {
        size_t global_idx = initial_offset + global_offset<Profile>(local_idx, data_size);
        f(global_idx, local_idx);
    });
}

template<typename Profile>
void load_hypercube(
        sycl::group<1> grp, index_type hc_index, grid<const Profile> grid, hypercube<Profile> hc) {
    using bits_type = typename Profile::bits_type;
    for_hypercube_indices<Profile>(
            grp, hc_index, grid.data.size(), [&](index_type global_idx, index_type local_idx) {
                hc[local_idx] = rotate_left_1(bit_cast<bits_type>(grid.data.data()[global_idx]));
                // TODO merge with block_transform to avoid LM round-trip?
                //  we could assign directly to private_memory
            });
}

template<typename Profile>
void store_hypercube(
        sycl::group<1> grp, index_type hc_index, grid<Profile> grid, hypercube<Profile> hc) {
    using data_type = typename Profile::data_type;
    for_hypercube_indices<Profile>(
            grp, hc_index, grid.data.size(), [&](index_type global_idx, index_type local_idx) {
                grid.data.data()[global_idx] = bit_cast<data_type>(rotate_right_1(hc[local_idx]));
            });
}


template<typename Profile>
void block_transform(sycl::group<1> grp, hypercube<Profile> hc) {
    using bits_type = typename Profile::bits_type;

    constexpr auto n = Profile::hypercube_side_length;
    constexpr auto n2 = n * n;
    constexpr auto dims = Profile::dimensions;
    constexpr auto hc_size = ipow(n, dims);

    sycl::private_memory<bits_type[hc_size / hypercube_group_size]> a{grp};

    for_range(grp, hypercube_group_size, hc_size,
            [&](index_type item, index_type iteration, sycl::logical_item<1> idx) {
                a(idx)[iteration] = hc[item];
            });
    for_range(grp, hypercube_group_size, hc_size,
            [&](index_type item, index_type iteration, sycl::logical_item<1> idx) {
                if (item % n != n - 1) { hc[item + 1] -= a(idx)[iteration]; }
            });

    if (dims >= 2) {
        for_range(grp, hypercube_group_size, hc_size,
                [&](index_type item, index_type iteration, sycl::logical_item<1> idx) {
                    a(idx)[iteration] = hc[item % n * n + item % n2 / n + item / n2 * n2];
                });
        for_range(grp, hypercube_group_size, hc_size,
                [&](index_type item, index_type iteration, sycl::logical_item<1> idx) {
                    if (item % n != n - 1) {
                        hc[item % n * n + item % n2 / n + item / n2 * n2 + n] -= a(idx)[iteration];
                    }
                });
    }

    if (dims >= 3) {
        for_range(grp, hypercube_group_size, hc_size,
                [&](index_type item, index_type iteration, sycl::logical_item<1> idx) {
                    a(idx)[iteration] = hc[item % n * n2 + item / n];
                });
        for_range(grp, hypercube_group_size, hc_size,
                [&](index_type item, index_type iteration, sycl::logical_item<1> idx) {
                    if (item % n != n - 1) {
                        hc[item % n * n2 + item / n + n2] -= a(idx)[iteration];
                    }
                });
    }

    // TODO move complement operation elsewhere to avoid local memory round-trip
    for_range(grp, hypercube_group_size, hc_size,
            [&](index_type item) { hc[item] = complement_negative(hc[item]); });
}


// TODO we might be able to avoid this kernel altogeher by writing the reduction results
// directly
//  to the stream header. Requires the stream format to be fixed first.
template<typename Profile>
void fill_stream_header(index_type num_hypercubes, global_write<stream_align_t> stream_acc,
        global_read<file_offset_type> offset_acc, sycl::handler &cgh) {
    const index_type group_size = 256;
    const index_type num_groups = div_ceil(num_hypercubes, group_size);
    cgh.parallel(sycl::range<1>{num_groups}, sycl::range<1>{group_size},
            [=](sycl::group<1> grp, sycl::physical_item<1>) {
                stream<Profile> stream{num_hypercubes, stream_acc.get_pointer()};
                const index_type base = static_cast<index_type>(grp.get_id(0)) * group_size;
                const index_type num_elements = std::min(group_size, num_hypercubes - base);
                for_range(grp, num_elements, [&](index_type i) {
                    stream.set_offset_after(base + i, offset_acc[base + i]);
                });
            });
}


template<typename Profile>
void compact_hypercubes(index_type num_hypercubes, global_write<stream_align_t> stream_acc,
        global_read<typename Profile::bits_type> chunks_acc, sycl::handler &cgh) {
    constexpr index_type num_threads_per_hc = 64;
    cgh.parallel(sycl::range<1>{num_hypercubes}, sycl::range<1>{num_threads_per_hc},
            [=](sycl::group<1> grp, sycl::physical_item<1>) {
                auto hc_index = static_cast<index_type>(grp.get_id(0));
                compressed_chunks<Profile> chunks{num_hypercubes, chunks_acc.get_pointer()};
                auto in = chunks.hypercube(hc_index);
                stream<Profile> stream{num_hypercubes, stream_acc.get_pointer()};
                auto out = stream.hypercube(hc_index);
                for_range(grp, stream.hypercube_size(hc_index),
                        [&](index_type i) { out[i] = in[i]; });
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
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;
    using sam = sycl::access::mode;

    const auto max_chunk_size
            = (profile::compressed_block_size_bound + sizeof(bits_type) - 1) / sizeof(bits_type);

    // TODO edge case w/ 0 hypercubes

    detail::file<profile> file(data.size());
    if (auto env = getenv("NDZIP_VERBOSE"); env && *env) {
        printf("Have %zu hypercubes\n", file.num_hypercubes());
    }

    sycl::buffer<data_type, dimensions> data_buffer{
            detail::gpu::extent_cast<sycl::range<dimensions>>(data.size())};
    sycl::buffer<bits_type, 1> stream_chunks_buffer{
            sycl::range<1>{file.num_hypercubes() * max_chunk_size}};
    sycl::buffer<detail::file_offset_type, 1> stream_chunk_lengths_buffer{
            sycl::range<1>{file.num_hypercubes()}};

    detail::gpu::submit_and_profile(_pimpl->q, "copy input to device", [&](sycl::handler &cgh) {
        cgh.copy(data.data(), data_buffer.template get_access<sam::discard_write>(cgh));
    });

    auto local_memory_words_per_hc = detail::gpu::local_memory_words_per_hypercube<profile>;
    auto max_hcs_per_work_group = detail::gpu::max_work_group_size(
            local_memory_words_per_hc * sizeof(bits_type), _pimpl->q.get_device());

    detail::gpu::submit_and_profile(_pimpl->q, "block compression", [&](sycl::handler &cgh) {
        auto data_acc = data_buffer.template get_access<sam::read>(cgh);
        auto stream_chunks_acc
                = stream_chunks_buffer.template get_access<sam::discard_read_write>(cgh);
        auto stream_chunk_lengths_acc
                = stream_chunk_lengths_buffer.get_access<sam::discard_write>(cgh);
        auto local_memory_acc = detail::gpu::local_accessor<bits_type>{
                local_memory_words_per_hc * max_hcs_per_work_group, cgh};
        auto data_size = data.size();
        auto hc_range = detail::gpu::hypercube_range{file.num_hypercubes(), max_hcs_per_work_group};
        cgh.parallel_for<detail::gpu::block_compression_kernel<T, Dims>>(
                hc_range.item_space(), [=](detail::gpu::hypercube_item item) {
                    if (hc_range.contains(item)) {
                        detail::gpu::compress_hypercubes<profile>(data_acc.get_pointer(),
                                stream_chunks_acc.get_pointer(),
                                stream_chunk_lengths_acc.get_pointer(),
                                local_memory_acc.get_pointer(), data_size, hc_range, item);
                    }
                });
    });

    sycl::buffer<detail::file_offset_type> stream_chunk_offsets_buffer{
            sycl::range<1>{file.num_hypercubes()}};
    detail::gpu::submit_and_profile(_pimpl->q, "dup lengths", [&](sycl::handler &cgh) {
        cgh.copy(stream_chunk_lengths_buffer.get_access<sam::read>(cgh),
                stream_chunk_offsets_buffer.get_access<sam::discard_write>(cgh));
    });

    detail::gpu::hierarchical_inclusive_prefix_sum<detail::file_offset_type> prefix_sum(
            file.num_hypercubes(), 256 /* local size */);
    prefix_sum(_pimpl->q, stream_chunk_offsets_buffer);

    detail::file_offset_type num_compressed_words;
    auto num_compressed_words_available = _pimpl->q.submit([&](sycl::handler &cgh) {
        cgh.copy(stream_chunk_offsets_buffer.template get_access<sam::read>(
                         cgh, sycl::range<1>{1}, sycl::id<1>{file.num_hypercubes() - 1}),
                &num_compressed_words);
    });

    sycl::buffer<detail::gpu::stream_align_t> stream_buffer(
            (compressed_size_bound<data_type, dimensions>(data.size())
                    + sizeof(detail::gpu::stream_align_t) - 1)
            / sizeof(detail::gpu::stream_align_t));

    detail::gpu::submit_and_profile(_pimpl->q, "fill header", [&](sycl::handler &cgh) {
        detail::gpu::fill_stream_header<profile>(file.num_hypercubes(),
                stream_buffer.get_access<sam::write>(cgh),  // TODO limit access range
                stream_chunk_offsets_buffer.get_access<sam::read>(cgh), cgh);
    });

    detail::gpu::submit_and_profile(_pimpl->q, "compact chunks", [&](sycl::handler &cgh) {
        detail::gpu::compact_hypercubes<profile>(file.num_hypercubes(),
                stream_buffer.get_access<sam::write>(cgh),  // TODO limit access range
                stream_chunks_buffer.template get_access<sam::read>(cgh), cgh);
    });

    num_compressed_words_available.wait();
    auto stream_pos = file.file_header_length() + num_compressed_words * sizeof(bits_type);
    auto n_aligned_words = (stream_pos + sizeof(detail::gpu::stream_align_t) - 1)
            / sizeof(detail::gpu::stream_align_t);
    auto stream_transferred = detail::gpu::submit_and_profile(
            _pimpl->q, "copy stream to host", [&](sycl::handler &cgh) {
                cgh.copy(stream_buffer.get_access<sam::read>(cgh, n_aligned_words),
                        static_cast<detail::gpu::stream_align_t *>(stream));
            });

    stream_transferred.wait();  // TODO I need to wait since I'm potentially overwriting the border
                                //  in the aligned copy
    stream_pos += detail::pack_border(
            static_cast<char *>(stream) + stream_pos, data, profile::hypercube_side_length);

    return stream_pos;
}


template<typename T, unsigned Dims>
size_t ndzip::gpu_encoder<T, Dims>::decompress(
        const void *stream, size_t bytes, const slice<data_type, dimensions> &data) const {
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;
    using sam = sycl::access::mode;

    detail::file<profile> file(data.size());

    auto local_memory_words_per_hc = detail::gpu::local_memory_words_per_hypercube<profile>;
    auto max_hcs_per_work_group = detail::gpu::max_work_group_size(
            local_memory_words_per_hc * sizeof(bits_type), _pimpl->q.get_device());

    // TODO the range computation here is questionable at best
    sycl::buffer<bits_type, 1> stream_buffer{sycl::range<1>{bytes / sizeof(bits_type)}};
    sycl::buffer<data_type, dimensions> data_buffer{
            detail::gpu::extent_cast<sycl::range<dimensions>>(data.size())};

    detail::gpu::submit_and_profile(_pimpl->q, "copy stream to device", [&](sycl::handler &cgh) {
        cgh.copy(static_cast<const bits_type *>(stream),
                stream_buffer.template get_access<sam::discard_write>(cgh));
    });

    detail::gpu::submit_and_profile(_pimpl->q, "decompress blocks", [&](sycl::handler &cgh) {
        auto stream_acc = stream_buffer.template get_access<sam::read>(cgh);
        auto data_acc = data_buffer.template get_access<sam::discard_write>(cgh);
        auto local_memory_acc = detail::gpu::local_accessor<bits_type>{
                local_memory_words_per_hc * max_hcs_per_work_group, cgh};
        auto data_size = data.size();
        auto hc_range = detail::gpu::hypercube_range{file.num_hypercubes(), max_hcs_per_work_group};
        cgh.parallel_for<detail::gpu::stream_decompression_kernel<T, Dims>>(
                hc_range.item_space(), [=](detail::gpu::hypercube_item item) {
                    if (hc_range.contains(item)) {
                        detail::gpu::decompress_hypercubes<profile>(stream_acc.get_pointer(),
                                data_acc.get_pointer(), local_memory_acc.get_pointer(), data_size,
                                hc_range, item);
                    }
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
