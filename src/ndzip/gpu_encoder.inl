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


template<typename Profile>
struct hypercube {
    constexpr static unsigned dimensions = Profile::dimensions;
    constexpr static unsigned side_length = Profile::hypercube_side_length;
    constexpr static index_type allocation_size = dimensions == 1 ? side_length
            : dimensions == 2
            ? side_length * (side_length + 1)
            : ipow(side_length, 3) + ipow(side_length, 3) / warp_size + side_length;

    using bits_type = typename Profile::bits_type;
    using extent = ndzip::extent<dimensions>;

    bits_type *bits;

    bits_type &operator[](index_type linear_idx) const {
        index_type pads = 0;
        if constexpr (dimensions == 2) { pads = linear_idx / side_length; }
        if constexpr (dimensions == 3) {
            pads = linear_idx / warp_size + linear_idx / ipow(side_length, 2);
        }
        return bits[linear_idx + pads];
    }
};

template<typename Profile, unsigned Direction>
struct directional_hypercube_accessor;

template<typename Profile>
struct directional_hypercube_accessor<Profile, 1> {
    hypercube<Profile> hc;
    index_type offset = 0;

    constexpr static index_type stride = 1;

    index_type linear_index(index_type i) const { return i; }

    typename Profile::bits_type &operator[](index_type i) const { return hc[i]; }
};

template<typename Profile>
struct directional_hypercube_accessor<Profile, 2> {
    hypercube<Profile> hc;
    index_type offset = 0;

    constexpr static index_type stride = Profile::hypercube_side_length;

    index_type linear_index(index_type i) const {
        constexpr auto n = Profile::hypercube_side_length;
        if constexpr (Profile::dimensions == 2) {
            return i % n * n + i / n;
        } else /* Profile::dimensions == 3 */ {
            // TODO include padding in LUT, do not calculate repeatedly from hc::operator[]
            static const uint16_t start_lut[] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7,
                    15, 256, 264, 257, 265, 258, 266, 259, 267, 260, 268, 261, 269, 262, 270, 263,
                    271, 512, 520, 513, 521, 514, 522, 515, 523, 516, 524, 517, 525, 518, 526, 519,
                    527, 768, 776, 769, 777, 770, 778, 771, 779, 772, 780, 773, 781, 774, 782, 775,
                    783, 1024, 1032, 1025, 1033, 1026, 1034, 1027, 1035, 1028, 1036, 1029, 1037,
                    1030, 1038, 1031, 1039, 1280, 1288, 1281, 1289, 1282, 1290, 1283, 1291, 1284,
                    1292, 1285, 1293, 1286, 1294, 1287, 1295, 1536, 1544, 1537, 1545, 1538, 1546,
                    1539, 1547, 1540, 1548, 1541, 1549, 1542, 1550, 1543, 1551, 1792, 1800, 1793,
                    1801, 1794, 1802, 1795, 1803, 1796, 1804, 1797, 1805, 1798, 1806, 1799, 1807,
                    2048, 2056, 2049, 2057, 2050, 2058, 2051, 2059, 2052, 2060, 2053, 2061, 2054,
                    2062, 2055, 2063, 2304, 2312, 2305, 2313, 2306, 2314, 2307, 2315, 2308, 2316,
                    2309, 2317, 2310, 2318, 2311, 2319, 2560, 2568, 2561, 2569, 2562, 2570, 2563,
                    2571, 2564, 2572, 2565, 2573, 2566, 2574, 2567, 2575, 2816, 2824, 2817, 2825,
                    2818, 2826, 2819, 2827, 2820, 2828, 2821, 2829, 2822, 2830, 2823, 2831, 3072,
                    3080, 3073, 3081, 3074, 3082, 3075, 3083, 3076, 3084, 3077, 3085, 3078, 3086,
                    3079, 3087, 3328, 3336, 3329, 3337, 3330, 3338, 3331, 3339, 3332, 3340, 3333,
                    3341, 3334, 3342, 3335, 3343, 3584, 3592, 3585, 3593, 3586, 3594, 3587, 3595,
                    3588, 3596, 3589, 3597, 3590, 3598, 3591, 3599, 3840, 3848, 3841, 3849, 3842,
                    3850, 3843, 3851, 3844, 3852, 3845, 3853, 3846, 3854, 3847, 3855};

            index_type b = i / n;
            index_type start = start_lut[b];
            return start + i % n * n;
        }
    }

    typename Profile::bits_type &operator[](index_type i) const { return hc[linear_index(i)]; }
};

template<typename Profile>
struct directional_hypercube_accessor<Profile, 3> {
    hypercube<Profile> hc;
    index_type offset = 0;

    constexpr static index_type hc_size = ipow(Profile::hypercube_side_length, 3);
    constexpr static index_type stride = ipow(Profile::hypercube_side_length, 2);

    index_type linear_index(index_type i) const {
        constexpr auto n = Profile::hypercube_side_length;
        // TODO include padding in LUT, do not calculate repeatedly from hc::operator[]
        static const uint16_t start_lut[] = {0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176,
                192, 208, 224, 240, 1, 17, 33, 49, 65, 81, 97, 113, 129, 145, 161, 177, 193, 209,
                225, 241, 2, 18, 34, 50, 66, 82, 98, 114, 130, 146, 162, 178, 194, 210, 226, 242, 3,
                19, 35, 51, 67, 83, 99, 115, 131, 147, 163, 179, 195, 211, 227, 243, 4, 20, 36, 52,
                68, 84, 100, 116, 132, 148, 164, 180, 196, 212, 228, 244, 5, 21, 37, 53, 69, 85,
                101, 117, 133, 149, 165, 181, 197, 213, 229, 245, 6, 22, 38, 54, 70, 86, 102, 118,
                134, 150, 166, 182, 198, 214, 230, 246, 7, 23, 39, 55, 71, 87, 103, 119, 135, 151,
                167, 183, 199, 215, 231, 247, 8, 24, 40, 56, 72, 88, 104, 120, 136, 152, 168, 184,
                200, 216, 232, 248, 9, 25, 41, 57, 73, 89, 105, 121, 137, 153, 169, 185, 201, 217,
                233, 249, 10, 26, 42, 58, 74, 90, 106, 122, 138, 154, 170, 186, 202, 218, 234, 250,
                11, 27, 43, 59, 75, 91, 107, 123, 139, 155, 171, 187, 203, 219, 235, 251, 12, 28,
                44, 60, 76, 92, 108, 124, 140, 156, 172, 188, 204, 220, 236, 252, 13, 29, 45, 61,
                77, 93, 109, 125, 141, 157, 173, 189, 205, 221, 237, 253, 14, 30, 46, 62, 78, 94,
                110, 126, 142, 158, 174, 190, 206, 222, 238, 254, 15, 31, 47, 63, 79, 95, 111, 127,
                143, 159, 175, 191, 207, 223, 239, 255};

        index_type b = i / n;
        index_type start = start_lut[b];
        return start + i % n * n * n;
    }

    typename Profile::bits_type &operator[](index_type i) const { return hc[linear_index(i)]; }
};

// Fine tuning block size. For block transform:
//    -  double precision, 256 >> 128
//    - single precision 1D, 512 >> 128 > 256.
//    - single precision forward 1D 2D, 512 >> 256.
// At least for sm_61 profile<double, 3d> exceeds maximum register usage with 512
inline constexpr index_type hypercube_group_size = 256;
using hypercube_group = known_size_group<hypercube_group_size>;

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


template<typename Profile, typename Accessor>
void forward_transform_step(hypercube_group grp, Accessor acc) {
    using bits_type = typename Profile::bits_type;
    constexpr auto n = Profile::hypercube_side_length;
    constexpr index_type hc_size = ipow(n, Profile::dimensions);

    sycl::private_memory<bits_type[div_ceil(hc_size, hypercube_group_size)]> linear{grp};
    sycl::private_memory<bits_type[div_ceil(hc_size, hypercube_group_size)]> a{grp};
    grp.distribute_for<hc_size>(
            [&](index_type item, index_type iteration, sycl::logical_item<1> idx) {
                linear(idx)[iteration] = acc.linear_index(item);
                a(idx)[iteration] = acc.hc[linear(idx)[iteration]];
            });
    grp.distribute_for<hc_size>(
            [&](index_type item, index_type iteration, sycl::logical_item<1> idx) {
                if (item % n != n - 1) {
                    acc.hc[linear(idx)[iteration] + Accessor::stride] -= a(idx)[iteration];
                }
            });
}


template<typename Profile>
void block_transform(hypercube_group grp, hypercube<Profile> hc) {
    constexpr auto dims = Profile::dimensions;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, dims);

    forward_transform_step<Profile>(grp, directional_hypercube_accessor<Profile, 1>{hc});
    if constexpr (dims >= 2) {
        forward_transform_step<Profile>(grp, directional_hypercube_accessor<Profile, 2>{hc});
    }
    if constexpr (dims >= 3) {
        forward_transform_step<Profile>(grp, directional_hypercube_accessor<Profile, 3>{hc});
    }

    // TODO move complement operation elsewhere to avoid local memory round-trip
    grp.distribute_for(hc_size, [&](index_type item) { hc[item] = complement_negative(hc[item]); });
}


template<typename Accessor>
void inverse_transform_lanes(
        hypercube_group grp, index_type n_lanes, index_type lane_length, Accessor acc) {
    grp.distribute_for(n_lanes, [&](index_type i) {
        // TODO this is almost exactly common.hh / inverse_transform_step => to "detail::seq::"
        auto a = acc[i * lane_length];
        for (size_t j = 1; j < lane_length; ++j) {
            a += acc[i * lane_length + j];
            acc[i * lane_length + j] = a;
        }
    });
}


template<typename Profile>
void inverse_block_transform(hypercube_group grp, hypercube<Profile> hc) {
    using bits_type = typename Profile::bits_type;
    constexpr auto dims = Profile::dimensions;
    constexpr auto n = Profile::hypercube_side_length;
    constexpr auto n2 = n * n;
    constexpr index_type hc_size = ipow(n, dims);

    // TODO move complement operation elsewhere to avoid local memory round-trip
    grp.distribute_for(hc_size, [&](index_type item) { hc[item] = complement_negative(hc[item]); });

    // TODO how to do 2D / 3D?
    //  - For 2D we have 64 parallel work items but we _probably_ want at least 256 threads per SM
    //    (= per HC for double) to hide latencies. Maybe hybrid approach - do (64/32)*64 sub-group
    //    prefix sums and optimize inclusive_prefix_sum to skip the recursion since the second
    //    level does not actually need a reduction. Benchmark against leaving 256-64 = 192 threads
    //    idle and going with a sequential-per-lane transform.
    //  - For 3D we have 256 parallel work items, so implementing that w/o prefix sum should be
    //    the most efficient

    if (dims == 1) { inclusive_scan<n>(grp, hc, sycl::plus<bits_type>{}); }
    if (dims == 2) {
        // TODO inefficient, see above
        inverse_transform_lanes(grp, n, n, directional_hypercube_accessor<Profile, 1>{hc});
        inverse_transform_lanes(grp, n, n, directional_hypercube_accessor<Profile, 2>{hc});
    }
    if (dims == 3) {
        inverse_transform_lanes(grp, n2, n, directional_hypercube_accessor<Profile, 1>{hc});
        inverse_transform_lanes(grp, n2, n, directional_hypercube_accessor<Profile, 2>{hc});
        inverse_transform_lanes(grp, n2, n, directional_hypercube_accessor<Profile, 3>{hc});
    }
}

template<typename Profile>
void write_transposed_chunks(hypercube_group grp, hypercube<Profile> hc,
        typename Profile::bits_type *out_heads, typename Profile::bits_type *out_columns,
        index_type *out_lengths) {
    using bits_type = typename Profile::bits_type;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    static_assert(hc_size % warp_size == 0);

    // One group per warp (for subgroup reductions)
    constexpr index_type chunk_size = bitsof<bits_type>;

    grp.distribute_for(hc_size,
            [&](index_type item, index_type iteration, sycl::logical_item<1> idx,
                    sycl::sub_group sg) {
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
                    head |= sycl::group_reduce(sg, hc[col] & mask, sycl::bit_or<bits_type>{});
                }

                index_type this_warp_size = 0;
                bits_type column = 0;
                if (head != 0) {
                    const auto chunk_base = floor(item, chunk_size);
                    const auto cell = item - chunk_base;
                    for (index_type i = 0; i < chunk_size; ++i) {
                        // TODO for double, can we still operate on 32 bit words? e.g split into
                        //  low / high loop
                        column |= (hc[chunk_base + i] >> (chunk_size - 1 - cell) & bits_type{1})
                                << (chunk_size - 1 - i);
                    }
                    if constexpr (sizeof(bits_type) == 4) {
                        this_warp_size = __builtin_popcount(head);
                    } else {
                        this_warp_size = __builtin_popcountl(head);
                    }
                    auto base = floor(item, warp_size);
                    auto relative_pos = sycl::group_exclusive_scan(
                            sg, index_type{column != 0}, sycl::plus<index_type>{});
                    if (column != 0) { out_columns[base + relative_pos] = column; }
                }
                if (warp_index % (chunk_size / warp_size) == 0) { this_warp_size += 1; }
                if (sg.leader()) {
                    // TODO collect in local memory, write coalesced - otherwise 3 full GM
                    //  transaction per HC instead of 1!
                    out_heads[warp_index] = head;
                    out_lengths[warp_index] = this_warp_size;
                }
            });
}


template<typename Bits>
void compact_chunks(sycl::group<1> grp, const Bits *heads, const Bits *columns,
        const index_type *offsets, Bits *stream) {
    // One group per warp (for subgroup reductions)
    constexpr index_type chunk_size = bitsof<Bits>;

    grp.distribute_for([&](sycl::sub_group sg, sycl::logical_item<1> idx) {
        auto item = idx.get_global_id(0);
        auto warp_index = item / warp_size;

        auto offset = offsets[warp_index];
        if (warp_index % (chunk_size / warp_size) == 0) {
            Bits head = 0;
            for (index_type i = 0; i < (chunk_size / warp_size); ++i) {
                head |= heads[warp_index + i];
            }
            if (sg.leader()) {
                stream[offset] = head;
            }
            offset += 1;
        }
        index_type tid = sg.get_local_id()[0];
        if (offset + tid < offsets[warp_index + 1]) {
            stream[offset + tid] = columns[item];
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
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;
    using sam = sycl::access::mode;

    const auto max_chunk_size
            = (profile::compressed_block_size_bound + sizeof(bits_type) - 1) / sizeof(bits_type);
    constexpr detail::gpu::index_type hc_size
            = detail::ipow(profile::hypercube_side_length, profile::dimensions);
    constexpr detail::gpu::index_type warps_per_hc = hc_size / detail::gpu::warp_size;

    // TODO edge case w/ 0 hypercubes

    detail::file<profile> file(data.size());
    auto num_hypercubes = file.num_hypercubes();
    if (auto env = getenv("NDZIP_VERBOSE"); env && *env) {
        printf("Have %zu hypercubes\n", num_hypercubes);
    }

    sycl::buffer<data_type, dimensions> data_buffer{
            detail::gpu::extent_cast<sycl::range<dimensions>>(data.size())};

    detail::gpu::submit_and_profile(_pimpl->q, "copy input to device", [&](sycl::handler &cgh) {
        cgh.copy(data.data(), data_buffer.template get_access<sam::discard_write>(cgh));
    });

    sycl::buffer<bits_type> columns_buf(num_hypercubes * hc_size);
    sycl::buffer<bits_type> heads_buf(num_hypercubes * warps_per_hc);
    sycl::buffer<detail::gpu::index_type> chunk_lengths_buf(1 + num_hypercubes * warps_per_hc);

    detail::gpu::submit_and_profile(_pimpl->q, "transform + chunk encode", [&](sycl::handler &cgh) {
        auto data_acc = data_buffer.template get_access<sam::read>(cgh);
        auto heads_acc = heads_buf.template get_access<sam::discard_write>(cgh);
        auto columns_acc = columns_buf.template get_access<sam::discard_write>(cgh);
        auto chunk_lengths_acc = chunk_lengths_buf.get_access<sam::discard_write>(cgh);
        auto data_size = data.size();
        cgh.parallel<detail::gpu::block_compression_kernel<T, Dims>>(
                sycl::range<1>{file.num_hypercubes()},
                sycl::range<1>{detail::gpu::hypercube_group_size},
                [=](detail::gpu::hypercube_group grp, sycl::physical_item<1> phys_idx) {
                    detail::gpu::index_type hc_index = grp.get_id(0);
                    slice<const data_type, dimensions> data{data_acc.get_pointer(), data_size};
                    sycl::local_memory<bits_type[detail::gpu::hypercube<profile>::allocation_size]>
                            lm{grp};
                    detail::gpu::hypercube<profile> hc{&lm[0]};

                    detail::gpu::load_hypercube(grp, hc_index, {data}, hc);
                    detail::gpu::block_transform(grp, hc);
                    detail::gpu::write_transposed_chunks(grp, hc,
                            &heads_acc[hc_index * warps_per_hc], &columns_acc[hc_index * hc_size],
                            &chunk_lengths_acc[1 + hc_index * warps_per_hc]);
                    // hack
                    if (phys_idx.get_global_linear_id() == 0) {
                        grp.single_item([&] { chunk_lengths_acc[0] = 0; });
                    }
                });
    });

    std::vector<detail::gpu::index_type> dbg_lengths(chunk_lengths_buf.get_range()[0]);
    _pimpl->q.submit([&](sycl::handler &cgh) {
                 cgh.copy(chunk_lengths_buf.get_access<sam::read>(cgh), dbg_lengths.data());
    }).wait();
    detail::gpu::hierarchical_inclusive_prefix_sum<detail::gpu::index_type> prefix_sum(
            1 + num_hypercubes * warps_per_hc, 256 /* local size */);
    prefix_sum(_pimpl->q, chunk_lengths_buf);
    std::vector<detail::gpu::index_type> dbg_offsets(chunk_lengths_buf.get_range()[0]);
    _pimpl->q.submit([&](sycl::handler &cgh) {
      cgh.copy(chunk_lengths_buf.get_access<sam::read>(cgh), dbg_offsets.data());
    }).wait();

    detail::gpu::index_type num_compressed_words;
    auto num_compressed_words_available = _pimpl->q.submit([&](sycl::handler &cgh) {
        cgh.copy(chunk_lengths_buf.template get_access<sam::read>(
                         cgh, sycl::range<1>{1}, sycl::id<1>{num_hypercubes * warps_per_hc}),
                &num_compressed_words);
    });

    sycl::buffer<detail::gpu::stream_align_t> stream_buf(
            (compressed_size_bound<data_type, dimensions>(data.size())
                    + sizeof(detail::gpu::stream_align_t) - 1)
            / sizeof(detail::gpu::stream_align_t));

    detail::gpu::submit_and_profile(_pimpl->q, "fill header", [&](sycl::handler &cgh) {
        detail::gpu::fill_stream_header<profile>(num_hypercubes,
                stream_buf.get_access<sam::write>(cgh),  // TODO limit access range
                chunk_lengths_buf.get_access<sam::read>(cgh), cgh);
    });

    detail::gpu::submit_and_profile(_pimpl->q, "compact chunks", [&](sycl::handler &cgh) {
        auto columns_acc = columns_buf.template get_access<sam::read>(cgh);
        auto heads_acc = heads_buf.template get_access<sam::read>(cgh);
        auto offsets_acc = chunk_lengths_buf.template get_access<sam::read>(cgh);
        auto stream_acc = stream_buf.template get_access<sam::discard_write>(cgh);
        constexpr size_t group_size = 1024;
        const size_t header_offset
                = file.file_header_length() / sizeof(detail::gpu::stream_align_t);
        cgh.parallel<detail::gpu::stream_compaction_kernel<T, Dims>>(
                sycl::range<1>{hc_size / group_size * num_hypercubes}, sycl::range<1>{group_size},
                [=](sycl::group<1> grp, sycl::physical_item<1>) {
                    detail::gpu::compact_chunks(grp,
                            static_cast<const bits_type *>(heads_acc.get_pointer()),
                            static_cast<const bits_type *>(columns_acc.get_pointer()),
                            static_cast<const detail::gpu::index_type *>(offsets_acc.get_pointer()),
                            reinterpret_cast<bits_type *>(
                                    static_cast<detail::gpu::stream_align_t *>(
                                            stream_acc.get_pointer())
                                    + header_offset));
                });
    });

    num_compressed_words_available.wait();
    auto stream_pos = file.file_header_length() + num_compressed_words * sizeof(bits_type);

    auto n_aligned_words = (stream_pos + sizeof(detail::gpu::stream_align_t) - 1)
            / sizeof(detail::gpu::stream_align_t);
    auto stream_transferred = detail::gpu::submit_and_profile(
            _pimpl->q, "copy stream to host", [&](sycl::handler &cgh) {
                cgh.copy(stream_buf.get_access<sam::read>(cgh, n_aligned_words),
                        static_cast<detail::gpu::stream_align_t *>(stream));
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
