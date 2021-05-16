#pragma once

#include "common.hh"


namespace ndzip::detail::gpu {


// TODO _should_ be a template parameter with selection based on queue (/device?) properties.
//  However a lot of code currently assumes that bitsof<uint32_t> == warp_size (e.g. we want to
//  use subgroup reductions for length-32 chunks of residuals)
inline constexpr index_type warp_size = 32;

// A memory bank on CUDA is 32-bit. This used to be configurable but is not any more.
// Hypercube layouts for 64-bit values need to introduce misalignment to pad shared memory accesses.
using uint_bank_t = uint32_t;

template<typename Bits>
inline constexpr index_type banks_of = bytes_of<Bits> / bytes_of<uint_bank_t>;


template<typename Profile>
NDZIP_UNIVERSAL index_type global_offset(
        index_type local_offset, extent<Profile::dimensions> global_size) {
    index_type global_offset = 0;
    index_type global_stride = 1;
    for (unsigned d = 0; d < Profile::dimensions; ++d) {
        global_offset += global_stride * (local_offset % Profile::hypercube_side_length);
        local_offset /= Profile::hypercube_side_length;
        global_stride *= global_size[Profile::dimensions - 1 - d];
    }
    return global_offset;
}


// We want to maintain a fixed number of threads per SM to control occupancy. Occupancy is
// primarily limited by local memory usage, so we adjust the group size to keep local memory
// requirement constant -- 256 threads/group for 32 bit, 512 threads/group for 64 bit.
template<typename Profile>
#ifdef NDZIP_GPU_GROUP_SIZE
inline constexpr index_type hypercube_group_size = NDZIP_GPU_GROUP_SIZE;
#else
inline constexpr index_type hypercube_group_size
        = bytes_of<typename Profile::bits_type> == 4 ? 256 : 512;
#endif

struct forward_transform_tag;
struct inverse_transform_tag;

template<typename Profile, typename Transform>
struct hypercube_layout;

template<typename Profile, unsigned Direction, typename Transform>
struct directional_accessor;

// std::optional is not allowed in kernels
inline constexpr index_type no_such_lane = ~index_type{};

template<typename T>
struct hypercube_layout<profile<T, 1>, forward_transform_tag> {
    constexpr static index_type side_length = 4096;
    constexpr static index_type hc_size = 4096;
    constexpr static index_type num_lanes = floor_power_of_two(hypercube_group_size<profile<T, 1>>);
    constexpr static index_type lane_length = hc_size / num_lanes;
    constexpr static index_type value_width = banks_of<typename profile<T, 1>::bits_type>;

    NDZIP_UNIVERSAL constexpr static index_type pad(index_type i) {
        return value_width * i + value_width * i / warp_size;
    }
};

template<typename T>
struct hypercube_layout<profile<T, 1>, inverse_transform_tag> {
    constexpr static index_type side_length = 4096;
    constexpr static index_type hc_size = 4096;
    constexpr static index_type num_lanes = 1;
    constexpr static index_type lane_length = hc_size;
    constexpr static index_type value_width = banks_of<typename profile<T, 1>::bits_type>;

    // Special case: 1D inverse transform uses prefix sum, which is optimal without padding.
};

template<typename T>
struct directional_accessor<profile<T, 1>, 0, forward_transform_tag> {
    using layout = hypercube_layout<profile<T, 1>, forward_transform_tag>;

    NDZIP_UNIVERSAL constexpr static index_type prev_lane_in_row(index_type lane) {
        return lane > 0 ? lane - 1 : no_such_lane;
    }

    NDZIP_UNIVERSAL constexpr static index_type offset(index_type lane) {
        return lane * layout::hc_size / layout::num_lanes;
    }

    static inline const index_type stride = 1;
};

// 1D inverse transform is a parallel scan, so no directional accessor is implemented.

template<typename T>
struct hypercube_layout<profile<T, 2>, forward_transform_tag> {
    constexpr static index_type num_lanes = floor_power_of_two(hypercube_group_size<profile<T, 2>>);
    constexpr static index_type side_length = 64;
    constexpr static index_type hc_size = 64 * 64;
    constexpr static index_type lane_length = hc_size / num_lanes;
    constexpr static index_type value_width = banks_of<typename profile<T, 2>::bits_type>;

    NDZIP_UNIVERSAL constexpr static index_type pad(index_type i) {
        return value_width * i + value_width * i / warp_size;
    }
};

template<typename T>
struct directional_accessor<profile<T, 2>, 0, forward_transform_tag> {
    using layout = hypercube_layout<profile<T, 2>, forward_transform_tag>;

    NDZIP_UNIVERSAL constexpr static index_type prev_lane_in_row(index_type lane) {
        if (lane % (layout::side_length / layout::lane_length) > 0) {
            return lane - 1;
        } else {
            return no_such_lane;
        }
    }
    NDZIP_UNIVERSAL constexpr static index_type offset(index_type lane) {
        return lane * (layout::hc_size / layout::num_lanes);
    }
    constexpr static index_type stride = 1;
};

template<typename T>
struct directional_accessor<profile<T, 2>, 1, forward_transform_tag> {
    using layout = hypercube_layout<profile<T, 2>, forward_transform_tag>;

    NDZIP_UNIVERSAL constexpr static index_type prev_lane_in_row(index_type lane) {
        if (lane >= layout::side_length) {
            return lane - layout::side_length;
        } else {
            return no_such_lane;
        }
    }
    NDZIP_UNIVERSAL constexpr static index_type offset(index_type lane) {
        return (lane / layout::side_length)
                * (layout::hc_size / layout::num_lanes * layout::side_length)
                + lane % layout::side_length;
    }
    constexpr static index_type stride = layout::side_length;
};

template<typename T>
struct hypercube_layout<profile<T, 2>, inverse_transform_tag> {
    constexpr static index_type side_length = 64;
    constexpr static index_type hc_size = 64 * 64;
    constexpr static index_type num_lanes = side_length;
    constexpr static index_type lane_length = hc_size / num_lanes;
    constexpr static index_type value_width = banks_of<typename profile<T, 2>::bits_type>;

    NDZIP_UNIVERSAL constexpr static index_type pad(index_type i) {
        if constexpr (value_width == 1) {
            return i + i / side_length;
        } else {
            return value_width * i + i / (warp_size / value_width) - i / side_length;
        }
    }
};

template<typename T>
struct directional_accessor<profile<T, 2>, 0, inverse_transform_tag> {
    using layout = hypercube_layout<profile<T, 2>, inverse_transform_tag>;
    NDZIP_UNIVERSAL constexpr static index_type prev_lane_in_row(index_type) {
        return no_such_lane;
    }
    NDZIP_UNIVERSAL constexpr static index_type offset(index_type lane) {
        return lane * layout::side_length;
    }
    constexpr static index_type stride = 1;
};

template<typename T>
struct directional_accessor<profile<T, 2>, 1, inverse_transform_tag> {
    using layout = hypercube_layout<profile<T, 2>, inverse_transform_tag>;
    NDZIP_UNIVERSAL constexpr static index_type prev_lane_in_row(index_type) {
        return no_such_lane;
    }
    NDZIP_UNIVERSAL constexpr static index_type offset(index_type lane) {
        return lane % layout::side_length;
    }
    constexpr static index_type stride = layout::side_length;
};

template<typename T, typename Transform>
struct hypercube_layout<profile<T, 3>, Transform> {
    constexpr static index_type side_length = 16;
    constexpr static index_type hc_size = ipow(side_length, 3);
    // TODO implement support for forward_transform with > 256 lanes (group size for doubles is 512)
    constexpr static index_type num_lanes = ipow(side_length, 2);
    constexpr static index_type lane_length = hc_size / num_lanes;
    constexpr static index_type value_width = banks_of<typename profile<T, 3>::bits_type>;

    NDZIP_UNIVERSAL constexpr static index_type pad(index_type i) {
        auto padded = value_width * i + value_width * i / warp_size;
        if (value_width == 2) { padded -= i / (value_width * num_lanes); }
        return padded;
    }
};

template<typename T, typename Transform>
struct directional_accessor<profile<T, 3>, 0, Transform> {
    using layout = hypercube_layout<profile<T, 3>, Transform>;
    NDZIP_UNIVERSAL constexpr static index_type prev_lane_in_row(index_type) {
        return no_such_lane;
    }
    NDZIP_UNIVERSAL constexpr static index_type offset(index_type lane) {
        return lane * layout::side_length;
    }
    constexpr static index_type stride = 1;
};

template<typename T, typename Transform>
struct directional_accessor<profile<T, 3>, 1, Transform> {
    using layout = hypercube_layout<profile<T, 3>, Transform>;

    NDZIP_UNIVERSAL constexpr static index_type prev_lane_in_row(index_type) {
        return no_such_lane;
    }
    NDZIP_UNIVERSAL constexpr static index_type offset(index_type lane) {
        return (lane / layout::side_length) * 2 * layout::num_lanes
                - (lane / (layout::num_lanes / 2))
                * (layout::hc_size - ipow(layout::side_length, 2))
                + lane % layout::side_length;
    }
    constexpr static index_type stride = layout::side_length;
};

template<typename T, typename Transform>
struct directional_accessor<profile<T, 3>, 2, Transform> {
    using layout = hypercube_layout<profile<T, 3>, Transform>;
    NDZIP_UNIVERSAL constexpr static index_type prev_lane_in_row(index_type) {
        return no_such_lane;
    }
    NDZIP_UNIVERSAL constexpr static index_type offset(index_type lane) { return lane; }
    constexpr static index_type stride = layout::side_length * layout::side_length;
};


template<typename Layout>
struct hypercube_allocation {
    using backing_type = uint_bank_t;
    constexpr static index_type size = ceil(Layout::pad(Layout::hc_size), Layout::value_width);
};

template<typename T>
struct hypercube_allocation<hypercube_layout<profile<T, 1>, inverse_transform_tag>> {
    using backing_type = typename profile<T, 1>::bits_type;
    constexpr static index_type size
            = hypercube_layout<profile<T, 1>, inverse_transform_tag>::hc_size;
};


template<typename Profile, typename Transform>
struct hypercube_ptr {
    using bits_type = typename Profile::bits_type;
    using layout = hypercube_layout<Profile, Transform>;

    uint_bank_t *memory;

    template<typename T = bits_type>
    NDZIP_UNIVERSAL T load(index_type i) const {
        static_assert(sizeof(T) == sizeof(bits_type));
        return load_aligned<alignof(uint_bank_t), T>(memory + layout::pad(i));
    }

    template<typename T = bits_type>
    NDZIP_UNIVERSAL void store(index_type i, std::common_type_t<T> bits) {
        static_assert(sizeof(T) == sizeof(bits_type));
        store_aligned<alignof(uint_bank_t), T>(memory + layout::pad(i), bits);
    }
};


template<typename Profile>
class border_map {
  public:
    constexpr static index_type dimensions = Profile::dimensions;

    constexpr explicit border_map(extent<dimensions> outer) {
        index_type outer_acc = 1, inner_acc = 1, edge_acc = 1;
        for (unsigned d = 0; d < dimensions; ++d) {
            unsigned dd = dimensions - 1 - d;
            _inner[dd] = floor(outer[dd], Profile::hypercube_side_length);
            _stride[dd] = outer_acc;
            _edge[dd] = _inner[dd] * edge_acc;
            outer_acc *= outer[dd];
            inner_acc *= _inner[dd];
            edge_acc = _border[dd] = outer_acc - inner_acc;
        }
    }

    NDZIP_UNIVERSAL constexpr extent<dimensions> operator[](index_type i) const {
        return index(dim_tag<dimensions>{}, i);
    }

    NDZIP_UNIVERSAL constexpr index_type size() const { return _border[0]; }

  private:
    extent<dimensions> _inner;
    extent<dimensions> _border;
    extent<dimensions> _edge;
    extent<dimensions> _stride;

    // We cannot write index() as a specialized template function because NVCC doesn't allow
    // specializations inside a class definition and border_map is a template itself, so
    // specialization cannot happen outside either. Instead we do overload selection by tag type.
    template<unsigned D> using dim_tag = std::integral_constant<unsigned, D>;

    NDZIP_UNIVERSAL constexpr extent<1> index(dim_tag<1>, index_type i) const {
        return {_edge[dimensions - 1] + i};
    }

    NDZIP_UNIVERSAL constexpr extent<2> index(dim_tag<2>, index_type i) const {
        if (i >= _edge[dimensions - 2]) {
            i -= _edge[dimensions - 2];
            auto y = _inner[dimensions - 2] + i / _stride[dimensions - 2];
            auto x = i % _stride[dimensions - 2];
            return {y, x};
        } else {
            auto y = i / _border[dimensions - 1];
            auto e_x = index(dim_tag<1>{}, i % _border[dimensions - 1]);
            return {y, e_x[0]};
        }
    }

    NDZIP_UNIVERSAL constexpr extent<3> index(dim_tag<3>, index_type i) const {
        if (i >= _edge[dimensions - 3]) {
            i -= _edge[dimensions - 3];
            auto z = _inner[dimensions - 3] + i / _stride[dimensions - 3];
            i %= _stride[dimensions - 3];
            auto y = i / _stride[dimensions - 2];
            auto x = i % _stride[dimensions - 2];
            return {z, y, x};
        } else {
            auto z = i / _border[dimensions - 2];
            auto e_yx = index(dim_tag<2>{}, i % _border[dimensions - 2]);
            return {z, e_yx[0], e_yx[1]};
        }
    }
};


// We guarantee that memory is laid out sequentially for 1D inverse transform, which is implemented
// using gpu_bits prefix_sum
template<typename Data>
struct hypercube_ptr<profile<Data, 1>, inverse_transform_tag> {
    using bits_type = typename profile<Data, 1>::bits_type;
    using layout = hypercube_layout<profile<bits_type, 1>, inverse_transform_tag>;

    bits_type *memory;

    NDZIP_UNIVERSAL bits_type load(index_type i) const { return memory[i]; }

    NDZIP_UNIVERSAL void store(index_type i, bits_type bits) { memory[i] = bits; }
};


}  // namespace ndzip::detail::gpu
