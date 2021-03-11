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
                    static_cast<index_type>(sizeof(Bits) / sizeof(uint_bank_t)));
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
        typename Profile::bits_type *out_chunks, index_type *out_lengths) {
    using bits_type = typename Profile::bits_type;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr index_type col_chunk_size = bitsof<bits_type>;
    constexpr index_type num_col_chunks = hc_size / col_chunk_size;
    constexpr index_type header_chunk_size = num_col_chunks;
    constexpr index_type warps_per_col_chunk = col_chunk_size / warp_size;

    static_assert(col_chunk_size % warp_size == 0);

    grp.single_item([&] {  // single_item does not introduce a barrier
        out_lengths[0] = header_chunk_size;
    });

    const auto out_header_chunk = out_chunks;

    // Schedule one warp per chunk to allow subgroup reductions
    grp.distribute_for(num_col_chunks * warp_size,
            [&](index_type item, index_type, sycl::logical_item<1>, sycl::sub_group sg) {
                const auto col_chunk_index = item / warp_size;
                const auto in_col_chunk_base = col_chunk_index * col_chunk_size;

                bits_type head = 0;
                for (index_type w = 0; w < warps_per_col_chunk; ++w) {
                    auto col = in_col_chunk_base + w * warp_size + item % warp_size;
                    head |= sycl::group_reduce(sg, hc.load(col), sycl::bit_or<bits_type>{});
                }

                index_type compact_warp_offset[1 + warps_per_col_chunk];
                compact_warp_offset[0] = 0;
                for (index_type w = 0; w < warps_per_col_chunk; ++w) {
                    static_assert(warp_size == bitsof<uint32_t>);
                    compact_warp_offset[1 + w] = compact_warp_offset[w]
                            + popcount(static_cast<uint32_t>(
                                    head >> ((warps_per_col_chunk - 1 - w) * warp_size)));
                }
                const auto chunk_compact_size = compact_warp_offset[warps_per_col_chunk];

                const auto out_col_chunk = out_chunks + header_chunk_size + in_col_chunk_base;
                for (index_type w = 0; w < warps_per_col_chunk; ++w) {
                    bits_type column = 0;
                    // TODO this shortcut does not improve performance - but why? Shortcut' warps
                    // are stalled on the final barrier, but given a large enough group_size, this
                    // should still result in much fewer wasted cycles if (head != 0) {
                    // ??????
                    const auto cell = w * warp_size + item % warp_size;
                    for (index_type i = 0; i < col_chunk_size; ++i) {
                        // TODO for double, can we still operate on 32 bit words? e.g split into
                        //  low / high loop
                        column |= (hc.load(in_col_chunk_base + i) >> (col_chunk_size - 1 - cell)
                                          & bits_type{1})
                                << (col_chunk_size - 1 - i);
                    }

                    auto pos_in_out_col_chunk = compact_warp_offset[w]
                            + sycl::group_exclusive_scan(
                                    sg, index_type{column != 0}, sycl::plus<index_type>{});
                    if (column != 0) { out_col_chunk[pos_in_out_col_chunk] = column; }
                }

                if (sg.leader()) {
                    // TODO collect in local memory, write coalesced - otherwise 3 full GM
                    //  transaction per HC instead of 1! => But only if WC does not resolve this
                    //  anyway
                    out_header_chunk[col_chunk_index] = head;
                    out_lengths[1 + col_chunk_index] = chunk_compact_size;
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
void compact_chunks(hypercube_group grp, const typename Profile::bits_type *chunks,
        const index_type *offsets, typename Profile::bits_type *stream) {
    using bits_type = typename Profile::bits_type;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr index_type col_chunk_size = bitsof<bits_type>;
    constexpr index_type header_chunk_size = hc_size / col_chunk_size;
    constexpr index_type hc_total_chunks_size = hc_size + header_chunk_size;
    constexpr index_type chunks_per_hc = 1 /* header */ + hc_size / col_chunk_size;

    sycl::local_memory<index_type[chunks_per_hc + 1]> hc_offsets{grp};
    grp.distribute_for(chunks_per_hc + 1, [&](index_type i) { hc_offsets[i] = offsets[i]; });

    grp.distribute_for(hc_total_chunks_size, [&](index_type item) {
        index_type chunk_rel_item = item;
        index_type chunk_index = 0;
        if (item >= header_chunk_size) {
            chunk_index += 1 + (item - header_chunk_size) / col_chunk_size;
            chunk_rel_item = (item - header_chunk_size) % col_chunk_size;
        }
        auto stream_offset = hc_offsets[chunk_index] + chunk_rel_item;
        if (stream_offset < hc_offsets[chunk_index + 1]) { stream[stream_offset] = chunks[item]; }
    });
}


// TODO we might be able to avoid this kernel altogether by writing the reduction results directly
//  to the stream header. Requires the stream format to be fixed first.
// => Probably not after the fine-grained compaction refactor
template<typename Profile>
void fill_stream_header(index_type num_hypercubes, global_write<stream_align_t> stream_acc,
        global_read<index_type> offset_acc, sycl::handler &cgh) {
    using bits_type = typename Profile::bits_type;
    constexpr index_type hc_size
            = detail::ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr index_type chunks_per_hc = 1 + hc_size / bitsof<bits_type>;
    constexpr index_type group_size = 256;
    const index_type num_groups = div_ceil(num_hypercubes, group_size);
    cgh.parallel(sycl::range<1>{num_groups}, sycl::range<1>{group_size},
            [=](known_size_group<group_size> grp, sycl::physical_item<1>) {
                stream<Profile> stream{num_hypercubes, stream_acc.get_pointer()};
                const index_type base = static_cast<index_type>(grp.get_id(0)) * group_size;
                const index_type num_elements = std::min(group_size, num_hypercubes - base);
                grp.distribute_for(num_elements, [&](index_type i) {
                    stream.set_offset_after(base + i, offset_acc[(base + i + 1) * chunks_per_hc]);
                });
            });
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
    using namespace detail;
    using namespace detail::gpu;

    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;
    using sam = sycl::access::mode;
    using hc_layout = hypercube_layout<profile::dimensions, forward_transform_tag>;

    constexpr index_type hc_size
            = detail::ipow(profile::hypercube_side_length, profile::dimensions);
    constexpr index_type col_chunk_size = detail::bitsof<bits_type>;
    constexpr index_type header_chunk_size = hc_size / col_chunk_size;
    constexpr index_type chunks_per_hc = 1 /* header */ + hc_size / col_chunk_size;
    constexpr index_type hc_total_chunks_size = hc_size + header_chunk_size;

    // TODO edge case w/ 0 hypercubes

    detail::file<profile> file(data.size());
    const auto num_hypercubes = file.num_hypercubes();
    const auto num_chunks = num_hypercubes * (1 + hc_size / col_chunk_size);
    if (auto env = getenv("NDZIP_VERBOSE"); env && *env) {
        printf("Have %zu hypercubes\n", num_hypercubes);
    }

    sycl::buffer<data_type, dimensions> data_buffer{
            extent_cast<sycl::range<dimensions>>(data.size())};

    submit_and_profile(_pimpl->q, "copy input to device", [&](sycl::handler &cgh) {
        cgh.copy(data.data(), data_buffer.template get_access<sam::discard_write>(cgh));
    });

    sycl::buffer<bits_type> chunks_buf(num_hypercubes * hc_total_chunks_size);
    sycl::buffer<index_type> chunk_lengths_buf(
            ceil(1 + num_chunks, hierarchical_inclusive_scan_granularity));

    submit_and_profile(_pimpl->q, "transform + chunk encode", [&](sycl::handler &cgh) {
        auto data_acc = data_buffer.template get_access<sam::read>(cgh);
        auto chunks_acc = chunks_buf.template get_access<sam::discard_write>(cgh);
        auto chunk_lengths_acc = chunk_lengths_buf.get_access<sam::discard_write>(cgh);
        auto data_size = data.size();
        cgh.parallel<block_compression_kernel<T, Dims>>(sycl::range<1>{file.num_hypercubes()},
                sycl::range<1>{hypercube_group_size},
                [=](hypercube_group grp, sycl::physical_item<1> phys_idx) {
                    slice<const data_type, dimensions> data{data_acc.get_pointer(), data_size};
                    hypercube_memory<bits_type, hc_layout> lm{grp};
                    hypercube_ptr<profile, forward_transform_tag> hc{&lm[0]};

                    auto hc_index = static_cast<index_type>(grp.get_id(0));
                    load_hypercube(grp, hc_index, {data}, hc);
                    forward_block_transform(grp, hc);
                    write_transposed_chunks(grp, hc, &chunks_acc[hc_index * hc_total_chunks_size],
                            &chunk_lengths_acc[1 + hc_index * chunks_per_hc]);
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
    hierarchical_inclusive_scan(_pimpl->q, chunk_lengths_buf, sycl::plus<index_type>{});
    std::vector<index_type> dbg_offsets(chunk_lengths_buf.get_range()[0]);
    _pimpl->q
            .submit([&](sycl::handler &cgh) {
                cgh.copy(chunk_lengths_buf.get_access<sam::read>(cgh), dbg_offsets.data());
            })
            .wait();

    index_type num_compressed_words;
    auto num_compressed_words_available = _pimpl->q.submit([&](sycl::handler &cgh) {
        cgh.copy(chunk_lengths_buf.template get_access<sam::read>(
                         cgh, sycl::range<1>{1}, sycl::id<1>{num_hypercubes * chunks_per_hc}),
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
        auto chunks_acc = chunks_buf.template get_access<sam::read>(cgh);
        auto offsets_acc = chunk_lengths_buf.template get_access<sam::read>(cgh);
        auto stream_acc = stream_buf.template get_access<sam::discard_write>(cgh);
        const size_t header_offset = file.file_header_length() / sizeof(stream_align_t);
        cgh.parallel<stream_compaction_kernel<T, Dims>>(sycl::range<1>{num_hypercubes},
                sycl::range<1>{hypercube_group_size},
                [=](hypercube_group grp, sycl::physical_item<1>) {
                    auto hc_index = static_cast<index_type>(grp.get_id(0));
                    compact_chunks<profile>(grp,
                            &chunks_acc.get_pointer()[hc_index * hc_total_chunks_size],
                            &offsets_acc.get_pointer()[hc_index * chunks_per_hc],
                            reinterpret_cast<bits_type *>(
                                    &stream_acc.get_pointer()[0] + header_offset));
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
        const void *raw_stream, size_t bytes, const slice<data_type, dimensions> &data) const {
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
        cgh.copy(static_cast<const stream_align_t *>(raw_stream),
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
                    detail::stream<const profile> stream{num_hypercubes, stream_acc.get_pointer()};
                    read_transposed_chunks<profile>(grp, hc, stream.hypercube(hc_index));
                    inverse_block_transform<profile>(grp, hc);
                    store_hypercube(grp, hc_index, {data}, hc);
                });
    });

    auto data_copy_event = detail::gpu::submit_and_profile(
            _pimpl->q, "copy output to host", [&](sycl::handler &cgh) {
                cgh.copy(data_buffer.template get_access<sam::read>(cgh), data.data());
            });

    detail::stream<const profile> stream{
            file.num_hypercubes(), static_cast<const detail::stream_align_t *>(raw_stream)};
    auto stream_pos = file.file_header_length();
    stream_pos += (stream.border() - stream.hypercube(0)) * sizeof(bits_type);

    data_copy_event.wait();

    // TODO GPU border handling!
    stream_pos
            += detail::unpack_border(data, static_cast<const std::byte *>(raw_stream) + stream_pos,
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
