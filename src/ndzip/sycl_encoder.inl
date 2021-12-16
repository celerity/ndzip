#pragma once

#include "gpu_common.hh"
#include "sycl_bits.hh"

#include <numeric>
#include <stdexcept>
#include <vector>

#include <ndzip/sycl.hh>
#include <ndzip/sycl_encoder.hh>


namespace ndzip::detail::gpu_sycl {

using namespace ndzip::detail::gpu;

template<typename Profile>
using hypercube_group = known_size_group<hypercube_group_size<Profile>>;

template<typename Profile>
using hypercube_item = known_group_size_item<hypercube_group_size<Profile>>;


template<typename Profile, typename F>
void for_hypercube_indices(
        hypercube_group<Profile> grp, index_type hc_index, extent<Profile::dimensions> data_size, F &&f) {
    const auto side_length = Profile::hypercube_side_length;
    const auto hc_size = ipow(side_length, Profile::dimensions);
    const auto hc_offset = detail::extent_from_linear_id(hc_index, data_size / side_length) * side_length;

    index_type initial_offset = linear_offset(hc_offset, data_size);
    distribute_for(hc_size, grp, [&](index_type local_idx) {
        index_type global_idx = initial_offset + global_offset<Profile>(local_idx, data_size);
        f(global_idx, local_idx);
    });
}


template<typename Profile>
void load_hypercube(sycl::group<1> grp, index_type hc_index,
        slice<const typename Profile::data_type, extent<Profile::dimensions>> data,
        hypercube_ptr<Profile, forward_transform_tag> hc) {
    using bits_type = typename Profile::bits_type;

    for_hypercube_indices<Profile>(grp, hc_index, data.size(), [&](index_type global_idx, index_type local_idx) {
        hc.store(local_idx, rotate_left_1(bit_cast<bits_type>(data.data()[global_idx])));
        // TODO merge with block_transform to avoid LM round-trip?
        //  we could assign directly to registers
    });
}

template<typename Profile>
void store_hypercube(sycl::group<1> grp, index_type hc_index,
        slice<typename Profile::data_type, extent<Profile::dimensions>> data,
        hypercube_ptr<Profile, inverse_transform_tag> hc) {
    using data_type = typename Profile::data_type;
    for_hypercube_indices<Profile>(grp, hc_index, data.size(), [&](index_type global_idx, index_type local_idx) {
        data.data()[global_idx] = bit_cast<data_type>(rotate_right_1(hc.load(local_idx)));
    });
}


template<unsigned Direction, typename Profile>
void forward_transform_lanes(hypercube_group<Profile> grp, hypercube_ptr<Profile, forward_transform_tag> hc) {
    using bits_type = typename Profile::bits_type;
    using accessor = directional_accessor<Profile, Direction, forward_transform_tag>;
    using layout = typename accessor::layout;

    // 1D and 2D transforms have multiple lanes per row, so a barrier is required to synchronize
    // the read of the last element from (lane-1) with the write to the last element of (lane)
    constexpr bool needs_carry = layout::side_length > layout::lane_length;
    // std::max: size might be zero if !needs_carry, but that is not a valid type
    bits_type carry[std::max(index_type{1}, layout::num_lanes / hypercube_group_size<Profile>)];  // per thread

    if constexpr (needs_carry) {
        distribute_for<layout::num_lanes>(grp, [&](index_type lane, index_type iteration) {
            if (auto prev_lane = accessor::prev_lane_in_row(lane); prev_lane != no_such_lane) {
                // TODO this load causes a bank conflict in the 1-dimensional case. Why?
                carry[iteration] = hc.load(accessor::offset(prev_lane) + (layout::lane_length - 1) * accessor::stride);
            } else {
                carry[iteration] = 0;
            }
        });
    }

    distribute_for<layout::num_lanes>(grp, [&](index_type lane, index_type iteration) {
        bits_type a = needs_carry ? carry[iteration] : 0;
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
void forward_block_transform(hypercube_group<Profile> grp, hypercube_ptr<Profile, forward_transform_tag> hc) {
    constexpr auto dims = Profile::dimensions;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, dims);

    // Why is there no constexpr for?
    forward_transform_lanes<0>(grp, hc);
    if constexpr (dims >= 2) { forward_transform_lanes<1>(grp, hc); }
    if constexpr (dims >= 3) { forward_transform_lanes<2>(grp, hc); }

    // TODO move complement operation elsewhere to avoid local memory round-trip
    distribute_for(hc_size, grp, [&](index_type item) { hc.store(item, complement_negative(hc.load(item))); });
}


template<unsigned Direction, typename Profile>
void inverse_transform_lanes(hypercube_group<Profile> grp, hypercube_ptr<Profile, inverse_transform_tag> hc) {
    using bits_type = typename Profile::bits_type;
    using accessor = directional_accessor<Profile, Direction, inverse_transform_tag>;
    using layout = typename accessor::layout;

    distribute_for<layout::num_lanes>(grp, [&](index_type lane) {
        index_type index = accessor::offset(lane);
        bits_type a = hc.load(index);
        for (index_type i = 1; i < layout::lane_length; ++i) {
            index += accessor::stride;
            a += hc.load(index);
            hc.store(index, a);
        }
    });
}


template<typename Profile>
struct inverse_transform_local_allocation {};

template<typename Data>
struct inverse_transform_local_allocation<profile<Data, 1>> {
    using profile = profile<Data, 1>;
    using bits_type = typename profile::bits_type;

    inclusive_scan_local_allocation<bits_type, profile::hypercube_side_length> scan;
};


template<typename Profile>
void inverse_block_transform(hypercube_item<Profile> item, hypercube_ptr<Profile, inverse_transform_tag> hc,
        inverse_transform_local_allocation<Profile> &lm) {
    using bits_type = typename Profile::bits_type;
    constexpr auto dims = Profile::dimensions;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, dims);

    // TODO move complement operation elsewhere to avoid local memory round-trip
    distribute_for(
            hc_size, item.get_group(), [&](index_type item) { hc.store(item, complement_negative(hc.load(item))); });

    // TODO how to do 2D?
    //   For 2D we have 64 parallel work items but we _probably_ want at least 256 threads per SM
    //   (= per HC for double) to hide latencies. Maybe hybrid approach - do (64/32)*64 sub-group
    //   prefix sums and optimize inclusive_prefix_sum to skip the recursion since the second
    //   level does not actually need a reduction. Benchmark against leaving 256-64 = 192 threads
    //   idle and going with a sequential-per-lane transform.

    if constexpr (dims == 1) {
        // 1D inverse hypercube_ptr guarantees linear layout of hc.memory
        inclusive_scan_over_group(item, hc.memory(), lm.scan, sycl::plus<bits_type>{});
    }
    if constexpr (dims == 2) {
        // TODO inefficient, see above
        inverse_transform_lanes<0>(item.get_group(), hc);
        inverse_transform_lanes<1>(item.get_group(), hc);
    }
    if constexpr (dims == 3) {
        inverse_transform_lanes<0>(item.get_group(), hc);
        inverse_transform_lanes<1>(item.get_group(), hc);
        inverse_transform_lanes<2>(item.get_group(), hc);
    }
}


template<typename Profile>
struct writer_local_allocation {
    using bits_type = typename Profile::bits_type;
    inline static constexpr index_type col_chunk_size = bits_of<bits_type>;
    inline static constexpr index_type warps_per_col_chunk = col_chunk_size / warp_size;

    sycl::vec<uint32_t, warps_per_col_chunk> row_stage[hypercube_group_size<Profile>];
};


template<typename Profile>
void write_transposed_chunks(hypercube_item<Profile> item, hypercube_ptr<Profile, forward_transform_tag> hc,
        typename Profile::bits_type *out_chunks, index_type *out_lengths, writer_local_allocation<Profile> &lm) {
    using bits_type = typename Profile::bits_type;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr index_type col_chunk_size = bits_of<bits_type>;
    constexpr index_type num_col_chunks = hc_size / col_chunk_size;
    constexpr index_type header_chunk_size = num_col_chunks;
    constexpr index_type warps_per_col_chunk = col_chunk_size / warp_size;

    static_assert(col_chunk_size % warp_size == 0);

    if (item.get_group().leader()) { out_lengths[0] = header_chunk_size; }

    const auto out_header_chunk = out_chunks;
    const auto row_stage = lm.row_stage;

    // Schedule one warp per chunk to allow subgroup reductions
    distribute_for(num_col_chunks * warp_size, item.get_group(), [&](index_type i) {
        auto sg = item.get_sub_group();

        const auto col_chunk_index = i / warp_size;
        const auto in_col_chunk_base = col_chunk_index * col_chunk_size;

        // Collectively determine the head for a chunk
        bits_type head = 0;
        for (index_type w = 0; w < warps_per_col_chunk; ++w) {
            auto row = hc.load(in_col_chunk_base + w * warp_size + i % warp_size);
            head |= sycl::reduce_over_group(sg, row, sycl::bit_or<bits_type>{});
        }

        index_type chunk_compact_size = 0;

        if (head != 0) {  // short-circuit the entire warp
            const auto out_col_chunk = out_chunks + header_chunk_size + in_col_chunk_base;

            // Each thread computes 1 column for 32 bit and 2 columns for 64 bit. For the
            // 2-iteration scenario, we compute the relative output position of the second
            // column directly, so that one warp-collective prefix sum is enough to
            // determine the final position within the chunk.
            index_type compact_warp_offset[1 + warps_per_col_chunk];
            compact_warp_offset[0] = 0;
#pragma unroll
            for (index_type w = 0; w < warps_per_col_chunk; ++w) {
                static_assert(warp_size == bits_of<uint32_t>);
                compact_warp_offset[1 + w] = compact_warp_offset[w]
                        + popcount(static_cast<uint32_t>(head >> ((warps_per_col_chunk - 1 - w) * warp_size)));
            }
            chunk_compact_size = compact_warp_offset[warps_per_col_chunk];

            // Transposition is relatively simple in the 32-bit case, but for 64-bit, we
            // achieve considerable speedup (> 2x) by operating on 32-bit values. This
            // requires two outer loop iterations, one computing the lower and one computing
            // the upper word of each column. The number of shifts/adds remains the same.
            // Note that this makes assumptions about the endianness of uint64_t.
            static_assert(warp_size == 32);  // implicit assumption with uint32_t
            sycl::vec<uint32_t, warps_per_col_chunk> columns[warps_per_col_chunk];

            for (index_type j = 0; j < warps_per_col_chunk; ++j) {
                // Stage rows through local memory to avoid repeated hypercube_ptr index
                // calculations inside the loop
                row_stage[item.get_local_id(0)] = hc.template load<sycl::vec<uint32_t, warps_per_col_chunk>>(
                        in_col_chunk_base + col_chunk_size - 1 - (warp_size * j + sg.get_local_linear_id()));
                sycl::group_barrier(sg);

                for (index_type k = 0; k < warp_size; ++k) {
                    auto row = row_stage[floor<index_type>(item.get_local_id(0), warp_size) + k];
#pragma unroll
                    for (index_type w = 0; w < warps_per_col_chunk; ++w) {
                        columns[w][j] |= ((row[warps_per_col_chunk - 1 - w] >> (31 - i % warp_size)) & uint32_t{1})
                                << k;
                    }
                }
            }

#pragma unroll
            for (index_type w = 0; w < warps_per_col_chunk; ++w) {
                auto column_bits = detail::bit_cast<bits_type>(columns[w]);
                auto pos_in_out_col_chunk = compact_warp_offset[w]
                        + sycl::exclusive_scan_over_group(sg, index_type{column_bits != 0}, sycl::plus<index_type>{});
                if (column_bits != 0) { out_col_chunk[pos_in_out_col_chunk] = column_bits; }
            }
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
struct reader_local_allocation {
    using bits_type = typename Profile::bits_type;
    using word_type = uint32_t;

    static inline constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    static inline constexpr index_type col_chunk_size = bits_of<typename Profile::bits_type>;
    static inline constexpr index_type num_col_chunks = hc_size / col_chunk_size;
    static inline constexpr index_type words_per_col = sizeof(bits_type) / sizeof(word_type);

    index_type chunk_offsets[1 + num_col_chunks];
    union {
        index_type stage[hypercube_group_size<Profile> * ipow(words_per_col, 2)];
        inclusive_scan_local_allocation<index_type, 1 + num_col_chunks> scan;
    };
};


template<typename Profile>
void read_transposed_chunks(hypercube_item<Profile> item, hypercube_ptr<Profile, inverse_transform_tag> hc,
        const typename Profile::bits_type *stream, reader_local_allocation<Profile> &lm) {
    using bits_type = typename Profile::bits_type;
    using word_type = uint32_t;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr index_type col_chunk_size = bits_of<bits_type>;
    constexpr index_type num_col_chunks = hc_size / col_chunk_size;
    constexpr index_type warps_per_col_chunk = col_chunk_size / warp_size;
    constexpr index_type word_size = bits_of<word_type>;
    constexpr index_type words_per_col = sizeof(bits_type) / sizeof(word_type);

    const auto chunk_offsets = lm.chunk_offsets;
    if (item.get_group().leader()) { chunk_offsets[0] = num_col_chunks; }
    distribute_for(num_col_chunks, item.get_group(),
            [&](index_type item) { chunk_offsets[1 + item] = popcount(stream[item]); });
    inclusive_scan_over_group(item, chunk_offsets, lm.scan, sycl::plus<index_type>());

    // See write_transposed_chunks for the optimization of the 64-bit case
    distribute_for(num_col_chunks * warp_size, item.get_group(), [&](index_type i0) {
        auto sg = item.get_sub_group();

        auto col_chunk_index = i0 / warp_size;
        auto head = stream[col_chunk_index];  // TODO this has been read before, stage?

        if (head != 0) {
            word_type head_col[words_per_col];
            __builtin_memcpy(head_col, &head, sizeof head_col);
            auto offset = chunk_offsets[col_chunk_index];

            // TODO this can be hoisted out of the loop within distribute_for. Maybe
            //  write that loop explicitly for this purpose?
            auto *stage = &lm.stage[ipow(words_per_col, 2)
                    * floor(static_cast<index_type>(item.get_local_linear_id()), warp_size)];

            // TODO There is an excellent opportunity to hide global memory latencies by
            //  starting to read values for outer-loop iteration n+1 into registers and then
            //  just committing the reads to shared memory in the next iteration
#pragma unroll
            for (index_type w = 0; w < ipow(words_per_col, 2); ++w) {
                auto i = w * warp_size + i0 % warp_size;
                if (offset + i / words_per_col < chunk_offsets[col_chunk_index + 1]) {
                    // TODO this load is uncoalesced since offsets are not warp-aligned
                    stage[i] = load_aligned<word_type>(reinterpret_cast<const word_type *>(stream + offset) + i);
                }
            }
            sycl::group_barrier(sg, sycl::sub_group::fence_scope /* TODO hipSYCL workaround, remove parameter */);

            for (index_type w = 0; w < warps_per_col_chunk; ++w) {
                index_type item = floor(i0, warp_size) * warps_per_col_chunk + w * warp_size + i0 % warp_size;
                const auto outer_cell = words_per_col - 1 - w;
                const auto inner_cell = word_size - 1 - i0 % warp_size;

                index_type local_offset = 0;
                word_type row[words_per_col] = {0};
#pragma unroll
                for (index_type i = 0; i < words_per_col; ++i) {
                    const auto ii = words_per_col - 1 - i;
#pragma unroll  // TODO this unroll significantly reduces the computational complexity of the loop,
                //  but I'm uncomfortable with the increased instruction cache pressure. We might
                //  resolve this by decoding in the opposite direction offsets[i+1] => offsets[i]
                //  which should allow us to get rid of the repeated N-1-i terms here.
                    for (index_type j = 0; j < word_size; ++j) {
                        const auto jj = word_size - 1 - j;
                        const auto stage_idx = words_per_col * local_offset + outer_cell;
                        auto col_word = stage[stage_idx];
                        if ((head_col[ii] >> jj) & word_type{1}) {
                            row[ii] |= ((col_word >> inner_cell) & word_type{1}) << jj;
                            local_offset += 1;
                        }
                    }
                }
                bits_type row_bits;
                __builtin_memcpy(&row_bits, row, sizeof row_bits);
                hc.store(item, row_bits);
            }
        } else {
            // TODO duplication of the `item` calculation above. The term can be simplified!
            for (index_type w = 0; w < warps_per_col_chunk; ++w) {
                index_type item = floor(i0, warp_size) * warps_per_col_chunk + w * warp_size + i0 % warp_size;
                hc.store(item, 0);
            }
        }
    });
}


template<typename Profile>
struct compaction_local_allocation {
    using bits_type = typename Profile::bits_type;

    inline static constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    inline static constexpr index_type col_chunk_size = bits_of<bits_type>;
    inline static constexpr index_type chunks_per_hc = 1 /* header */ + hc_size / col_chunk_size;

    index_type hc_offsets[chunks_per_hc + 1];
};


template<typename Profile>
void compact_chunks(hypercube_group<Profile> grp, const typename Profile::bits_type *chunks, const index_type *offsets,
        index_type *stream_header_entry, typename Profile::bits_type *stream_hc,
        compaction_local_allocation<Profile> &lm) {
    using bits_type = typename Profile::bits_type;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr index_type col_chunk_size = bits_of<bits_type>;
    constexpr index_type header_chunk_size = hc_size / col_chunk_size;
    constexpr index_type hc_total_chunks_size = hc_size + header_chunk_size;
    constexpr index_type chunks_per_hc = 1 /* header */ + hc_size / col_chunk_size;

    const auto hc_offsets = lm.hc_offsets;
    distribute_for(chunks_per_hc + 1, grp, [&](index_type i) { hc_offsets[i] = offsets[i]; });

    if (grp.leader()) {
        *stream_header_entry = hc_offsets[chunks_per_hc];  // header encodes offset *after*
    }

    distribute_for(hc_total_chunks_size, grp, [&](index_type i) {
        index_type chunk_rel_item = i;
        index_type chunk_index = 0;
        if (i >= header_chunk_size) {
            chunk_index += 1 + (i - header_chunk_size) / col_chunk_size;
            chunk_rel_item = (i - header_chunk_size) % col_chunk_size;
        }
        auto stream_offset = hc_offsets[chunk_index] + chunk_rel_item;
        if (stream_offset < hc_offsets[chunk_index + 1]) { stream_hc[stream_offset] = chunks[i]; }
    });
}


template<typename Profile>
struct compressor_local_allocation {
    hypercube_allocation<Profile, forward_transform_tag> hc;
    writer_local_allocation<Profile> writer;
};


template<typename Profile>
struct decompressor_local_allocation {
    hypercube_allocation<Profile, inverse_transform_tag> hc;
    union {
        reader_local_allocation<Profile> reader;
        inverse_transform_local_allocation<Profile> transform;
    };
};


template<typename Buffer>
void force_device_allocation(Buffer &buf, sycl::queue &q) {
    q.submit([&](sycl::handler &cgh) {
        cgh.fill(buf.template get_access<sycl::access::mode::discard_write>(cgh), typename Buffer::value_type{});
    });
};


// SYCL kernel names
template<typename, unsigned>
class block_compression_kernel;

template<typename, unsigned>
class chunk_compaction_kernel;

template<typename, unsigned>
class border_compaction_kernel;

template<typename, unsigned>
class block_decompression_kernel;

template<typename, unsigned>
class border_expansion_kernel;

}  // namespace ndzip::detail::gpu_sycl


template<int Dims>
void ndzip::sycl_compressor_requirements<Dims>::include(extent<Dims> data_size) {
    using profile = detail::profile<float, Dims>;  // TODO value_type does not matter here, refactor
    const auto file = detail::file<profile>(data_size);
    _max_num_hypercubes = std::max(_max_num_hypercubes, file.num_hypercubes());
}

template<typename T, int Dims>
static std::pair<ndzip::index_type, ndzip::index_type>
get_chunks_and_length_buf_size(ndzip::index_type num_hypercubes) {
    using namespace ndzip;

    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;

    constexpr index_type hc_size = detail::ipow(profile::hypercube_side_length, profile::dimensions);
    constexpr index_type col_chunk_size = detail::bits_of<bits_type>;
    constexpr index_type header_chunk_size = hc_size / col_chunk_size;
    constexpr index_type hc_total_chunks_size = hc_size + header_chunk_size;

    const auto num_chunks = num_hypercubes * (1 + hc_size / col_chunk_size);
    const auto chunks_buf_size = num_hypercubes * hc_total_chunks_size;
    const auto length_buf_size = detail::ceil(1 + num_chunks, detail::gpu::hierarchical_inclusive_scan_granularity);

    return {chunks_buf_size, length_buf_size};
}

template<typename T, int Dims>
ndzip::sycl_compressor<T, Dims>::sycl_compressor(sycl::queue &q, sycl_compressor_requirements<Dims> req)
    : sycl_compressor{q, get_chunks_and_length_buf_size<T, Dims>(req._max_num_hypercubes)} {
}

template<typename T, int Dims>
ndzip::sycl_compressor<T, Dims>::sycl_compressor(
        sycl::queue &q, std::pair<index_type, index_type> chunks_and_length_buf_sizes)
    : _q{&q}
    , _chunks_buf{chunks_and_length_buf_sizes.first}
    , _chunk_lengths_buf{chunks_and_length_buf_sizes.second}
    , _hierarchical_scan_bufs{
              detail::gpu_sycl::hierarchical_inclusive_scan_allocate<index_type>(chunks_and_length_buf_sizes.second)} {
}

template<typename T, int Dims>
ndzip::sycl_compress_events ndzip::sycl_compressor<T, Dims>::compress(sycl::buffer<value_type, Dims> &in_data,
        sycl::buffer<compressed_type> &out_stream, sycl::buffer<index_type> *out_stream_length) {
    using namespace ndzip;
    using namespace detail;
    using namespace detail::gpu_sycl;

    constexpr unsigned dimensions = static_cast<unsigned>(Dims);
    using profile = detail::profile<T, dimensions>;
    using bits_type = typename profile::bits_type;
    using sam = sycl::access::mode;

    constexpr index_type hc_size = ipow(profile::hypercube_side_length, profile::dimensions);
    constexpr index_type col_chunk_size = bits_of<bits_type>;
    constexpr index_type header_chunk_size = hc_size / col_chunk_size;
    constexpr index_type chunks_per_hc = 1 /* header */ + hc_size / col_chunk_size;
    constexpr index_type hc_total_chunks_size = hc_size + header_chunk_size;

    // TODO edge case w/ 0 hypercubes

    const auto data_size = extent_cast<extent<dimensions>>(in_data.get_range());
    detail::file<profile> file(data_size);
    const auto num_hypercubes = file.num_hypercubes();

    sycl_compress_events events;
    events.start = submit_and_profile(*_q, "transform + chunk encode", [&](sycl::handler &cgh) {
        const auto data_acc = in_data.template get_access<sam::read>(cgh);
        const auto chunks_acc = _chunks_buf.template get_access<sam::discard_write>(cgh);
        const auto chunk_lengths_acc = _chunk_lengths_buf.template get_access<sam::discard_write>(cgh);
        const auto group_size = hypercube_group_size<profile>;
        const auto nd_range = make_nd_range(num_hypercubes, group_size);

        sycl::local_accessor<compressor_local_allocation<profile>> lm{1, cgh};
        cgh.parallel_for<block_compression_kernel<T, dimensions>>(nd_range, [=](hypercube_item<profile> item) {
            slice<const T, extent<dimensions>> data{data_acc.get_pointer(), data_size};
            hypercube_ptr<profile, forward_transform_tag> hc{lm[0].hc};

            auto hc_index = static_cast<index_type>(item.get_group_id(0));
            load_hypercube(item.get_group(), hc_index, {data}, hc);
            forward_block_transform(item.get_group(), hc);
            write_transposed_chunks(item, hc, &chunks_acc[hc_index * hc_total_chunks_size],
                    &chunk_lengths_acc[1 + hc_index * chunks_per_hc], lm[0].writer);
            // hack
            if (item.get_global_linear_id() == 0) { chunk_lengths_acc[0] = 0; }
        });
    });

    hierarchical_inclusive_scan(*_q, _chunk_lengths_buf, _hierarchical_scan_bufs, sycl::plus<index_type>{});

    const auto num_compressed_words_offset = sycl::id<1>{num_hypercubes * chunks_per_hc};

    auto compact_kernel_evt = submit_and_profile(*_q, "compact chunks", [&](sycl::handler &cgh) {
        const auto chunks_acc = _chunks_buf.template get_access<sam::read>(cgh);
        const auto offsets_acc = _chunk_lengths_buf.template get_access<sam::read>(cgh);
        const auto stream_acc = out_stream.template get_access<sam::discard_write>(cgh);
        const auto num_header_fields
                = detail::ceil(num_hypercubes, static_cast<uint32_t>(sizeof(bits_type) / sizeof(index_type)));
        const auto nd_range = make_nd_range(num_header_fields, hypercube_group_size<profile>);

        sycl::local_accessor<compaction_local_allocation<profile>> lm{1, cgh};
        cgh.parallel_for<chunk_compaction_kernel<T, dimensions>>(nd_range, [=](hypercube_item<profile> item) {
            auto hc_index = static_cast<index_type>(item.get_group_id(0));
            detail::stream<profile> stream{num_hypercubes, stream_acc.get_pointer()};
            if (hc_index == num_hypercubes) {
                // For 64-bit data types and an odd number of hypercubes, we insert a
                // padding word in the header to guarantee correct alignment. To keep the
                // compressed stream deterministic we round gridDim.x up to the next
                // multiple of 2 and initialize the padding to zero.
                if (item.get_group().leader()) { stream.header()[num_hypercubes] = 0; }
            } else {
                compact_chunks(item.get_group(), &chunks_acc.get_pointer()[hc_index * hc_total_chunks_size],
                        &offsets_acc.get_pointer()[hc_index * chunks_per_hc], stream.header() + hc_index,
                        stream.hypercube(0), lm[0]);
            }
        });
    });
    events.stream_available.push_back(compact_kernel_evt);

    const auto border_map = gpu::border_map<profile>{data_size};
    const auto num_border_words = border_map.size();

    detail::stream<profile> stream{num_hypercubes, nullptr};
    const auto num_header_words = stream.hypercube(0) - stream.buffer;
    // TODO num_header_words == num_header_fields ??

    if (num_border_words > 0) {
        auto compact_border_evt = submit_and_profile(*_q, "compact border", [&](sycl::handler &cgh) {
            auto data_acc = in_data.template get_access<sam::read>(cgh);
            auto offsets_acc = _chunk_lengths_buf.template get_access<sam::read>(cgh);
            // TODO ranged accessor to allow overlapping with compact_chunks kernel
            auto stream_acc = out_stream.template get_access<sam::discard_write>(cgh);
            cgh.parallel_for<border_compaction_kernel<T, dimensions>>(  // TODO leverage ILP
                    sycl::range<1>{num_border_words}, [=](sycl::item<1> item) {
                        slice<const T, extent<dimensions>> data{data_acc.get_pointer(), data_size};
                        const auto num_compressed_words = offsets_acc[num_compressed_words_offset];
                        const auto border_offset = num_header_words + num_compressed_words;
                        const auto i = static_cast<index_type>(item.get_linear_id());
                        stream_acc[border_offset + i] = detail::bit_cast<bits_type>(data[border_map[i]]);
                    });
        });
        events.stream_available.push_back(compact_border_evt);
    }

    if (out_stream_length) {
        events.stream_length_available = _q->submit([&](sycl::handler &cgh) {
            auto length_acc = out_stream_length->get_access<sam::discard_write>(cgh);
            auto offsets_acc = _chunk_lengths_buf.template get_access<sam::read>(cgh);
            cgh.single_task([=] {
                const auto num_compressed_words = offsets_acc[num_compressed_words_offset];
                length_acc[0] = num_header_words + num_compressed_words + num_border_words;
            });
        });
    }

    return events;
}

template<typename T, int Dims>
ndzip::sycl_decompress_events ndzip::sycl_decompressor<T, Dims>::decompress(
        sycl::buffer<compressed_type> &in_stream, sycl::buffer<value_type, Dims> &out_data) {
    using namespace ndzip;
    using namespace detail;
    using namespace detail::gpu_sycl;

    constexpr unsigned dimensions = static_cast<unsigned>(Dims);
    using profile = detail::profile<T, dimensions>;
    using sam = sycl::access::mode;

    const auto data_size = extent_cast<extent<dimensions>>(out_data.get_range());
    const auto file = detail::file<profile>(data_size);
    const auto num_hypercubes = file.num_hypercubes();

    sycl_decompress_events events;
    auto decompress_kernel_evt = submit_and_profile(*_q, "decompress blocks", [&](sycl::handler &cgh) {
        auto stream_acc = in_stream.template get_access<sam::read>(cgh);
        auto data_acc = out_data.template get_access<sam::discard_write>(cgh);
        auto nd_range = make_nd_range(num_hypercubes, hypercube_group_size<profile>);

        sycl::local_accessor<decompressor_local_allocation<profile>> lm{1, cgh};
        cgh.parallel_for<block_decompression_kernel<T, dimensions>>(nd_range, [=](hypercube_item<profile> item) {
            slice<T, extent<Dims>> data{data_acc.get_pointer(), data_size};
            hypercube_ptr<profile, inverse_transform_tag> hc{lm[0].hc};

            const auto hc_index = static_cast<index_type>(item.get_group_id(0));
            detail::stream<const profile> stream{num_hypercubes, stream_acc.get_pointer()};
            read_transposed_chunks<profile>(item, hc, stream.hypercube(hc_index), lm[0].reader);
            inverse_block_transform<profile>(item, hc, lm[0].transform);
            store_hypercube(item.get_group(), hc_index, {data}, hc);
        });
    });
    events.start = decompress_kernel_evt;
    events.data_available.push_back(decompress_kernel_evt);

    const auto border_map = gpu::border_map<profile>{data_size};
    const auto num_border_words = border_map.size();

    if (num_border_words > 0) {
        auto expand_border_evt = submit_and_profile(*_q, "expand border", [&](sycl::handler &cgh) {
            auto stream_acc = in_stream.template get_access<sam::read>(cgh);
            auto data_acc = out_data.template get_access<sam::discard_write>(cgh);
            cgh.parallel_for<border_expansion_kernel<T, Dims>>(  // TODO leverage ILP
                    sycl::range<1>{num_border_words}, [=](sycl::item<1> item) {
                        detail::stream<const profile> stream{file.num_hypercubes(), stream_acc.get_pointer()};
                        const auto border_offset = static_cast<index_type>(stream.border() - stream.buffer);

                        slice<T, extent<Dims>> data{data_acc.get_pointer(), data_size};
                        auto i = static_cast<index_type>(item.get_linear_id());
                        data[border_map[i]] = bit_cast<T>(stream_acc[border_offset + i]);
                    });
        });
        events.data_available.push_back(expand_border_evt);
    }

    return events;
}


template<typename T, unsigned Dims>
struct ndzip::sycl_encoder<T, Dims>::impl {
    sycl::queue q;

    static sycl::property_list make_queue_properties(bool profile) {
        if (profile) { return sycl::property_list{sycl::property::queue::enable_profiling{}}; }
        return sycl::property_list{};
    }

    impl(bool report_kernel_duration, bool verbose)
        : q{sycl::gpu_selector{}, make_queue_properties(report_kernel_duration || verbose)} {
        if (verbose) {
            auto device = q.get_device();
            printf("SYCL backend is %s on %s %s (%lu bytes of local memory)\n",
                    device.get_platform().get_info<sycl::info::platform::name>().c_str(),
                    device.get_info<sycl::info::device::vendor>().c_str(),
                    device.get_info<sycl::info::device::name>().c_str(),
                    (unsigned long) device.get_info<sycl::info::device::local_mem_size>());
        }
    }

    bool is_profiling() const { return q.template has_property<sycl::property::queue::enable_profiling>(); }
};

template<typename T, unsigned Dims>
ndzip::sycl_encoder<T, Dims>::sycl_encoder(bool report_kernel_duration)
    : _pimpl(std::make_unique<impl>(report_kernel_duration, detail::verbose())) {
}

template<typename T, unsigned Dims>
ndzip::sycl_encoder<T, Dims>::~sycl_encoder() = default;


template<typename T, unsigned Dims>
size_t ndzip::sycl_encoder<T, Dims>::compress(
        const slice<const data_type, extent<dimensions>> &data, void *raw_stream, kernel_duration *out_kernel_duration) const {
    using namespace detail;
    using namespace detail::gpu_sycl;

    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;
    using sam = sycl::access::mode;

    // TODO edge case w/ 0 hypercubes

    detail::file<profile> file(data.size());
    const auto num_hypercubes = file.num_hypercubes();
    if (verbose()) { printf("Have %u hypercubes\n", num_hypercubes); }

    sycl::buffer<data_type, dimensions> data_buf{extent_cast<sycl::range<dimensions>>(data.size())};

    submit_and_profile(_pimpl->q, "copy input to device",
            [&](sycl::handler &cgh) { cgh.copy(data.data(), data_buf.template get_access<sam::discard_write>(cgh)); });

    sycl::buffer<bits_type> stream_buf(
            div_ceil(compressed_size_bound<data_type, dimensions>(data.size()), sizeof(bits_type)));
    sycl::buffer<index_type> stream_length_buf(1);

    ndzip::sycl_compressor<T, Dims> compressor{_pimpl->q, data.size()};

    if (_pimpl->is_profiling()) {
        force_device_allocation(stream_buf, _pimpl->q);
        force_device_allocation(stream_length_buf, _pimpl->q);
        force_device_allocation(compressor._chunks_buf, _pimpl->q);
        force_device_allocation(compressor._chunk_lengths_buf, _pimpl->q);
        for (auto &buf : compressor._hierarchical_scan_bufs) {
            force_device_allocation(buf, _pimpl->q);
        }

        _pimpl->q.wait();
    }

    auto events = compressor.compress(data_buf, stream_buf, &stream_length_buf);

    index_type host_size;
    _pimpl->q
            .submit([&](sycl::handler &cgh) {
                cgh.copy(stream_length_buf.template get_access<sam::read>(cgh, sycl::range<1>{1}), &host_size);
            })
            .wait();

    auto stream_copy_evt = submit_and_profile(_pimpl->q, "copy stream to host", [&](sycl::handler &cgh) {
        cgh.copy(stream_buf.template get_access<sam::read>(cgh, host_size), static_cast<bits_type *>(raw_stream));
    });

    if (_pimpl->is_profiling()) {
        auto [early, late, kernel_duration] = measure_duration(events.start, events.stream_available);
        if (verbose()) {
            printf("[profile] %8lu %8lu total kernel time %.3fms\n", early, late, kernel_duration.count() * 1e-6);
        }
        if (out_kernel_duration) { *out_kernel_duration = kernel_duration; }
    } else if (out_kernel_duration) {
        *out_kernel_duration = {};
    }

    stream_copy_evt.wait();

    return host_size * sizeof(bits_type);
}


template<typename T, unsigned Dims>
size_t ndzip::sycl_encoder<T, Dims>::decompress(const void *raw_stream, size_t bytes,
        const slice<data_type, extent<dimensions>> &data, kernel_duration *out_kernel_duration) const {
    using namespace detail;
    using namespace detail::gpu_sycl;

    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;
    using sam = sycl::access::mode;

    detail::file<profile> file(data.size());

    // TODO the range computation here is questionable at best
    sycl::buffer<bits_type> stream_buf{div_ceil(bytes, sizeof(bits_type))};
    sycl::buffer<data_type, dimensions> data_buf{extent_cast<sycl::range<dimensions>>(data.size())};

    submit_and_profile(_pimpl->q, "copy stream to device", [&](sycl::handler &cgh) {
        cgh.copy(static_cast<const bits_type *>(raw_stream), stream_buf.template get_access<sam::discard_write>(cgh));
    });

    if (_pimpl->is_profiling()) {
        force_device_allocation(data_buf, _pimpl->q);

        _pimpl->q.wait();
    }

    auto events = sycl_decompressor<T, Dims>{_pimpl->q}.decompress(stream_buf, data_buf);

    submit_and_profile(_pimpl->q, "copy output to host", [&](sycl::handler &cgh) {
        cgh.copy(data_buf.template get_access<sam::read>(cgh), data.data());
    }).wait();

    if (_pimpl->is_profiling()) {
        auto [early, late, kernel_duration] = measure_duration(events.start, events.data_available);
        if (verbose()) {
            printf("[profile] %8lu %8lu total kernel time %.3fms\n", early, late, kernel_duration.count() * 1e-6);
        }
        if (out_kernel_duration) { *out_kernel_duration = kernel_duration; }
    } else if (out_kernel_duration) {
        *out_kernel_duration = {};
    }


    // TODO all this just to return the size? Maybe have a compressed-stream-size-query instead
    //  -- or a stream verification function!
    const auto num_hypercubes = file.num_hypercubes();
    const auto border_map = gpu::border_map<profile>{data.size()};
    const auto num_border_words = border_map.size();

    detail::stream<const profile> stream{num_hypercubes, static_cast<const bits_type *>(raw_stream)};
    const auto border_offset = static_cast<index_type>(stream.border() - stream.buffer);
    const auto num_stream_words = border_offset + num_border_words;
    return num_stream_words * sizeof(bits_type);
}


namespace ndzip {

extern template class sycl_encoder<float, 1>;
extern template class sycl_encoder<float, 2>;
extern template class sycl_encoder<float, 3>;
extern template class sycl_encoder<double, 1>;
extern template class sycl_encoder<double, 2>;
extern template class sycl_encoder<double, 3>;

#ifdef SPLIT_CONFIGURATION_sycl_encoder
template class sycl_encoder<DATA_TYPE, DIMENSIONS>;
#endif

}  // namespace ndzip
