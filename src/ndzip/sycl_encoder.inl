#pragma once

#include "gpu_common.hh"
#include "sycl_bits.hh"

#include <numeric>
#include <stdexcept>
#include <vector>

#include <ndzip/sycl_encoder.hh>


namespace ndzip::detail::gpu_sycl {

using namespace ndzip::detail::gpu;

template<typename Profile>
using hypercube_group = known_size_group<hypercube_group_size<Profile>>;

template<typename Profile, typename Transform>
using hypercube_memory = sycl::local_memory<
        typename hypercube_allocation<hypercube_layout<Profile, Transform>>::backing_type
        [hypercube_allocation<hypercube_layout<Profile, Transform>>::size]>;


template<typename Profile, typename F>
void for_hypercube_indices(hypercube_group<Profile> grp, index_type hc_index,
                           extent<Profile::dimensions> data_size, F &&f) {
    const auto side_length = Profile::hypercube_side_length;
    const auto hc_size = ipow(side_length, Profile::dimensions);
    const auto hc_offset
            = detail::extent_from_linear_id(hc_index, data_size / side_length) * side_length;

    index_type initial_offset = linear_offset(hc_offset, data_size);
    grp.distribute_for(hc_size, [&](index_type local_idx) {
      index_type global_idx = initial_offset + global_offset<Profile>(local_idx, data_size);
      f(global_idx, local_idx);
    });
}


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
        hypercube_group<Profile> grp, hypercube_ptr<Profile, forward_transform_tag> hc) {
    using bits_type = typename Profile::bits_type;
    using accessor = directional_accessor<Profile, Direction, forward_transform_tag>;
    using layout = typename accessor::layout;

    // 1D and 2D transforms have multiple lanes per row, so a barrier is required to synchronize
    // the read of the last element from (lane-1) with the write to the last element of (lane)
    constexpr bool needs_carry = layout::side_length > layout::lane_length;
    // std::max: size might be zero if !needs_carry, but that is not a valid type
    sycl::private_memory<
            bits_type[std::max(index_type{1}, layout::num_lanes / hypercube_group_size<Profile>)]>
            carry{grp};

    if constexpr (needs_carry) {
        grp.template distribute_for<layout::num_lanes>([&](index_type lane, index_type iteration,
                                                               sycl::logical_item<1> idx) {
            if (auto prev_lane = accessor::prev_lane_in_row(lane); prev_lane != no_such_lane) {
                // TODO this load causes a bank conflict in the 1-dimensional case. Why?
                carry(idx)[iteration] = hc.load(
                        accessor::offset(prev_lane) + (layout::lane_length - 1) * accessor::stride);
            } else {
                carry(idx)[iteration] = 0;
            }
        });
    }

    grp.template distribute_for<layout::num_lanes>(
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
        hypercube_group<Profile> grp, hypercube_ptr<Profile, forward_transform_tag> hc) {
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
        hypercube_group<Profile> grp, hypercube_ptr<Profile, inverse_transform_tag> hc) {
    using bits_type = typename Profile::bits_type;
    using accessor = directional_accessor<Profile, Direction, inverse_transform_tag>;
    using layout = typename accessor::layout;

    grp.template distribute_for<layout::num_lanes>([&](index_type lane) {
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
void inverse_block_transform(
        hypercube_group<Profile> grp, hypercube_ptr<Profile, inverse_transform_tag> hc) {
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
void write_transposed_chunks(hypercube_group<Profile> grp,
        hypercube_ptr<Profile, forward_transform_tag> hc, typename Profile::bits_type *out_chunks,
        index_type *out_lengths) {
    using bits_type = typename Profile::bits_type;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr index_type col_chunk_size = bits_of<bits_type>;
    constexpr index_type num_col_chunks = hc_size / col_chunk_size;
    constexpr index_type header_chunk_size = num_col_chunks;
    constexpr index_type warps_per_col_chunk = col_chunk_size / warp_size;

    static_assert(col_chunk_size % warp_size == 0);

    grp.single_item([&] {  // single_item does not introduce a barrier
        out_lengths[0] = header_chunk_size;
    });

    const auto out_header_chunk = out_chunks;
    sycl::local_memory<sycl::vec<uint32_t, warps_per_col_chunk>[hypercube_group_size<Profile>]>
            row_stage { grp };

    // Schedule one warp per chunk to allow subgroup reductions
    grp.distribute_for(num_col_chunks * warp_size,
            [&](index_type item, index_type, sycl::logical_item<1> idx, sycl::sub_group sg) {
                const auto col_chunk_index = item / warp_size;
                const auto in_col_chunk_base = col_chunk_index * col_chunk_size;

                // Collectively determine the head for a chunk
                bits_type head = 0;
                for (index_type w = 0; w < warps_per_col_chunk; ++w) {
                    auto row = hc.load(in_col_chunk_base + w * warp_size + item % warp_size);
                    head |= sycl::group_reduce(sg, row, sycl::bit_or<bits_type>{});
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
                                + popcount(static_cast<uint32_t>(
                                        head >> ((warps_per_col_chunk - 1 - w) * warp_size)));
                    }
                    chunk_compact_size = compact_warp_offset[warps_per_col_chunk];

                    // Transposition is relatively simple in the 32-bit case, but for 64-bit, we
                    // achieve considerable speedup (> 2x) by operating on 32-bit values. This
                    // requires two outer loop iterations, one computing the lower and one computing
                    // the upper word of each column. The number of shifts/adds remains the same.
                    // Note that this makes assumptions about the endianness of uint64_t.
                    static_assert(warp_size == 32);  // implicit assumption with uint32_t
                    sycl::vec<uint32_t, warps_per_col_chunk> columns[warps_per_col_chunk];

                    for (index_type i = 0; i < warps_per_col_chunk; ++i) {
                        // Stage rows through local memory to avoid repeated hypercube_ptr index
                        // calculations inside the loop
                        row_stage[idx.get_local_id(0)]
                                = hc.template load<sycl::vec<uint32_t, warps_per_col_chunk>>(
                                        in_col_chunk_base + col_chunk_size - 1
                                        - (warp_size * i + sg.get_local_linear_id()));
                        sycl::group_barrier(sg);

                        for (index_type j = 0; j < warp_size; ++j) {
                            auto row = row_stage[floor<index_type>(idx.get_local_id(0), warp_size)
                                    + j];
#pragma unroll
                            for (index_type w = 0; w < warps_per_col_chunk; ++w) {
                                columns[w][i] |= ((row[warps_per_col_chunk - 1 - w]
                                                          >> (31 - item % warp_size))
                                                         & uint32_t{1})
                                        << j;
                            }
                        }
                    }

#pragma unroll
                    for (index_type w = 0; w < warps_per_col_chunk; ++w) {
                        auto column_bits = bit_cast<bits_type>(columns[w]);
                        auto pos_in_out_col_chunk = compact_warp_offset[w]
                                + sycl::group_exclusive_scan(
                                        sg, index_type{column_bits != 0}, sycl::plus<index_type>{});
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
void read_transposed_chunks(hypercube_group<Profile> grp,
        hypercube_ptr<Profile, inverse_transform_tag> hc,
        const typename Profile::bits_type *stream) {
    using bits_type = typename Profile::bits_type;
    using word_type = uint32_t;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr index_type col_chunk_size = bits_of<bits_type>;
    constexpr index_type num_col_chunks = hc_size / col_chunk_size;
    constexpr index_type warps_per_col_chunk = col_chunk_size / warp_size;
    constexpr index_type word_size = bits_of<word_type>;
    constexpr index_type words_per_col = sizeof(bits_type) / sizeof(word_type);

    sycl::local_memory<index_type[1 + num_col_chunks]> chunk_offsets{grp};
    grp.single_item([&] { chunk_offsets[0] = num_col_chunks; });  // single_item has no barrier
    grp.distribute_for(num_col_chunks,
            [&](index_type item) { chunk_offsets[1 + item] = popcount(stream[item]); });
    inclusive_scan<num_col_chunks + 1>(grp, chunk_offsets(), sycl::plus<index_type>());

    sycl::local_memory<index_type[hypercube_group_size<Profile> * ipow(words_per_col, 2)]>
            stage_mem{grp};

    // See write_transposed_chunks for the optimization of the 64-bit case
    grp.distribute_for(num_col_chunks * warp_size,
            [&](index_type item0, index_type, sycl::logical_item<1> idx, sycl::sub_group sg) {
                auto col_chunk_index = item0 / warp_size;
                auto head = stream[col_chunk_index]; // TODO this has been read before, stage?

                if (head != 0) {
                    word_type head_col[words_per_col];
                    __builtin_memcpy(head_col, &head, sizeof head_col);
                    auto offset = chunk_offsets[col_chunk_index];

                    // TODO this can be hoisted out of the loop within distribute_for. Maybe
                    //  write that loop explicitly for this purpose?
                    auto *stage = &stage_mem[ipow(words_per_col, 2)
                            * floor(static_cast<index_type>(idx.get_local_linear_id()), warp_size)];

            // TODO There is an excellent opportunity to hide global memory latencies by
            //  starting to read values for outer-loop iteration n+1 into registers and then
            //  just committing the reads to shared memory in the next iteration
#pragma unroll
                    for (index_type w = 0; w < ipow(words_per_col, 2); ++w) {
                        auto i = w * warp_size + item0 % warp_size;
                        if (offset + i / words_per_col < chunk_offsets[col_chunk_index + 1]) {
                            // TODO this load is uncoalesced since offsets are not warp-aligned
                            stage[i] = load_aligned<word_type>(
                                    reinterpret_cast<const word_type *>(stream + offset) + i);
                        }
                    }
                    group_barrier(sg);

                    for (index_type w = 0; w < warps_per_col_chunk; ++w) {
                        index_type item = floor(item0, warp_size) * warps_per_col_chunk
                                + w * warp_size + item0 % warp_size;
                        const auto outer_cell = words_per_col - 1 - w;
                        const auto inner_cell = word_size - 1 - item0 % warp_size;

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
                        index_type item = floor(item0, warp_size) * warps_per_col_chunk
                                + w * warp_size + item0 % warp_size;
                        hc.store(item, 0);
                    }
                }
            });
}


template<typename Profile>
void compact_chunks(hypercube_group<Profile> grp, const typename Profile::bits_type *chunks,
        const index_type *offsets, index_type *stream_header_entry,
        typename Profile::bits_type *stream_hc) {
    using bits_type = typename Profile::bits_type;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr index_type col_chunk_size = bits_of<bits_type>;
    constexpr index_type header_chunk_size = hc_size / col_chunk_size;
    constexpr index_type hc_total_chunks_size = hc_size + header_chunk_size;
    constexpr index_type chunks_per_hc = 1 /* header */ + hc_size / col_chunk_size;

    sycl::local_memory<index_type[chunks_per_hc + 1]> hc_offsets{grp};
    grp.distribute_for(chunks_per_hc + 1, [&](index_type i) { hc_offsets[i] = offsets[i]; });

    grp.single_item([&] {
        *stream_header_entry = hc_offsets[chunks_per_hc];  // header encodes offset *after*
    });

    grp.distribute_for(hc_total_chunks_size, [&](index_type item) {
        index_type chunk_rel_item = item;
        index_type chunk_index = 0;
        if (item >= header_chunk_size) {
            chunk_index += 1 + (item - header_chunk_size) / col_chunk_size;
            chunk_rel_item = (item - header_chunk_size) % col_chunk_size;
        }
        auto stream_offset = hc_offsets[chunk_index] + chunk_rel_item;
        if (stream_offset < hc_offsets[chunk_index + 1]) {
            stream_hc[stream_offset] = chunks[item];
        }
    });
}


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

}  // namespace ndzip::detail::gpu


template<typename T, unsigned Dims>
struct ndzip::sycl_encoder<T, Dims>::impl {
    bool profiling_enabled;
    sycl::queue q;

    impl(bool report_kernel_duration, bool verbose)
        : profiling_enabled(report_kernel_duration || verbose)
        , q{sycl::gpu_selector{},
                report_kernel_duration || verbose
                        ? sycl::property_list{sycl::property::queue::enable_profiling{}}
                        : sycl::property_list{}}
    {
        if (verbose) {
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
ndzip::sycl_encoder<T, Dims>::sycl_encoder(bool report_kernel_duration)
    : _pimpl(std::make_unique<impl>(report_kernel_duration, detail::verbose())) {
}

template<typename T, unsigned Dims>
ndzip::sycl_encoder<T, Dims>::~sycl_encoder() = default;


template<typename T, unsigned Dims>
size_t ndzip::sycl_encoder<T, Dims>::compress(const slice<const data_type, dimensions> &data,
        void *raw_stream, kernel_duration *out_kernel_duration) const {
    using namespace detail;
    using namespace detail::gpu_sycl;

    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;
    using sam = sycl::access::mode;

    constexpr index_type hc_size = ipow(profile::hypercube_side_length, profile::dimensions);
    constexpr index_type col_chunk_size = bits_of<bits_type>;
    constexpr index_type header_chunk_size = hc_size / col_chunk_size;
    constexpr index_type chunks_per_hc = 1 /* header */ + hc_size / col_chunk_size;
    constexpr index_type hc_total_chunks_size = hc_size + header_chunk_size;

    // TODO edge case w/ 0 hypercubes

    detail::file<profile> file(data.size());
    const auto num_hypercubes = file.num_hypercubes();
    const auto num_chunks = num_hypercubes * (1 + hc_size / col_chunk_size);
    if (verbose()) { printf("Have %u hypercubes\n", num_hypercubes); }

    sycl::buffer<data_type, dimensions> data_buffer{
            extent_cast<sycl::range<dimensions>>(data.size())};

    submit_and_profile(_pimpl->q, "copy input to device", [&](sycl::handler &cgh) {
        cgh.copy(data.data(), data_buffer.template get_access<sam::discard_write>(cgh));
    }).wait();

    sycl::buffer<bits_type> chunks_buf(num_hypercubes * hc_total_chunks_size);
    sycl::buffer<index_type> chunk_lengths_buf(
            ceil(1 + num_chunks, hierarchical_inclusive_scan_granularity));
    sycl::buffer<bits_type> stream_buf(
            div_ceil(compressed_size_bound<data_type, dimensions>(data.size()), sizeof(bits_type)));
    _pimpl->q.submit([&](sycl::handler &cgh) {
        cgh.fill(stream_buf.template get_access<sam::discard_write>(cgh, sycl::range<1>{1}), bits_type{});
    }).wait();

    auto encode_kernel = [&](sycl::handler &cgh) {
        auto data_acc = data_buffer.template get_access<sam::read>(cgh);
        auto chunks_acc = chunks_buf.template get_access<sam::discard_write>(cgh);
        auto chunk_lengths_acc = chunk_lengths_buf.get_access<sam::discard_write>(cgh);
        auto data_size = data.size();
        cgh.parallel<block_compression_kernel<T, Dims>>(sycl::range<1>{file.num_hypercubes()},
                sycl::range<1>{hypercube_group_size<profile>},
                [=](hypercube_group<profile> grp, sycl::physical_item<1> phys_idx) {
                    slice<const data_type, dimensions> data{data_acc.get_pointer(), data_size};
                    hypercube_memory<profile, forward_transform_tag> lm{grp};
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
    };
    auto encode_kernel_evt
            = submit_and_profile(_pimpl->q, "transform + chunk encode", encode_kernel);

    auto intermediate_bufs_keepalive = hierarchical_inclusive_scan(_pimpl->q, chunk_lengths_buf, sycl::plus<index_type>{});

    auto num_compressed_words_offset = sycl::id<1>{num_hypercubes * chunks_per_hc};

    submit_and_profile(_pimpl->q, "compact chunks", [&](sycl::handler &cgh) {
        auto chunks_acc = chunks_buf.template get_access<sam::read>(cgh);
        auto offsets_acc = chunk_lengths_buf.template get_access<sam::read>(cgh);
        auto stream_acc = stream_buf.template get_access<sam::discard_write>(cgh);
        cgh.parallel<chunk_compaction_kernel<T, Dims>>(sycl::range<1>{num_hypercubes},
                sycl::range<1>{hypercube_group_size<profile>},
                [=](hypercube_group<profile> grp, sycl::physical_item<1>) {
                    auto hc_index = static_cast<index_type>(grp.get_id(0));
                    detail::stream<profile> stream{num_hypercubes, stream_acc.get_pointer()};
                    compact_chunks<profile>(grp,
                            &chunks_acc.get_pointer()[hc_index * hc_total_chunks_size],
                            &offsets_acc.get_pointer()[hc_index * chunks_per_hc],
                            stream.header() + hc_index, stream.hypercube(0));
                });
    });

    const auto border_map = gpu::border_map<profile>{data.size()};
    const auto num_border_words = border_map.size();

    detail::stream<profile> stream{num_hypercubes, static_cast<bits_type *>(raw_stream)};
    const auto num_header_words = stream.hypercube(0) - stream.buffer;

    auto compact_border_kernel = [&](sycl::handler &cgh) {
        auto data_acc = data_buffer.template get_access<sam::read>(cgh);
        auto offsets_acc = chunk_lengths_buf.template get_access<sam::read>(cgh);
        auto stream_acc = stream_buf.template get_access<sam::discard_write>(cgh);
        const auto data_size = data.size();
        cgh.parallel_for<border_compaction_kernel<T, Dims>>(  // TODO leverage ILP
                sycl::range<1>{num_border_words}, [=](sycl::item<1> item) {
                    slice<const data_type, dimensions> data{data_acc.get_pointer(), data_size};
                    auto num_compressed_words = offsets_acc[num_compressed_words_offset];
                    auto border_offset = num_header_words + num_compressed_words;
                    auto i = static_cast<index_type>(item.get_linear_id());
                    stream_acc[border_offset + i] = bit_cast<bits_type>(data[border_map[i]]);
                });
    };
    auto compact_border_evt
            = submit_and_profile(_pimpl->q, "compact border", compact_border_kernel);
    compact_border_evt.wait();

    index_type host_num_compressed_words;
    auto num_compressed_words_available = _pimpl->q.submit([&](sycl::handler &cgh) {
        cgh.copy(chunk_lengths_buf.template get_access<sam::read>(cgh, sycl::range<1>{1},
                    num_compressed_words_offset),
                &host_num_compressed_words);
    });

    _pimpl->q.submit([&](sycl::handler &cgh) {
        cgh.copy(chunk_lengths_buf.template get_access<sam::read>(cgh, sycl::range<1>{1},
                    num_compressed_words_offset),
                &host_num_compressed_words);
    }).wait();

    const auto host_border_offset = num_header_words + host_num_compressed_words;
    const auto host_num_stream_words = host_border_offset + num_border_words;

    submit_and_profile(_pimpl->q, "copy stream to host", [&](sycl::handler &cgh) {
        cgh.copy(stream_buf.template get_access<sam::read>(cgh, host_num_stream_words), stream.buffer);
    }).wait();

    if (_pimpl->profiling_enabled) {
        auto [early, late, kernel_duration] = measure_duration(encode_kernel_evt, compact_border_evt);
        if (verbose()) {
            printf("[profile] %8lu %8lu total kernel time %.3fms\n", early, late, kernel_duration.count() * 1e-6);
        }
        if (out_kernel_duration) {
            *out_kernel_duration = kernel_duration;
        }
    } else if (out_kernel_duration) {
        *out_kernel_duration = {};
    }

    return host_num_stream_words * sizeof(bits_type);
}


template<typename T, unsigned Dims>
size_t ndzip::sycl_encoder<T, Dims>::decompress(const void *raw_stream, size_t bytes,
        const slice<data_type, dimensions> &data, kernel_duration *out_kernel_duration) const {
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
        cgh.copy(static_cast<const bits_type *>(raw_stream),
                stream_buf.template get_access<sam::discard_write>(cgh));
    }).wait();

    auto kernel_evt = submit_and_profile(_pimpl->q, "decompress blocks", [&](sycl::handler &cgh) {
        auto stream_acc = stream_buf.template get_access<sam::read>(cgh);
        auto data_acc = data_buf.template get_access<sam::discard_write>(cgh);
        auto data_size = data.size();
        auto num_hypercubes = file.num_hypercubes();
        cgh.parallel<block_decompression_kernel<T, Dims>>(sycl::range<1>{num_hypercubes},
                sycl::range<1>{hypercube_group_size<profile>},
                [=](hypercube_group<profile> grp, sycl::physical_item<1>) {
                    slice<data_type, dimensions> data{data_acc.get_pointer(), data_size};
                    hypercube_memory<profile, inverse_transform_tag> lm{grp};
                    hypercube_ptr<profile, inverse_transform_tag> hc{lm()};

                    const auto hc_index = static_cast<index_type>(grp.get_id(0));
                    detail::stream<const profile> stream{num_hypercubes, stream_acc.get_pointer()};
                    read_transposed_chunks<profile>(grp, hc, stream.hypercube(hc_index));
                    inverse_block_transform<profile>(grp, hc);
                    store_hypercube(grp, hc_index, {data}, hc);
                });
    });

    const auto border_map = gpu::border_map<profile>{data.size()};
    const auto num_border_words = border_map.size();

    detail::stream<const profile> stream{
            file.num_hypercubes(), static_cast<const bits_type *>(raw_stream)};
    const auto border_offset = static_cast<index_type>(stream.border() - stream.buffer);
    const auto num_stream_words = border_offset + num_border_words;

    auto expand_border_kernel = [&](sycl::handler &cgh) {
        auto stream_acc = stream_buf.template get_access<sam::read>(cgh);
        auto data_acc = data_buf.template get_access<sam::discard_write>(cgh);
        const auto data_size = data.size();
        cgh.parallel_for<border_expansion_kernel<T, Dims>>(  // TODO leverage ILP
                sycl::range<1>{num_border_words}, [=](sycl::item<1> item) {
                    slice<data_type, dimensions> data{data_acc.get_pointer(), data_size};
                    auto i = static_cast<index_type>(item.get_linear_id());
                    data[border_map[i]] = bit_cast<data_type>(stream_acc[border_offset + i]);
                });
    };
    auto expand_border_evt = submit_and_profile(_pimpl->q, "expand border", expand_border_kernel);
    expand_border_evt.wait();

    submit_and_profile(_pimpl->q, "copy output to host", [&](sycl::handler &cgh) {
        cgh.copy(data_buf.template get_access<sam::read>(cgh), data.data());
    }).wait();

    if (_pimpl->profiling_enabled) {
        auto [early, late, kernel_duration] = measure_duration(kernel_evt, expand_border_evt);
        if (verbose()) {
            printf("[profile] %8lu %8lu total kernel time %.3fms\n", early, late, kernel_duration.count() * 1e-6);
        }
        if (out_kernel_duration) {
            *out_kernel_duration = kernel_duration;
        }
    } else if (out_kernel_duration) {
        *out_kernel_duration = {};
    }

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
