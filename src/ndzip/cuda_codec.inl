#pragma once

#include "cuda_bits.cuh"
#include "gpu_common.hh"

#include <numeric>
#include <stdexcept>
#include <vector>

#include <ndzip/cuda.hh>
#include <ndzip/offload.hh>


namespace ndzip::detail::gpu_cuda {

using namespace ndzip::detail::gpu;

template<typename T, index_type N>
struct vec {
    alignas(sizeof(T) * N) T elements[N];

    __device__ T &operator[](index_type i) { return elements[i]; }
    __device__ const T &operator[](index_type i) const { return elements[i]; }
};

template<typename Profile>
using hypercube_block = known_size_block<hypercube_group_size<Profile>>;


template<typename Profile, typename F>
__device__ void for_hypercube_indices(hypercube_block<Profile> block, index_type hc_index,
        const static_extent<Profile::dimensions> &data_size, F &&f) {
    const auto side_length = Profile::hypercube_side_length;
    const auto hc_size = ipow(side_length, Profile::dimensions);
    const auto hc_offset = detail::extent_from_linear_id(hc_index, data_size / side_length) * side_length;

    index_type initial_offset = linear_index(data_size, hc_offset);
    distribute_for(hc_size, block, [&](index_type local_idx) {
        index_type global_idx = initial_offset + global_offset<Profile>(local_idx, data_size);
        f(global_idx, local_idx);
    });
}


template<typename Profile>
__device__ void
load_hypercube(hypercube_block<Profile> block, index_type hc_index, const typename Profile::value_type *data,
        const static_extent<Profile::dimensions> &data_size, hypercube_ptr<Profile, forward_transform_tag> hc) {
    using bits_type = typename Profile::bits_type;

    for_hypercube_indices<Profile>(block, hc_index, data_size, [&](index_type global_idx, index_type local_idx) {
        hc.store(local_idx, rotate_left_1(bit_cast<bits_type>(data[global_idx])));
        // TODO merge with block_transform to avoid LM round-trip?
        //  we could assign directly to registers
    });
}

template<typename Profile>
__device__ void store_hypercube(hypercube_block<Profile> block, index_type hc_index, typename Profile::value_type *data,
        const static_extent<Profile::dimensions> &data_size, hypercube_ptr<Profile, inverse_transform_tag> hc) {
    using value_type = typename Profile::value_type;
    for_hypercube_indices<Profile>(block, hc_index, data_size, [&](index_type global_idx, index_type local_idx) {
        data[global_idx] = bit_cast<value_type>(rotate_right_1(hc.load(local_idx)));
    });
}


template<dim_type Direction, typename Profile>
__device__ void
forward_transform_lanes(hypercube_block<Profile> block, hypercube_ptr<Profile, forward_transform_tag> hc) {
    using bits_type = typename Profile::bits_type;
    using accessor = directional_accessor<Profile, Direction, forward_transform_tag>;
    using layout = typename accessor::layout;

    // 1D and 2D transforms have multiple lanes per row, so a barrier is required to synchronize
    // the read of the last element from (lane-1) with the write to the last element of (lane)
    constexpr bool needs_carry = layout::side_length > layout::lane_length;
    // std::max: size might be zero if !needs_carry, but that is not a valid type
    bits_type carry[std::max(index_type{1}, layout::num_lanes / hypercube_group_size<Profile>)];

    if constexpr (needs_carry) {
        distribute_for<layout::num_lanes>(block, [&](index_type lane, index_type iteration) {
            if (auto prev_lane = accessor::prev_lane_in_row(lane); prev_lane != no_such_lane) {
                // TODO this load causes a bank conflict in the 1-dimensional case. Why?
                carry[iteration] = hc.load(accessor::offset(prev_lane) + (layout::lane_length - 1) * accessor::stride);
            } else {
                carry[iteration] = 0;
            }
        });
        __syncthreads();
    }

    distribute_for<layout::num_lanes>(block, [&](index_type lane, index_type iteration) {
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
__device__ void
forward_block_transform(hypercube_block<Profile> block, hypercube_ptr<Profile, forward_transform_tag> hc) {
    constexpr auto dims = Profile::dimensions;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, dims);

    // Why is there no constexpr for?
    forward_transform_lanes<0>(block, hc);
    __syncthreads();
    if constexpr (dims >= 2) {
        forward_transform_lanes<1>(block, hc);
        __syncthreads();
    }
    if constexpr (dims >= 3) {
        forward_transform_lanes<2>(block, hc);
        __syncthreads();
    }

    // TODO move complement operation elsewhere to avoid local memory round-trip
    distribute_for(hc_size, block, [&](index_type item) { hc.store(item, complement_negative(hc.load(item))); });
}


template<dim_type Direction, typename Profile>
__device__ void
inverse_transform_lanes(hypercube_block<Profile> block, hypercube_ptr<Profile, inverse_transform_tag> hc) {
    using bits_type = typename Profile::bits_type;
    using accessor = directional_accessor<Profile, Direction, inverse_transform_tag>;
    using layout = typename accessor::layout;

    distribute_for<layout::num_lanes>(block, [&](index_type lane) {
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
__device__ void
inverse_block_transform(hypercube_block<Profile> block, hypercube_ptr<Profile, inverse_transform_tag> hc) {
    using bits_type = typename Profile::bits_type;
    constexpr auto dims = Profile::dimensions;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, dims);

    // TODO move complement operation elsewhere to avoid local memory round-trip
    distribute_for(hc_size, block, [&](index_type item) { hc.store(item, complement_negative(hc.load(item))); });
    __syncthreads();

    // TODO how to do 2D?
    //   For 2D we have 64 parallel work items but we _probably_ want at least 256 threads per SM
    //   (= per HC for double) to hide latencies. Maybe hybrid approach - do (64/32)*64 sub-group
    //   prefix sums and optimize inclusive_prefix_sum to skip the recursion since the second
    //   level does not actually need a reduction. Benchmark against leaving 256-64 = 192 threads
    //   idle and going with a sequential-per-lane transform.

    if constexpr (dims == 1) {
        // 1D inverse hypercube_ptr guarantees linear layout of hc.memory
        inclusive_scan<Profile::hypercube_side_length>(block, hc.memory(), plus<bits_type>{});
    }
    if constexpr (dims == 2) {
        // TODO inefficient, see above
        inverse_transform_lanes<0>(block, hc);
        __syncthreads();
        inverse_transform_lanes<1>(block, hc);
    }
    if constexpr (dims == 3) {
        inverse_transform_lanes<0>(block, hc);
        __syncthreads();
        inverse_transform_lanes<1>(block, hc);
        __syncthreads();
        inverse_transform_lanes<2>(block, hc);
    }
}

template<typename Profile>
__device__ void
write_transposed_chunks(hypercube_block<Profile> block, hypercube_ptr<Profile, forward_transform_tag> hc,
        typename Profile::bits_type *out_chunks, index_type *out_lengths) {
    using bits_type = typename Profile::bits_type;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr index_type col_chunk_size = bits_of<bits_type>;
    constexpr index_type num_col_chunks = hc_size / col_chunk_size;
    constexpr index_type header_chunk_size = num_col_chunks;
    constexpr index_type warps_per_col_chunk = col_chunk_size / warp_size;

    static_assert(col_chunk_size % warp_size == 0);

    if (threadIdx.x == 0) { out_lengths[0] = header_chunk_size; }

    const auto out_header_chunk = out_chunks;

    // Schedule one warp per chunk to allow subgroup reductions
    distribute_for(num_col_chunks * warp_size, block, [&](index_type item) {
        const auto col_chunk_index = item / warp_size;
        const auto in_col_chunk_base = col_chunk_index * col_chunk_size;

        // Collectively determine the head for a chunk
        bits_type head = 0;
        for (index_type w = 0; w < warps_per_col_chunk; ++w) {
            auto row = hc.load(in_col_chunk_base + w * warp_size + item % warp_size);
            head |= warp_reduce(row, bit_or<bits_type>{});
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
            vec<uint32_t, warps_per_col_chunk> columns[warps_per_col_chunk];
            memset(&columns, 0, sizeof columns);
#pragma unroll
            for (index_type i = 0; i < warps_per_col_chunk; ++i) {
                for (index_type j = 0; j < warp_size; ++j) {
                    auto row = hc.template load<vec<uint32_t, warps_per_col_chunk>>(
                            in_col_chunk_base + col_chunk_size - 1 - (warp_size * i + j));
#pragma unroll
                    for (index_type w = 0; w < warps_per_col_chunk; ++w) {
                        columns[w][i] |= ((row[warps_per_col_chunk - 1 - w] >> (31 - item % warp_size)) & uint32_t{1})
                                << j;
                    }
                }
            }

#pragma unroll
            for (index_type w = 0; w < warps_per_col_chunk; ++w) {
                auto column_bits = bit_cast<bits_type>(columns[w]);
                auto pos_in_out_col_chunk = compact_warp_offset[w]
                        + warp_exclusive_scan(index_type{column_bits != 0}, index_type{0}, plus<index_type>{});
                if (column_bits != 0) { out_col_chunk[pos_in_out_col_chunk] = column_bits; }
            }
        }

        if (threadIdx.x % warp_size == 0) {
            // TODO collect in local memory, write coalesced - otherwise 3 full GM
            //  transaction per HC instead of 1! => But only if WC does not resolve this
            //  anyway
            out_header_chunk[col_chunk_index] = head;
            out_lengths[1 + col_chunk_index] = chunk_compact_size;
        }
    });
}


template<typename Profile>
__device__ void read_transposed_chunks(hypercube_block<Profile> block, hypercube_ptr<Profile, inverse_transform_tag> hc,
        const typename Profile::bits_type *stream) {
    using bits_type = typename Profile::bits_type;
    using word_type = uint32_t;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr index_type col_chunk_size = bits_of<bits_type>;
    constexpr index_type num_col_chunks = hc_size / col_chunk_size;
    constexpr index_type warps_per_col_chunk = col_chunk_size / warp_size;
    constexpr index_type word_size = bits_of<word_type>;
    constexpr index_type words_per_col = sizeof(bits_type) / sizeof(word_type);

    __shared__ index_type chunk_offsets[1 + num_col_chunks];
    if (threadIdx.x == 0) { chunk_offsets[0] = num_col_chunks; }
    distribute_for(num_col_chunks, block, [&](index_type item) { chunk_offsets[1 + item] = popcount(stream[item]); });
    __syncthreads();

    inclusive_scan<num_col_chunks + 1>(block, chunk_offsets, plus<index_type>());
    __syncthreads();

    __shared__ index_type stage_mem[hypercube_group_size<Profile> * ipow(words_per_col, 2)];

    // See write_transposed_chunks for the optimization of the 64-bit case
    distribute_for(num_col_chunks * warp_size, block, [&](index_type item0) {
        auto col_chunk_index = item0 / warp_size;
        auto head = stream[col_chunk_index];  // TODO this has been read before, stage?

        if (head != 0) {
            word_type head_col[words_per_col];
            __builtin_memcpy(head_col, &head, sizeof head_col);
            auto offset = chunk_offsets[col_chunk_index];

            // TODO this can be hoisted out of the loop within distribute_for. Maybe
            //  write that loop explicitly for this purpose?
            auto *stage = &stage_mem[ipow(words_per_col, 2) * floor(static_cast<index_type>(threadIdx.x), warp_size)];

            // TODO There is an excellent opportunity to hide global memory latencies by
            //  starting to read values for outer-loop iteration n+1 into registers and then
            //  just committing the reads to shared memory in the next iteration
#pragma unroll
            for (index_type w = 0; w < ipow(words_per_col, 2); ++w) {
                auto i = w * warp_size + item0 % warp_size;
                if (offset + i / words_per_col < chunk_offsets[col_chunk_index + 1]) {
                    // TODO this load is uncoalesced since offsets are not warp-aligned
                    stage[i] = load_aligned<word_type>(reinterpret_cast<const word_type *>(stream + offset) + i);
                }
            }
            __syncwarp();

            for (index_type w = 0; w < warps_per_col_chunk; ++w) {
                index_type item = floor(item0, warp_size) * warps_per_col_chunk + w * warp_size + item0 % warp_size;
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
                index_type item = floor(item0, warp_size) * warps_per_col_chunk + w * warp_size + item0 % warp_size;
                hc.store(item, 0);
            }
        }
    });
}


template<typename Profile>
__device__ void compact_chunks(hypercube_block<Profile> block, const typename Profile::bits_type *chunks,
        const index_type *offsets, index_type *stream_header_entry, typename Profile::bits_type *stream_hc) {
    using bits_type = typename Profile::bits_type;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr index_type col_chunk_size = bits_of<bits_type>;
    constexpr index_type header_chunk_size = hc_size / col_chunk_size;
    constexpr index_type hc_total_chunks_size = hc_size + header_chunk_size;
    constexpr index_type chunks_per_hc = 1 /* header */ + hc_size / col_chunk_size;

    __shared__ index_type hc_offsets[chunks_per_hc + 1];
    distribute_for(chunks_per_hc + 1, block, [&](index_type i) { hc_offsets[i] = offsets[i]; });
    __syncthreads();

    if (threadIdx.x == 0) {
        *stream_header_entry = hc_offsets[chunks_per_hc];  // header encodes offset *after*
    }

    distribute_for(hc_total_chunks_size, block, [&](index_type item) {
        index_type chunk_rel_item = item;
        index_type chunk_index = 0;
        if (item >= header_chunk_size) {
            chunk_index += 1 + (item - header_chunk_size) / col_chunk_size;
            chunk_rel_item = (item - header_chunk_size) % col_chunk_size;
        }
        auto stream_offset = hc_offsets[chunk_index] + chunk_rel_item;
        if (stream_offset < hc_offsets[chunk_index + 1]) { stream_hc[stream_offset] = chunks[item]; }
    });
}


template<typename Profile>
__global__ void compress_block(const typename Profile::value_type *data, static_extent<Profile::dimensions> data_size,
        typename Profile::bits_type *chunks, index_type *chunk_lengths) {
    using bits_type = typename Profile::bits_type;

    constexpr index_type dimensions = Profile::dimensions;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, dimensions);
    constexpr index_type col_chunk_size = bits_of<bits_type>;
    constexpr index_type header_chunk_size = hc_size / col_chunk_size;
    constexpr index_type chunks_per_hc = 1 /* header */ + hc_size / col_chunk_size;
    constexpr index_type hc_total_chunks_size = hc_size + header_chunk_size;

    __shared__ hypercube_allocation<Profile, forward_transform_tag> lm;
    hypercube_ptr<Profile, forward_transform_tag> hc{lm};

    auto hc_index = static_cast<index_type>(blockIdx.x);
    auto block = hypercube_block<Profile>{};
    load_hypercube(block, hc_index, data, data_size, hc);
    __syncthreads();
    forward_block_transform(block, hc);
    __syncthreads();
    write_transposed_chunks(
            block, hc, chunks + hc_index * hc_total_chunks_size, chunk_lengths + 1 + hc_index * chunks_per_hc);
    // hack
    if (blockIdx.x == 0 && threadIdx.x == 0) { chunk_lengths[0] = 0; }
}


template<typename Profile>
__global__ void compact_all_chunks(index_type num_hypercubes, const typename Profile::bits_type *chunks,
        const index_type *offsets, typename Profile::bits_type *stream_buf) {
    using bits_type = typename Profile::bits_type;

    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr index_type col_chunk_size = bits_of<bits_type>;
    constexpr index_type header_chunk_size = hc_size / col_chunk_size;
    constexpr index_type chunks_per_hc = 1 /* header */ + hc_size / col_chunk_size;
    constexpr index_type hc_total_chunks_size = hc_size + header_chunk_size;

    const auto hc_index = static_cast<index_type>(blockIdx.x);
    auto block = hypercube_block<Profile>{};

    detail::stream<Profile> stream{num_hypercubes, stream_buf};
    if (hc_index == num_hypercubes) {
        // For 64-bit data types and an odd number of hypercubes, we insert a padding word in the
        // header to guarantee correct alignment. To keep the compressed stream deterministic we
        // round gridDim.x up to the next multiple of 2 and initialize the padding to zero.
        if (threadIdx.x == 0) { stream.header()[num_hypercubes] = 0; }
    } else {
        compact_chunks<Profile>(block, chunks + hc_index * hc_total_chunks_size, offsets + hc_index * chunks_per_hc,
                stream.header() + hc_index, stream.hypercube(0));
    }
}


constexpr static index_type border_threads_per_block = 256;


template<typename Profile>
__global__ void compact_border(const typename Profile::value_type *data, static_extent<Profile::dimensions> data_size,
        const index_type *num_compressed_words, typename Profile::bits_type *stream_buf, index_type num_header_words,
        border_map<Profile> border_map) {
    using bits_type = typename Profile::bits_type;

    auto border_offset = num_header_words + *num_compressed_words;
    auto i = static_cast<index_type>(blockIdx.x * border_threads_per_block + threadIdx.x);
    if (i < border_map.size()) {
        stream_buf[border_offset + i] = bit_cast<bits_type>(data[linear_index(data_size, border_map[i])]);
    }
}


template<typename Profile>
__global__ void decompress_block(const typename Profile::bits_type *stream_buf, typename Profile::value_type *data,
        static_extent<Profile::dimensions> data_size) {
    auto block = hypercube_block<Profile>{};
    __shared__ hypercube_allocation<Profile, inverse_transform_tag> lm;
    hypercube_ptr<Profile, inverse_transform_tag> hc{lm};

    const auto num_hypercubes = static_cast<index_type>(gridDim.x);
    const auto hc_index = static_cast<index_type>(blockIdx.x);
    detail::stream<const Profile> stream{num_hypercubes, stream_buf};
    read_transposed_chunks<Profile>(block, hc, stream.hypercube(hc_index));
    __syncthreads();
    inverse_block_transform<Profile>(block, hc);
    __syncthreads();
    store_hypercube(block, hc_index, data, data_size, hc);
}


template<typename Profile>
__global__ void expand_border(const typename Profile::bits_type *stream_buf, typename Profile::value_type *data,
        static_extent<Profile::dimensions> data_size, border_map<Profile> border_map, index_type num_hypercubes) {
    using value_type = typename Profile::value_type;
    if (auto i = static_cast<index_type>(blockIdx.x * border_threads_per_block + threadIdx.x); i < border_map.size()) {
        detail::stream<const Profile> stream{num_hypercubes, stream_buf};
        const auto border_offset = static_cast<index_type>(stream.border() - stream.buffer);
        data[linear_index(data_size, border_map[i])] = bit_cast<value_type>(stream_buf[border_offset + i]);
    }
}


template<typename>  // Does not need to be a template, but inline does not work on kernels
__global__ void store_stream_length(
        index_type *stream_length, const index_type *num_compressed_words, index_type num_header_and_border_words) {
    *stream_length = num_header_and_border_words + *num_compressed_words;
}


template<typename Profile>
class cuda_compressor_impl final : public cuda_compressor<typename Profile::value_type> {
  public:
    using value_type = typename Profile::value_type;
    using bits_type = typename Profile::bits_type;

    explicit cuda_compressor_impl(cudaStream_t stream, const compressor_requirements &reqs);

    void compress(const value_type *in_device_data, const extent &data_size, bits_type *out_device_stream,
            index_type *out_device_stream_length) override;

  private:
    constexpr static auto dimensions = Profile::dimensions;
    constexpr static index_type hc_size = detail::ipow(Profile::hypercube_side_length, dimensions);
    constexpr static index_type col_chunk_size = detail::bits_of<bits_type>;
    constexpr static index_type header_chunk_size = hc_size / col_chunk_size;
    constexpr static index_type hc_total_chunks_size = hc_size + header_chunk_size;
    constexpr static index_type chunks_per_hc = 1 /* header */ + hc_size / col_chunk_size;

    template<typename>
    friend class cuda_offloader;

    cuda_buffer<bits_type> chunks_buf;
    cuda_buffer<index_type> chunk_lengths_buf;
    std::vector<detail::gpu_cuda::cuda_buffer<index_type>> intermediate_bufs;
    cudaStream_t _stream;
};


template<typename Profile>
cuda_compressor_impl<Profile>::cuda_compressor_impl(cudaStream_t stream, const compressor_requirements &reqs)
    : _stream{stream} {
    const auto num_hypercubes = detail::get_num_hypercubes(reqs);
    const auto num_chunks = num_hypercubes * (1 + hc_size / col_chunk_size);

    chunks_buf.allocate(num_hypercubes * hc_total_chunks_size);
    chunk_lengths_buf.allocate(detail::ceil(1 + num_chunks, detail::gpu_cuda::hierarchical_inclusive_scan_granularity));
    intermediate_bufs = detail::gpu_cuda::hierarchical_inclusive_scan_allocate<index_type>(chunk_lengths_buf.size());
}

template<typename Profile>
void cuda_compressor_impl<Profile>::compress(const value_type *in_device_data, const extent &data_size,
        bits_type *out_device_stream, index_type *out_device_stream_length) {
    if (data_size.dimensions() != dimensions) {
        throw std::runtime_error{"data dimensionality does not match compressor dimensionality"};
    }

    // TODO edge case w/ 0 hypercubes

    const auto static_size = detail::static_extent<dimensions>{data_size};
    const auto num_hypercubes = detail::num_hypercubes(static_size);
    if (verbose()) { printf("Have %u hypercubes\n", num_hypercubes); }

    compress_block<Profile><<<num_hypercubes, (hypercube_group_size<Profile>), 0, _stream>>>(
            in_device_data, static_size, chunks_buf.get(), chunk_lengths_buf.get());

    hierarchical_inclusive_scan(
            chunk_lengths_buf.get(), intermediate_bufs, chunk_lengths_buf.size(), plus<index_type>{}, _stream);

    const auto num_compressed_words_offset = num_hypercubes * chunks_per_hc;

    const auto num_header_fields
            = detail::ceil(num_hypercubes, static_cast<uint32_t>(sizeof(bits_type) / sizeof(index_type)));
    compact_all_chunks<Profile><<<num_header_fields, (hypercube_group_size<Profile>), 0, _stream>>>(
            num_hypercubes, chunks_buf.get(), chunk_lengths_buf.get(), static_cast<bits_type *>(out_device_stream));

    const auto border_map = gpu::border_map<Profile>{static_size};
    const auto num_border_words = border_map.size();

    detail::stream<Profile> stream{num_hypercubes, static_cast<bits_type *>(out_device_stream)};
    const auto num_header_words = stream.hypercube(0) - stream.buffer;
    // TODO num_header_words == num_header_fields ??

    if (num_border_words > 0) {
        const index_type border_blocks = div_ceil(num_border_words, border_threads_per_block);
        compact_border<Profile><<<border_blocks, border_threads_per_block, 0, _stream>>>(in_device_data, static_size,
                chunk_lengths_buf.get() + num_compressed_words_offset, static_cast<bits_type *>(out_device_stream),
                num_header_words, border_map);
    }

    if (out_device_stream_length) {
        store_stream_length<Profile><<<1, 1, 0, _stream>>>(out_device_stream_length,
                chunk_lengths_buf.get() + num_compressed_words_offset, num_header_words + num_border_words);
    }
}

template<typename Profile>
class cuda_decompressor_impl final : public cuda_decompressor<typename Profile::value_type> {
  public:
    using value_type = typename Profile::value_type;
    using bits_type = typename Profile::bits_type;

    cuda_decompressor_impl() = default;

    explicit cuda_decompressor_impl(cudaStream_t stream) : _stream(stream) {}

    void decompress(const bits_type *in_device_stream, value_type *out_device_data, const extent &data_size) override;

  private:
    constexpr static auto dimensions = Profile::dimensions;
    constexpr static index_type hc_size = detail::ipow(Profile::hypercube_side_length, dimensions);
    constexpr static index_type col_chunk_size = detail::bits_of<bits_type>;
    constexpr static index_type header_chunk_size = hc_size / col_chunk_size;
    constexpr static index_type hc_total_chunks_size = hc_size + header_chunk_size;
    constexpr static index_type chunks_per_hc = 1 /* header */ + hc_size / col_chunk_size;

    cudaStream_t _stream = nullptr;
};

template<typename Profile>
void cuda_decompressor_impl<Profile>::decompress(
        const bits_type *in_device_stream, value_type *out_device_data, const extent &data_size) {
    if (data_size.dimensions() != dimensions) {
        throw std::runtime_error{"data dimensionality does not match compressor dimensionality"};
    }

    const auto static_size = detail::static_extent<dimensions>{data_size};
    const auto num_hypercubes = detail::num_hypercubes(static_size);

    decompress_block<Profile><<<num_hypercubes, (hypercube_group_size<Profile>), 0, _stream>>>(
            static_cast<const bits_type *>(in_device_stream), out_device_data, static_size);

    const auto border_map = gpu::border_map<Profile>{static_size};
    const auto num_border_words = border_map.size();

    if (num_border_words > 0) {
        const index_type border_blocks = div_ceil(num_border_words, border_threads_per_block);
        expand_border<Profile><<<border_blocks, border_threads_per_block, 0, _stream>>>(
                static_cast<const bits_type *>(in_device_stream), out_device_data, static_size, border_map,
                num_hypercubes);
    }
}

template<typename Profile>
class cuda_offloader final : public offloader<typename Profile::value_type> {
  public:
    using value_type = typename Profile::value_type;
    using bits_type = typename Profile::bits_type;
    constexpr static dim_type dimensions = Profile::dimensions;

  protected:
    index_type do_compress(
            const value_type *data, const extent &data_size, bits_type *stream, kernel_duration *duration) override;

    index_type do_decompress(const bits_type *stream, index_type length, value_type *data, const extent &data_size,
            kernel_duration *duration) override;
};

template<typename Profile>
index_type cuda_offloader<Profile>::do_compress(
        const value_type *data, const extent &data_size, bits_type *raw_stream, kernel_duration *out_kernel_duration) {
    // TODO edge case w/ 0 hypercubes

    const auto num_hypercubes = detail::num_hypercubes(data_size);
    if (verbose()) { printf("Have %u hypercubes\n", num_hypercubes); }

    cuda_buffer<value_type> data_buffer{num_elements(data_size)};
    CHECKED_CUDA_CALL(
            cudaMemcpy, data_buffer.get(), data, data_buffer.size() * bytes_of<value_type>, cudaMemcpyHostToDevice);

    cuda_compressor_impl<Profile> compressor{nullptr /* stream */, data_size};
    cuda_buffer<bits_type> stream_buf{compressed_length_bound<value_type>(data_size)};
    cuda_buffer<index_type> stream_length_buf{1};

    cuda_event start, stop;
    bool record_events = out_kernel_duration || verbose();
    if (record_events) { start.record(); }

    compressor.compress(data_buffer.get(), data_size, stream_buf.get(), stream_length_buf.get());

    if (record_events) {
        stop.record();
        auto duration = stop - start;
        if (out_kernel_duration) { *out_kernel_duration = duration; }
        if (verbose()) { printf("[profile] total kernel time %.3fms\n", duration.count() * 1e-6); }
    }

    index_type stream_length;
    CHECKED_CUDA_CALL(
            cudaMemcpy, &stream_length, stream_length_buf.get(), sizeof stream_length, cudaMemcpyDeviceToHost);

    const auto stream_size = stream_length * sizeof(bits_type);
    CHECKED_CUDA_CALL(cudaMemcpy, raw_stream, stream_buf.get(), stream_size, cudaMemcpyDeviceToHost);

    return stream_length;
}

template<typename Profile>
index_type cuda_offloader<Profile>::do_decompress(const bits_type *raw_stream, index_type length, value_type *data,
        const extent &data_size, kernel_duration *out_kernel_duration) {
    if (data_size.dimensions() != dimensions) {
        throw std::runtime_error{"data dimensionality does not match decompressor dimensionality"};
    }

    const auto static_size = static_extent<dimensions>{data_size};
    const auto num_hypercubes = detail::num_hypercubes(static_size);

    // TODO the range computation here is questionable at best
    cuda_buffer<bits_type> stream_buf{length};
    cuda_buffer<value_type> data_buf{num_elements(data_size)};

    CHECKED_CUDA_CALL(cudaMemcpy, stream_buf.get(), raw_stream, length * sizeof(bits_type), cudaMemcpyHostToDevice);

    cuda_event start, stop;
    bool record_events = out_kernel_duration || verbose();
    if (record_events) { start.record(); }

    cuda_decompressor_impl<Profile>{nullptr /* stream */}.decompress(stream_buf.get(), data_buf.get(), data_size);

    const auto border_map = gpu::border_map<Profile>{static_size};
    const auto num_border_words = border_map.size();

    detail::stream<const Profile> stream{num_hypercubes, static_cast<const bits_type *>(raw_stream)};
    const auto border_offset = static_cast<index_type>(stream.border() - stream.buffer);
    const auto num_stream_words = border_offset + num_border_words;

    if (record_events) {
        stop.record();
        auto duration = stop - start;
        if (out_kernel_duration) { *out_kernel_duration = duration; }
        if (verbose()) { printf("[profile] total kernel time %.3fms\n", duration.count() * 1e-6); }
    }

    CHECKED_CUDA_CALL(cudaMemcpy, data, data_buf.get(), data_buf.size() * sizeof(value_type), cudaMemcpyDeviceToHost);

    return num_stream_words;
}

extern template class cuda_offloader<profile<float, 1>>;
extern template class cuda_offloader<profile<float, 2>>;
extern template class cuda_offloader<profile<float, 3>>;
extern template class cuda_offloader<profile<double, 1>>;
extern template class cuda_offloader<profile<double, 2>>;
extern template class cuda_offloader<profile<double, 3>>;

#ifdef SPLIT_CONFIGURATION_cuda_encoder
template class cuda_offloader<profile<DATA_TYPE, DIMENSIONS>>;
#endif

}  // namespace ndzip::detail::gpu_cuda
