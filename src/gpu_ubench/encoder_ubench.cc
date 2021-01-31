#include "ubench.hh"

#include <ndzip/gpu_encoder.inl>

using namespace ndzip::detail;
using namespace ndzip::detail::gpu;
using sam = sycl::access::mode;


#define ALL_PROFILES \
    (profile<float, 1>), (profile<float, 2>), (profile<float, 3>), (profile<double, 1>), \
            (profile<double, 2>), (profile<double, 3>)

// Kernel names (for profiler)
template<typename>
class block_transform_reference_kernel;
template<typename>
class block_forward_transform_kernel;
template<typename>
class block_inverse_transform_kernel;
template<typename>
class encode_reference_kernel;
template<typename>
class chunk_transpose_kernel;
template<typename>
class chunk_compact_kernel;
template<typename>
class chunk_encode_kernel;


TEMPLATE_TEST_CASE("Block transform", "[transform]", ALL_PROFILES) {
    constexpr index_type n_blocks = 16384;

    SYCL_BENCHMARK("Reference: rotate only")(sycl::queue & q) {
        using bits_type = typename TestType::bits_type;
        constexpr auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);

        sycl::buffer<bits_type> out(n_blocks * hc_size);
        return q.submit([&](sycl::handler &cgh) {
            auto g = out.template get_access<sam::discard_write>(cgh);
            cgh.parallel<block_transform_reference_kernel<TestType>>(sycl::range<1>{n_blocks},
                    sycl::range<1>{hypercube_group_size},
                    [=](hypercube_group grp, sycl::physical_item<1>) {
                        sycl::local_memory<bits_type[hypercube<TestType>::allocation_size]> lm{grp};
                        hypercube<TestType> hc{&lm[0]};
                        grp.distribute_for(hc_size, [&](index_type i) { hc[i] = i; });
                        const auto hc_index = grp.get_id(0);
                        grp.distribute_for(hc_size, [&](index_type i) {
                            g[hc_index * hc_size + i] = rotate_left_1(hc[i]);
                        });
                    });
        });
    };

    SYCL_BENCHMARK("Forward transform")(sycl::queue & q) {
        using bits_type = typename TestType::bits_type;
        constexpr auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);

        sycl::buffer<bits_type> out(n_blocks * hc_size);
        return q.submit([&](sycl::handler &cgh) {
            auto g = out.template get_access<sam::discard_write>(cgh);
            cgh.parallel<block_forward_transform_kernel<TestType>>(sycl::range<1>{n_blocks},
                    sycl::range<1>{hypercube_group_size},
                    [=](hypercube_group grp, sycl::physical_item<1>) {
                        sycl::local_memory<bits_type[hypercube<TestType>::allocation_size]> lm{grp};
                        hypercube<TestType> hc{&lm[0]};
                        grp.distribute_for(hc_size, [&](index_type i) { hc[i] = i; });
                        block_transform(grp, hc);
                        const auto hc_index = grp.get_id(0);
                        grp.distribute_for(
                                hc_size, [&](index_type i) { g[hc_index * hc_size + i] = hc[i]; });
                    });
        });
    };

    SYCL_BENCHMARK("Inverse transform")(sycl::queue & q) {
        using bits_type = typename TestType::bits_type;
        constexpr auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);

        sycl::buffer<bits_type> out(n_blocks * hc_size);
        return q.submit([&](sycl::handler &cgh) {
            auto g = out.template get_access<sam::discard_write>(cgh);
            cgh.parallel<block_inverse_transform_kernel<TestType>>(sycl::range<1>{n_blocks},
                    sycl::range<1>{hypercube_group_size},
                    [=](hypercube_group grp, sycl::physical_item<1>) {
                        sycl::local_memory<bits_type[hypercube<TestType>::allocation_size]> lm{grp};
                        hypercube<TestType> hc{&lm[0]};
                        grp.distribute_for(hc_size, [&](index_type i) { hc[i] = i; });
                        inverse_block_transform(grp, hc);
                        const auto hc_index = grp.get_id(0);
                        grp.distribute_for(
                                hc_size, [&](index_type i) { g[hc_index * hc_size + i] = hc[i]; });
                    });
        });
    };
}


template<typename Profile>
void write_transposed_chunks(hypercube_group grp, hypercube<Profile> hc,
        typename Profile::bits_type *out_heads,
        typename Profile::bits_type *out_columns, index_type *out_lengths) {
    using bits_type = typename Profile::bits_type;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    static_assert(hc_size % warp_size == 0);

    // One group per warp (for subgroup reductions)
    constexpr index_type chunk_size = bitsof<bits_type>;

    // TODO even if there is a use case for index_type > 32 bits, this should always be 32 bits
    grp.distribute_for(hc_size, [&](index_type item, index_type iteration,
                                        sycl::logical_item<1> idx, sycl::sub_group sg) {
        auto warp_index = item / warp_size;
        auto head = sycl::group_reduce(sg, hc[item], sycl::bit_or<bits_type>{});
        index_type this_warp_size = 0;
        bits_type column = 0;
        if (head != 0) {
            const auto chunk_base = floor(item, chunk_size);
            const auto cell = item - chunk_base;
            for (index_type i = 0; i < chunk_size; ++i) {
                column |= (hc[chunk_base + i] >> (chunk_size - 1 - cell) & bits_type{1})
                        << (chunk_size - 1 - i);
            }
            if constexpr (sizeof(bits_type) == 4) {
                this_warp_size = __builtin_popcount(head);
            } else {
                this_warp_size = __builtin_popcountl(head);
            }
        }
        if (warp_index % (chunk_size / warp_size) == 0) {
            this_warp_size += 1;
        }
        out_columns[item] = column;
        if (sg.leader()) {
            out_heads[warp_index] = head;
            out_lengths[warp_index] = this_warp_size;
        }
    });
}


template<typename Profile>
void compact_chunks(hypercube_group grp, const typename Profile::bits_type *heads,
        const typename Profile::bits_type *columns, const index_type *offsets,
        typename Profile::bits_type *stream) {
    using bits_type = typename Profile::bits_type;
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    static_assert(hc_size % warp_size == 0);

    // One group per warp (for subgroup reductions)
    constexpr index_type chunk_size = bitsof<bits_type>;

    grp.distribute_for(hc_size, [&](index_type item, index_type iteration,
                                        sycl::logical_item<1> idx, sycl::sub_group sg) {
        auto warp_index = item / warp_size;

        auto offset = offsets[warp_index];
        if (warp_index % (chunk_size / warp_size) == 0) {
            bits_type head = 0;
            for (index_type i = 0; i < (chunk_size / warp_size); ++i) {
                head |= heads[warp_index + i];
            }
            if (grp.leader()) {
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


// Impact of dimensionality should not be that large, but the hc padding could hold surprises
TEMPLATE_TEST_CASE("Chunk encoding", "[encode]", (profile<float, 1>), (profile<double, 1>)) {
    constexpr index_type n_blocks = 16384;
    using bits_type = typename TestType::bits_type;
    constexpr auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);

    SYCL_BENCHMARK("Reference: serialize")(sycl::queue & q) {
        sycl::buffer<bits_type> out(n_blocks * hc_size);
        return q.submit([&](sycl::handler &cgh) {
            auto g = out.template get_access<sam::discard_write>(cgh);
            cgh.parallel<encode_reference_kernel<TestType>>(sycl::range<1>{n_blocks},
                    sycl::range<1>{hypercube_group_size},
                    [=](hypercube_group grp, sycl::physical_item<1>) {
                        sycl::local_memory<bits_type[hypercube<TestType>::allocation_size]> lm{grp};
                        hypercube<TestType> hc{&lm[0]};
                        grp.distribute_for(hc_size, [&](index_type i) { hc[i] = i; });
                        const auto hc_index = grp.get_id(0);
                        grp.distribute_for(
                                hc_size, [&](index_type i) { g[hc_index * hc_size + i] = hc[i]; });
                    });
        });
    };

    sycl::buffer<bits_type> columns(n_blocks * hc_size);
    sycl::buffer<bits_type> heads(n_blocks * hc_size / warp_size);
    sycl::buffer<index_type> lengths(n_blocks * hc_size / warp_size);

    SYCL_BENCHMARK("Transpose only")(sycl::queue & q) {
        return q.submit([&](sycl::handler &cgh) {
            auto c = columns.template get_access<sam::discard_write>(cgh);
            auto h = heads.template get_access<sam::discard_write>(cgh);
            auto l = lengths.template get_access<sam::discard_write>(cgh);
            cgh.parallel<chunk_transpose_kernel<TestType>>(sycl::range<1>{n_blocks},
                    sycl::range<1>{hypercube_group_size},
                    [=](hypercube_group grp, sycl::physical_item<1>) {
                        sycl::local_memory<bits_type[hypercube<TestType>::allocation_size]> lm{grp};
                        hypercube<TestType> hc{&lm[0]};
                        grp.distribute_for(hc_size, [&](index_type i) { hc[i] = i * 199; });
                        const auto hc_index = grp.get_id(0);
                        write_transposed_chunks(grp, hc, &h[hc_index], &c[hc_index * hc_size], &l[hc_index]);
                    });
        });
    };

    hierarchical_inclusive_prefix_sum<index_type> prefix_sum(
            n_blocks * hc_size / warp_size, 256 /* local size */);
    sycl::queue q;
    prefix_sum(q, lengths);
    sycl::buffer<bits_type> stream(n_blocks * (hc_size + hc_size / warp_size));

    SYCL_BENCHMARK("Compact transposed")(sycl::queue & q) {
        return q.submit([&](sycl::handler &cgh) {
            auto c = columns.template get_access<sam::read_write>(cgh);
            auto h = heads.template get_access<sam::read_write>(cgh);
            auto l = lengths.template get_access<sam::read_write>(cgh);
            auto s = stream.template get_access<sam::discard_write>(cgh);
            cgh.parallel<chunk_compact_kernel<TestType>>(sycl::range<1>{n_blocks},
                    sycl::range<1>{hypercube_group_size},
                    [=](hypercube_group grp, sycl::physical_item<1>) {
                        const auto hc_index = grp.get_id(0);
                        compact_chunks<TestType>(grp, &h[hc_index], &c[hc_index * hc_size], &l[hc_index],
                                static_cast<bits_type*>(s.get_pointer()));
                    });
        });
    };

    SYCL_BENCHMARK("Encode")(sycl::queue & q) {
        const auto max_chunk_size = (TestType::compressed_block_size_bound + sizeof(bits_type) - 1)
                / sizeof(bits_type);
        sycl::buffer<bits_type> out(n_blocks * max_chunk_size);
        sycl::buffer<file_offset_type> lengths(n_blocks);
        return q.submit([&](sycl::handler &cgh) {
            auto g = out.template get_access<sam::discard_write>(cgh);
            auto l = lengths.template get_access<sam::discard_write>(cgh);
            cgh.parallel<chunk_encode_kernel<TestType>>(sycl::range<1>{n_blocks},
                    sycl::range<1>{hypercube_group_size},
                    [=](hypercube_group grp, sycl::physical_item<1>) {
                        sycl::local_memory<bits_type[hypercube<TestType>::allocation_size]> lm{grp};
                        hypercube<TestType> hc{&lm[0]};
                        grp.distribute_for(hc_size, [&](index_type i) { hc[i] = i * 199; });
                        const auto hc_index = grp.get_id(0);
                        encode_chunks(grp, hc, &g[hc_index * max_chunk_size], &l[hc_index]);
                    });
        });
    };
}
