#include "ubench.hh"

#include <ndzip/sycl_encoder.inl>
#include <test/test_utils.hh>

using namespace ndzip;
using namespace ndzip::detail;
using namespace ndzip::detail::gpu_sycl;
using sam = sycl::access::mode;


#define ALL_PROFILES (profile<DATA_TYPE, DIMENSIONS>)


// Kernel names (for profiler)
template<typename>
class load_hypercube_kernel;
template<typename>
class block_transform_reference_kernel;
template<typename>
class block_forward_transform_kernel;
template<typename>
class block_inverse_transform_kernel;
template<typename>
class encode_reference_kernel;
template<typename>
class chunk_transpose_write_kernel;
template<typename>
class chunk_transpose_read_kernel;
template<typename>
class chunk_compact_kernel;


TEMPLATE_TEST_CASE("Loading", "[load]", ALL_PROFILES) {
    using data_type = typename TestType::data_type;
    using bits_type = typename TestType::bits_type;

    constexpr unsigned dimensions = TestType::dimensions;
    constexpr index_type n_blocks = 16384;

    const auto grid_extent = [] {
        extent<TestType::dimensions> grid_extent;
        const auto n_blocks_regular = static_cast<index_type>(pow(n_blocks, 1.f / dimensions));
        auto n_blocks_to_distribute = n_blocks;
        for (unsigned d = 0; d < dimensions; ++d) {
            auto n_blocks_this_dim = std::min(n_blocks_regular, n_blocks_to_distribute);
            grid_extent[d] = n_blocks_this_dim * TestType::hypercube_side_length + 3 /* border */;
            n_blocks_to_distribute /= n_blocks_this_dim;
        }
        assert(n_blocks_to_distribute == 0);
        return grid_extent;
    }();

    const auto data = make_random_vector<data_type>(num_elements(grid_extent));
    sycl::buffer<data_type> data_buffer(data.data(), data.size());

    SYCL_BENCHMARK("Load hypercube")(sycl::queue & q) {
        constexpr auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);

        sycl::buffer<bits_type> out(n_blocks * hc_size);
        return q.submit([&](sycl::handler &cgh) {
            auto data_acc = data_buffer.template get_access<sam::read>(cgh);
            cgh.parallel<load_hypercube_kernel<TestType>>(sycl::range<1>{n_blocks},
                    sycl::range<1>{hypercube_group_size<TestType>},
                    [=](hypercube_group<TestType> grp, sycl::physical_item<1>) {
                        hypercube_memory<TestType, gpu::forward_transform_tag> lm{grp};
                        hypercube_ptr<TestType, gpu::forward_transform_tag> hc{lm()};
                        index_type hc_index = grp.get_id(0);
                        slice<const data_type, dimensions> data{
                                data_acc.get_pointer(), grid_extent};

                        load_hypercube(grp, hc_index, data, hc);

                        black_hole(hc.memory);
                    });
        });
    };
}


TEMPLATE_TEST_CASE("Block transform", "[transform]", ALL_PROFILES) {
    constexpr index_type n_blocks = 16384;

    SYCL_BENCHMARK("Reference: rotate only")(sycl::queue & q) {
        constexpr auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);

        return q.submit([&](sycl::handler &cgh) {
            cgh.parallel<block_transform_reference_kernel<TestType>>(sycl::range<1>{n_blocks},
                    sycl::range<1>{hypercube_group_size<TestType>},
                    [=](hypercube_group<TestType> grp, sycl::physical_item<1>) {
                        hypercube_memory<TestType, forward_transform_tag> lm{grp};
                        gpu::hypercube_ptr<TestType, forward_transform_tag> hc{lm()};
                        grp.distribute_for(hc_size, [&](index_type i) { hc.store(i, i); });
                        black_hole(hc.memory);
                    });
        });
    };

    SYCL_BENCHMARK("Forward transform")(sycl::queue & q) {
        constexpr auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);

        return q.submit([&](sycl::handler &cgh) {
            cgh.parallel<block_forward_transform_kernel<TestType>>(sycl::range<1>{n_blocks},
                    sycl::range<1>{hypercube_group_size<TestType>},
                    [=](hypercube_group<TestType> grp, sycl::physical_item<1>) {
                        hypercube_memory<TestType, forward_transform_tag> lm{grp};
                        gpu::hypercube_ptr<TestType, forward_transform_tag> hc{lm()};
                        grp.distribute_for(hc_size, [&](index_type i) { hc.store(i, i); });
                        forward_block_transform(grp, hc);
                        black_hole(hc.memory);
                    });
        });
    };

    SYCL_BENCHMARK("Inverse transform")(sycl::queue & q) {
        constexpr auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);

        return q.submit([&](sycl::handler &cgh) {
            cgh.parallel<block_inverse_transform_kernel<TestType>>(sycl::range<1>{n_blocks},
                    sycl::range<1>{hypercube_group_size<TestType>},
                    [=](hypercube_group<TestType> grp, sycl::physical_item<1>) {
                        hypercube_memory<TestType, inverse_transform_tag> lm{grp};
                        gpu::hypercube_ptr<TestType, inverse_transform_tag> hc{lm()};
                        grp.distribute_for(hc_size, [&](index_type i) { hc.store(i, i); });
                        inverse_block_transform(grp, hc);
                        black_hole(hc.memory);
                    });
        });
    };
}


// Impact of dimensionality should not be that large, but the hc padding could hold surprises
TEMPLATE_TEST_CASE("Chunk encoding", "[encode]", ALL_PROFILES) {
    constexpr index_type n_blocks = 16384;
    using bits_type = typename TestType::bits_type;
    constexpr auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);
    constexpr index_type col_chunk_size = bits_of<bits_type>;
    constexpr index_type header_chunk_size = hc_size / col_chunk_size;
    constexpr index_type hc_total_chunks_size = hc_size + header_chunk_size;
    constexpr index_type chunks_per_hc = 1 /* header */ + hc_size / col_chunk_size;

    SYCL_BENCHMARK("Reference: serialize")(sycl::queue & q) {
        return q.submit([&](sycl::handler &cgh) {
            cgh.parallel<encode_reference_kernel<TestType>>(sycl::range<1>{n_blocks},
                    sycl::range<1>{hypercube_group_size<TestType>},
                    [=](hypercube_group<TestType> grp, sycl::physical_item<1>) {
                        hypercube_memory<TestType, forward_transform_tag> lm{grp};
                        gpu::hypercube_ptr<TestType, forward_transform_tag> hc{lm()};
                        grp.distribute_for(hc_size, [&](index_type i) { hc.store(i, i); });
                        black_hole(hc.memory);
                    });
        });
    };

    sycl::buffer<bits_type> chunks(n_blocks * hc_total_chunks_size);
    sycl::buffer<index_type> lengths(
            ceil(1 + n_blocks * chunks_per_hc, hierarchical_inclusive_scan_granularity));

    SYCL_BENCHMARK("Transpose chunks")(sycl::queue & q) {
        return q.submit([&](sycl::handler &cgh) {
            auto c = chunks.template get_access<sam::discard_write>(cgh);
            auto l = lengths.template get_access<sam::discard_write>(cgh);
            cgh.parallel<chunk_transpose_write_kernel<TestType>>(sycl::range<1>{n_blocks},
                    sycl::range<1>{hypercube_group_size<TestType>},
                    [=](hypercube_group<TestType> grp, sycl::physical_item<1> phys_idx) {
                        hypercube_memory<TestType, forward_transform_tag> lm{grp};
                        gpu::hypercube_ptr<TestType, forward_transform_tag> hc{lm()};
                        // Set some to zero - test zero-head shortcut optimization
                        grp.distribute_for(hc_size,
                                [&](index_type i) { hc.store(i, (i > 512 ? i * 199 : 0)); });
                        const auto hc_index = grp.get_id(0);
                        write_transposed_chunks(grp, hc, &c[hc_index * hc_total_chunks_size],
                                &l[1 + hc_index * chunks_per_hc]);
                        // hack
                        if (phys_idx.get_global_linear_id() == 0) {
                            grp.single_item([&] { l[0] = 0; });
                        }
                    });
        });
    };

    {
        sycl::queue q;
        hierarchical_inclusive_scan(q, lengths, sycl::plus<index_type>{});
    }

    sycl::buffer<bits_type> stream(n_blocks * hc_total_chunks_size);

    SYCL_BENCHMARK("Compact transposed")(sycl::queue & q) {
        return q.submit([&](sycl::handler &cgh) {
            auto c = chunks.template get_access<sam::read>(cgh);
            auto l = lengths.template get_access<sam::read>(cgh);
            auto s = stream.template get_access<sam::discard_write>(cgh);
            cgh.parallel<chunk_compact_kernel<TestType>>(sycl::range<1>{n_blocks},
                    sycl::range<1>{hypercube_group_size<TestType>},
                    [=](hypercube_group<TestType> grp, sycl::physical_item<1>) {
                        auto hc_index = static_cast<index_type>(grp.get_id(0));
                        index_type offset_after;
                        compact_chunks<TestType>(grp,
                                &c.get_pointer()[hc_index * hc_total_chunks_size],
                                &l.get_pointer()[hc_index * chunks_per_hc], &offset_after,
                                &s.get_pointer()[0]);
                    });
        });
    };
}


// Impact of dimensionality should not be that large, but the hc padding could hold surprises
TEMPLATE_TEST_CASE("Chunk decoding", "[decode]", ALL_PROFILES) {
    constexpr index_type n_blocks = 16384;
    using bits_type = typename TestType::bits_type;
    constexpr auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);

    sycl::buffer<bits_type> columns(n_blocks * hc_size);
    sycl::queue{}.submit([&](sycl::handler &cgh) {
        cgh.fill(columns.template get_access<sam::discard_write>(cgh),
                static_cast<bits_type>(7948741984121192831 /* whatever */));
    });

    SYCL_BENCHMARK("Read and transpose")(sycl::queue & q) {
        return q.submit([&](sycl::handler &cgh) {
            auto c = columns.template get_access<sam::read>(cgh);
            cgh.parallel<chunk_transpose_read_kernel<TestType>>(sycl::range<1>{n_blocks},
                    sycl::range<1>{hypercube_group_size<TestType>},
                    [=](hypercube_group<TestType> grp, sycl::physical_item<1>) {
                        hypercube_memory<TestType, inverse_transform_tag> lm{grp};
                        gpu::hypercube_ptr<TestType, gpu::inverse_transform_tag> hc{lm()};
                        const auto hc_index = grp.get_id(0);
                        const bits_type *column = c.get_pointer();
                        read_transposed_chunks(grp, hc, &column[hc_index * hc_size]);
                    });
        });
    };
}
