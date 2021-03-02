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


// Impact of dimensionality should not be that large, but the hc padding could hold surprises
TEMPLATE_TEST_CASE("Chunk encoding", "[encode]", (profile<float, 1>), (profile<double, 1>)) {
    constexpr index_type n_blocks = 16384;
    using bits_type = typename TestType::bits_type;
    constexpr auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);
    constexpr auto warps_per_hc = hc_size / warp_size;

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
    sycl::buffer<bits_type> heads(n_blocks * warps_per_hc);
    sycl::buffer<index_type> lengths(1 + n_blocks * warps_per_hc);

    SYCL_BENCHMARK("Transpose only")(sycl::queue & q) {
        return q.submit([&](sycl::handler &cgh) {
            auto c = columns.template get_access<sam::discard_write>(cgh);
            auto h = heads.template get_access<sam::discard_write>(cgh);
            auto l = lengths.template get_access<sam::discard_write>(cgh);
            cgh.parallel<chunk_transpose_kernel<TestType>>(sycl::range<1>{n_blocks},
                    sycl::range<1>{hypercube_group_size},
                    [=](hypercube_group grp, sycl::physical_item<1> phys_idx) {
                        sycl::local_memory<bits_type[hypercube<TestType>::allocation_size]> lm{grp};
                        hypercube<TestType> hc{&lm[0]};
                        grp.distribute_for(hc_size, [&](index_type i) { hc[i] = i * 199; });
                        const auto hc_index = grp.get_id(0);
                        write_transposed_chunks(grp, hc, &h[hc_index * warps_per_hc],
                                &c[hc_index * hc_size], &l[1 + hc_index * warps_per_hc]);
                        // hack
                        if (phys_idx.get_global_linear_id() == 0) {
                            grp.single_item([&] { l[0] = 0; });
                        }
                    });
        });
    };

    {
        hierarchical_inclusive_prefix_sum<index_type> prefix_sum(
                1 + n_blocks * hc_size / warp_size, 256 /* local size */);
        sycl::queue q;
        prefix_sum(q, lengths);
    }

    sycl::buffer<bits_type> stream(n_blocks * (hc_size + hc_size / warp_size));

    SYCL_BENCHMARK("Compact transposed")(sycl::queue & q) {
        return q.submit([&](sycl::handler &cgh) {
            auto c = columns.template get_access<sam::read>(cgh);
            auto h = heads.template get_access<sam::read>(cgh);
            auto l = lengths.template get_access<sam::read>(cgh);
            auto s = stream.template get_access<sam::discard_write>(cgh);
            constexpr size_t group_size = 1024;
            cgh.parallel<chunk_compact_kernel<TestType>>(
                    sycl::range<1>{hc_size / group_size * n_blocks}, sycl::range<1>{group_size},
                    [=](sycl::group<1> grp, sycl::physical_item<1>) {
                        compact_chunks(grp, static_cast<const bits_type *>(h.get_pointer()),
                                static_cast<const bits_type *>(c.get_pointer()),
                                static_cast<const index_type *>(l.get_pointer()),
                                static_cast<bits_type *>(s.get_pointer()));
                    });
        });
    };
}
