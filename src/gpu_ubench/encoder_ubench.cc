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
