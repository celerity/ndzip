#include "ubench.hh"

#include <ndzip/gpu_bits.hh>

using namespace ndzip::detail::gpu;
using sam = sycl::access::mode;

// Kernel names (for profiler)
template<typename>
class inclusive_scan_reference_kernel;
template<typename>
class inclusive_scan_sycl_subgroup_kernel;
template<typename>
class inclusive_scan_sycl_group_kernel;
template<typename>
class inclusive_scan_hierarchical_subgroup_kernel;

TEMPLATE_TEST_CASE("Register to global memory inclusive scan", "[scan]", uint32_t, uint64_t) {
    constexpr index_type group_size = 1024;
    constexpr index_type n_groups = 16384;

    SYCL_BENCHMARK("Reference: memset")(sycl::queue & q) {
        sycl::buffer<TestType> out(group_size * n_groups);
        return q.submit([&](sycl::handler &cgh) {
            auto g = out.template get_access<sam::discard_write>(cgh);
            cgh.parallel<inclusive_scan_reference_kernel<TestType>>(sycl::range<1>{n_groups},
                    sycl::range<1>{group_size}, [=](sycl::group<1> &grp, sycl::physical_item<1>) {
                        grp.distribute_for([&](sycl::sub_group, sycl::logical_item<1> idx) {
                            g[idx.get_global_linear_id()] = idx.get_global_linear_id();
                        });
                    });
        });
    };

    SYCL_BENCHMARK("SYCL subgroup scan")(sycl::queue & q) {
        sycl::buffer<TestType> out(group_size * n_groups);
        return q.submit([&](sycl::handler &cgh) {
            auto g = out.template get_access<sam::discard_write>(cgh);
            cgh.parallel<inclusive_scan_sycl_subgroup_kernel<TestType>>(sycl::range<1>{n_groups},
                    sycl::range<1>{group_size}, [=](sycl::group<1> &grp, sycl::physical_item<1>) {
                        grp.distribute_for([&](sycl::sub_group sg, sycl::logical_item<1> idx) {
                            g[idx.get_global_linear_id()] = sycl::group_inclusive_scan(
                                    sg, idx.get_global_linear_id(), sycl::plus<TestType>{});
                        });
                    });
        });
    };

    SYCL_BENCHMARK("SYCL group scan")(sycl::queue & q) {
        sycl::buffer<TestType> out(group_size * n_groups);
        return q.submit([&](sycl::handler &cgh) {
            auto g = out.template get_access<sam::discard_write>(cgh);
            cgh.parallel<inclusive_scan_sycl_group_kernel<TestType>>(sycl::range<1>{n_groups},
                    sycl::range<1>{group_size}, [=](sycl::group<1> &grp, sycl::physical_item<1>) {
                        grp.distribute_for([&](sycl::sub_group, sycl::logical_item<1> idx) {
                            g[idx.get_global_linear_id()] = sycl::group_inclusive_scan(
                                    grp, idx.get_global_linear_id(), sycl::plus<TestType>{});
                        });
                    });
        });
    };

    SYCL_BENCHMARK("hierarchical subgroup scan")(sycl::queue & q) {
        sycl::buffer<TestType> out(group_size * n_groups);
        return q.submit([&](sycl::handler &cgh) {
            auto g = out.template get_access<sam::discard_write>(cgh);
            cgh.parallel<inclusive_scan_hierarchical_subgroup_kernel<TestType>>(
                    sycl::range<1>{n_groups}, sycl::range<1>{group_size},
                    [=](known_size_group<group_size> grp, sycl::physical_item<1>) {
                        sycl::local_memory<TestType[group_size]> lm{grp};
                        grp.distribute_for(group_size,
                                [&](index_type item, index_type, sycl::logical_item<1> idx) {
                                    lm[item] = idx.get_global_linear_id();
                                });
                        inclusive_scan<group_size>(grp, lm(), sycl::plus<TestType>{});
                        grp.distribute_for(group_size,
                                [&](index_type item, index_type, sycl::logical_item<1> idx) {
                                    g[idx.get_global_linear_id()] = lm[item];
                                });
                    });
        });
    };
}
