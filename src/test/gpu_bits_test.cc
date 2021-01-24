#include <test/test_utils.hh>
#include <ndzip/gpu_bits.hh>

using namespace ndzip::detail::gpu;
using sam = sycl::access::mode;


TEMPLATE_TEST_CASE("Subgroup hierarchical inclusive scan works", "[gpu_bits]", uint32_t, uint64_t) {
    constexpr index_type group_size = 1024;
    constexpr index_type n_groups = 2;

    sycl::queue q;
    sycl::buffer<TestType> out(group_size * n_groups);
    q.submit([&](sycl::handler &cgh) {
        auto g = out.template get_access<sam::discard_write>(cgh);
        cgh.parallel(sycl::range<1>{n_groups}, sycl::range<1>{group_size},
                [=](known_size_group<group_size> grp, sycl::physical_item<1>) {
                    sycl::local_memory<TestType[group_size]> lm{grp};
                    grp.distribute_for(group_size, [&](index_type item) { lm[item] = 1; });
                    inclusive_scan<group_size>(grp, lm(), sycl::plus<TestType>{});
                    grp.distribute_for(group_size,
                            [&](index_type item, index_type, sycl::logical_item<1> idx) {
                                g[idx.get_global_linear_id()] = lm[item];
                            });
                });
    });

    std::vector<TestType> gpu_result(group_size * n_groups);
    auto gpu_result_available = q.submit([&](sycl::handler &cgh) {
        cgh.copy(out.template get_access<sam::read>(cgh), gpu_result.data());
    });

    std::vector<TestType> cpu_input(group_size * n_groups, TestType{1});
    std::vector<TestType> cpu_result(group_size * n_groups);
    for (index_type i = 0; i < n_groups; ++i) {
        std::inclusive_scan(cpu_input.begin() + i * group_size,
                cpu_input.begin() + (i + 1) * group_size, cpu_result.begin() + i * group_size);
    }

    gpu_result_available.wait();
    check_for_vector_equality(cpu_result, gpu_result);
}
