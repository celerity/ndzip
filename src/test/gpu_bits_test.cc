#define CATCH_CONFIG_MAIN
#include <test/test_utils.hh>
#include <ndzip/gpu_bits.hh>

using namespace ndzip::detail::gpu;
using sam = sycl::access::mode;


TEMPLATE_TEST_CASE("Subgroup hierarchical inclusive scan works", "[gpu_bits]", uint32_t, uint64_t) {
    constexpr index_type group_size = 64;
    constexpr index_type n_groups = 2;
    constexpr index_type range = group_size * n_groups;

    sycl::queue q;
    sycl::buffer<TestType> out(range);
    q.submit([&](sycl::handler &cgh) {
        auto g = out.template get_access<sam::discard_write>(cgh);
        cgh.parallel(sycl::range<1>{1}, sycl::range<1>{group_size},
                [=](known_size_group<group_size> grp, sycl::physical_item<1>) {
                    grp.distribute_for(range, [&](index_type item) { g[item] = 1; });
                    inclusive_scan<range>(grp, g, sycl::plus<TestType>{});
                });
    });

    std::vector<TestType> gpu_result(range);
    auto gpu_result_available = q.submit([&](sycl::handler &cgh) {
        cgh.copy(out.template get_access<sam::read>(cgh), gpu_result.data());
    });

    std::vector<TestType> cpu_input(range, TestType{1});
    std::vector<TestType> cpu_result(range);
    std::inclusive_scan(cpu_input.begin(), cpu_input.end(), cpu_result.begin());

    gpu_result_available.wait();
    check_for_vector_equality(cpu_result, gpu_result);
}
