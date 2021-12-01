#include "test_utils.hh"

#include <complex>  // we don't use <complex>, but not including it triggers a CUDA error

#include <ndzip/sycl_bits.hh>

using namespace ndzip;
using namespace ndzip::detail;
using namespace ndzip::detail::gpu_sycl;
using sam = sycl::access::mode;


TEMPLATE_TEST_CASE("Subgroup hierarchical inclusive scan works", "[sycl][scan]", uint32_t, uint64_t) {
    constexpr index_type group_size = 1024;
    constexpr index_type n_groups = 9;
    constexpr index_type range = group_size * n_groups;

    sycl::queue q;
    sycl::buffer<TestType> out(range);
    q.submit([&](sycl::handler &cgh) {
        auto out_acc = out.template get_access<sam::discard_write>(cgh);
        sycl::local_accessor<inclusive_scan_local_allocation<TestType, range>> lm{1, cgh};
        cgh.parallel(sycl::range<1>{1}, sycl::range<1>{group_size},
                [=](known_size_group<group_size> grp, sycl::physical_item<1>) {
                    TestType *out = out_acc.get_pointer();
                    grp.distribute_for(range, [&](index_type item) { out[item] = 1; });
                    inclusive_scan_over_group<range>(grp, out, lm[0], sycl::plus<TestType>{});
                });
    });

    std::vector<TestType> gpu_result(range);
    auto gpu_result_available = q.submit(
            [&](sycl::handler &cgh) { cgh.copy(out.template get_access<sam::read>(cgh), gpu_result.data()); });

    std::vector<TestType> cpu_input(range, TestType{1});
    std::vector<TestType> cpu_result(range);
    iter_inclusive_scan(cpu_input.begin(), cpu_input.end(), cpu_result.begin());

    gpu_result_available.wait();
    CHECK_FOR_VECTOR_EQUALITY(cpu_result, gpu_result);
}


TEMPLATE_TEST_CASE("hierarchical_inclusive_scan produces the expected results", "[sycl][scan]", sycl::plus<uint32_t>,
        sycl::logical_or<uint32_t>) {
    std::vector<uint32_t> input(size_t{1} << 24u);
    std::iota(input.begin(), input.end(), uint32_t{});

    std::vector<uint32_t> cpu_prefix_sum(input.size());
    iter_inclusive_scan(input.begin(), input.end(), cpu_prefix_sum.begin(), TestType{});

    sycl::queue q{sycl::gpu_selector{}};
    sycl::buffer<uint32_t> prefix_sum_buffer(sycl::range<1>(input.size()));
    q.submit(
            [&](sycl::handler &cgh) { cgh.copy(input.data(), prefix_sum_buffer.get_access<sam::discard_write>(cgh)); });

    auto scratch = hierarchical_inclusive_scan_allocate<uint32_t>(input.size());
    hierarchical_inclusive_scan(q, prefix_sum_buffer, scratch, TestType{});

    std::vector<uint32_t> gpu_prefix_sum(input.size());
    q.submit(
            [&](sycl::handler &cgh) { cgh.copy(prefix_sum_buffer.get_access<sam::read>(cgh), gpu_prefix_sum.data()); });
    q.wait();

    CHECK_FOR_VECTOR_EQUALITY(gpu_prefix_sum, cpu_prefix_sum);
}
