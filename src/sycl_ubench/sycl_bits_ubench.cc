#include "ubench.hh"

#include <ndzip/sycl_bits.hh>
#include <test/test_utils.hh>

using namespace ndzip;
using namespace ndzip::detail;
using namespace ndzip::detail::gpu_sycl;
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
            cgh.parallel_for<inclusive_scan_reference_kernel<TestType>>(make_nd_range(n_groups, group_size),
                    [=](sycl::nd_item<1> item) { g[item.get_global_id()] = item.get_global_linear_id(); });
        });
    };

    SYCL_BENCHMARK("SYCL subgroup scan")(sycl::queue & q) {
        sycl::buffer<TestType> out(group_size * n_groups);
        return q.submit([&](sycl::handler &cgh) {
            auto g = out.template get_access<sam::discard_write>(cgh);
            cgh.parallel_for<inclusive_scan_sycl_subgroup_kernel<TestType>>(
                    make_nd_range(n_groups, group_size), [=](sycl::nd_item<1> item) {
                        g[item.get_global_id()] = sycl::inclusive_scan_over_group(
                                item.get_sub_group(), item.get_global_linear_id(), sycl::plus<TestType>{});
                    });
        });
    };

    SYCL_BENCHMARK("SYCL group scan")(sycl::queue & q) {
        sycl::buffer<TestType> out(group_size * n_groups);
        return q.submit([&](sycl::handler &cgh) {
            auto g = out.template get_access<sam::discard_write>(cgh);
            cgh.parallel_for<inclusive_scan_sycl_group_kernel<TestType>>(
                    make_nd_range(n_groups, group_size), [=](sycl::nd_item<1> item) {
                        g[item.get_global_id()] = sycl::inclusive_scan_over_group(
                                item.get_group(), item.get_global_linear_id(), sycl::plus<TestType>{});
                    });
        });
    };

    SYCL_BENCHMARK("hierarchical subgroup scan")(sycl::queue & q) {
        sycl::buffer<TestType> out(group_size * n_groups);
        return q.submit([&](sycl::handler &cgh) {
            auto g = out.template get_access<sam::discard_write>(cgh);
            sycl::local_accessor<inclusive_scan_local_allocation<TestType, group_size>> lm{1, cgh};
            cgh.parallel_for<inclusive_scan_hierarchical_subgroup_kernel<TestType>>(
                    make_nd_range(n_groups, group_size), [=](known_group_size_item<group_size> item) {
                        distribute_for(group_size, item.get_group(),
                                [&](index_type i) { g[i] = item.get_global_linear_id(); });
                        inclusive_scan_over_group<group_size>(item, g, lm[0], sycl::plus<TestType>{});
                    });
        });
    };
}

template<typename>
class ndzip_inclusive_scan_kernel;

template<typename>
class joint_inclusive_scan_kernel;

TEMPLATE_TEST_CASE("Joint inclusive scan", "[scan][memory]", uint32_t, uint64_t) {
    SYCL_BENCHMARK("ndzip::detail::gpu_sycl::inclusive_scan")(sycl::queue & q) {
        constexpr index_type block_size = 4096;
        constexpr index_type group_size = 256;
        sycl::buffer<TestType> buf(block_size);
        return q.submit([&](sycl::handler &cgh) {
            sycl::accessor g{buf, cgh, sycl::read_write, sycl::no_init};
            sycl::local_accessor<inclusive_scan_local_allocation<TestType, block_size>> lm{1, cgh};
            cgh.parallel_for<ndzip_inclusive_scan_kernel<TestType>>(
                    sycl::nd_range<1>{group_size, group_size}, [=](known_group_size_item<group_size> item) {
                        distribute_for(block_size, item.get_group(),
                                [&](index_type i) { g[i] = item.get_global_linear_id(); });
                        inclusive_scan_over_group(item, g, lm[0], sycl::plus<TestType>{});
                    });
        });
    };

    SYCL_BENCHMARK("sycl::joint_inclusive_scan")(sycl::queue & q) {
        constexpr index_type block_size = 4096;
        constexpr index_type group_size = 256;
        sycl::buffer<TestType> buf(block_size);
        return q.submit([&](sycl::handler &cgh) {
            sycl::accessor g{buf, cgh, sycl::read_write, sycl::no_init};
            cgh.parallel_for<joint_inclusive_scan_kernel<TestType>>(
                    sycl::nd_range<1>{group_size, group_size}, [=](known_group_size_item<group_size> item) {
                        distribute_for(block_size, item.get_group(),
                                [&](index_type i) { g[i] = item.get_global_linear_id(); });
                        sycl::joint_inclusive_scan(
                                item.get_sycl_group(), &g[0], &g[block_size], &g[0], sycl::plus<TestType>{});
                    });
        });
    };
}


template<typename>
class local_memory_scan_spillage_kernel;

// Verify in Nsight Compute that this does not spill to device memory
TEMPLATE_TEST_CASE("Local-memory scan spillage", "[scan][memory]", uint32_t, uint64_t) {
    SYCL_BENCHMARK("4096-element inclusive scan")(sycl::queue & q) {
        constexpr index_type block_size = 4096;
        constexpr index_type group_size = 256;
        sycl::buffer<TestType> buf(block_size);
        return q.submit([&](sycl::handler &cgh) {
            sycl::accessor g{buf, cgh, sycl::read_write, sycl::no_init};
            sycl::local_accessor<inclusive_scan_local_allocation<TestType, block_size>> lm{1, cgh};
            cgh.parallel_for<local_memory_scan_spillage_kernel<TestType>>(
                    sycl::nd_range<1>{group_size, group_size}, [=](known_group_size_item<group_size> item) {
                        distribute_for(block_size, item.get_group(),
                                [&](index_type i) { g[i] = item.get_global_linear_id(); });
                        inclusive_scan_over_group<block_size>(item, g, lm[0], sycl::plus<TestType>{});
                    });
        });
    };
}


template<typename>
class transpose_each_row_kernel;
template<typename>
class transpose_via_subgroups_kernel;

TEMPLATE_TEST_CASE("Local memory transpose bits", "[transpose]", uint32_t) {
    constexpr static index_type num_groups = 16384;
    constexpr static index_type num_group_items = 4096;
    constexpr static index_type group_size = 256;
    constexpr static index_type chunk_size = 32;

    const auto transpose_harness = [](sycl::handler &cgh, auto &&transpose) {
        sycl::local_accessor<TestType> lm{num_group_items, cgh};
        return [=](known_group_size_item<group_size> item) {
            distribute_for(
                    num_group_items, item.get_group(), [&](index_type i) { lm[i] = item.get_global_linear_id(); });
            TestType column;  // per-thread
            distribute_for(num_group_items, item.get_group(),
                    [&](index_type i) { transpose(i, item.get_sub_group(), &lm[0], column); });
            distribute_for(num_group_items, item.get_group(), [&](index_type item) { lm[item] = column; });
            black_hole(&lm[0]);
        };
    };

    SYCL_BENCHMARK("Every thread loads each row")(sycl::queue & q) {
        auto transpose = [&](index_type i, sycl::sub_group, TestType *rows, TestType &column) {
            column = 0;
            const auto chunk_base = floor(i, chunk_size);
            const auto cell = i - chunk_base;
            for (index_type j = 0; j < chunk_size; ++j) {
                column |= (rows[chunk_base + j] >> (chunk_size - 1 - cell) & TestType{1}) << (chunk_size - 1 - j);
            }
        };
        return q.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<transpose_each_row_kernel<TestType>>(
                    make_nd_range(num_groups, group_size), transpose_harness(cgh, transpose));
        });
    };

    // Not that smart...
    SYCL_BENCHMARK("Per-column subgroup reductions")(sycl::queue & q) {
        auto transpose = [&](index_type i, sycl::sub_group sg, TestType *rows, TestType &column) {
            const auto chunk_base = floor(i, chunk_size);
            const auto cell = i - chunk_base;
            const auto row = rows[i];
            for (index_type j = 0; j < chunk_size; ++j) {
                TestType this_column
                        = sycl::reduce_over_group(sg, ((row >> j) & TestType{1}) << cell, sycl::bit_or<TestType>{});
                if (cell == j) { column = this_column; }
            }
        };
        return q.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<transpose_via_subgroups_kernel<TestType>>(
                    make_nd_range(num_groups, group_size), transpose_harness(cgh, transpose));
        });
    };
}
