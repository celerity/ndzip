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
                            g[idx.get_global_linear_id()] = sycl::inclusive_scan_over_group(
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
                            g[idx.get_global_linear_id()] = sycl::inclusive_scan_over_group(
                                    grp, idx.get_global_linear_id(), sycl::plus<TestType>{});
                        });
                    });
        });
    };

    SYCL_BENCHMARK("hierarchical subgroup scan")(sycl::queue & q) {
        sycl::buffer<TestType> out(group_size * n_groups);
        return q.submit([&](sycl::handler &cgh) {
            auto g = out.template get_access<sam::discard_write>(cgh);
            sycl::local_accessor<inclusive_scan_local_allocation<TestType, group_size>> lm{1, cgh};
            cgh.parallel<inclusive_scan_hierarchical_subgroup_kernel<TestType>>(sycl::range<1>{n_groups},
                    sycl::range<1>{group_size}, [=](known_size_group<group_size> grp, sycl::physical_item<1>) {
                        grp.distribute_for(group_size, [&](index_type item, index_type, sycl::logical_item<1> idx) {
                            g[item] = idx.get_global_linear_id();
                        });
                        inclusive_scan_over_group<group_size>(grp, g, lm[0], sycl::plus<TestType>{});
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
            cgh.parallel<ndzip_inclusive_scan_kernel<TestType>>(sycl::range<1>{1}, sycl::range<1>{group_size},
                    [=](known_size_group<group_size> grp, sycl::physical_item<1>) {
                        distribute_for(block_size, grp, [&](index_type item, index_type, sycl::logical_item<1> idx) {
                            g[item] = idx.get_global_linear_id();
                        });
                        inclusive_scan_over_group<block_size>(grp, g, lm[0], sycl::plus<TestType>{});
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
                    sycl::nd_range<1>{group_size, group_size}, [=](sycl::nd_item<1> item) {
                        known_size_group<group_size> grp{item.get_group()};
                        distribute_for(block_size, grp, [&](index_type item, index_type, sycl::logical_item<1> idx) {
                            g[item] = idx.get_global_linear_id();
                        });
                        sycl::joint_inclusive_scan(
                                item.get_group(), &g[0], &g[block_size], &g[0], sycl::plus<TestType>{});
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
            cgh.parallel<local_memory_scan_spillage_kernel<TestType>>(sycl::range<1>{1}, sycl::range<1>{group_size},
                    [=](known_size_group<group_size> grp, sycl::physical_item<1>) {
                        grp.distribute_for(block_size, [&](index_type item, index_type, sycl::logical_item<1> idx) {
                            g[item] = idx.get_global_linear_id();
                        });
                        inclusive_scan_over_group<block_size>(grp, g, lm[0], sycl::plus<TestType>{});
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
        return [=](known_size_group<group_size> grp, sycl::physical_item<1>) {
            grp.distribute_for(num_group_items, [&](index_type item, index_type, sycl::logical_item<1> idx) {
                lm[item] = idx.get_global_linear_id();
            });
            sycl::private_memory<TestType> column{grp};
            grp.distribute_for(
                    num_group_items, [&](index_type item, index_type, sycl::logical_item<1> idx, sycl::sub_group sg) {
                        transpose(item, idx, sg, &lm[0], column);
                    });
            grp.distribute_for(num_group_items,
                    [&](index_type item, index_type, sycl::logical_item<1> idx) { lm[item] = column(idx); });
            black_hole(&lm[0]);
        };
    };

    SYCL_BENCHMARK("Every thread loads each row")(sycl::queue & q) {
        auto transpose = [&](index_type item, sycl::logical_item<1> idx, sycl::sub_group, TestType *rows,
                                 sycl::private_memory<TestType> &column) {
            column(idx) = 0;
            const auto chunk_base = floor(item, chunk_size);
            const auto cell = item - chunk_base;
            for (index_type i = 0; i < chunk_size; ++i) {
                column(idx) |= (rows[chunk_base + i] >> (chunk_size - 1 - cell) & TestType{1}) << (chunk_size - 1 - i);
            }
        };
        return q.submit([&](sycl::handler &cgh) {
            cgh.parallel<transpose_each_row_kernel<TestType>>(
                    sycl::range<1>{num_groups}, sycl::range<1>{group_size}, transpose_harness(cgh, transpose));
        });
    };

    // Not that smart...
    SYCL_BENCHMARK("Per-column subgroup reductions")(sycl::queue & q) {
        auto transpose = [&](index_type item, sycl::logical_item<1> idx, sycl::sub_group sg,
                                                TestType *rows, sycl::private_memory<TestType> &column) {
            const auto chunk_base = floor(item, chunk_size);
            const auto cell = item - chunk_base;
            const auto row = rows[item];
            for (index_type i = 0; i < chunk_size; ++i) {
                TestType this_column
                        = sycl::reduce_over_group(sg, ((row >> i) & TestType{1}) << cell, sycl::bit_or<TestType>{});
                if (cell == i) { column(idx) = this_column; }
            }
        };
        return q.submit([&](sycl::handler &cgh) {
            cgh.parallel<transpose_via_subgroups_kernel<TestType>>(
                    sycl::range<1>{num_groups}, sycl::range<1>{group_size}, transpose_harness(cgh, transpose));
        });
    };
}
