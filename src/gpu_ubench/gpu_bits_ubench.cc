#include <iomanip>

#include <ndzip/gpu_bits.hh>

#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

using namespace ndzip::detail::gpu;
using sam = sycl::access::mode;


struct SyclBenchmark {
    std::string name;
};

template<typename Lambda>
void operator<<=(SyclBenchmark &&bench, Lambda &&lambda) {
    const size_t warmup_runs = 1;
    sycl::queue q{sycl::property::queue::enable_profiling{}};

    Catch::IConfigPtr cfg = Catch::getCurrentContext().getConfig();

    using duration = std::chrono::duration<double, std::nano>;
    Catch::Benchmark::Environment<duration> env{{duration{100}, {}}, {duration{0.0}, {}}};

    Catch::getResultCapture().benchmarkPreparing(bench.name);

    Catch::BenchmarkInfo info{std::move(bench.name), 0.0, 1, cfg->benchmarkSamples(),
            cfg->benchmarkResamples(), env.clock_resolution.mean.count(),
            env.clock_cost.mean.count()};

    Catch::getResultCapture().benchmarkStarting(info);

    std::vector<duration> samples(static_cast<size_t>(info.samples));
    for (unsigned i = 0; i < warmup_runs + samples.size(); ++i) {
        sycl::event evt = lambda(q);
        evt.wait();
        if (i >= warmup_runs) {
            samples[i - warmup_runs] = std::chrono::duration_cast<
                    duration>(std::chrono::duration<uint64_t, std::nano>(
                    evt.get_profiling_info<sycl::info::event_profiling::command_end>()
                    - evt.get_profiling_info<sycl::info::event_profiling::command_start>()));
        }
    }

    auto analysis = Catch::Benchmark::Detail::analyse(*cfg, env, samples.begin(), samples.end());
    Catch::BenchmarkStats<duration> stats{info, analysis.samples, analysis.mean,
            analysis.standard_deviation, analysis.outliers, analysis.outlier_variance};

    Catch::getResultCapture().benchmarkEnded(stats);
}

#define SYCL_BENCHMARK(name) SyclBenchmark{name} <<= [&]


TEMPLATE_TEST_CASE("Register to global memory inclusive scan", "[scan]", uint32_t, uint64_t) {
    constexpr index_type group_size = 1024;
    constexpr index_type n_groups = 16384;

    SYCL_BENCHMARK("Reference: memset")(sycl::queue & q) {
        sycl::buffer<TestType> out(group_size * n_groups);
        return q.submit([&](sycl::handler &cgh) {
            auto g = out.template get_access<sam::discard_write>(cgh);
            cgh.parallel(sycl::range<1>{n_groups}, sycl::range<1>{group_size},
                    [=](sycl::group<1> &grp, sycl::physical_item<1>) {
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
            cgh.parallel(sycl::range<1>{n_groups}, sycl::range<1>{group_size},
                    [=](sycl::group<1> &grp, sycl::physical_item<1>) {
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
            cgh.parallel(sycl::range<1>{n_groups}, sycl::range<1>{group_size},
                    [=](sycl::group<1> &grp, sycl::physical_item<1>) {
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
            cgh.parallel(sycl::range<1>{n_groups}, sycl::range<1>{group_size},
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
