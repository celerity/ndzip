#pragma once

#include <complex>  // we don't use <complex>, but not including it triggers a CUDA error

#include <SYCL/sycl.hpp>
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>


struct SyclBenchmark {
    std::string name;
};

template<typename Lambda>
void operator<<=(SyclBenchmark &&bench, Lambda &&lambda) {
    sycl::queue q{sycl::property::queue::enable_profiling{}};

    Catch::IConfigPtr cfg = Catch::getCurrentContext().getConfig();
    size_t warmup_runs = cfg->benchmarkWarmupTime() < std::chrono::milliseconds(1) ? 0 : 1;

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
