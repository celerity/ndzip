#pragma once

#include "gpu_common.hh"

#include <type_traits>

#include <SYCL/sycl.hpp>
#include <ndzip/sycl_encoder.hh>


namespace ndzip::detail::gpu_sycl {

using namespace ndzip::detail::gpu;

template<typename ...Events>
std::tuple<uint64_t, uint64_t, kernel_duration> measure_duration(const Events &...events) {
    auto early = std::min<uint64_t>({events.template get_profiling_info<
            sycl::info::event_profiling::command_start>()...});
    auto late = std::max<uint64_t>({events.template get_profiling_info<
            sycl::info::event_profiling::command_end>()}...);
    return {early, late, kernel_duration{late - early}};
}

template<typename CGF>
auto submit_and_profile(sycl::queue &q, const char *label, CGF &&cgf) {
    if (verbose()) {
        auto evt = q.submit(std::forward<CGF>(cgf));
        auto [early, late, duration] = measure_duration(evt, evt);
        printf("[profile] %8lu %8lu %s: %.3fms\n", early, late, label, duration.count() * 1e-6);
        return evt;
    } else {
        return q.submit(std::forward<CGF>(cgf));
    }
}


template<index_type LocalSize>
class known_size_group : public sycl::group<1> {
  public:
    using sycl::group<1>::group;
    using sycl::group<1>::distribute_for;

    known_size_group(sycl::group<1> grp)  // NOLINT(google-explicit-constructor)
        : sycl::group<1>{grp} {}

    template<typename F>
    [[gnu::always_inline]] void distribute_for(index_type range, F &&f) {
        distribute_for([&](sycl::sub_group sg, sycl::logical_item<1> idx) {
            const index_type num_full_iterations = range / LocalSize;
            const auto tid = static_cast<index_type>(idx.get_local_id(0));

            for (index_type iteration = 0; iteration < num_full_iterations; ++iteration) {
                auto item = iteration * LocalSize + tid;
                invoke_f(f, item, iteration, idx, sg);
            }
            distribute_for_partial_iteration(range, sg, idx, f);
        });
    }

    template<index_type Range, typename F>
    [[gnu::always_inline]] void distribute_for(F &&f) {
        distribute_for([&](sycl::sub_group sg, sycl::logical_item<1> idx) {
            constexpr index_type num_full_iterations = Range / LocalSize;
            const auto tid = static_cast<index_type>(idx.get_local_id(0));

#pragma unroll
            for (index_type iteration = 0; iteration < num_full_iterations; ++iteration) {
                auto item = iteration * LocalSize + tid;
                invoke_f(f, item, iteration, idx, sg);
            }
            distribute_for_partial_iteration(Range, sg, idx, f);
        });
    }

  private:
    template<typename F>
    [[gnu::always_inline]] void invoke_f(F &&f, index_type item, index_type iteration,
            sycl::logical_item<1> idx, sycl::sub_group sg) const {
        if constexpr (std::is_invocable_v<F, index_type, index_type, sycl::logical_item<1>,
                              sycl::sub_group>) {
            f(item, iteration, idx, sg);
        } else if constexpr (std::is_invocable_v<F, index_type, index_type,
                                     sycl::logical_item<1>>) {
            f(item, iteration, idx);
        } else if constexpr (std::is_invocable_v<F, index_type, index_type>) {
            f(item, iteration);
        } else {
            f(item);
        }
    }

    template<typename F>
    [[gnu::always_inline]] void distribute_for_partial_iteration(
            index_type range, sycl::sub_group sg, sycl::logical_item<1> idx, F &&f) {
        const index_type num_full_iterations = range / LocalSize;
        const index_type partial_iteration_length = range % LocalSize;
        const auto tid = static_cast<index_type>(idx.get_local_id(0));
        if (tid < partial_iteration_length) {
            auto iteration = num_full_iterations;
            auto item = iteration * LocalSize + tid;
            invoke_f(f, item, iteration, idx, sg);
        }
    }
};

template<index_type Range, index_type LocalSize, typename Accessor, typename BinaryOp>
std::enable_if_t<(Range <= warp_size)>
inclusive_scan(known_size_group<LocalSize> grp, Accessor acc, BinaryOp op) {
    static_assert(LocalSize % warp_size == 0);
    grp.template distribute_for<ceil(Range, warp_size)>(
            [&](index_type item, index_type, sycl::logical_item<1>, sycl::sub_group sg) {
                auto a = item < Range ? acc[item] : 0;
                auto b = sycl::group_inclusive_scan(sg, a, op);
                if (item < Range) { acc[item] = b; }
            });
}

template<index_type Range, index_type LocalSize, typename Accessor, typename BinaryOp>
std::enable_if_t<(Range > warp_size)>
inclusive_scan(known_size_group<LocalSize> grp, Accessor acc, BinaryOp op) {
    static_assert(LocalSize % warp_size == 0);
    using value_type = std::decay_t<decltype(acc[index_type{}])>;

    sycl::private_memory<value_type[div_ceil(Range, LocalSize)]> fine{grp};
    sycl::local_memory<value_type[div_ceil(Range, warp_size)]> coarse{grp};
    grp.template distribute_for<ceil(Range, warp_size)>([&](index_type item, index_type iteration,
                                                                sycl::logical_item<1> idx,
                                                                sycl::sub_group sg) {
        fine(idx)[iteration] = sycl::group_inclusive_scan(sg, item < Range ? acc[item] : 0, op);
        if (item % warp_size == warp_size - 1) { coarse[item / warp_size] = fine(idx)[iteration]; }
    });
    inclusive_scan<div_ceil(Range, warp_size)>(grp, coarse(), op);
    grp.template distribute_for<Range>(
            [&](index_type item, index_type iteration, sycl::logical_item<1> idx) {
                auto value = fine(idx)[iteration];
                if (item >= warp_size) { value = op(value, coarse[item / warp_size - 1]); }
                acc[item] = value;
            });
}

template<typename, typename>
class hierarchical_inclusive_scan_reduction_kernel;

template<typename, typename>
class hierarchical_inclusive_scan_expansion_kernel;

template<typename Scalar, typename BinaryOp>
auto hierarchical_inclusive_scan(
        sycl::queue &queue, sycl::buffer<Scalar> &in_out_buffer, BinaryOp op = {}) {
    using sam = sycl::access::mode;

    constexpr index_type granularity = hierarchical_inclusive_scan_granularity;
    constexpr index_type local_size = 256;

    std::vector<sycl::buffer<Scalar>> intermediate_bufs;
    {
        auto n_elems = static_cast<index_type>(in_out_buffer.get_count());
        assert(n_elems % granularity == 0);  // otherwise we will overrun the in_out buffer bounds

        while (n_elems > 1) {
            n_elems = div_ceil(n_elems, granularity);
            intermediate_bufs.emplace_back(ceil(n_elems, granularity));
        }
    }

    for (index_type i = 0; i < intermediate_bufs.size(); ++i) {
        auto &big_buffer = i > 0 ? intermediate_bufs[i - 1] : in_out_buffer;
        auto &small_buffer = intermediate_bufs[i];
        const auto group_range = sycl::range<1>{
                div_ceil(static_cast<index_type>(big_buffer.get_count()), granularity)};
        const auto local_range = sycl::range<1>{local_size};

        char label[50];
        sprintf(label, "hierarchical_inclusive_scan reduce %u", i);
        submit_and_profile(queue, label, [&](sycl::handler &cgh) {
            auto big_acc = big_buffer.template get_access<sam::read_write>(cgh);
            auto small_acc = small_buffer.template get_access<sam::discard_write>(cgh);
            cgh.parallel<hierarchical_inclusive_scan_reduction_kernel<Scalar, BinaryOp>>(
                    group_range, local_range,
                    [big_acc, small_acc, op](
                            known_size_group<local_size> grp, sycl::physical_item<1>) {
                        auto group_index = static_cast<index_type>(grp.get_id(0));
                        Scalar *big = &big_acc[group_index * granularity];
                        Scalar &small = small_acc[group_index];
                        inclusive_scan<granularity>(grp, big, op);
                        // TODO unnecessary GM read from big -- maybe return final sum
                        //  in the last item? Or allow additional accessor in inclusive_scan?
                        grp.single_item([&] { small = big[granularity - 1]; });
                    });
        });
    }

    for (index_type i = 1; i < intermediate_bufs.size(); ++i) {
        auto ii = static_cast<index_type>(intermediate_bufs.size()) - 1 - i;
        auto &small_buffer = intermediate_bufs[ii];
        auto &big_buffer = ii > 0 ? intermediate_bufs[ii - 1] : in_out_buffer;
        const auto group_range = sycl::range<1>{
                div_ceil(static_cast<index_type>(big_buffer.get_count()), granularity) - 1};
        const auto local_range = sycl::range<1>{local_size};

        char label[50];
        sprintf(label, "hierarchical_inclusive_scan expand %u", ii);
        submit_and_profile(queue, label, [&](sycl::handler &cgh) {
            auto small_acc = small_buffer.template get_access<sam::read>(cgh);
            auto big_acc = big_buffer.template get_access<sam::read_write>(cgh);
            cgh.parallel<hierarchical_inclusive_scan_expansion_kernel<Scalar, BinaryOp>>(
                    group_range, local_range,
                    [small_acc, big_acc, op](
                            known_size_group<local_size> grp, sycl::physical_item<1>) {
                        auto group_index = static_cast<index_type>(grp.get_id(0));
                        Scalar *big = &big_acc[(group_index + 1) * granularity];
                        Scalar small = small_acc[group_index];
                        grp.distribute_for(
                                granularity, [&](index_type i) { big[i] = op(big[i], small); });
                    });
        });
    }

    // (optionally) keep buffers alive so that cudaFree does not mess up profiling
    // TODO keep state in `scanner` type to avoid delayed allocation
    return intermediate_bufs;
}

template<unsigned Dims, typename U, typename T>
U extent_cast(const T &e) {
    U v;
    for (unsigned i = 0; i < Dims; ++i) {
        v[i] = e[i];
    }
    return v;
}

template<typename U, unsigned Dims>
U extent_cast(const extent<Dims> &e) {
    return extent_cast<Dims, U>(e);
}

template<typename T, int Dims>
T extent_cast(const sycl::range<Dims> &r) {
    return extent_cast<static_cast<unsigned>(Dims), T>(r);
}

template<typename T, int Dims>
T extent_cast(const sycl::id<Dims> &r) {
    return extent_cast<static_cast<unsigned>(Dims), T>(r);
}

}  // namespace ndzip::detail::gpu
