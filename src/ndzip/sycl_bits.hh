#pragma once

#include "gpu_common.hh"

#include <type_traits>

#include <SYCL/sycl.hpp>


namespace ndzip::detail::gpu_sycl {

using namespace ndzip::detail::gpu;

inline uint64_t earliest_event_start(const sycl::event &evt) {
    return evt.get_profiling_info<sycl::info::event_profiling::command_start>();
}

inline uint64_t earliest_event_start(const std::vector<sycl::event> &events) {
    uint64_t start = UINT64_MAX;
    for (auto &evt : events) {
        start = std::min(start, evt.get_profiling_info<sycl::info::event_profiling::command_start>());
    }
    return start;
}

inline uint64_t latest_event_end(const sycl::event &evt) {
    return evt.get_profiling_info<sycl::info::event_profiling::command_end>();
}

inline uint64_t latest_event_end(const std::vector<sycl::event> &events) {
    uint64_t end = 0;
    for (auto &evt : events) {
        end = std::max(end, evt.get_profiling_info<sycl::info::event_profiling::command_end>());
    }
    return end;
}

template<typename... Events>
std::tuple<uint64_t, uint64_t, kernel_duration> measure_duration(const Events &...events) {
    auto early = std::min<uint64_t>({earliest_event_start(events)...});
    auto late = std::max<uint64_t>({latest_event_end(events)...});
    return {early, late, kernel_duration{late - early}};
}

template<typename CGF>
auto submit_and_profile(sycl::queue &q, const char *label, CGF &&cgf) {
    if (verbose() && q.has_property<sycl::property::queue::enable_profiling>()) {
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
    known_size_group(sycl::group<1> grp)  // NOLINT(google-explicit-constructor)
        : sycl::group<1>{grp} {}

    size_t get_local_linear_range() const { return LocalSize; }

    sycl::range<1> get_local_range() const { return LocalSize; }
};

template<index_type LocalSize>
void group_barrier(known_size_group<LocalSize> grp,
        sycl::memory_scope fence_scope = known_size_group<LocalSize>::fence_scope) {
    group_barrier(static_cast<sycl::group<1>&>(grp), fence_scope);
}

template<typename F>
[[gnu::always_inline]] void distribute_for_invoke(F &&f, index_type item, index_type iteration) {
    if constexpr (std::is_invocable_v<F, index_type, index_type>) {
        f(item, iteration);
    } else {
        f(item);
    }
}

template<index_type LocalSize, typename F>
[[gnu::always_inline]] void distribute_for(index_type range, known_size_group<LocalSize> grp, F &&f) {
    const index_type num_full_iterations = range / LocalSize;
    const auto tid = static_cast<index_type>(grp.get_local_id(0));

    for (index_type iteration = 0; iteration < num_full_iterations; ++iteration) {
        auto item = iteration * LocalSize + tid;
        distribute_for_invoke(f, item, iteration);
    }
    distribute_for_partial_iteration(range, grp, f);
    group_barrier(grp);
}

template<index_type Range, index_type LocalSize, typename F>
[[gnu::always_inline]] void distribute_for(known_size_group<LocalSize> grp, F &&f) {
    constexpr index_type num_full_iterations = Range / LocalSize;
    const auto tid = static_cast<index_type>(grp.get_local_id(0));

#pragma unroll
    for (index_type iteration = 0; iteration < num_full_iterations; ++iteration) {
        auto item = iteration * LocalSize + tid;
        distribute_for_invoke(f, item, iteration);
    }
    distribute_for_partial_iteration(Range, grp, f);
    group_barrier(grp);
}

template<index_type LocalSize, typename F>
[[gnu::always_inline]] void distribute_for_partial_iteration(index_type range, known_size_group<LocalSize> grp, F &&f) {
    const index_type num_full_iterations = range / LocalSize;
    const index_type partial_iteration_length = range % LocalSize;
    const auto tid = static_cast<index_type>(grp.get_local_id(0));
    if (tid < partial_iteration_length) {
        auto iteration = num_full_iterations;
        auto item = iteration * LocalSize + tid;
        distribute_for_invoke(f, item, iteration);
    }
}


template<index_type LocalSize>
class known_group_size_item : public sycl::nd_item<1> {
  public:
    known_group_size_item(sycl::nd_item<1> item)  // NOLINT(google-explicit-constructor)
        : sycl::nd_item<1>{item} {}

    size_t get_local_linear_range() const { return LocalSize; }

    sycl::range<1> get_local_range() const { return LocalSize; }

    sycl::id<1> get_group_id() const { return get_group().get_group_id(); }

    size_t get_group_id(int dimension) const { return get_group().get_group_id(dimension); }

    known_size_group<LocalSize> get_group() const { return sycl::nd_item<1>::get_group(); }

    sycl::group<1> get_sycl_group() const { return sycl::nd_item<1>::get_group(); }
};


inline sycl::nd_range<1> make_nd_range(index_type n_groups, index_type local_size) {
    return sycl::nd_range<1>{n_groups * local_size, local_size};
}


template<typename Value, index_type Range, typename Enable = void>
struct inclusive_scan_local_allocation {};

template<typename Value, index_type Range>
struct inclusive_scan_local_allocation<Value, Range, std::enable_if_t<(Range > warp_size)>> {
    Value memory[div_ceil(Range, warp_size)];
    inclusive_scan_local_allocation<Value, div_ceil(Range, warp_size)> next;
};


// TODO rename (confusion with sycl::inclusive_scan_over_group + it receives an item, not a group!)
template<index_type Range, index_type LocalSize, typename Accessor, typename BinaryOp>
std::enable_if_t<(Range <= warp_size)> inclusive_scan_over_group(known_group_size_item<LocalSize> item, Accessor acc,
        inclusive_scan_local_allocation<std::decay_t<decltype(acc[index_type{}])>, Range> &, BinaryOp op) {
    static_assert(LocalSize % warp_size == 0);
    distribute_for<ceil(Range, warp_size)>(item.get_group(), [&](index_type i) {
        auto a = i < Range ? acc[i] : 0;
        auto b = sycl::inclusive_scan_over_group(item.get_sub_group(), a, op);
        if (i < Range) { acc[i] = b; }
    });
}

template<index_type Range, index_type LocalSize, typename Accessor, typename BinaryOp>
std::enable_if_t<(Range > warp_size)> inclusive_scan_over_group(known_group_size_item<LocalSize> item, Accessor acc,
        inclusive_scan_local_allocation<std::decay_t<decltype(acc[index_type{}])>, Range> &lm, BinaryOp op) {
    static_assert(LocalSize % warp_size == 0);
    using value_type = std::decay_t<decltype(acc[index_type{}])>;

    value_type fine[div_ceil(Range, LocalSize)];  // per-thread
    const auto coarse = lm.memory;
    distribute_for<ceil(Range, warp_size)>(item.get_group(), [&](index_type i, index_type iteration) {
        fine[iteration] = sycl::inclusive_scan_over_group(item.get_sub_group(), i < Range ? acc[i] : 0, op);
        if (i % warp_size == warp_size - 1) { coarse[i / warp_size] = fine[iteration]; }
    });
    inclusive_scan_over_group(item, coarse, lm.next, op);
    distribute_for<Range>(item.get_group(), [&](index_type i, index_type iteration) {
        auto value = fine[iteration];
        if (i >= warp_size) { value = op(value, coarse[i / warp_size - 1]); }
        acc[i] = value;
    });
}

template<typename Scalar>
std::vector<sycl::buffer<Scalar>> hierarchical_inclusive_scan_allocate(index_type in_out_buffer_size) {
    constexpr index_type granularity = hierarchical_inclusive_scan_granularity;

    std::vector<sycl::buffer<Scalar>> intermediate_bufs;
    assert(in_out_buffer_size % granularity == 0);  // otherwise we will overrun the in_out buffer bounds

    auto n_elems = in_out_buffer_size;
    while (n_elems > 1) {
        n_elems = div_ceil(n_elems, granularity);
        intermediate_bufs.emplace_back(ceil(n_elems, granularity));
    }

    return intermediate_bufs;
}

template<typename, typename>
class hierarchical_inclusive_scan_reduction_kernel;

template<typename, typename>
class hierarchical_inclusive_scan_expansion_kernel;

template<typename Scalar, typename BinaryOp>
void hierarchical_inclusive_scan(sycl::queue &queue, sycl::buffer<Scalar> &in_out_buffer,
        std::vector<sycl::buffer<Scalar>> &intermediate_bufs, BinaryOp op = {}) {
    using sam = sycl::access::mode;

    constexpr index_type granularity = hierarchical_inclusive_scan_granularity;
    constexpr index_type local_size = 256;

    for (index_type i = 0; i < intermediate_bufs.size(); ++i) {
        auto &big_buffer = i > 0 ? intermediate_bufs[i - 1] : in_out_buffer;
        auto &small_buffer = intermediate_bufs[i];

        char label[50];
        sprintf(label, "hierarchical_inclusive_scan reduce %u", i);
        submit_and_profile(queue, label, [&](sycl::handler &cgh) {
            auto big_acc = big_buffer.template get_access<sam::read_write>(cgh);
            auto small_acc = small_buffer.template get_access<sam::discard_write>(cgh);
            sycl::local_accessor<inclusive_scan_local_allocation<Scalar, granularity>> lm{1, cgh};

          const auto n_groups = div_ceil(static_cast<index_type>(big_buffer.get_count()), granularity);
          const auto nd_range = make_nd_range(n_groups, local_size);
            cgh.parallel_for<hierarchical_inclusive_scan_reduction_kernel<Scalar, BinaryOp>>(
                    nd_range, [big_acc, small_acc, lm, op](known_group_size_item<local_size> item) {
                        auto group_index = static_cast<index_type>(item.get_group_id(0));
                        Scalar *big = &big_acc[group_index * granularity];
                        Scalar &small = small_acc[group_index];
                        inclusive_scan_over_group(item, big, lm[0], op);
                        // TODO unnecessary GM read from big -- maybe return final sum
                        //  in the last item? Or allow additional accessor in inclusive_scan?
                        if (item.get_group().leader()) { small = big[granularity - 1]; }
                    });
        });
    }

    for (index_type i = 1; i < intermediate_bufs.size(); ++i) {
        auto ii = static_cast<index_type>(intermediate_bufs.size()) - 1 - i;
        auto &small_buffer = intermediate_bufs[ii];
        auto &big_buffer = ii > 0 ? intermediate_bufs[ii - 1] : in_out_buffer;

        char label[50];
        sprintf(label, "hierarchical_inclusive_scan expand %u", ii);
        submit_and_profile(queue, label, [&](sycl::handler &cgh) {
            auto small_acc = small_buffer.template get_access<sam::read>(cgh);
            auto big_acc = big_buffer.template get_access<sam::read_write>(cgh);

          const auto n_groups = div_ceil(static_cast<index_type>(big_buffer.get_count()), granularity) - 1;
          const auto nd_range = make_nd_range(n_groups, local_size);
            cgh.parallel_for<hierarchical_inclusive_scan_expansion_kernel<Scalar, BinaryOp>>(nd_range,
                    [small_acc, big_acc, op](known_group_size_item<local_size> item) {
                        auto group_index = static_cast<index_type>(item.get_group_id(0));
                        Scalar *big = &big_acc[(group_index + 1) * granularity];
                        Scalar small = small_acc[group_index];
                        distribute_for(granularity, item.get_group(), [&](index_type i) { big[i] = op(big[i], small); });
                    });
        });
    }
}

template<dim_type Dims, typename U, typename T>
U extent_cast(const T &e) {
    U v;
    for (dim_type d = 0; d < Dims; ++d) {
        v[d] = e[d];
    }
    return v;
}

template<typename U, dim_type Dims>
U extent_cast(const extent<Dims> &e) {
    return extent_cast<Dims, U>(e);
}

template<typename T, int Dims>
T extent_cast(const sycl::range<Dims> &r) {
    return extent_cast<Dims, T>(r);
}

template<typename T, int Dims>
T extent_cast(const sycl::id<Dims> &r) {
    return extent_cast<Dims, T>(r);
}

}  // namespace ndzip::detail::gpu_sycl
