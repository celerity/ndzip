#pragma once

#include <type_traits>

#include <SYCL/sycl.hpp>


namespace ndzip::detail::gpu {

template<typename Integer>
constexpr Integer div_ceil(Integer p, Integer q) {
    return (p + q - 1) / q;
}

template<typename Integer>
constexpr Integer ceil(Integer x, Integer multiple) {
    return div_ceil(x, multiple) * multiple;
}

template<typename Integer>
constexpr Integer floor(Integer x, Integer multiple) {
    return x / multiple * multiple;
}


using index_type = uint64_t;

// TODO _should_ be a template parameter with selection based on queue (/device?) properties.
//  However a lot of code currently assumes that bitsof<uint32_t> == warp_size (e.g. we want to
//  use subgroup reductions for length-32 chunks of residuals)
inline constexpr index_type warp_size = 32;

template<typename T>
using global_read = sycl::accessor<T, 1, sycl::access::mode::read>;
template<typename T>
using global_write = sycl::accessor<T, 1, sycl::access::mode::write>;
template<typename T>
using global_read_write = sycl::accessor<T, 1, sycl::access::mode::read_write>;


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
[[gnu::always_inline]]
std::enable_if_t<(Range <= warp_size)>
inclusive_scan(known_size_group<LocalSize> grp, Accessor acc, BinaryOp op) {
    static_assert(LocalSize % warp_size == 0);
    grp.template distribute_for<ceil(Range, warp_size)>(
            [&](index_type item, index_type, sycl::logical_item<1>, sycl::sub_group sg) __attribute__((always_inline)){
                auto a = item < Range ? acc[item] : 0;
                auto b = sycl::group_inclusive_scan(sg, a, op);
                if (item < Range) { acc[item] = b; }
            });
}

template<index_type Range, index_type LocalSize, typename Accessor, typename BinaryOp>
[[gnu::always_inline]]
std::enable_if_t<(Range > warp_size)>
inclusive_scan(known_size_group<LocalSize> grp, Accessor acc, BinaryOp op) {
    static_assert(LocalSize % warp_size == 0);
    using value_type = std::decay_t<decltype(acc[index_type{}])>;

    sycl::private_memory<value_type[div_ceil(Range, LocalSize)]> fine{grp};
    sycl::local_memory<value_type[div_ceil(Range, warp_size)]> coarse{grp};
    grp.template distribute_for<ceil(Range, warp_size)>([&](index_type item, index_type iteration,
                                                                sycl::logical_item<1> idx,
                                                                sycl::sub_group sg) __attribute__((always_inline)) {
        fine(idx)[iteration] = sycl::group_inclusive_scan(sg, item < Range ? acc[item] : 0, op);
        if (item % warp_size == warp_size - 1) { coarse[item / warp_size] = fine(idx)[iteration]; }
    });
    inclusive_scan<div_ceil(Range, warp_size)>(grp, coarse(), op);
    grp.template distribute_for<Range>(
            [&](index_type item, index_type iteration, sycl::logical_item<1> idx) __attribute__((always_inline)) {
                auto value = fine(idx)[iteration];
                if (item >= warp_size) { value = op(value, coarse[item / warp_size - 1]); }
                acc[item] = value;
            });
}

}  // namespace ndzip::detail::gpu
