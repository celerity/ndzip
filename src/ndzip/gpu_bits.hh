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

// TODO must be a template parameter with selection based on queue (/device?) properties
inline constexpr index_type warp_size = 32;

template<typename T>
using global_read = sycl::accessor<T, 1, sycl::access::mode::read>;
template<typename T>
using global_write = sycl::accessor<T, 1, sycl::access::mode::write>;
template<typename T>
using global_read_write = sycl::accessor<T, 1, sycl::access::mode::read_write>;


template<index_type LocalSize>
struct known_size_group : public sycl::group<1> {
    using sycl::group<1>::group;
    using sycl::group<1>::distribute_for;

    known_size_group(sycl::group<1> grp)  // NOLINT(google-explicit-constructor)
        : sycl::group<1>{grp} {}

    template<typename F>
    [[gnu::always_inline]] void distribute_for(index_type range, F &&f) {
        auto invoke_f = [&](index_type item, index_type iteration, sycl::logical_item<1> idx,
                                sycl::sub_group sg) {
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
        };

        distribute_for([&](sycl::sub_group sg, sycl::logical_item<1> idx) {
            const index_type num_full_iterations = range / LocalSize;
            const index_type partial_iteration_length = range % LocalSize;
            const auto tid = static_cast<index_type>(idx.get_local_id(0));

#pragma unroll
            for (index_type iteration = 0; iteration < num_full_iterations; ++iteration) {
                auto item = iteration * LocalSize + tid;
                invoke_f(item, iteration, idx, sg);
            }

            if (tid < partial_iteration_length) {
                auto iteration = num_full_iterations;
                auto item = iteration * LocalSize + tid;
                invoke_f(item, iteration, idx, sg);
            }
        });
    }
};

template<index_type Range, index_type LocalSize, typename Accessor, typename BinaryOp>
std::enable_if_t<(Range <= warp_size)>
inclusive_scan(known_size_group<LocalSize> grp, Accessor acc, BinaryOp op) {
    grp.distribute_for(
            Range, [&](index_type item, index_type, sycl::logical_item<1>, sycl::sub_group sg) {
                acc[item] = sycl::group_inclusive_scan(sg, acc[item], op);
            });
}

template<index_type Range, index_type LocalSize, typename Accessor, typename BinaryOp>
std::enable_if_t<(Range > warp_size)>
inclusive_scan(known_size_group<LocalSize> grp, Accessor acc, BinaryOp op) {
    using value_type = std::decay_t<decltype(acc[index_type{}])>;

    sycl::private_memory<value_type[div_ceil(Range, LocalSize)]> fine{grp};
    sycl::local_memory<value_type[div_ceil(Range, warp_size)]> coarse{grp};
    grp.distribute_for(Range,
            [&](index_type item, index_type iteration, sycl::logical_item<1> idx,
                    sycl::sub_group sg) {
                fine(idx)[iteration] = sycl::group_inclusive_scan(sg, acc[item], op);
                if (item % warp_size == warp_size - 1) {
                    coarse[item / warp_size] = fine(idx)[iteration];
                }
            });
    inclusive_scan<div_ceil(Range, warp_size)>(grp, coarse(), op);
    grp.distribute_for(
            Range, [&](index_type item, index_type iteration, sycl::logical_item<1> idx) {
                auto value = fine(idx)[iteration];
                if (item >= warp_size) { value = op(value, coarse[item / warp_size - 1]); }
                acc[item] = value;
            });
}

}  // namespace ndzip::detail::gpu
