#pragma once

// We don't use <complex>, but a Clang/glibcxx bug causes the build to fail when <random> is
// included before <complex> (__host__/__device__ mismatch for glibcxx internal __failed_assertion()
// prototype)
#include <complex>

#include <algorithm>
#include <random>
#include <sstream>
#include <vector>

#include <catch2/catch.hpp>


template<typename Arithmetic>
static std::vector<Arithmetic> make_random_vector(size_t size) {
    std::vector<Arithmetic> vector(size);
    auto gen = std::minstd_rand();
    if constexpr (std::is_floating_point_v<Arithmetic>) {
        auto dist = std::uniform_real_distribution<Arithmetic>();
        std::generate(vector.begin(), vector.end(), [&] { return dist(gen); });
    } else {
        auto dist = std::uniform_int_distribution<Arithmetic>();
        std::generate(vector.begin(), vector.end(), [&] { return dist(gen); });
    }
    return vector;
}


template<typename T>
void check_for_vector_equality(const T *lhs, const T *rhs, size_t size, const char *file, int line) {
    size_t first_mismatch = SIZE_MAX, last_mismatch = 0;
    for (size_t i = 0; i < size; ++i) {
        if (lhs[i] != rhs[i]) {
            first_mismatch = std::min(first_mismatch, i);
            last_mismatch = std::max(last_mismatch, i);
        }
    }

    if (first_mismatch <= last_mismatch) {
        std::ostringstream ss;
        ss << file << ":" << line << ": vectors mismatch between index " << first_mismatch << " and " << last_mismatch
           << ":\n    {";
        for (auto *vec : {&lhs, &rhs}) {
            for (size_t i = first_mismatch; i <= last_mismatch;) {
                ss << (*vec)[i];
                if (i < last_mismatch) { ss << ", "; }
                if (i >= first_mismatch + 20 && i < last_mismatch - 20) {
                    i = last_mismatch - 20;
                    ss << "..., ";
                } else {
                    ++i;
                }
            }
            if (vec == &lhs) { ss << "}\n != {"; }
        }
        ss << "}\n";
        FAIL_CHECK(ss.str());
    }
}


template<typename T>
void check_for_vector_equality(const std::vector<T> &lhs, const std::vector<T> &rhs,
        const char *file, int line) {
    if (lhs.size() != rhs.size()) {
        FAIL_CHECK(file << ":" << line << ": vectors differ in size: " << lhs.size() << " vs. " << rhs.size() << " elements\n");
    } else {
        check_for_vector_equality(lhs.data(), rhs.data(), lhs.size(), file, line);
    }
}


#define CHECK_FOR_VECTOR_EQUALITY(...) check_for_vector_equality(__VA_ARGS__, __FILE__, __LINE__)


template<class T>
void black_hole(T *datum) {
    __asm__ __volatile__("" ::"m"(datum));
}


// std::inclusive_scan is not available on all platforms
template<typename InputIt, typename OutputIt, typename BinaryOperation = std::plus<>,
        typename T = typename std::iterator_traits<OutputIt>::value_type>
constexpr OutputIt iter_inclusive_scan(InputIt first, InputIt last, OutputIt d_first,
                                       BinaryOperation binary_op = {}, T init = {}) {
    while (first != last) {
        *d_first = init = binary_op(init, *first);
        ++first;
        ++d_first;
    }
    return d_first;
}

// std::exclusive_scan is not available on all platforms
template<typename InputIt, typename OutputIt, typename BinaryOperation = std::plus<>,
        typename T = typename std::iterator_traits<OutputIt>::value_type>
constexpr OutputIt iter_exclusive_scan(InputIt first, InputIt last, OutputIt d_first,
        BinaryOperation binary_op = {}, T init = {}) {
    while (first != last) {
        *d_first = init;
        init = binary_op(init, *first);
        ++first;
        ++d_first;
    }
    return d_first;
}
