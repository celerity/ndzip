#pragma once

#include <algorithm>
#include <vector>
#include <random>
#include <sstream>

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
void check_for_vector_equality(const T *lhs, const T *rhs, size_t size) {
    size_t first_mismatch = SIZE_MAX, last_mismatch = 0;
    for (size_t i = 0; i < size; ++i) {
        if (lhs[i] != rhs[i]) {
            first_mismatch = std::min(first_mismatch, i);
            last_mismatch = std::max(last_mismatch, i);
        }
    }

    if (first_mismatch <= last_mismatch) {
        std::ostringstream ss;
        ss << "vectors mismatch between index " << first_mismatch << " and " << last_mismatch
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
void check_for_vector_equality(const std::vector<T> &lhs, const std::vector<T> &rhs) {
    if (lhs.size() != rhs.size()) {
        FAIL_CHECK("vectors differ in size: " << lhs.size() << " vs. " << rhs.size()
                                              << " elements\n");
    } else {
        check_for_vector_equality(lhs.data(), rhs.data(), lhs.size());
    }
}


template<class T>
void black_hole(T *datum) {
    __asm__ __volatile__("" :: "m"(datum));
}
