#include "test_utils.hh"

#include <iostream>

#include <ndzip/common.hh>
#include <ndzip/cpu_codec.inl>


using namespace ndzip;
using namespace ndzip::detail;


TEST_CASE("floor_power_of_two works as advertised") {
    CHECK(floor_power_of_two(0) == 0);
    CHECK(floor_power_of_two(1) == 1);
    CHECK(floor_power_of_two(2) == 2);
    CHECK(floor_power_of_two(3) == 2);
    CHECK(floor_power_of_two(4) == 4);
    CHECK(floor_power_of_two(63) == 32);
    CHECK(floor_power_of_two(64) == 64);
    CHECK(floor_power_of_two(65) == 64);
    CHECK(floor_power_of_two(1023) == 512);
    CHECK(floor_power_of_two(1024) == 1024);
    CHECK(floor_power_of_two(1025) == 1024);
}


template<typename Bits>
Bits zero_map_from_transposed(const Bits *transposed) {
    Bits zero_map = 0;
    for (index_type i = 0; i < bits_of<Bits>; ++i) {
        zero_map = (zero_map << 1u) | (transposed[i] != 0);
    }
    return zero_map;
}


TEMPLATE_TEST_CASE("CPU zero-word compaction is reversible", "[cpu]", uint32_t, uint64_t) {
    cpu::simd_aligned_buffer<TestType> input(bits_of<TestType>);
    auto gen = std::minstd_rand();  // NOLINT(cert-msc51-cpp)
    auto dist = std::uniform_int_distribution<TestType>();
    index_type zeroes = 0;
    for (index_type i = 0; i < bits_of<TestType>; ++i) {
        auto r = dist(gen);
        if (r % 5 == 2) {
            input[i] = 0;
            zeroes++;
        } else {
            input[i] = r;
        }
    }

    std::vector<std::byte> compact((bits_of<TestType> + 1) * sizeof(TestType));
    auto head = zero_map_from_transposed(input.data());
    auto bytes_written = cpu::compact_zero_words(input.data(), compact.data());
    CHECK(bytes_written == (bits_of<TestType> - zeroes) * sizeof(TestType));

    std::vector<TestType> output(bits_of<TestType>);
    auto bytes_read = cpu::expand_zero_words(compact.data(), output.data(), head);
    CHECK(bytes_read == bytes_written);
    CHECK(output == std::vector<TestType>(input.data(), input.data() + bits_of<TestType>));
}


TEMPLATE_TEST_CASE("CPU bit transposition is reversible", "[cpu]", uint32_t, uint64_t) {
    alignas(cpu::simd_width_bytes) TestType input[bits_of<TestType>];
    auto rng = std::minstd_rand(1);
    auto bit_dist = std::uniform_int_distribution<TestType>();
    auto shift_dist = std::uniform_int_distribution<index_type>(0, bits_of<TestType> - 1);
    for (auto &value : input) {
        value = bit_dist(rng) >> shift_dist(rng);
    }

    alignas(cpu::simd_width_bytes) TestType transposed[bits_of<TestType>];
    cpu::transpose_bits(input, transposed);

    alignas(cpu::simd_width_bytes) TestType output[bits_of<TestType>];
    cpu::transpose_bits(transposed, output);

    CHECK(memcmp(input, output, sizeof input) == 0);
}


using border_slice = std::pair<index_type, index_type>;
using slice_vec = std::vector<border_slice>;

namespace std {
ostream &operator<<(ostream &os, const border_slice &s) {
    return os << "(" << s.first << ", " << s.second << ")";
}
}  // namespace std

template<dim_type Dims>
static auto dump_border_slices(const static_extent<Dims> &size, index_type side_length) {
    slice_vec v;
    for_each_border_slice(
            size, side_length, [&](index_type offset, index_type count) { v.emplace_back(offset, count); });
    return v;
}


TEST_CASE("for_each_border_slice iterates correctly") {
    CHECK(dump_border_slices(static_extent<2>{4, 4}, 4) == slice_vec{});
    CHECK(dump_border_slices(static_extent<2>{4, 6}, 2) == slice_vec{});
    CHECK(dump_border_slices(static_extent<2>{5, 4}, 4) == slice_vec{{16, 4}});
    CHECK(dump_border_slices(static_extent<2>{4, 5}, 4) == slice_vec{{4, 1}, {9, 1}, {14, 1}, {19, 1}});
    CHECK(dump_border_slices(static_extent<2>{4, 5}, 2) == slice_vec{{4, 1}, {9, 1}, {14, 1}, {19, 1}});
    CHECK(dump_border_slices(static_extent<2>{4, 6}, 4) == slice_vec{{4, 2}, {10, 2}, {16, 2}, {22, 2}});
    CHECK(dump_border_slices(static_extent<2>{4, 6}, 5) == slice_vec{{0, 24}});
    CHECK(dump_border_slices(static_extent<2>{6, 4}, 5) == slice_vec{{0, 24}});
}


TEMPLATE_TEST_CASE("file produces a sane hypercube / header layout", "[file]", (std::integral_constant<dim_type, 1>),
        (std::integral_constant<dim_type, 2>), (std::integral_constant<dim_type, 3>) ) {
    constexpr dim_type dims = TestType::value;
    using profile = detail::profile<float, dims>;
    const index_type n = 100;
    const auto n_hypercubes_per_dim = n / profile::hypercube_side_length;
    const auto side_length = profile::hypercube_side_length;

    static_extent<dims> size;
    for (dim_type d = 0; d < dims; ++d) {
        size[d] = n;
    }

    std::vector<std::vector<static_extent<dims>>> superblocks;
    std::vector<bool> visited(ipow(n_hypercubes_per_dim, dims));

    std::vector<static_extent<dims>> blocks;
    index_type hypercube_index = 0;
    for_each_hypercube(size, [&](auto hc_offset, auto hc_index) {
        CHECK(hc_index == hypercube_index);

        auto off = hc_offset;
        for (dim_type d = 0; d < dims; ++d) {
            CHECK(off[d] < n);
            CHECK(off[d] % side_length == 0);
        }

        auto cell_index = off[0] / side_length;
        for (dim_type d = 1; d < dims; ++d) {
            cell_index = cell_index * n_hypercubes_per_dim + off[d] / side_length;
        }
        CHECK(!visited[cell_index]);
        visited[cell_index] = true;

        blocks.push_back(off);
        ++hypercube_index;
    });
    CHECK(blocks.size() == num_hypercubes(size));

    CHECK(std::all_of(visited.begin(), visited.end(), [](auto b) { return b; }));

    // CHECK(f.file_header_length() == f.num_hypercubes() * sizeof(index_type));
    CHECK(num_hypercubes(size) == ipow(n_hypercubes_per_dim, dims));
}
