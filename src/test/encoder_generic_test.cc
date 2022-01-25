#include "test_utils.hh"

#include <iostream>

#include <ndzip/common.hh>
#include <ndzip/cpu_encoder.inl>


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
    auto shift_dist = std::uniform_int_distribution<unsigned>(0, bits_of<TestType> - 1);
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

template<unsigned Dims>
static auto dump_border_slices(const extent<Dims> &size, unsigned side_length) {
    slice_vec v;
    for_each_border_slice(
            size, side_length, [&](index_type offset, index_type count) { v.emplace_back(offset, count); });
    return v;
}


TEST_CASE("for_each_border_slice iterates correctly") {
    CHECK(dump_border_slices(extent<2>{4, 4}, 4) == slice_vec{});
    CHECK(dump_border_slices(extent<2>{4, 6}, 2) == slice_vec{});
    CHECK(dump_border_slices(extent<2>{5, 4}, 4) == slice_vec{{16, 4}});
    CHECK(dump_border_slices(extent<2>{4, 5}, 4) == slice_vec{{4, 1}, {9, 1}, {14, 1}, {19, 1}});
    CHECK(dump_border_slices(extent<2>{4, 5}, 2) == slice_vec{{4, 1}, {9, 1}, {14, 1}, {19, 1}});
    CHECK(dump_border_slices(extent<2>{4, 6}, 4) == slice_vec{{4, 2}, {10, 2}, {16, 2}, {22, 2}});
    CHECK(dump_border_slices(extent<2>{4, 6}, 5) == slice_vec{{0, 24}});
    CHECK(dump_border_slices(extent<2>{6, 4}, 5) == slice_vec{{0, 24}});
}


TEMPLATE_TEST_CASE("file produces a sane hypercube / header layout", "[file]", (std::integral_constant<unsigned, 1>),
        (std::integral_constant<unsigned, 2>), (std::integral_constant<unsigned, 3>),
        (std::integral_constant<unsigned, 4>) ) {
    constexpr unsigned dims = TestType::value;
    using profile = detail::profile<float, dims>;
    const index_type n = 100;
    const auto n_hypercubes_per_dim = n / profile::hypercube_side_length;
    const auto side_length = profile::hypercube_side_length;

    extent<dims> size;
    for (unsigned d = 0; d < dims; ++d) {
        size[d] = n;
    }

    std::vector<std::vector<extent<dims>>> superblocks;
    std::vector<bool> visited(ipow(n_hypercubes_per_dim, dims));

    file<profile> f(size);
    std::vector<extent<dims>> blocks;
    index_type hypercube_index = 0;
    f.for_each_hypercube([&](auto hc_offset, auto hc_index) {
        CHECK(hc_index == hypercube_index);

        auto off = hc_offset;
        for (unsigned d = 0; d < dims; ++d) {
            CHECK(off[d] < n);
            CHECK(off[d] % side_length == 0);
        }

        auto cell_index = off[0] / side_length;
        for (unsigned d = 1; d < dims; ++d) {
            cell_index = cell_index * n_hypercubes_per_dim + off[d] / side_length;
        }
        CHECK(!visited[cell_index]);
        visited[cell_index] = true;

        blocks.push_back(off);
        ++hypercube_index;
    });
    CHECK(blocks.size() == f.num_hypercubes());

    CHECK(std::all_of(visited.begin(), visited.end(), [](auto b) { return b; }));

    CHECK(f.file_header_length() == f.num_hypercubes() * sizeof(index_type));
    CHECK(f.num_hypercubes() == ipow(n_hypercubes_per_dim, dims));
}


/* requires Profile parameters
TEMPLATE_TEST_CASE("encoder produces the expected bit stream", "[encoder]",
    (cpu_encoder<float, 2>), (cpu_encoder<float, 3>),
    (mt_cpu_encoder<float, 2>), (mt_cpu_encoder<float, 3>)
) {
    using profile = detail::profile<typename TestType::data_type, TestType::dimensions>;
    using value_type = typename profile::value_type;
    using hc_offset_type = typename profile::hypercube_offset_type;

    const size_t n = 199;
    const auto cell = 3.141592f;
    const auto border = 2.71828f;
    constexpr auto dims = profile::dimensions;

    std::vector<float> data(ipow(n, dims));
    slice<float, dims> array(data.data(), extent<dims>::broadcast(n));

    const auto border_start = n / profile::hypercube_side_length * profile::hypercube_side_length;
    for_each_in_hyperslab(array.size(), [=](auto index) {
        if (std::all_of(index.begin(), index.end(), [=](auto s) { return s < border_start; })) {
            array[index] = cell;
        } else {
            array[index] = border;
        }
    });

    file<profile> f(array.size());
    REQUIRE(f.num_hypercubes() > 1);

    TestType encoder;
    std::vector<std::byte> stream(ndzip::compressed_size_bound<data_type>(array.size()));
    size_t size = encoder.compress(array, stream.data());

    CHECK(size <= stream.size());
    stream.resize(size);

    const size_t hc_size = sizeof(float) * ipow(profile::hypercube_side_length, dims);

    const auto *file_header = stream.data();
    f.for_each_hypercube([&](auto hc, auto hc_index) {
        const auto hc_offset = hc_index * hc_size;
        if (hc_index > 0) {
            const void *hc_offset_address = file_header + (hc_index - 1) * sizeof(hc_offset_type);
            CHECK(endian_transform(load_unaligned<hc_offset_type>(hc_offset_address))
                == hc_offset);
        }
        for (size_t i = 0; i < ipow(profile::hypercube_side_length, dims); ++i) {
            float value;
            const void *value_offset_address = file_header + hc_offset + i * sizeof value;
            detail::store_value<profile>(&value, load_unaligned<value_type>(value_offset_address));
            CHECK(memcmp(&value, &cell, sizeof value) == 0);
        }
    });

    const auto border_offset = f.file_header_length() + f.num_hypercubes() * hc_size;
    const void *border_offset_address = file_header + (f.num_hypercubes() - 1) * sizeof(uint64_t);
    CHECK(endian_transform(load_unaligned<uint64_t>(border_offset_address)) == border_offset);
    size_t n_border_elems = 0;
    for_each_border_slice(array.size(), profile::hypercube_side_length, [&](auto, auto count) {
        for (unsigned i = 0; i < count; ++i) {
            float value;
            const void *value_offset_address = stream.data() + border_offset
                + (n_border_elems + i) * sizeof value;
            detail::store_value<profile>(&value, load_unaligned<value_type>(value_offset_address));
            CHECK(memcmp(&value, &border, sizeof value) == 0);
        }
        n_border_elems += count;
    });
    CHECK(n_border_elems == num_elements(array.size()) - ipow(border_start, dims));
}
*/
