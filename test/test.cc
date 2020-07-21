#include <hcde.hh>
#include "../src/common.hh"

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>


using namespace hcde;
using namespace hcde::detail;


const float float_data_2d[16] = {
    0.1f, 1.1f, 2.3f, 4.9f,
    -0.1f, 1.1f, 2.3f, 4.9f,
    0.1f, 2.3f, 9.3f, 0.9f,
    -0.1f, 1.1f, 0.3f, 0.0f,
};

const float float_data_2d_with_border[40] = {
    0.1f, 1.1f, 2.3f, 4.9f, 1.f,
    -0.1f, 1.1f, 2.3f, 4.9f, 2.f,
    0.1f, 2.3f, 9.3f, 0.9f, 3.f,
    -0.1f, 1.1f, 0.3f, 0.0f, 4.f,
    -0.1f, 1.1f, 2.3f, 4.9f, 2.f,
    0.1f, 1.1f, 2.3f, 4.9f, 1.f,
    -0.1f, 1.1f, 0.3f, 0.0f, 4.f,
    0.1f, 2.3f, 9.3f, 0.9f, 3.f,
};


TEST_CASE("load_bits") {
    SECTION("for 8-bit integers") {
        alignas(uint8_t) const uint8_t bits[2] = {0b1010'0100, 0b1000'0000};
        CHECK(load_bits<uint8_t>(const_bit_ptr<1>(bits, 0), 2) == 0b10);
        CHECK(load_bits<uint8_t>(const_bit_ptr<1>(bits, 2), 3) == 0b100);
        CHECK(load_bits<uint8_t>(const_bit_ptr<1>(bits, 5), 4) == 0b1001);
    }
    SECTION("for 16-bit integers") {
        alignas(uint16_t) const uint8_t bits[4] = {0b1010'0100, 0b1000'0000};
        CHECK(load_bits<uint16_t>(const_bit_ptr<2>(bits, 0), 2) == 0b10);
        CHECK(load_bits<uint16_t>(const_bit_ptr<2>(bits, 2), 3) == 0b100);
        CHECK(load_bits<uint16_t>(const_bit_ptr<2>(bits, 5), 4) == 0b1001);
    }
    SECTION("for 32-bit integers") {
        alignas(uint32_t) const uint8_t bits[8] = {0b1010'0100, 0b1000'0000};
        CHECK(load_bits<uint32_t>(const_bit_ptr<4>(bits, 0), 2) == 0b10);
        CHECK(load_bits<uint32_t>(const_bit_ptr<4>(bits, 2), 3) == 0b100);
        CHECK(load_bits<uint32_t>(const_bit_ptr<4>(bits, 5), 4) == 0b1001);
    }
    SECTION("for 64-bit integers") {
        alignas(uint64_t) const uint8_t bits[16] = {0b1010'0100, 0b1000'0000};
        CHECK(load_bits<uint64_t>(const_bit_ptr<8>(bits, 0), 2) == 0b10);
        CHECK(load_bits<uint64_t>(const_bit_ptr<8>(bits, 2), 3) == 0b100);
        CHECK(load_bits<uint64_t>(const_bit_ptr<8>(bits, 5), 4) == 0b1001);
    }
}

TEST_CASE("store_bits_linear") {
    SECTION("for 8-bit integers") {
        alignas(uint8_t) const uint8_t expected_bits[2] = {0b1010'0100, 0b1000'0000};
        alignas(uint8_t) uint8_t bits[2] = {0};
        store_bits_linear<uint8_t>(bit_ptr<1>(bits, 0), 2, 0b10);
        store_bits_linear<uint8_t>(bit_ptr<1>(bits, 2), 3, 0b100);
        store_bits_linear<uint8_t>(bit_ptr<1>(bits, 5), 4, 0b1001);
        CHECK(memcmp(bits, expected_bits, sizeof bits) == 0);
    }
    SECTION("for 32-bit integers") {
        alignas(uint16_t) const uint8_t expected_bits[4] = {0b1010'0100, 0b1000'0000};
        alignas(uint16_t) uint8_t bits[4] = {0};
        store_bits_linear<uint16_t>(bit_ptr<2>(bits, 0), 2, 0b10);
        store_bits_linear<uint16_t>(bit_ptr<2>(bits, 2), 3, 0b100);
        store_bits_linear<uint16_t>(bit_ptr<2>(bits, 5), 4, 0b1001);
        CHECK(memcmp(bits, expected_bits, sizeof bits) == 0);
    }
    SECTION("for 32-bit integers") {
        alignas(uint32_t) const uint8_t expected_bits[8] = {0b1010'0100, 0b1000'0000};
        alignas(uint32_t) uint8_t bits[8] = {0};
        store_bits_linear<uint32_t>(bit_ptr<4>(bits, 0), 2, 0b10);
        store_bits_linear<uint32_t>(bit_ptr<4>(bits, 2), 3, 0b100);
        store_bits_linear<uint32_t>(bit_ptr<4>(bits, 5), 4, 0b1001);
        CHECK(memcmp(bits, expected_bits, sizeof bits) == 0);
    }
    SECTION("for 32-bit integers") {
        alignas(uint64_t) const uint8_t expected_bits[16] = {0b1010'0100, 0b1000'0000};
        alignas(uint64_t) uint8_t bits[16] = {0};
        store_bits_linear<uint64_t>(bit_ptr<8>(bits, 0), 2, 0b10);
        store_bits_linear<uint64_t>(bit_ptr<8>(bits, 2), 3, 0b100);
        store_bits_linear<uint64_t>(bit_ptr<8>(bits, 5), 4, 0b1001);
        CHECK(memcmp(bits, expected_bits, sizeof bits) == 0);
    }
}


TEST_CASE("fast_profile value en-/decode") {
    uint32_t bits[16];
    fast_profile<float, 2> p;
    for (unsigned i = 0; i < 16; ++i) {
        bits[i] = p.load_value(&float_data_2d[i]);
    }
    float data[16];
    for (unsigned i = 0; i < 16; ++i) {
        p.store_value(&data[i], bits[i]);
    }
    CHECK(memcmp(float_data_2d, data, sizeof data) == 0);
}


TEST_CASE("fast_profile block en-/decode") {
    uint32_t bits[16];
    fast_profile<float, 2> p;
    for (unsigned i = 0; i < 16; ++i) {
        bits[i] = p.load_value(&float_data_2d[i]);
    }
    char stream[100]={0};
    p.encode_block(bits, stream);
    uint32_t bits2[16]={0};
    p.decode_block(stream, bits2);
    CHECK(memcmp(bits, bits2, sizeof bits) == 0);
}


using border_slice = std::pair<size_t, size_t>;
using slice_vec = std::vector<border_slice>;

namespace std {
    ostream &operator<<(ostream &os, const border_slice &s) {
        return os << "(" << s.first << ", " << s.second << ")";
    }
}

template<unsigned Dims>
static auto dump_border_slices(const extent<Dims> &size, unsigned side_length) {
    slice_vec v;
    for_each_border_slice(size, side_length, [&](size_t offset, size_t count) {
        v.emplace_back(offset, count);
    });
    return v;
}


TEST_CASE("for_each_border_slice") {
    CHECK(dump_border_slices(extent<2>{4, 4}, 4) == slice_vec{});
    CHECK(dump_border_slices(extent<2>{4, 6}, 2) == slice_vec{});
    CHECK(dump_border_slices(extent<2>{5, 4}, 4) == slice_vec{{16, 4}});
    CHECK(dump_border_slices(extent<2>{4, 5}, 4) == slice_vec{{4, 1}, {9, 1}, {14, 1}, {19, 1}});
    CHECK(dump_border_slices(extent<2>{4, 5}, 2) == slice_vec{{4, 1}, {9, 1}, {14, 1}, {19, 1}});
    CHECK(dump_border_slices(extent<2>{4, 6}, 4) == slice_vec{{4, 2}, {10, 2}, {16, 2}, {22, 2}});
    CHECK(dump_border_slices(extent<2>{4, 6}, 5) == slice_vec{{0, 24}});
    CHECK(dump_border_slices(extent<2>{6, 4}, 5) == slice_vec{{0, 24}});
}


TEMPLATE_TEST_CASE("singlethread_cpu_encoder", "", (fast_profile<float, 2>)) {
    std::string stream;
    singlethread_cpu_encoder<TestType> p;
    slice<const float, 2> data(float_data_2d_with_border, extent<2>{8, 5});
    stream.resize(p.compressed_size_bound(data.extent()));
    auto cursor = p.compress(data, stream.data());
    CHECK(cursor <= stream.size());

    float restore_data[sizeof(float_data_2d_with_border) / sizeof(float)];
    slice<float, 2> restore(restore_data, extent<2>{8, 5});
    auto de_cursor = p.decompress(stream.data(), cursor, restore);
    CHECK(de_cursor == cursor);
    CHECK(memcmp(float_data_2d_with_border, restore_data, sizeof restore_data) == 0);
}

