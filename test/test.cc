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


TEST_CASE("load_bits") {
    SECTION("for 8-bit integers") {
        alignas(uint8_t) const uint8_t bits[2] = {0b1010'0100, 0b1000'0000};
        CHECK(load_bits<uint8_t>(bits, 0, 2) == 0b10);
        CHECK(load_bits<uint8_t>(bits, 2, 3) == 0b100);
        CHECK(load_bits<uint8_t>(bits, 5, 4) == 0b1001);
    }
    SECTION("for 16-bit integers") {
        alignas(uint16_t) const uint8_t bits[4] = {0b1010'0100, 0b1000'0000};
        CHECK(load_bits<uint16_t>(bits, 0, 2) == 0b10);
        CHECK(load_bits<uint16_t>(bits, 2, 3) == 0b100);
        CHECK(load_bits<uint16_t>(bits, 5, 4) == 0b1001);
    }
    SECTION("for 32-bit integers") {
        alignas(uint32_t) const uint8_t bits[8] = {0b1010'0100, 0b1000'0000};
        CHECK(load_bits<uint32_t>(bits, 0, 2) == 0b10);
        CHECK(load_bits<uint32_t>(bits, 2, 3) == 0b100);
        CHECK(load_bits<uint32_t>(bits, 5, 4) == 0b1001);
    }
    SECTION("for 64-bit integers") {
        alignas(uint64_t) const uint8_t bits[16] = {0b1010'0100, 0b1000'0000};
        CHECK(load_bits<uint64_t>(bits, 0, 2) == 0b10);
        CHECK(load_bits<uint64_t>(bits, 2, 3) == 0b100);
        CHECK(load_bits<uint64_t>(bits, 5, 4) == 0b1001);
    }
}

TEST_CASE("store_bits_linear") {
    SECTION("for 8-bit integers") {
        alignas(uint8_t) const uint8_t expected_bits[2] = {0b1010'0100, 0b1000'0000};
        alignas(uint8_t) uint8_t bits[2] = {0};
        store_bits_linear<uint8_t>(bits, 0, 2, 0b10);
        store_bits_linear<uint8_t>(bits, 2, 3, 0b100);
        store_bits_linear<uint8_t>(bits, 5, 4, 0b1001);
        CHECK(memcmp(bits, expected_bits, sizeof bits) == 0);
    }
    SECTION("for 32-bit integers") {
        alignas(uint16_t) const uint8_t expected_bits[4] = {0b1010'0100, 0b1000'0000};
        alignas(uint16_t) uint8_t bits[4] = {0};
        store_bits_linear<uint16_t>(bits, 0, 2, 0b10);
        store_bits_linear<uint16_t>(bits, 2, 3, 0b100);
        store_bits_linear<uint16_t>(bits, 5, 4, 0b1001);
        CHECK(memcmp(bits, expected_bits, sizeof bits) == 0);
    }
    SECTION("for 32-bit integers") {
        alignas(uint32_t) const uint8_t expected_bits[8] = {0b1010'0100, 0b1000'0000};
        alignas(uint32_t) uint8_t bits[8] = {0};
        store_bits_linear<uint32_t>(bits, 0, 2, 0b10);
        store_bits_linear<uint32_t>(bits, 2, 3, 0b100);
        store_bits_linear<uint32_t>(bits, 5, 4, 0b1001);
        CHECK(memcmp(bits, expected_bits, sizeof bits) == 0);
    }
    SECTION("for 32-bit integers") {
        alignas(uint64_t) const uint8_t expected_bits[16] = {0b1010'0100, 0b1000'0000};
        alignas(uint64_t) uint8_t bits[16] = {0};
        store_bits_linear<uint64_t>(bits, 0, 2, 0b10);
        store_bits_linear<uint64_t>(bits, 2, 3, 0b100);
        store_bits_linear<uint64_t>(bits, 5, 4, 0b1001);
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


