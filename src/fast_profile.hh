#pragma once

#include "common.hh"


namespace hcde {

template<typename T, unsigned Dims>
auto fast_profile<T, Dims>::load_value(const data_type *data) const -> bits_type {
    bits_type bits;
    memcpy(&bits, data, sizeof bits);
    if constexpr (std::is_floating_point_v<data_type>) {
        bits = (bits << 1u) | (bits >> (detail::bitsof<bits_type> - 1));
    } else if constexpr (std::is_signed_v<data_type>) {
        bits += static_cast<bits_type>(std::numeric_limits<data_type>::min());
    }
    return bits;
}


template<typename T, unsigned Dims>
void fast_profile<T, Dims>::store_value(data_type *data, bits_type bits) const {
    if constexpr (std::is_floating_point_v<data_type>) {
        bits = (bits >> 1u) | (bits << (detail::bitsof<bits_type> - 1));
    } else if constexpr (std::is_signed_v<data_type>) {
        bits -= static_cast<bits_type>(std::numeric_limits<data_type>::min());
    }
    memcpy(data, &bits, sizeof bits);
}


template<typename T, unsigned Dims>
size_t fast_profile<T, Dims>::encode_block(const bits_type *bits, void *stream) const {
    auto ref = bits[0];
    bits_type domain = 0;
    for (size_t i = 1; i < detail::ipow(hypercube_side_length, Dims); ++i) {
        domain |= bits[i] ^ ref;
    }
    auto remainder_width = detail::significant_bits(domain);
    auto width_width = sizeof(T) == 1 ? 4 : sizeof(T) == 2 ? 5 : sizeof(T) == 4 ? 6 : 7;
    memcpy(stream, &ref, sizeof ref);
    auto dest = detail::bit_ptr<sizeof(bits_type)>::from_unaligned_pointer(stream);
    dest.advance(detail::bitsof<T>);
    detail::store_bits_linear<bits_type>(dest, width_width, remainder_width);
    dest.advance(width_width);
    if (remainder_width > 0) { // store_bits_linear does not allow n_bits == 0
        for (size_t i = 1; i < detail::ipow(hypercube_side_length, Dims); ++i) {
            detail::store_bits_linear<bits_type>(dest, remainder_width, bits[i] ^ ref);
            dest.advance(remainder_width);
        }
    }
    return ceil_byte_offset(stream, dest);
}


template<typename T, unsigned Dims>
size_t fast_profile<T, Dims>::decode_block(const void *stream, bits_type *bits) const {
    bits_type ref;
    memcpy(&ref, stream, sizeof ref);
    auto src = detail::const_bit_ptr<sizeof(bits_type)>::from_unaligned_pointer(stream);
    src.advance(detail::bitsof<T>);
    auto width_width = sizeof(T) == 1 ? 4 : sizeof(T) == 2 ? 5 : sizeof(T) == 4 ? 6 : 7;
    auto remainder_width = detail::load_bits<bits_type>(src, width_width);
    src.advance(width_width);
    if (remainder_width > 0) { // load_bits does not allow n_bits == 0
        bits[0] = ref;
        for (size_t i = 1; i < detail::ipow(hypercube_side_length, Dims); ++i) {
            bits[i] = detail::load_bits<bits_type>(src, remainder_width) ^ ref;
            src.advance(remainder_width);
        }
    } else {
        for (size_t i = 0; i < detail::ipow(hypercube_side_length, Dims); ++i) {
            bits[i] = ref;
        }
    }
    return ceil_byte_offset(stream, src);
}

} // namespace hcde

