#pragma once

#include "common.hh"

namespace hcde::detail {

template<typename T, typename Enable=void>
struct positional_bits_repr;

template<typename T>
struct positional_bits_repr<T, std::enable_if_t<std::is_integral_v<T> && std::is_unsigned_v<T>>> {
    using numeric_type = T;
    using bits_type = T;

    static bits_type to_bits(numeric_type x) {
        return x;
    }

    static numeric_type from_bits(bits_type x) {
        return x;
    }
};

template<typename T>
struct positional_bits_repr<T, std::enable_if_t<std::is_integral_v<T> && std::is_signed_v<T>>> {
    using numeric_type = T;
    using bits_type = std::make_unsigned_t<T>;

    static bits_type to_bits(numeric_type x) {
        return static_cast<bits_type>(x)
                - static_cast<bits_type>(std::numeric_limits<numeric_type>::min());
    }

    static numeric_type from_bits(bits_type x) {
        return static_cast<numeric_type>(
                x + static_cast<bits_type>(std::numeric_limits<numeric_type>::min()));
    }
};


template<typename T>
struct float_bits_traits;

template<>
struct float_bits_traits<float> {
    static_assert(sizeof(float) == 4);
    static_assert(std::numeric_limits<float>::is_iec559); // IEE 754

    using bits_type = uint32_t;

    constexpr static unsigned sign_bits = 1;
    constexpr static unsigned exponent_bits = 8;
    constexpr static unsigned mantissa_bits = 23;
};

template<>
struct float_bits_traits<double> {
    static_assert(sizeof(double) == 8);
    static_assert(std::numeric_limits<double>::is_iec559); // IEE 754

    using bits_type = uint64_t;

    constexpr static unsigned sign_bits = 1;
    constexpr static unsigned exponent_bits = 11;
    constexpr static unsigned mantissa_bits = 52;
};


template<typename T>
struct positional_bits_repr<T, std::enable_if_t<std::is_floating_point_v<T>>> {
    constexpr static unsigned sign_bits = float_bits_traits<T>::sign_bits;
    constexpr static unsigned exponent_bits = float_bits_traits<T>::exponent_bits;
    constexpr static unsigned mantissa_bits = float_bits_traits<T>::mantissa_bits;

    using numeric_type = T;
    using bits_type = typename float_bits_traits<T>::bits_type;

    static bits_type to_bits(numeric_type x) {
        bits_type bits;
        memcpy(&bits, &x, sizeof bits);
        auto exponent = (bits << 1u) & ~(~bits_type{0} >> exponent_bits);
        auto sign = (bits & ~(~bits_type{0} >> sign_bits)) >> exponent_bits;
        auto mantissa = bits & ~(~bits_type{0} << mantissa_bits);
        return exponent | sign | mantissa;
    }

    static numeric_type from_bits(bits_type bits) {
        auto sign = (bits << exponent_bits) & ~(~bits_type{0} >> sign_bits);
        auto exponent = (bits & ~(~bits_type{0} >> exponent_bits)) >> sign_bits;
        auto mantissa = bits & ~(~bits_type{0} << mantissa_bits);
        bits = sign | exponent | mantissa;
        numeric_type x;
        memcpy(&x, &bits, sizeof bits);
        return x;
    }
};


template<typename Bits, unsigned Dims, unsigned SideLength>
[[gnu::noinline]]
void xor_neighborhood_encode(const Bits *__restrict in, Bits * __restrict out) {
    unsigned dim3_stride = 1;
    unsigned dim2_stride = Dims > 3 ? SideLength : 1;
    unsigned dim1_stride = Dims > 2 ? dim2_stride * SideLength : 1;
    unsigned dim0_stride = Dims > 1 ? dim1_stride * SideLength : 1;
    unsigned strides[] = {dim0_stride, dim1_stride, dim2_stride, dim3_stride};

    if constexpr (Dims >= 1) {
        for (unsigned d = 0; d < Dims; ++d) {
            size_t offset = 0;
            auto left = in[offset];
            for (unsigned i = 1; i < SideLength; ++i) {
                offset += strides[d];
                auto right = in[offset];
                out[offset] = left ^ right;
                left = right;
            }
        }
    }
    if constexpr (Dims >= 2) {
        for (unsigned d0 = 0; d0 < Dims; ++d0) {
            for (unsigned d1 = d0 + 1; d1 < Dims; ++d1) {
                size_t offset0 = 0;
                for (unsigned i = 1; i < SideLength; ++i) {
                    offset0 += strides[d0];
                    auto offset1 = offset0;
                    auto l00 = offset1 -  strides[d0];
                    auto l10 = offset1;
                    auto left = in[l00] ^ in[l10];
                    for (unsigned j = 1; j < SideLength; ++j) {
                        offset1 += strides[d1];
                        auto r01 = offset1 - strides[d0];
                        auto r11 = offset1;
                        auto right = in[r01] ^ in[r11];
                        out[offset1] = left ^ right;
                        left = right;
                    }
                }
            }
        }
    }
    if constexpr (Dims >= 3) {
        for (unsigned d0 = 0; d0 < Dims; ++d0) {
            for (unsigned d1 = d0 + 1; d1 < Dims; ++d1) {
                for (unsigned d2 = d1 + 1; d2 < Dims; ++d2) {
                    size_t offset0 = 0;
                    for (unsigned i = 1; i < SideLength; ++i) {
                        offset0 += strides[d0];
                        auto offset1 = offset0;
                        for (unsigned j = 1; j < SideLength; ++j) {
                            offset1 += strides[d1];
                            auto offset2 = offset1;
                            auto l000 = offset2 - strides[d1] - strides[d0];
                            auto l010 = offset2 - strides[d0];
                            auto l100 = offset2 - strides[d1];
                            auto l110 = offset2;
                            auto left = in[l000] ^ in[l010] ^ in[l100] ^ in[l110];
                            for (unsigned k = 1; k < SideLength; ++k) {
                                offset2 += strides[d2];
                                auto r001 = offset2 - strides[d1] - strides[d0];
                                auto r011 = offset2 - strides[d0];
                                auto r101 = offset2 - strides[d1];
                                auto r111 = offset2;
                                auto right = in[r001] ^ in[r011] ^ in[r101] ^ in[r111];
                                out[offset2] = left ^ right;
                                left = right;
                            }
                        }
                    }
                }
            }
        }
    }
}


template<typename Bits, unsigned Dims, unsigned SideLength>
[[gnu::noinline]]
void xor_neighborhood_decode(const Bits *__restrict in, Bits * __restrict out) {
    unsigned dim3_stride = 1;
    unsigned dim2_stride = Dims > 3 ? SideLength : 1;
    unsigned dim1_stride = Dims > 2 ? dim2_stride * SideLength : 1;
    unsigned dim0_stride = Dims > 1 ? dim1_stride * SideLength : 1;
    unsigned strides[] = {dim0_stride, dim1_stride, dim2_stride, dim3_stride};

    if constexpr (Dims >= 1) {
        for (unsigned d = 0; d < Dims; ++d) {
            size_t offset = 0;
            auto left = out[offset];
            for (unsigned i = 1; i < SideLength; ++i) {
                offset += strides[d];
                auto result = left ^ in[offset];
                out[offset] = result;
                left = result;
            }
        }
    }
    if constexpr (Dims >= 2) {
        for (unsigned d0 = 0; d0 < Dims; ++d0) {
            for (unsigned d1 = d0 + 1; d1 < Dims; ++d1) {
                size_t offset0 = 0;
                for (unsigned i = 1; i < SideLength; ++i) {
                    offset0 += strides[d0];
                    auto offset1 = offset0;
                    auto l00 = offset1 -  strides[d0];
                    auto l10 = offset1;
                    auto left = out[l00] ^ out[l10];
                    for (unsigned j = 1; j < SideLength; ++j) {
                        offset1 += strides[d1];
                        auto r01 = offset1 - strides[d0];
                        auto imm = out[r01];
                        auto right = left ^ in[offset1];
                        out[offset1] = right ^ imm ;
                        left = right;
                    }
                }
            }
        }
    }
    if constexpr (Dims >= 3) {
        for (unsigned d0 = 0; d0 < Dims; ++d0) {
            for (unsigned d1 = d0 + 1; d1 < Dims; ++d1) {
                for (unsigned d2 = d1 + 1; d2 < Dims; ++d2) {
                    size_t offset0 = 0;
                    for (unsigned i = 1; i < SideLength; ++i) {
                        offset0 += strides[d0];
                        auto offset1 = offset0;
                        for (unsigned j = 1; j < SideLength; ++j) {
                            offset1 += strides[d1];
                            auto offset2 = offset1;
                            auto l000 = offset2 - strides[d1] - strides[d0];
                            auto l010 = offset2 - strides[d0];
                            auto l100 = offset2 - strides[d1];
                            auto l110 = offset2;
                            auto left = out[l000] ^ out[l010] ^ out[l100] ^ out[l110];
                            for (unsigned k = 1; k < SideLength; ++k) {
                                offset2 += strides[d2];
                                auto r001 = offset2 - strides[d1] - strides[d0];
                                auto r011 = offset2 - strides[d0];
                                auto r101 = offset2 - strides[d1];
                                auto imm = out[r001] ^ out[r011] ^ out[r101];
                                auto right = left ^ in[offset2];
                                out[offset2] = right ^ imm;
                                left = right;
                            }
                        }
                    }
                }
            }
        }
    }
}

template<typename Profile>
[[gnu::noinline]]
bit_ptr<sizeof(typename Profile::bits_type)> encode_difference_bits(
    const typename Profile::bits_type *difference,
    bit_ptr<sizeof(typename Profile::bits_type)> dest) {
    using bits_type = typename Profile::bits_type;
    constexpr unsigned width_width = 4;
    auto remainder_dest = dest;
    remainder_dest.advance(width_width * (detail::ipow(Profile::hypercube_side_length, Profile::dimensions) - 1));
    for (unsigned i = 1; i < detail::ipow(Profile::hypercube_side_length, Profile::dimensions); ++i) {
        auto width = detail::significant_bits(difference[i]);
        if (width == 0) {
            dest.advance(width_width);
        } else {
            auto width_code = width < 19 ? bits_type{1} : width - 17;
            auto verbatim_bits = width < 19 ? 18 : width - 1;
            detail::store_bits_linear<bits_type>(dest, width_width, width_code);
            dest.advance(width_width);
            detail::store_bits_linear<bits_type>(remainder_dest, verbatim_bits,
                difference[i] & ~(~bits_type{} << (verbatim_bits)));
            remainder_dest.advance(verbatim_bits);
        }
    }
    return remainder_dest;
}

template<typename Profile>
[[gnu::noinline]]
const_bit_ptr<sizeof(typename Profile::bits_type)>
decode_difference_bits(const_bit_ptr<sizeof(typename Profile::bits_type)> src,
    typename Profile::bits_type *difference) {
    using bits_type = typename Profile::bits_type;
    constexpr unsigned width_width = 4;
    auto remainder_src = src;
    remainder_src.advance(width_width * (detail::ipow(Profile::hypercube_side_length, Profile::dimensions) - 1));
    for (unsigned i = 1; i < detail::ipow(Profile::hypercube_side_length, Profile::dimensions); ++i) {
        auto width_code = detail::load_bits<bits_type>(src, width_width);
        src.advance(width_width);
        if (width_code == 0) {
            difference[i] = 0;
        } else if (width_code == 1) {
            difference[i] = detail::load_bits<bits_type>(remainder_src, 18);
            remainder_src.advance(18);
        } else {
            auto width = width_code + 17;
            difference[i] = (1 << (width - 1)) | detail::load_bits<bits_type>(remainder_src, width - 1);
            remainder_src.advance(width - 1);
        }
    }
    return remainder_src;
}

} // namespace hcde::detail

namespace hcde {

template<typename T, unsigned Dims>
auto strong_profile<T, Dims>::load_value(const data_type *data) const -> bits_type {
    return detail::positional_bits_repr<T>::to_bits(*data);
}


template<typename T, unsigned Dims>
void strong_profile<T, Dims>::store_value(data_type *data, bits_type bits) const {
    *data = detail::positional_bits_repr<T>::from_bits(bits);
}


template<typename T, unsigned Dims>
[[gnu::noinline]]
size_t strong_profile<T, Dims>::encode_block(const bits_type *bits, void *stream) const {
    assert((std::is_same_v<T, float>)); // TODO substitute generic constants below

    memcpy(stream, &bits[0], sizeof(bits_type));
    auto dest = detail::bit_ptr<sizeof(bits_type)>::from_unaligned_pointer(stream);
    dest.advance(detail::bitsof<T>);

    bits_type difference[detail::ipow(hypercube_side_length, Dims)];
    detail::xor_neighborhood_encode<bits_type, Dims, hypercube_side_length>(bits, difference);

    dest = detail::encode_difference_bits<strong_profile>(difference, dest);
    return ceil_byte_offset(stream, dest);
}


template<typename T, unsigned Dims>
[[gnu::noinline]]
size_t strong_profile<T, Dims>::decode_block(const void *stream, bits_type *bits) const {
    assert((std::is_same_v<T, float>)); // TODO substitute generic constants below

    memcpy(&bits[0], stream, sizeof(bits_type));
    auto src = detail::const_bit_ptr<sizeof(bits_type)>::from_unaligned_pointer(stream);
    src.advance(detail::bitsof<T>);

    bits_type difference[detail::ipow(hypercube_side_length, Dims)];
    src = detail::decode_difference_bits<strong_profile>(src, difference);

    detail::xor_neighborhood_decode<bits_type, Dims, hypercube_side_length>(difference, bits);
    return ceil_byte_offset(stream, src);
}

} // namespace hcde

