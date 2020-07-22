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


template<typename Bits, unsigned Dims>
void xor_neighborhood(unsigned side_length, const Bits *neighborhood, const Bits *in, Bits *out) {
    unsigned dim3_stride = 1;
    unsigned dim2_stride = Dims > 3 ? side_length : 1;
    unsigned dim1_stride = Dims > 2 ? dim2_stride * side_length : 1;
    unsigned dim0_stride = Dims > 1 ? dim1_stride * side_length : 1;
    unsigned strides[] = {dim0_stride, dim1_stride, dim2_stride, dim3_stride    };

    if constexpr (Dims >= 1) {
        for (unsigned d = 0; d < Dims; ++d) {
            size_t offset = 0;
            for (unsigned i = 1; i < side_length; ++i) {
                size_t prev_offset = offset;
                offset += strides[d];
                out[offset] = in[offset] ^ neighborhood[prev_offset];
            }
        }
    }
    if constexpr (Dims >= 2) {
        for (unsigned d0 = 0; d0 < Dims; ++d0) {
            for (unsigned d1 = d0 + 1; d1 < Dims; ++d1) {
                size_t offset0 = 0;
                for (unsigned i = 1; i < side_length; ++i) {
                    offset0 += strides[d0];
                    auto offset1 = offset0;
                    for (unsigned j = 1; j < side_length; ++j) {
                        offset1 += strides[d1];
                        auto n00 = offset1 - strides[d1] - strides[d0];
                        auto n01 = offset1 - strides[d0];
                        auto n10 = offset1 - strides[d1];
                        out[offset1] = in[offset1] ^ neighborhood[n00] ^ neighborhood[n01]
                            ^ neighborhood[n10];
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
                    for (unsigned i = 1; i < side_length; ++i) {
                        offset0 += strides[d0];
                        auto offset1 = offset0;
                        for (unsigned j = 1; j < side_length; ++j) {
                            offset1 += strides[d1];
                            auto offset2 = offset1;
                            for (unsigned k = 1; k < side_length; ++k) {
                                offset2 += strides[d2];
                                auto n000 = offset1 - strides[d2] - strides[d1] - strides[d0];
                                auto n001 = offset1 - strides[d1] - strides[d0];
                                auto n010 = offset1 - strides[d2] - strides[d0];
                                auto n100 = offset1 - strides[d2] - strides[d1];
                                auto n011 = offset1 - strides[d0];
                                auto n101 = offset1 - strides[d1];
                                auto n110 = offset1 - strides[d2];
                                out[offset2] = in[offset2] ^ neighborhood[n000]
                                    ^ neighborhood[n001] ^ neighborhood[n010] ^ neighborhood[n100]
                                    ^ neighborhood[n011] ^ neighborhood[n101] ^ neighborhood[n110];
                            }
                        }
                    }
                }
            }
        }
    }
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
size_t strong_profile<T, Dims>::encode_block(const bits_type *bits, void *stream) const {
    assert((std::is_same_v<T, float>)); // TODO substitute generic constants below

    memcpy(stream, &bits[0], sizeof(bits_type));
    auto dest = detail::bit_ptr<sizeof(bits_type)>::from_unaligned_pointer(stream);
    dest.advance(detail::bitsof<T>);

    bits_type difference[detail::ipow(hypercube_side_length, Dims)];
    detail::xor_neighborhood<bits_type, Dims>(hypercube_side_length, bits, bits, difference);

    constexpr unsigned width_width = 4;
    for (unsigned i = 1; i < detail::ipow(hypercube_side_length, Dims); ++i) {
        auto width = detail::significant_bits(difference[i]);
        if (width == 0) {
            dest.advance(width_width);
        } else if (width < 19) {
            detail::store_bits_linear<bits_type>(dest, width_width, 1);
            dest.advance(width_width);
            detail::store_bits_linear<bits_type>(dest, 18, difference[i]);
            dest.advance(18);
        } else {
            detail::store_bits_linear<bits_type>(dest, width_width, width - 17);
            dest.advance(width_width);
            detail::store_bits_linear<bits_type>(dest, width - 1,
                    difference[i] & ~(~bits_type{} << (width - 1)));
            dest.advance(width - 1);
        }
    }

    return ceil_byte_offset(stream, dest);
}


template<typename T, unsigned Dims>
size_t strong_profile<T, Dims>::decode_block(const void *stream, bits_type *bits) const {
    assert((std::is_same_v<T, float>)); // TODO substitute generic constants below

    memcpy(&bits[0], stream, sizeof(bits_type));
    auto src = detail::const_bit_ptr<sizeof(bits_type)>::from_unaligned_pointer(stream);
    src.advance(detail::bitsof<T>);

    bits_type difference[detail::ipow(hypercube_side_length, Dims)];

    constexpr unsigned width_width = 4;
    for (unsigned i = 1; i < detail::ipow(hypercube_side_length, Dims); ++i) {
        auto width_code = detail::load_bits<bits_type>(src, width_width);
        src.advance(width_width);
        if (width_code == 0) {
            difference[i] = 0;
        } else if (width_code == 1) {
            difference[i] = detail::load_bits<bits_type>(src, 18);
            src.advance(18);
        } else {
            auto width = width_code + 17;
            difference[i] = (1 << (width - 1)) | detail::load_bits<bits_type>(src, width - 1);
            src.advance(width - 1);
        }
    }

    detail::xor_neighborhood<bits_type, Dims>(hypercube_side_length, bits, difference, bits);

    return ceil_byte_offset(stream, src);
}

} // namespace hcde

