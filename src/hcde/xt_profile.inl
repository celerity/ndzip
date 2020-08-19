#pragma once

#include "common.hh"
#include "strong_profile.inl"

namespace hcde::detail {

template<typename T>
void xt_step(T *x, size_t n, size_t s) {
    T a, b;
    b = x[0*s];
    for (size_t i = 1; i < n; ++i) {
        a = b;
        b = x[i*s];
        x[i*s] = a ^ b;
    }
}

template<typename T>
void ixt_step(T *x, size_t n, size_t s) {
    for (size_t i = 1; i < n; ++i) {
        x[i*s] ^= x[(i-1)*s];
    }
}

template<typename T>
void xt(T *x, unsigned dims, size_t n) {
    if (dims == 1) {
        xt_step(x, n, 1);
    } else if (dims == 2) {
        for (size_t i = 0; i < n*n; i += n) {
            xt_step(x + i, n, 1);
        }
        for (size_t i = 0; i < n; ++i) {
            xt_step(x + i, n, n);
        }
    } else if (dims == 3) {
        for (size_t i = 0; i < n*n*n; i += n*n) {
            for (size_t j = 0; j < n; ++j) {
                xt_step(x + i + j, n, n);
            }
        }
        for (size_t i = 0; i < n*n*n; i += n) {
            xt_step(x + i, n, 1);
        }
        for (size_t i = 0; i < n*n; ++i) {
            xt_step(x + i, n, n*n);
        }
    }
}

template<typename T>
void ixt(T *x, unsigned dims, size_t n) {
    if (dims == 1) {
        ixt_step(x, n, 1);
    } else if (dims == 2) {
        for (size_t i = 0; i < n; ++i) {
            ixt_step(x + i, n, n);
        }
        for (size_t i = 0; i < n*n; i += n) {
            ixt_step(x + i, n, 1);
        }
    } else if (dims == 3) {
        for (size_t i = 0; i < n*n; ++i) {
            ixt_step(x + i, n, n*n);
        }
        for (size_t i = 0; i < n*n*n; i += n) {
            ixt_step(x + i, n, 1);
        }
        for (size_t i = 0; i < n*n*n; i += n*n) {
            for (size_t j = 0; j < n; ++j) {
                ixt_step(x + i + j, n, n);
            }
        }
    }
}

template<typename Profile>
[[gnu::noinline]]
size_t encode_coefficients(const typename Profile::bits_type *coeffs, void *stream) {
    using bits_type = typename Profile::bits_type;

    auto dest = detail::bit_ptr<sizeof(bits_type)>::from_unaligned_pointer(stream);

    unsigned width_width;
    unsigned width_min;
    if (sizeof(bits_type) <= 4) {
        width_width = 4;
        width_min = 19;
    } else {
        width_width = 5;
        width_min = 35;
    }

    auto remainder_dest = dest;
    remainder_dest.advance(width_width * detail::ipow(Profile::hypercube_side_length, Profile::dimensions));
    for (unsigned i = 0; i < detail::ipow(Profile::hypercube_side_length, Profile::dimensions); ++i) {
        auto width = detail::significant_bits(coeffs[i]);
        if (width == 0) {
            dest.advance(width_width);
        } else {
            auto width_code = width < width_min ? bits_type{1} : width + 2 - width_min;
            auto verbatim_bits = width < width_min ? width_min - 1 : width - 1;
            detail::store_bits_linear<bits_type>(dest, width_width, width_code);
            dest.advance(width_width);
            detail::store_bits_linear<bits_type>(remainder_dest, verbatim_bits,
                coeffs[i] & ~(~bits_type{} << (verbatim_bits)));
            remainder_dest.advance(verbatim_bits);
        }
    }

    return ceil_byte_offset(stream, remainder_dest);
}

template<typename Profile>
[[gnu::noinline]]
size_t decode_coefficients(const void *stream, typename Profile::bits_type *coeffs) {
    using bits_type = typename Profile::bits_type;

    auto src = detail::const_bit_ptr<sizeof(bits_type)>::from_unaligned_pointer(stream);

    unsigned width_width;
    unsigned width_min;
    if (sizeof(bits_type) <= 4) {
        width_width = 4;
        width_min = 19;
    } else {
        width_width = 5;
        width_min = 35;
    }

    auto remainder_src = src;
    remainder_src.advance(width_width * detail::ipow(Profile::hypercube_side_length, Profile::dimensions));
    for (unsigned i = 0; i < detail::ipow(Profile::hypercube_side_length, Profile::dimensions); ++i) {
        auto width_code = detail::load_bits<bits_type>(src, width_width);
        src.advance(width_width);
        if (width_code == 0) {
            coeffs[i] = 0;
        } else if (width_code == 1) {
            coeffs[i] = detail::load_bits<bits_type>(remainder_src, width_min - 1);
            remainder_src.advance(width_min - 1);
        } else {
            auto width = width_code + width_min - 2;
            coeffs[i] = (1 << (width - 1)) | detail::load_bits<bits_type>(remainder_src, width - 1);
            remainder_src.advance(width - 1);
        }
    }

    return ceil_byte_offset(stream, remainder_src);
}

} // namespace hcde::detail

namespace hcde {

template<typename T, unsigned Dims>
auto xt_profile<T, Dims>::load_value(const data_type *data) const -> bits_type {
    return detail::positional_bits_repr<T>::to_bits(*data);
}


template<typename T, unsigned Dims>
void xt_profile<T, Dims>::store_value(data_type *data, bits_type bits) const {
    *data = detail::positional_bits_repr<T>::from_bits(bits);
}


template<typename T, unsigned Dims>
[[gnu::noinline]]
size_t xt_profile<T, Dims>::encode_block(bits_type *bits, void *stream) const {
    detail::xt(bits, dimensions, hypercube_side_length);
    return detail::encode_coefficients<xt_profile>(bits, stream);
}


template<typename T, unsigned Dims>
[[gnu::noinline]]
size_t xt_profile<T, Dims>::decode_block(const void *stream, bits_type *bits) const {
    auto end = detail::decode_coefficients<xt_profile>(stream, bits);
    detail::ixt(bits, dimensions, hypercube_side_length);
    return end;
}

} // namespace hcde

