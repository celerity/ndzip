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
size_t xt_profile<T, Dims>::encode_block(const bits_type *bits, void *stream) const {
    assert((std::is_same_v<T, float>)); // TODO substitute generic constants below

    memcpy(stream, &bits[0], sizeof(bits_type));
    auto dest = detail::bit_ptr<sizeof(bits_type)>::from_unaligned_pointer(stream);
    dest.advance(detail::bitsof<T>);

    bits_type difference[detail::ipow(hypercube_side_length, Dims)]; // TODO in-place
    memcpy(difference, bits, sizeof difference);
    detail::xt(difference, dimensions, hypercube_side_length);

    dest = detail::encode_difference_bits<xt_profile>(difference, dest);
    return ceil_byte_offset(stream, dest);
}


template<typename T, unsigned Dims>
[[gnu::noinline]]
size_t xt_profile<T, Dims>::decode_block(const void *stream, bits_type *bits) const {
    assert((std::is_same_v<T, float>)); // TODO substitute generic constants below

    memcpy(&bits[0], stream, sizeof(bits_type));
    auto src = detail::const_bit_ptr<sizeof(bits_type)>::from_unaligned_pointer(stream);
    src.advance(detail::bitsof<T>);

    src = detail::decode_difference_bits<xt_profile>(src, bits);

    detail::ixt(bits, dimensions, hypercube_side_length);

    return ceil_byte_offset(stream, src);
}

} // namespace hcde

