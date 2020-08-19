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

inline unsigned count_significant_bytes(uint32_t x) {
    unsigned lz = x == 0 ? 32 : __builtin_clz(x);
    return 4 - lz/8;
}
inline unsigned count_significant_bytes(uint64_t x) {
    unsigned lz = x == 0 ? 64 : __builtin_clzl(x);
    return 8 - lz/8;
}

template<typename Profile>
size_t encode_coefficients(const typename Profile::bits_type *coeffs, void *stream) {
    using bits_type = typename Profile::bits_type;

    constexpr auto n_values = ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr auto header_density = sizeof(bits_type) <= 4 ? 4 : 2;

    auto *byte_stream = static_cast<std::byte*>(stream);
    static_assert(n_values % header_density == 0);

    auto header_pos = size_t{0};
    auto body_pos = n_values / header_density;
    for (unsigned i = 0; i < n_values; i += header_density) {
        unsigned lengths[header_density];
        unsigned length_code = 0;
        for (unsigned j = 0; j < header_density; ++j) {
            auto residual_bytes = count_significant_bytes(coeffs[i + j]);
            auto header_encoded_length = residual_bytes;
            if (sizeof(bits_type) == 4) {
                if (residual_bytes == 1) {
                    residual_bytes = 2;
                }
                if (header_encoded_length > 1) {
                    header_encoded_length -= 1;
                }
            }
            lengths[j] = residual_bytes;
            length_code = (length_code << (8u/header_density)) | header_encoded_length;
        }
        store_unaligned(byte_stream + header_pos, static_cast<std::uint8_t>(length_code));
        header_pos += 1;

        for (unsigned j = 0; j < header_density; ++j) {
            alignas(bits_type) std::byte coeff_bytes[2 * sizeof(bits_type)] = {};
            store_aligned<sizeof(bits_type)>(coeff_bytes, endian_transform(coeffs[i + j]));
            store_unaligned(byte_stream + body_pos,
                load_unaligned<bits_type>(coeff_bytes + sizeof(bits_type) - lengths[j]));
            body_pos += lengths[j];
        }
    }

    return body_pos;
}

template<typename Profile>
size_t decode_coefficients(const void *stream, typename Profile::bits_type *coeffs) {
    using bits_type = typename Profile::bits_type;

    constexpr auto n_values = ipow(Profile::hypercube_side_length, Profile::dimensions);
    constexpr auto header_density = sizeof(bits_type) <= 4 ? 4 : 2;
    constexpr auto header_length_bits = 8u/header_density;

    auto *byte_stream = static_cast<const std::byte*>(stream);
    static_assert(n_values % header_density == 0);

    auto header_pos = size_t{0};
    auto body_pos = n_values / header_density;
    for (unsigned i = 0; i < n_values; i += header_density) {
        unsigned length_code = load_unaligned<std::uint8_t>(byte_stream + header_pos);
        header_pos += 1;
        for (unsigned j = 0; j < header_density; ++j) {
            auto header_encoded_length = (length_code >> (8u - (header_length_bits * (1 + j))))
                & ~(~0u << header_length_bits);
            auto residual_bytes = header_encoded_length;
            if (sizeof(bits_type) == 4 && header_encoded_length > 0) {
                residual_bytes += 1;
            }
            alignas(bits_type) std::byte coeff_bytes[2 * sizeof(bits_type)] = {};
            store_unaligned(coeff_bytes + sizeof(bits_type) - residual_bytes,
                load_aligned<bits_type>(byte_stream + body_pos));
            coeffs[i + j] = endian_transform(load_aligned<bits_type>(coeff_bytes));
            body_pos += residual_bytes;
        }
    }

    return body_pos;
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

