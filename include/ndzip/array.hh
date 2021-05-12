#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <type_traits>


#ifdef __CUDA__
#define NDZIP_UNIVERSAL __host__ __device__
#else
#define NDZIP_UNIVERSAL
#endif


namespace ndzip {

using index_type = uint32_t;
using stream_size_type = size_t;

template<unsigned Dims>
class extent {
  public:
    using const_iterator = const index_type *;
    using iterator = index_type *;

    NDZIP_UNIVERSAL constexpr extent() noexcept = default;

    template<typename... Init,
            std::enable_if_t<((sizeof...(Init) == Dims) && ...
                                     && std::is_convertible_v<Init, index_type>),
                    int> = 0>
    NDZIP_UNIVERSAL constexpr extent(const Init &...components) noexcept
        : _components{static_cast<index_type>(components)...} {}

    NDZIP_UNIVERSAL static extent broadcast(index_type scalar) {
        extent e;
        for (unsigned d = 0; d < Dims; ++d) {
            e[d] = scalar;
        }
        return e;
    }

    NDZIP_UNIVERSAL index_type &operator[](unsigned d) { return _components[d]; }

    NDZIP_UNIVERSAL index_type operator[](unsigned d) const { return _components[d]; }

    NDZIP_UNIVERSAL extent &operator+=(const extent &other) {
        for (unsigned d = 0; d < Dims; ++d) {
            _components[d] += other._components[d];
        }
        return *this;
    }

    NDZIP_UNIVERSAL friend extent operator+(const extent &left, const extent &right) {
        auto result = left;
        result += right;
        return result;
    }

    NDZIP_UNIVERSAL extent &operator-=(const extent &other) {
        for (unsigned d = 0; d < Dims; ++d) {
            _components[d] -= other._components[d];
        }
        return *this;
    }

    NDZIP_UNIVERSAL friend extent operator-(const extent &left, const extent &right) {
        auto result = left;
        result -= right;
        return result;
    }

    NDZIP_UNIVERSAL extent &operator*=(index_type other) {
        for (unsigned d = 0; d < Dims; ++d) {
            _components[d] *= other;
        }
        return *this;
    }

    NDZIP_UNIVERSAL friend extent operator*(const extent &left, index_type right) {
        auto result = left;
        result *= right;
        return result;
    }

    NDZIP_UNIVERSAL friend extent operator*(index_type left, const extent &right) {
        auto result = right;
        result *= left;
        return result;
    }

    NDZIP_UNIVERSAL extent &operator/=(index_type other) {
        for (unsigned d = 0; d < Dims; ++d) {
            _components[d] /= other;
        }
        return *this;
    }

    NDZIP_UNIVERSAL friend extent operator/(const extent &left, index_type right) {
        auto result = left;
        result /= right;
        return result;
    }

    NDZIP_UNIVERSAL friend bool operator==(const extent &left, const extent &right) {
        bool eq = true;
        for (unsigned d = 0; d < Dims; ++d) {
            eq &= left[d] == right[d];
        }
        return eq;
    }

    NDZIP_UNIVERSAL friend bool operator!=(const extent &left, const extent &right) {
        return !operator==(left, right);
    }

    NDZIP_UNIVERSAL iterator begin() { return _components; }

    NDZIP_UNIVERSAL iterator end() { return _components + Dims; }

    NDZIP_UNIVERSAL const_iterator begin() const { return _components; }

    NDZIP_UNIVERSAL const_iterator end() const { return _components + Dims; }

  private:
    index_type _components[Dims] = {};
};

template<typename... Init>
extent(const Init &...) -> extent<sizeof...(Init)>;

template<unsigned Dims>
NDZIP_UNIVERSAL index_type num_elements(extent<Dims> size) {
    index_type n = 1;
    for (unsigned d = 0; d < Dims; ++d) {
        n *= size[d];
    }
    return n;
}

template<unsigned Dims>
NDZIP_UNIVERSAL index_type linear_offset(extent<Dims> position, extent<Dims> space) {
    index_type offset = 0;
    index_type stride = 1;
    for (unsigned nd = 0; nd < Dims; ++nd) {
        auto d = Dims - 1 - nd;
        offset += stride * position[d];
        stride *= space[d];
    }
    return offset;
}

}  // namespace ndzip

namespace ndzip::detail {

template<unsigned Dims>
NDZIP_UNIVERSAL index_type linear_index(
        const ndzip::extent<Dims> &size, const ndzip::extent<Dims> &pos) {
    index_type l = pos[0];
    for (unsigned d = 1; d < Dims; ++d) {
        l = l * size[d] + pos[d];
    }
    return l;
}

}  // namespace ndzip::detail

namespace ndzip {

template<typename T, unsigned Dims>
class slice {
  public:
    NDZIP_UNIVERSAL explicit slice(T *data, extent<Dims> size) : _data(data), _size(size) {}

    template<typename U,
            std::enable_if_t<std::is_const_v<T> && std::is_same_v<std::remove_const_t<T>, U>,
                    int> = 0>
    NDZIP_UNIVERSAL slice(slice<U, Dims> other) : _data(other._data), _size(other._size) {}

    NDZIP_UNIVERSAL const extent<Dims> &size() const { return _size; }

    NDZIP_UNIVERSAL T *data() const { return _data; }

    NDZIP_UNIVERSAL index_type linear_index(const ndzip::extent<Dims> &pos) const {
        return detail::linear_index(_size, pos);
    }

    NDZIP_UNIVERSAL T &operator[](const ndzip::extent<Dims> &pos) const {
        return _data[linear_index(pos)];
    }

  private:
    T *_data;
    extent<Dims> _size;

    friend class slice<const T, Dims>;
};

template<typename T, unsigned Dims>
stream_size_type compressed_size_bound(const extent<Dims> &e);

using kernel_duration = std::chrono::duration<uint64_t, std::nano>;

}  // namespace ndzip
