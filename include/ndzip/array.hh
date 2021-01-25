#pragma once

#include <cstdlib>
#include <type_traits>


namespace ndzip {

template<unsigned Dims>
class extent {
  public:
    using const_iterator = const size_t *;
    using iterator = size_t *;

    constexpr extent() noexcept = default;

    template<typename... Init,
            std::enable_if_t<((sizeof...(Init) == Dims) && ...
                                     && std::is_convertible_v<Init, size_t>),
                    int> = 0>
    constexpr extent(const Init &...components) noexcept
        : _components{static_cast<size_t>(components)...} {}

    static extent broadcast(size_t scalar) {
        extent e;
        for (unsigned d = 0; d < Dims; ++d) {
            e[d] = scalar;
        }
        return e;
    }

    size_t &operator[](unsigned d) { return _components[d]; }

    size_t operator[](unsigned d) const { return _components[d]; }

    extent &operator+=(const extent &other) {
        for (unsigned d = 0; d < Dims; ++d) {
            _components[d] += other._components[d];
        }
        return *this;
    }

    friend extent operator+(const extent &left, const extent &right) {
        auto result = left;
        result += right;
        return result;
    }

    extent &operator-=(const extent &other) {
        for (unsigned d = 0; d < Dims; ++d) {
            _components[d] -= other._components[d];
        }
        return *this;
    }

    friend extent operator-(const extent &left, const extent &right) {
        auto result = left;
        result -= right;
        return result;
    }

    extent &operator*=(size_t other) {
        for (unsigned d = 0; d < Dims; ++d) {
            _components[d] *= other;
        }
        return *this;
    }

    friend extent operator*(const extent &left, size_t right) {
        auto result = left;
        result *= right;
        return result;
    }

    friend extent operator*(size_t left, const extent &right) {
        auto result = right;
        result *= left;
        return result;
    }

    extent &operator/=(size_t other) {
        for (unsigned d = 0; d < Dims; ++d) {
            _components[d] /= other;
        }
        return *this;
    }

    friend extent operator/(const extent &left, size_t right) {
        auto result = left;
        result /= right;
        return result;
    }

    friend bool operator==(const extent &left, const extent &right) {
        bool eq = true;
        for (unsigned d = 0; d < Dims; ++d) {
            eq &= left[d] == right[d];
        }
        return eq;
    }

    friend bool operator!=(const extent &left, const extent &right) {
        return !operator==(left, right);
    }

    iterator begin() { return _components; }

    iterator end() { return _components + Dims; }

    const_iterator begin() const { return _components; }

    const_iterator end() const { return _components + Dims; }

  private:
    size_t _components[Dims] = {};
};

template<typename... Init>
extent(const Init &...) -> extent<sizeof...(Init)>;

template<unsigned Dims>
size_t num_elements(extent<Dims> size) {
    size_t n = 1;
    for (unsigned d = 0; d < Dims; ++d) {
        n *= size[d];
    }
    return n;
}

template<unsigned Dims>
size_t linear_offset(extent<Dims> position, extent<Dims> space) {
    size_t offset = 0;
    size_t stride = 1;
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
size_t linear_index(const ndzip::extent<Dims> &size, const ndzip::extent<Dims> &pos) {
    size_t l = pos[0];
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
    explicit slice(T *data, extent<Dims> size) : _data(data), _size(size) {}

    template<typename U,
            std::enable_if_t<std::is_const_v<T> && std::is_same_v<std::remove_const_t<T>, U>,
                    int> = 0>
    slice(slice<U, Dims> other) : _data(other._data), _size(other._size) {}

    const extent<Dims> &size() const { return _size; }

    T *data() const { return _data; }

    size_t linear_index(const ndzip::extent<Dims> &pos) const {
        return detail::linear_index(_size, pos);
    }

    T &operator[](const ndzip::extent<Dims> &pos) const { return _data[linear_index(pos)]; }

  private:
    T *_data;
    extent<Dims> _size;

    friend class slice<const T, Dims>;
};

template<typename T, unsigned Dims>
size_t compressed_size_bound(const extent<Dims> &e);

}  // namespace ndzip
