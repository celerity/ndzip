#pragma once

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <memory>


namespace hcde::detail {

template<typename Integer>
constexpr inline Integer ipow(Integer base, unsigned exponent) {
    Integer power{1};
    while (exponent) {
        if (exponent & 1u) {
            power *= base;
        }
        base *= base;
        exponent >>= 1;
    }
    return power;
}

template<typename T>
using bits_type = std::conditional_t<sizeof(T) == 1, uint8_t,
      std::conditional_t<sizeof(T) == 2, uint16_t,
      std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;

template<typename T>
constexpr inline size_t bitsof = CHAR_BIT * sizeof(T);

} // namespace hcde::detail


namespace hcde {

template<unsigned Dims>
class extent {
    public:
        using const_iterator = const size_t *;
        using iterator = size_t *;

        constexpr extent() noexcept = default;

        template<typename ...Init, std::enable_if_t<((sizeof...(Init) == Dims)
            && ... && std::is_convertible_v<Init, size_t>), int> = 0>
        constexpr extent(const Init &...components) noexcept
            : _components{static_cast<size_t>(components)...}
        {
        }

        static extent broadcast(size_t scalar) {
            extent e;
            for (unsigned d = 0; d < Dims; ++d) {
                e[d] = scalar;
            }
            return e;
        }

        size_t &operator[](unsigned d) {
            return _components[d];
        }

        size_t operator[](unsigned d) const {
            return _components[d];
        }

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

        iterator begin() {
            return _components;
        }

        iterator end() {
            return _components + Dims;
        }

        const_iterator begin() const {
            return _components;
        }

        const_iterator end() const {
            return _components + Dims;
        }

    private:
        size_t _components[Dims] = {};
};

template<typename ...Init>
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
size_t linear_offset(extent<Dims> space, extent<Dims> position) {
    size_t offset = 0;
    size_t stride = 1;
    for (unsigned nd = 0; nd < Dims; ++nd) {
        auto d = Dims - 1 - nd;
        offset += stride * position[d];
        stride *= space[d];
    }
    return offset;
}

} // namespace hcde

namespace hcde::detail {

    template<unsigned Dims>
    size_t linear_index(const hcde::extent<Dims> &size, const hcde::extent<Dims> &pos) {
        size_t l = pos[0];
        for (unsigned d = 1; d < Dims; ++d) {
            l = l * size[d] + pos[d];
        }
        return l;
    }

} // namespace hcde::detail

namespace hcde {

template<typename T, unsigned Dims>
class slice {
    public:
        explicit slice(T *data, extent<Dims> size)
            : _data(data)
            , _size(size)
        {
        }

        template<typename U, std::enable_if_t<std::is_const_v<T>
            && std::is_same_v<std::remove_const_t<T>, U>, int> = 0>
        slice(slice<U, Dims> other)
            : _data(other._data)
            , _size(other._size)
        {
        }

        const extent<Dims> &size() const {
            return _size;
        }

        T *data() const {
            return _data;
        }

        size_t linear_index(const hcde::extent<Dims> &pos) const {
            return detail::linear_index(_size, pos);
        }

        T &operator[](const hcde::extent<Dims> &pos) const {
            return _data[linear_index(pos)];
        }

    private:
        T *_data;
        extent<Dims> _size;

        friend class slice<const T, Dims>;
};

template<typename T, unsigned Dims>
size_t compressed_size_bound(const extent<Dims> &e);

template<typename T, unsigned Dims>
class cpu_encoder {
public:
    using data_type = T;
    constexpr static unsigned dimensions = Dims;

    size_t compress(const slice<const data_type, dimensions> &data, void *stream) const;

    size_t decompress(const void *stream, size_t bytes, const slice<data_type, dimensions> &data) const;
};

#if HCDE_OPENMP_SUPPORT

template<typename T, unsigned Dims>
class mt_cpu_encoder {
public:
    using data_type = T;
    constexpr static unsigned dimensions = Dims;

    size_t compress(const slice<const data_type, dimensions> &data, void *stream) const;

    size_t decompress(const void *stream, size_t bytes, const slice<data_type, dimensions> &data) const;
};

#endif // HCDE_OPENMP_SUPPORT

#if HCDE_GPU_SUPPORT

template<typename T, unsigned Dims>
using gpu_encoder = cpu_encoder<T, Dims>;

#endif // HCDE_GPU_SUPPORT

} // namespace hcde

