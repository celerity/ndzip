#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <type_traits>


#if defined(__CUDA__) || defined(__NVCC__)
#define NDZIP_UNIVERSAL __host__ __device__
#else
#define NDZIP_UNIVERSAL
#endif


namespace ndzip {

using dim_type = int;
using index_type = uint32_t;
using stream_size_type = size_t;

inline constexpr dim_type max_dimensionality = 3;

template<dim_type Dims>
class extent;

class dynamic_extent {
  public:
    using const_iterator = const index_type *;
    using iterator = index_type *;

    constexpr dynamic_extent() noexcept = default;

    template<dim_type Dims>
    NDZIP_UNIVERSAL constexpr dynamic_extent(extent<Dims> extent) : _dims{Dims}, _components{extent._components} {
        static_assert(Dims <= max_dimensionality);
    }

    NDZIP_UNIVERSAL constexpr dynamic_extent(dim_type dims) noexcept : _dims{dims} {
        assert(dims > 0);
        assert(dims <= max_dimensionality);
    }

    NDZIP_UNIVERSAL static constexpr dynamic_extent broadcast(dim_type dims, index_type scalar) {
        dynamic_extent e{dims};
        for (dim_type d = 0; d < dims; ++d) {
            e[d] = scalar;
        }
        return e;
    }

    NDZIP_UNIVERSAL constexpr dim_type dimensions() const { return _dims; }

    NDZIP_UNIVERSAL index_type &operator[](dim_type d) {
        assert(d < _dims);
        return _components[d];
    }

    NDZIP_UNIVERSAL index_type operator[](dim_type d) const {
        assert(d < _dims);
        return _components[d];
    }

    NDZIP_UNIVERSAL dynamic_extent &operator+=(const dynamic_extent &other) {
        assert(other._dims == _dims);
        for (dim_type d = 0; d < _dims; ++d) {
            _components[d] += other._components[d];
        }
        return *this;
    }

    NDZIP_UNIVERSAL friend dynamic_extent operator+(const dynamic_extent &left, const dynamic_extent &right) {
        auto result = left;
        result += right;
        return result;
    }

    NDZIP_UNIVERSAL dynamic_extent &operator-=(const dynamic_extent &other) {
        assert(other._dims == _dims);
        for (dim_type d = 0; d < _dims; ++d) {
            _components[d] -= other._components[d];
        }
        return *this;
    }

    NDZIP_UNIVERSAL friend dynamic_extent operator-(const dynamic_extent &left, const dynamic_extent &right) {
        auto result = left;
        result -= right;
        return result;
    }

    NDZIP_UNIVERSAL dynamic_extent &operator*=(index_type other) {
        for (dim_type d = 0; d < _dims; ++d) {
            _components[d] *= other;
        }
        return *this;
    }

    NDZIP_UNIVERSAL friend dynamic_extent operator*(const dynamic_extent &left, index_type right) {
        auto result = left;
        result *= right;
        return result;
    }

    NDZIP_UNIVERSAL friend dynamic_extent operator*(index_type left, const dynamic_extent &right) {
        auto result = right;
        result *= left;
        return result;
    }

    NDZIP_UNIVERSAL dynamic_extent &operator/=(index_type other) {
        for (dim_type d = 0; d < _dims; ++d) {
            _components[d] /= other;
        }
        return *this;
    }

    NDZIP_UNIVERSAL friend dynamic_extent operator/(const dynamic_extent &left, index_type right) {
        auto result = left;
        result /= right;
        return result;
    }

    NDZIP_UNIVERSAL friend bool operator==(const dynamic_extent &left, const dynamic_extent &right) {
        assert(left._dims == right._dims);
        bool eq = true;
        for (dim_type d = 0; d < left._dims; ++d) {
            eq &= left[d] == right[d];
        }
        return eq;
    }

    NDZIP_UNIVERSAL friend bool operator!=(const dynamic_extent &left, const dynamic_extent &right) {
        return !operator==(left, right);
    }

    NDZIP_UNIVERSAL iterator begin() { return _components; }

    NDZIP_UNIVERSAL iterator end() { return _components + _dims; }

    NDZIP_UNIVERSAL const_iterator begin() const { return _components; }

    NDZIP_UNIVERSAL const_iterator end() const { return _components + _dims; }

  private:
    template<dim_type Dims>
    friend class extent;

    dim_type _dims = 1;
    index_type _components[max_dimensionality] = {};
};

template<dim_type Dims>
class extent {
  public:
    static_assert(Dims > 0);
    static_assert(Dims <= max_dimensionality);

    using const_iterator = const index_type *;
    using iterator = index_type *;

    constexpr extent() noexcept = default;

    template<typename... Init,
            std::enable_if_t<((sizeof...(Init) == Dims) && ... && std::is_convertible_v<Init, index_type>), int> = 0>
    NDZIP_UNIVERSAL constexpr extent(const Init &...components) noexcept
        : _components{static_cast<index_type>(components)...} {}

    NDZIP_UNIVERSAL extent(const dynamic_extent &dyn) {
        assert(dyn._dims == Dims);
        for (dim_type d = 0; d < Dims; ++d) {
            _components[d] = dyn._components[d];
        }
    }

    NDZIP_UNIVERSAL static extent broadcast(index_type scalar) {
        extent e;
        for (dim_type d = 0; d < Dims; ++d) {
            e[d] = scalar;
        }
        return e;
    }

    NDZIP_UNIVERSAL constexpr dim_type dimensions() const { return Dims; }

    NDZIP_UNIVERSAL index_type &operator[](dim_type d) { return _components[d]; }

    NDZIP_UNIVERSAL index_type operator[](dim_type d) const { return _components[d]; }

    NDZIP_UNIVERSAL extent &operator+=(const extent &other) {
        for (dim_type d = 0; d < Dims; ++d) {
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
        for (dim_type d = 0; d < Dims; ++d) {
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
        for (dim_type d = 0; d < Dims; ++d) {
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
        for (dim_type d = 0; d < Dims; ++d) {
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
        for (dim_type d = 0; d < Dims; ++d) {
            eq &= left[d] == right[d];
        }
        return eq;
    }

    NDZIP_UNIVERSAL friend bool operator!=(const extent &left, const extent &right) { return !operator==(left, right); }

    NDZIP_UNIVERSAL iterator begin() { return _components; }

    NDZIP_UNIVERSAL iterator end() { return _components + Dims; }

    NDZIP_UNIVERSAL const_iterator begin() const { return _components; }

    NDZIP_UNIVERSAL const_iterator end() const { return _components + Dims; }

  private:
    friend class dynamic_extent;
    index_type _components[Dims] = {};
};

template<typename... Init>
extent(const Init &...) -> extent<sizeof...(Init)>;

template<typename Extent>
NDZIP_UNIVERSAL index_type num_elements(const Extent &size) {
    index_type n = 1;
    for (dim_type d = 0; d < size.dimensions(); ++d) {
        n *= size[d];
    }
    return n;
}

template<typename Extent>
NDZIP_UNIVERSAL index_type linear_offset(const Extent &position, const Extent &space) {
    assert(position.dimensions() == space.dimensions());
    index_type offset = 0;
    index_type stride = 1;
    for (dim_type nd = 0; nd < position.dimensions(); ++nd) {
        auto d = position.dimensions() - 1 - nd;
        offset += stride * position[d];
        stride *= space[d];
    }
    return offset;
}

}  // namespace ndzip

namespace ndzip::detail {

template<typename Extent>
NDZIP_UNIVERSAL index_type linear_index(const Extent &size, const Extent &pos) {
    assert(size.dimensions() == pos.dimensions());
    index_type l = pos[0];
    for (dim_type d = 1; d < size.dimensions(); ++d) {
        l = l * size[d] + pos[d];
    }
    return l;
}

template<size_t Size>
struct bits_type_s;

template<>
struct bits_type_s<1> {
    using type = uint8_t;
};

template<>
struct bits_type_s<2> {
    using type = uint16_t;
};

template<>
struct bits_type_s<4> {
    using type = uint32_t;
};

template<>
struct bits_type_s<8> {
    using type = uint64_t;
};

template<typename T>
using bits_type = typename bits_type_s<sizeof(T)>::type;

}  // namespace ndzip::detail

namespace ndzip {

template<typename T>
using compressed_type = ndzip::detail::bits_type<T>;

template<typename T, typename Extent>
class slice {
  public:
    NDZIP_UNIVERSAL explicit slice(T *data, const Extent &size) : _data(data), _size(size) {}

    template<typename OtherExtent>
    NDZIP_UNIVERSAL explicit slice(slice<T, OtherExtent> other) : _data(other.data()), _size(other.size()) {}

    NDZIP_UNIVERSAL const Extent &size() const { return _size; }

    NDZIP_UNIVERSAL T *data() const { return _data; }

    NDZIP_UNIVERSAL index_type linear_index(const Extent &pos) const { return detail::linear_index(_size, pos); }

    NDZIP_UNIVERSAL T &operator[](const Extent &pos) const { return _data[linear_index(pos)]; }

#ifdef __NVCC__
#pragma diag_suppress 554  //  will not be called if T == const T
#endif
    // NVCC breaks on a SFINAE'd constructor slice(slice<U, Dims>) with U == std::remove_const_t<T>
    NDZIP_UNIVERSAL operator slice<const T, Extent>() const { return slice<const T, Extent>{_data, _size}; }
#ifdef __NVCC__
#pragma diag_default 554
#endif

  private:
    T *_data;
    Extent _size;

    friend class slice<const T, Extent>;
};

template<typename T, dim_type Dims>
index_type compressed_length_bound(const extent<Dims> &e);

template<typename T>
index_type compressed_length_bound(const dynamic_extent &e);

using kernel_duration = std::chrono::duration<uint64_t, std::nano>;

template<typename T>
class basic_compressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~basic_compressor() = default;

    virtual index_type compress(const slice<const value_type, dynamic_extent> &data, compressed_type *stream) = 0;
};

template<typename T, int Dims>
class compressor final : public basic_compressor<T> {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    inline constexpr static int dimensions = Dims;

    compressor();

    explicit compressor(size_t num_threads);

    index_type compress(const slice<const value_type, extent<Dims>> &data, compressed_type *stream) {
        return _pimpl->compress(data, stream);
    }

    index_type compress(const slice<const value_type, dynamic_extent> &data, compressed_type *stream) override {
        return compress(slice<const value_type, extent<Dims>>{data}, stream);
    }

  private:
    struct impl {
        virtual ~impl() = default;
        virtual index_type compress(const slice<const value_type, extent<Dims>> &data, compressed_type *stream) = 0;
    };
    struct st_impl;
    struct mt_impl;

    std::unique_ptr<impl> _pimpl;
};

template<typename T>
class basic_decompressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~basic_decompressor() = default;

    virtual index_type decompress(const compressed_type *stream, const slice<value_type, dynamic_extent> &data) = 0;
};

template<typename T, int Dims>
class decompressor final : public basic_decompressor<T> {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    inline constexpr static int dimensions = Dims;

    decompressor();

    explicit decompressor(size_t num_threads);

    index_type decompress(const compressed_type *stream, const slice<value_type, extent<Dims>> &data) {
        return _pimpl->decompress(stream, data);
    }

    index_type decompress(const compressed_type *stream, const slice<value_type, dynamic_extent> &data) override {
        return decompress(stream, slice<value_type, extent<Dims>>{data});
    }

  private:
    // TODO these can be proper subclasses (with a factory method)
    struct impl {
        virtual ~impl() = default;
        virtual index_type decompress(const compressed_type *stream, const slice<value_type, extent<Dims>> &data) = 0;
    };
    struct st_impl;
    struct mt_impl;

    std::unique_ptr<impl> _pimpl;
};

}  // namespace ndzip