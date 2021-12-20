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

}  // namespace ndzip

namespace ndzip::detail {

template<dim_type Dims>
class static_extent;

}  // namespace ndzip::detail

namespace ndzip {

inline constexpr dim_type max_dimensionality = 3;

class extent {
  public:
    using const_iterator = const index_type *;
    using iterator = index_type *;

    constexpr extent() noexcept = default;

    template<dim_type Dims>
    NDZIP_UNIVERSAL constexpr extent(const detail::static_extent<Dims> &extent)
        : _dims{Dims}, _components{extent._components} {
        static_assert(Dims <= max_dimensionality);
    }

    NDZIP_UNIVERSAL constexpr extent(dim_type dims) noexcept : _dims{dims} {
        assert(dims > 0);
        assert(dims <= max_dimensionality);
    }

    NDZIP_UNIVERSAL static constexpr extent broadcast(dim_type dims, index_type scalar) {
        extent e{dims};
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

    NDZIP_UNIVERSAL extent &operator+=(const extent &other) {
        assert(other._dims == _dims);
        for (dim_type d = 0; d < _dims; ++d) {
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
        assert(other._dims == _dims);
        for (dim_type d = 0; d < _dims; ++d) {
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
        for (dim_type d = 0; d < _dims; ++d) {
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
        for (dim_type d = 0; d < _dims; ++d) {
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
        assert(left._dims == right._dims);
        bool eq = true;
        for (dim_type d = 0; d < left._dims; ++d) {
            eq &= left[d] == right[d];
        }
        return eq;
    }

    NDZIP_UNIVERSAL friend bool operator!=(const extent &left, const extent &right) { return !operator==(left, right); }

    NDZIP_UNIVERSAL iterator begin() { return _components; }

    NDZIP_UNIVERSAL iterator end() { return _components + _dims; }

    NDZIP_UNIVERSAL const_iterator begin() const { return _components; }

    NDZIP_UNIVERSAL const_iterator end() const { return _components + _dims; }

  private:
    template<dim_type Dims>
    friend class detail::static_extent;

    dim_type _dims = 1;
    index_type _components[max_dimensionality] = {};
};


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

class compressor_requirements;

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

index_type get_num_hypercubes(compressor_requirements req);

}  // namespace ndzip::detail

namespace ndzip {

template<typename T>
using compressed_type = ndzip::detail::bits_type<T>;

template<typename T>
index_type compressed_length_bound(const extent &e);

using kernel_duration = std::chrono::duration<uint64_t, std::nano>;

template<typename T>
class basic_compressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~basic_compressor() = default;

    virtual index_type compress(const value_type *data, const extent &data_size, compressed_type *stream) = 0;
};

template<typename T, int Dims>
class compressor final : public basic_compressor<T> {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    inline constexpr static int dimensions = Dims;

    compressor();

    explicit compressor(size_t num_threads);

    index_type compress(const value_type *data, const extent &data_size, compressed_type *stream) override {
        return _pimpl->compress(data, data_size, stream);
    }

  private:
    struct impl {
        virtual ~impl() = default;
        virtual index_type compress(const value_type *data, const extent &data_size, compressed_type *stream) = 0;
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

    virtual index_type decompress(const compressed_type *stream, value_type *data, const extent &data_size) = 0;
};

template<typename T, int Dims>
class decompressor final : public basic_decompressor<T> {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    inline constexpr static int dimensions = Dims;

    decompressor();

    explicit decompressor(size_t num_threads);

    index_type decompress(const compressed_type *stream, value_type *data, const extent &data_size) override {
        return _pimpl->decompress(stream, data, data_size);
    }

  private:
    // TODO these can be proper subclasses (with a factory method)
    struct impl {
        virtual ~impl() = default;
        virtual index_type decompress(const compressed_type *stream, value_type *data, const extent &data_size) = 0;
    };
    struct st_impl;
    struct mt_impl;

    std::unique_ptr<impl> _pimpl;
};

class compressor_requirements {
  public:
    compressor_requirements() = default;
    compressor_requirements(const ndzip::extent &single_data_size);  // NOLINT(google-explicit-constructor)
    compressor_requirements(std::initializer_list<extent> data_sizes);

    void include(const extent &data_size);

  private:
    friend index_type detail::get_num_hypercubes(compressor_requirements);

    index_type _max_num_hypercubes = 0;
};

}  // namespace ndzip