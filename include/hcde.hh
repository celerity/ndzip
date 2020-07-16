#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <type_traits>


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

} // namespace hcde::detail


namespace hcde {

template<unsigned Dims>
class extent {
    public:
        constexpr extent() noexcept = default;

        template<typename ...Tail>
        constexpr explicit extent(size_t head, Tail ...tail) noexcept
            : _components{head, tail...}
        {
            static_assert(1 + sizeof...(Tail) == Dims);
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

    private:
        size_t _components[Dims] = {};
};

template<typename T, unsigned Dims>
class slice {
    public:
        explicit slice(T *data, hcde::extent<Dims> e)
            : _data(data)
            , _extent(e)
        {
        }

        const hcde::extent<Dims> &extent() const {
            return _extent;
        }

        T *data() const {
            return _data;
        }

        size_t linear_index(const hcde::extent<Dims> &e) const {
            assert(e[0] < _extent[0]);
            size_t l = e[0];
            for (unsigned d = 1; d < Dims; ++d) {
                assert(e[d] < _extent[d]);
                l = l * _extent[d] + e[d];
            }
            return l;
        }

        T &operator[](const hcde::extent<Dims> &e) const {
            return _data[linear_index(e)];
        }

    private:
        T *_data;
        hcde::extent<Dims> _extent;
};

template<typename T, unsigned Dims>
class fast_profile {
    public:
        using data_type = T;
        using bits_type = detail::bits_type<T>;

        constexpr static unsigned dimensions = Dims;
        constexpr static unsigned hypercube_side_length = 4;
        constexpr static unsigned superblock_size = 64;
        constexpr static size_t compressed_block_size_bound
            = sizeof(data_type) * detail::ipow(hypercube_side_length, Dims);

        size_t encode_block(const bits_type *bits, void *stream) const;

        void decode_block(const void *stream, bits_type *bits) const;

        bits_type load_value(const data_type *data) const;

        void store_value(data_type *data, bits_type bits) const;

        size_t store_superblock(const bits_type *superblock, T *data, const extent<Dims> offset) const;
};

template<typename Profile>
struct singlethread_cpu_encoder {
    using data_type = typename Profile::data_type;

    constexpr static unsigned dimensions = Profile::dimensions;

    size_t compressed_size_bound(const extent<dimensions> &e) const;

    size_t compress(const slice<data_type, dimensions> &data, void *stream) const;

    void decompress(const void *stream, size_t bytes, const extent<dimensions> &size) const;
};

} // namespace hcde

