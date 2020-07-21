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

        template<typename ...Init, std::enable_if_t<((sizeof...(Init) == Dims)
            && ... && std::is_convertible_v<Init, size_t>), int> = 0>
        constexpr explicit extent(const Init &...components) noexcept
            : _components{static_cast<size_t>(components)...}
        {
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

        size_t linear_offset() const {
            size_t offset = 1;
            for (unsigned d = 0; d < Dims; ++d) {
                offset *= _components[d];
            }
            return offset;
        }

    private:
        size_t _components[Dims] = {};
};

} // namespace extent

namespace hcde::detail {

    template<unsigned Dims>
    size_t linear_index(const hcde::extent<Dims> &size, const hcde::extent<Dims> &pos) {
        // assert(pos[0] < size[0]);
        size_t l = pos[0];
        for (unsigned d = 1; d < Dims; ++d) {
            // assert(pos[d] < size[d]);
            l = l * size[d] + pos[d];
        }
        return l;
    }

} // namespace hcde::detail

namespace hcde {

template<typename T, unsigned Dims>
class slice {
    public:
        explicit slice(T *data, hcde::extent<Dims> e)
            : _data(data)
            , _extent(e)
        {
        }

        template<typename U, std::enable_if_t<std::is_const_v<T>
            && std::is_same_v<std::remove_const_t<T>, U>, int> = 0>
        slice(slice<U, Dims> other)
            : _data(other._data)
            , _extent(other._extent)
        {
        }

        const hcde::extent<Dims> &extent() const {
            return _extent;
        }

        T *data() const {
            return _data;
        }

        size_t linear_index(const hcde::extent<Dims> &pos) const {
            return detail::linear_index(_extent, pos);
        }

        T &operator[](const hcde::extent<Dims> &pos) const {
            return _data[linear_index(pos)];
        }

    private:
        T *_data;
        hcde::extent<Dims> _extent;

        friend class slice<const T, Dims>;
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
            = 1 + sizeof(data_type) * detail::ipow(hypercube_side_length, Dims);

        size_t encode_block(const bits_type *bits, void *stream) const;

        size_t decode_block(const void *stream, bits_type *bits) const;

        bits_type load_value(const data_type *data) const;

        void store_value(data_type *data, bits_type bits) const;

        size_t store_superblock(const bits_type *superblock, T *data, const extent<Dims> offset) const;
};

template<typename Profile>
struct singlethread_cpu_encoder {
    using profile = Profile;
    using data_type = typename Profile::data_type;

    constexpr static unsigned dimensions = Profile::dimensions;

    size_t compressed_size_bound(const extent<dimensions> &e) const;

    size_t compress(const slice<const data_type, dimensions> &data, void *stream) const;

    size_t decompress(const void *stream, size_t bytes,
            const slice<data_type, dimensions> &data) const;
};

} // namespace hcde

