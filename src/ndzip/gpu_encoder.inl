#pragma once

#include "common.hh"

#include <stdexcept>
#include <vector>

#include <SYCL/sycl.hpp>


namespace ndzip::detail::gpu {

template<unsigned Dims, typename U, typename T>
U extent_cast(const T &e) {
    U v;
    for (unsigned i = 0; i < Dims; ++i) {
        v[i] = e[i];
    }
    return v;
}

template<typename U, unsigned Dims>
U extent_cast(const extent<Dims> &e) {
    return extent_cast<Dims, U>(e);
}

template<typename T, int Dims>
T extent_cast(const sycl::range<Dims> &r) {
    return extent_cast<static_cast<unsigned>(Dims), T>(r);
}

template<typename T, int Dims>
T extent_cast(const sycl::id<Dims> &r) {
    return extent_cast<static_cast<unsigned>(Dims), T>(r);
}

template<typename Profile, typename DataAccessor, typename BlockTransformAccessor>
void load_hypercube(const DataAccessor &data_acc, const BlockTransformAccessor &block_transform_acc,
    sycl::nd_item<2> item)
{
    using bits_type = typename Profile::bits_type;

    constexpr auto dimensions = Profile::dimensions;
    constexpr auto side_length = Profile::hypercube_side_length;

    auto hc_index = item.get_global_range(0);
    auto data_size = detail::gpu::extent_cast<extent<dimensions>>(data_acc.get_range());
    auto hc_offset = detail::extent_from_linear_id(hc_index, data_size / side_length) * side_length;
    auto tid = item.get_local_range(1);

    // Parallel global -> local load
}


template<typename T>
void block_transform_step(T *x, size_t n, size_t s) {
    T a, b;
    b = x[0*s];
    for (size_t i = 1; i < n; ++i) {
        a = b;
        b = x[i*s];
        x[i*s] = a ^ b;
    }
}


template<typename Profile, typename BlockTransformAccessor>
void block_transform(const BlockTransformAccessor &block_transform_acc, sycl::nd_item<2> item) {
    constexpr size_t n = Profile::hypercube_side_length;

    auto tid = item.get_local_range(1);
    typename Profile::bits_type *x = block_transform_acc.get_pointer();

    if constexpr (Profile::dimensions == 1) {
        if (tid == 0) {
            block_transform_step(x, n, 1);
        }
    } else if constexpr (Profile::dimensions == 2) {
        for (size_t i = tid; i < n*n; i += n*NDZIP_WARP_SIZE) {
            block_transform_step(x + i, n, 1);
        }
        for (size_t i = tid; i < n; i += NDZIP_WARP_SIZE) {
            block_transform_step(x + i, n, n);
        }
    } else if constexpr (Profile::dimensions == 3) {
        for (size_t i = tid; i < n*n*n; i += n*n*NDZIP_WARP_SIZE) {
            for (size_t j = 0; j < n; ++j) {
                block_transform_step(x + i + j, n, n);
            }
        }
        for (size_t i = tid; i < n*n*n; i += n*NDZIP_WARP_SIZE) {
            block_transform_step(x + i, n, 1);
        }
        for (size_t i = tid; i < n*n; i += NDZIP_WARP_SIZE) {
            block_transform_step(x + i, n, n * n);
        }
    }
}


template<typename Profile, typename BlockTransformAccessor, typename BlockCompactionAccessor>
size_t zero_bit_encode(const BlockTransformAccessor &block_transform_acc,
    const BlockCompactionAccessor &block_compaction_acc, sycl::nd_item<2> item)
{
    using bits_type = typename Profile::bits_type;

    constexpr static auto side_length = Profile::hypercube_side_length;
    constexpr static auto hc_size = detail::ipow(side_length, Profile::dimensions);

    // zero bit encode

    return 0;
}

template<typename Profile, typename BlockCompactionAccessor, typename CompressedBlocksAccessor>
void store_compressed_block(BlockCompactionAccessor block_compaction_acc,
    CompressedBlocksAccessor compressed_blocks_acc, sycl::nd_item<2> nd_item)
{
}


// SYCL kernel names
template<typename, unsigned> class block_compression_kernel;
template<typename, unsigned> class length_sum_kernel;
template<typename, unsigned> class block_compaction_kernel;
template<typename, unsigned> class border_compaction_kernel;

}


template<typename T, unsigned Dims>
struct ndzip::gpu_encoder<T, Dims>::impl {
    sycl::queue q;
};

template<typename T, unsigned Dims>
ndzip::gpu_encoder<T, Dims>::gpu_encoder()
    : _pimpl(std::make_unique<impl>())
{
}

template<typename T, unsigned Dims>
ndzip::gpu_encoder<T, Dims>::~gpu_encoder() = default;


template<typename T, unsigned Dims>
size_t ndzip::gpu_encoder<T, Dims>::compress(const slice<const data_type, dimensions> &data, void *stream) const {
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;

    constexpr auto side_length = profile::hypercube_side_length;
    constexpr auto hc_size = detail::ipow(side_length, profile::dimensions);
    constexpr auto max_compressed_block_uints = hc_size / detail::bitsof<bits_type> * (detail::bitsof<bits_type> + 1);

    detail::file<profile> file(data.size());

    sycl::buffer<data_type, dimensions> data_buffer{data.data(),
        detail::gpu::extent_cast<sycl::range<dimensions>>(data.size())};
    sycl::buffer<bits_type, 2> compressed_blocks_buffer{
        sycl::range<2>{file.num_hypercubes(), max_compressed_block_uints}};
    sycl::buffer<detail::file_offset_type , 1> compressed_block_lengths_buffer{sycl::range<1>{file.num_hypercubes()}};

    _pimpl->q.submit([&](sycl::handler &cgh) {
        // global memory
        auto data_acc = data_buffer.template get_access<sycl::access::mode::read>(cgh);
        auto compressed_blocks_acc = compressed_blocks_buffer.template get_access<sycl::access::mode::discard_write>(
            cgh);
        auto compressed_block_lengths_acc = compressed_block_lengths_buffer.get_access<
            sycl::access::mode::discard_write>(cgh);

        // local memory
        auto block_transform_acc = sycl::accessor<bits_type, 1, sycl::access::mode::discard_read_write,
            sycl::access::target::local>{hc_size, cgh};
        auto block_compaction_acc = sycl::accessor<bits_type, 1, sycl::access::mode::discard_read_write,
            sycl::access::target::local>{max_compressed_block_uints, cgh};

        cgh.parallel_for<detail::gpu::block_compression_kernel<T, Dims>>(
            sycl::nd_range<2>{sycl::range<2>{file.num_hypercubes(), NDZIP_WARP_SIZE}, sycl::range<2>{1, NDZIP_WARP_SIZE}},
            [=](sycl::nd_item<2> item) {
                auto hc_index = item.get_global_range(0);

                detail::gpu::load_hypercube<profile>(data_acc, block_transform_acc, item);
                detail::gpu::block_transform<profile>(block_transform_acc, item);
                compressed_block_lengths_acc[sycl::id<1>{hc_index}] = detail::gpu::zero_bit_encode<profile>(
                    block_transform_acc, block_compaction_acc, item);
                detail::gpu::store_compressed_block<profile>(block_compaction_acc, compressed_blocks_acc, item);
            });
    });

    // Re-use stream buffer? It must receive the offsets as header anyway
    sycl::buffer<detail::file_offset_type , 1> compressed_block_offsets_buffer{
        sycl::range<1>{file.num_hypercubes() + 1}};

    _pimpl->q.submit([&](sycl::handler &cgh) {
        auto compressed_block_lengths_acc = compressed_block_lengths_buffer.get_access<sycl::access::mode::read>(cgh);
        auto compressed_block_offsets_acc = compressed_block_offsets_buffer.get_access<
            sycl::access::mode::discard_read_write>(cgh);

        cgh.parallel_for<detail::gpu::length_sum_kernel<T, Dims>>(sycl::range<1>{NDZIP_WARP_SIZE},
            [=](sycl::item<1> item) {
            // parallel prefix sum lengths -> offsets
        });
    });

    sycl::buffer<std::byte, 1> stream_buffer{static_cast<std::byte*>(stream),
        sycl::range<1>{compressed_size_bound<data_type>(data.size())}};

    _pimpl->q.submit([&](sycl::handler &cgh) {
        auto compressed_block_lengths_acc = compressed_block_lengths_buffer.get_access<sycl::access::mode::read>(cgh);
        auto compressed_block_offsets_acc = compressed_block_offsets_buffer.get_access<sycl::access::mode::read>(cgh);
        auto compressed_blocks_acc = compressed_blocks_buffer.template get_access<sycl::access::mode::read>(cgh);
        auto stream_acc = stream_buffer.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<detail::gpu::block_compaction_kernel<T, Dims>>(sycl::range<1>{file.num_hypercubes()},
            [=](sycl::item<1> item) {
                // compaction
            });
    });

    _pimpl->q.submit([&](sycl::handler &cgh) {
        auto compressed_block_offsets_acc = compressed_block_offsets_buffer.get_access<sycl::access::mode::read>(cgh);
        auto data_acc = stream_buffer.template get_access<sycl::access::mode::write>(cgh);
        auto stream_acc = stream_buffer.get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<detail::gpu::border_compaction_kernel<T, Dims>>(sycl::range<1>{NDZIP_WARP_SIZE},
            [=](sycl::item<1> item) {
                // sequentially append border slices
            });
    });

    detail::file_offset_type compressed_size;

    _pimpl->q.submit([&](sycl::handler &cgh) {
        auto compressed_block_offsets_acc = compressed_block_offsets_buffer.get_access<sycl::access::mode::read>(cgh,
            sycl::range<1>{1}, sycl::id<1>{file.num_hypercubes()});
        cgh.copy(compressed_block_offsets_acc, &compressed_size);
    });

    _pimpl->q.wait();
    return compressed_size;
}


template<typename T, unsigned Dims>
size_t ndzip::gpu_encoder<T, Dims>::decompress(
    const void *stream, size_t bytes, const slice<data_type, dimensions> &data) const {
    return 0;
}


namespace ndzip {
    extern template class gpu_encoder<float, 1>;
    extern template class gpu_encoder<float, 2>;
    extern template class gpu_encoder<float, 3>;
    extern template class gpu_encoder<double, 1>;
    extern template class gpu_encoder<double, 2>;
    extern template class gpu_encoder<double, 3>;
}
