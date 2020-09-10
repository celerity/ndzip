#pragma once

#include "common.hh"
#include <cstdlib>

#ifdef __GNUC__
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wunused-parameter"
#   pragma GCC diagnostic ignored "-Wundef"
#endif
#include <SYCL/sycl.hpp>
#ifdef __GNUC__
#   pragma GCC diagnostic pop
#endif


namespace hcde::detail {

template<size_t ...Index>
sycl::range<sizeof...(Index)> to_sycl_range(
    const extent<sizeof...(Index)> &e, std::index_sequence<Index...>) {
    return sycl::range<sizeof...(Index)>(e[Index]...);
}

template<unsigned Dims>
sycl::range<Dims> to_sycl_range(const extent<Dims> &e) {
    return extent_to_sycl_range(e, std::make_index_sequence<Dims>{});
}

template<typename profile>
std::vector<superblock<profile>> collect_superblocks(const file<profile> &f) {
    std::vector<superblock<profile>> sbs;
    sbs.reserve(f.num_superblocks());
    f.for_each_superblock([&](auto sb) { sbs.push_back(sb); });
    return sbs;
}

// SYCL kernel id types
template<typename T, unsigned Dims> class compression_kernel;
template<typename T, unsigned Dims> class compaction_kernel;
template<typename T, unsigned Dims> class decode_kernel;

}


template<typename T, unsigned Dims>
struct hcde::gpu_encoder<T, Dims>::impl {
    sycl::queue q;
    sycl::exception_list async_exceptions;

    impl()
        : q(sycl::async_handler([this](sycl::exception_list l) {
            async_exceptions.insert(async_exceptions.end(), l.begin(), l.end()); }),
        sycl::property_list{sycl::property::queue::enable_profiling{}})
    {
    }

    void rethrow_async_exceptions() {
        sycl::exception_list ex;
        std::swap(ex, async_exceptions);
        for (auto &e: ex) {
            std::rethrow_exception(e);
        }
    }
};


template<typename T, unsigned Dims>
hcde::gpu_encoder<T, Dims>::gpu_encoder()
    : _pimpl(std::make_unique<impl>())
{
}


template<typename T, unsigned Dims>
hcde::gpu_encoder<T, Dims>::~gpu_encoder() = default;


template<typename T, unsigned Dims>
size_t hcde::gpu_encoder<T, Dims>::compressed_size_bound(const extent<dimensions> &size) const {
    using profile = detail::profile<T, Dims>;

    detail::file<profile> file(size);
    size_t bound = file.combined_length_of_all_headers();
    bound += file.num_hypercubes() * profile::compressed_block_size_bound;
    bound += detail::border_element_count(size, profile::hypercube_side_length) * sizeof(data_type);
    return bound;
}


template<typename T, unsigned Dims>
size_t hcde::gpu_encoder<T, Dims>::compress(const slice<const data_type, dimensions> &data, void *stream) const {
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;
    using hypercube_offset_type = typename profile::hypercube_offset_type;

    constexpr static auto side_length = profile::hypercube_side_length;
    const auto hc_size = detail::ipow(side_length, profile::dimensions);

    sycl::buffer<data_type> data_buffer(data.data(), sycl::range<1>(num_elements(data.size())));

    const detail::file<profile> file(data.size());
    const auto superblocks = detail::collect_superblocks(file);
    sycl::buffer<detail::superblock<profile>> superblock_buffer(superblocks.data(), sycl::range<1>(superblocks.size()));

    std::vector<uint64_t> sb_initial_offset_data;
    sb_initial_offset_data.reserve(file.num_superblocks());
    {
        uint64_t offset = file.file_header_length();
        for (auto &sb: superblocks) {
            sb_initial_offset_data.push_back(offset);
            offset += file.superblock_header_length() + sb.num_hypercubes() * profile::compressed_block_size_bound;
        }
    }

    sycl::buffer<uint64_t> sb_initial_offsets(static_cast<const uint64_t*>(sb_initial_offset_data.data()),
        sycl::range<1>{sb_initial_offset_data.size()});
    sycl::buffer<std::byte> sb_stream_buffer{sycl::range<1>(compressed_size_bound(data.size()))};
    sycl::buffer<uint64_t> sb_lengths(sycl::range<1>{superblocks.size()});

    static_assert(std::is_trivially_copyable_v<hcde::detail::superblock<profile>>);
    static_assert(std::is_trivially_copyable_v<hcde::detail::file<profile>>);

    auto compress_event = _pimpl->q.submit([&](sycl::handler &cgh) {
        auto data_access = data_buffer.template get_access<sycl::access::mode::read>(cgh);
        auto superblock_access = superblock_buffer.template get_access<sycl::access::mode::read>(cgh);
        auto sb_stream_access = sb_stream_buffer.get_access<sycl::access::mode::discard_write>(cgh);
        auto sb_initial_offset_access = sb_initial_offsets.get_access<sycl::access::mode::read>(cgh);
        auto sb_length_access = sb_lengths.get_access<sycl::access::mode::discard_write>(cgh);
        auto bits_access = sycl::accessor<bits_type, 1, sycl::access::mode::read_write,
            sycl::access::target::local>(sycl::range<1>{file.max_num_hypercubes_per_superblock() * hc_size}, cgh);
        auto hc_stream_access = sycl::accessor<std::byte, 1, sycl::access::mode::read_write,
            sycl::access::target::local>(sycl::range<1>{file.max_num_hypercubes_per_superblock()
            * (profile::compressed_block_size_bound + sizeof(bits_type))}, cgh);
        auto hc_length_access = sycl::accessor<hypercube_offset_type, 1, sycl::access::mode::read_write,
            sycl::access::target::local>(sycl::range<1>{file.max_num_hypercubes_per_superblock()}, cgh);
        auto hc_offset_access = sycl::accessor<hypercube_offset_type, 1, sycl::access::mode::read_write,
            sycl::access::target::local>(sycl::range<1>{file.max_num_hypercubes_per_superblock()}, cgh);

        cgh.parallel_for<detail::compression_kernel<T, Dims>>(sycl::nd_range<1>{
            sycl::range<1>{superblocks.size() * file.max_num_hypercubes_per_superblock()},
                sycl::range<1>{file.max_num_hypercubes_per_superblock()}},
            [file, superblock_access, data_access, sb_stream_access, bits_access, sb_length_access,
                sb_initial_offset_access, hc_stream_access, hc_length_access, hc_offset_access,
                data_size = data.size()](sycl::nd_item<1> nd_item) {
                const auto sb_id = nd_item.get_group().get_id();
                const auto sb = superblock_access[sb_id];
                const auto hc_id = nd_item.get_local_id();
                const auto hc_index = hc_id[0];
                const auto n_hcs = sb.num_hypercubes();

                slice<const data_type, dimensions> device_data(data_access.get_pointer(), data_size);
                bits_type *bits = bits_access.get_pointer();

                detail::load_superblock_warp(hc_index, sb, device_data, bits);

                nd_item.barrier(sycl::access::fence_space::local_space);

                std::byte *hc_stream = hc_stream_access.get_pointer()
                    + hc_index * (profile::compressed_block_size_bound + sizeof(bits_type));
                __builtin_memset(hc_stream, 0, profile::compressed_block_size_bound + sizeof(bits_type));
                if (hc_index < n_hcs) {
                    hc_length_access[hc_id] = detail::encode_block<profile>(bits + hc_size * hc_index, hc_stream);
                }

                nd_item.barrier(sycl::access::fence_space::local_space);

                if (hc_index == 0) {
                    auto offset = file.superblock_header_length();
                    for (size_t i = 0; i < n_hcs; ++i) {
                        hc_offset_access[sycl::id<1>{i}] = offset;
                        offset += static_cast<hypercube_offset_type>(hc_length_access[sycl::id<1>{i}]);
                    }

                    sb_length_access[sb_id] = static_cast<uint64_t>(offset);
                }

                nd_item.barrier(sycl::access::fence_space::local_space);

                if (hc_index < n_hcs) {
                    std::byte *sb_stream = sb_stream_access.get_pointer() + sb_initial_offset_access[sb_id];
                    if (hc_index > 0) {
                        detail::store_unaligned(sb_stream + sizeof(hypercube_offset_type) * (hc_index - 1),
                            detail::endian_transform(hc_offset_access[hc_id]));
                    }
                    memcpy(sb_stream + hc_offset_access[hc_id], hc_stream, hc_length_access[hc_id]);
                }
            });
    });

    auto compress_duration = compress_event.template get_profiling_info<sycl::info::event_profiling::command_end>()
        - compress_event.template get_profiling_info<sycl::info::event_profiling::command_start>();
    printf("compression %g ms\n", 1e-6*compress_duration);

    sycl::buffer<uint64_t> sb_offsets(sycl::range<1>{superblocks.size() + 1});

    {
        auto sb_length_access = sb_lengths.get_access<sycl::access::mode::read>();
        auto sb_offset_access = sb_offsets.get_access<sycl::access::mode::discard_write>();
        auto offset = file.file_header_length();
        for (size_t i = 0; i < superblocks.size(); ++i) {
            sb_offset_access[sycl::id<1>{i}] = offset;
            offset += sb_length_access[sycl::id<1>{i}];
        }
        sb_offset_access[sycl::id<1>{superblocks.size()}] = offset;
    }

    std::optional<sycl::buffer<std::byte>> file_stream_buffer{std::in_place, static_cast<std::byte *>(stream),
        sycl::range<1>(compressed_size_bound(data.size()))};

    auto compact_event = _pimpl->q.submit([&](sycl::handler &cgh) {
        auto sb_initial_offset_access = sb_initial_offsets.get_access<sycl::access::mode::read>(cgh);
        auto sb_length_access = sb_lengths.get_access<sycl::access::mode::read>(cgh);
        auto sb_offset_access = sb_offsets.get_access<sycl::access::mode::read>(cgh);
        auto sb_stream_access = sb_stream_buffer.get_access<sycl::access::mode::read>(cgh);
        auto file_stream_access = file_stream_buffer->get_access<sycl::access::mode::discard_write>(cgh);

        cgh.parallel_for<detail::compaction_kernel<T, Dims>>(
            sycl::nd_range<2>{
                sycl::range<2>{superblocks.size(), HCDE_WARP_SIZE},
                sycl::range<2>{1, HCDE_WARP_SIZE}},
            [sb_stream_access, file_stream_access, sb_initial_offset_access, sb_offset_access, sb_length_access](
                sycl::nd_item<2> nd_item) {
                const auto sb_index = nd_item.get_global_id(0);
                const auto sb_id = sycl::id<1>{sb_index};
                const auto thread = nd_item.get_global_id(1);
                const std::byte *const sb_stream = sb_stream_access.get_pointer();
                std::byte *const file_stream = file_stream_access.get_pointer();

                auto src = sb_stream + sb_initial_offset_access[sb_id];
                auto dest = file_stream + sb_offset_access[sb_id];
                auto length = sb_length_access[sb_id];
                const size_t copy_size = 4;
                for (size_t j = copy_size * thread; j + copy_size <= length; j += copy_size * HCDE_WARP_SIZE) {
                    __builtin_memcpy(dest + j, src + j, copy_size);
                }
                for (size_t j = length - length % copy_size + thread; j < length; ++j) {
                    dest[j] = src[j];
                }
                if (thread == 0) {
                    auto dest = file_stream + sb_index * sizeof(uint64_t);
                    auto shift_id = sycl::id<1>{sb_index + 1}; // we don't write offset of sb 0, but the border offset
                    detail::store_unaligned(dest, detail::endian_transform(sb_offset_access[shift_id]));
                }
            });

        // cgh.update_host(file_stream_access);
    });

    compact_event.wait_and_throw();
    _pimpl->rethrow_async_exceptions();

    file_stream_buffer.reset(); // write back

    auto compact_duration = compact_event.template get_profiling_info<sycl::info::event_profiling::command_end>()
        - compact_event.template get_profiling_info<sycl::info::event_profiling::command_start>();
    printf("compaction %g ms\n", 1e-6*compact_duration);

    uint64_t file_pos = sb_offsets.get_access<sycl::access::mode::read>()[sycl::id<1>{superblocks.size()}];
    file_pos += detail::pack_border(static_cast<char *>(stream) + file_pos, data, side_length);
    return file_pos;
}


template<typename T, unsigned Dims>
size_t hcde::gpu_encoder<T, Dims>::decompress(const void *stream, size_t bytes,
    const slice<data_type, dimensions> &data) const
{
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;
    using hypercube_offset_type = typename profile::hypercube_offset_type;
    constexpr static auto side_length = profile::hypercube_side_length;

    const detail::file<profile> file(data.size());
    const auto superblocks = detail::collect_superblocks(file);
    sycl::buffer<detail::superblock<profile>> superblock_buffer(superblocks.data(), sycl::range<1>(superblocks.size()));

    sycl::buffer<std::byte> stream_buffer(static_cast<const std::byte*>(stream), sycl::range<1>(bytes));
    sycl::buffer<data_type> data_buffer(sycl::range<1>(num_elements(data.size())));

    auto event = _pimpl->q.submit([&](sycl::handler &cgh) {
        auto stream_access = stream_buffer.get_access<sycl::access::mode::read>(cgh);
        auto data_access = data_buffer.template get_access<sycl::access::mode::discard_write>(cgh);
        auto superblock_access = superblock_buffer.template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for<detail::decode_kernel<T, Dims>>(
            sycl::range<2>{superblocks.size(), file.max_num_hypercubes_per_superblock()},
            [superblock_access, data_access, stream_access, file, data_size = data.size()](sycl::item<2> item) {
                const auto sb_index = item.get_id(0);
                const auto hc_index = item.get_id(1);
                const auto sb = superblock_access[sycl::id<1>{sb_index}];

                if (hc_index >= sb.num_hypercubes()) {
                    return;
                }

                slice<data_type, dimensions> device_data(data_access.get_pointer(), data_size);
                std::byte *device_stream = stream_access.get_pointer();

                size_t sb_offset = file.file_header_length();
                if (sb_index > 0) {
                    auto sb_offset_address = device_stream + (sb_index - 1) * sizeof(uint64_t);
                    sb_offset = detail::endian_transform(detail::load_unaligned<uint64_t>(sb_offset_address));
                }

                size_t hc_offset = file.superblock_header_length();
                if (hc_index > 0) {
                    auto hc_offset_address = device_stream + sb_offset
                        + (hc_index - 1) * sizeof(hypercube_offset_type);
                    hc_offset = detail::endian_transform(
                        detail::load_unaligned<hypercube_offset_type>(hc_offset_address));
                }

                bits_type cube[detail::ipow(side_length, profile::dimensions)] = {};
                detail::decode_block<profile>(device_stream + sb_offset + hc_offset, cube);
                bits_type *cube_ptr = cube;
                sb.hypercube_at(hc_index).for_each_cell(device_data, [&](auto *cell) {
                    detail::store_value<profile>(cell, *cube_ptr++);
                });
        });
    });

    event.wait_and_throw();
    _pimpl->rethrow_async_exceptions();

    auto host_data = data_buffer.template get_access<sycl::access::mode::read>();
    memcpy(data.data(), host_data.get_pointer(), num_elements(data.size()) * sizeof(data_type));

    auto border_pos = file.file_header_length();
    if (file.num_superblocks() > 0) {
        auto border_pos_address = static_cast<const char *>(stream)
            + (file.num_superblocks() - 1) * sizeof(uint64_t);
        border_pos = static_cast<size_t>(detail::endian_transform(
            detail::load_unaligned<uint64_t>(border_pos_address)));
    }
    auto border_length = detail::unpack_border(data, static_cast<const char *>(stream) + border_pos,
        side_length);
    return border_pos + border_length;
}


namespace hcde {
    extern template class gpu_encoder<float, 1>;
    extern template class gpu_encoder<float, 2>;
    extern template class gpu_encoder<float, 3>;
    extern template class gpu_encoder<double, 1>;
    extern template class gpu_encoder<double, 2>;
    extern template class gpu_encoder<double, 3>;
}
