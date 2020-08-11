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

template<typename Profile>
std::vector<superblock<Profile>> collect_superblocks(const file<Profile> &f) {
    std::vector<superblock<Profile>> sbs;
    sbs.reserve(f.num_superblocks());
    f.for_each_superblock([&](auto sb) { sbs.push_back(sb); });
    return sbs;
}

// SYCL kernel id type
template<typename Profile>
class encode_kernel;

// SYCL kernel id type
template<typename Profile>
class decode_kernel;

}


template<typename Profile>
struct hcde::gpu_encoder<Profile>::impl {
    sycl::queue q;
};


template<typename Profile>
hcde::gpu_encoder<Profile>::gpu_encoder()
    : _pimpl(std::make_unique<impl>())
{
}


template<typename Profile>
hcde::gpu_encoder<Profile>::~gpu_encoder() = default;


template<typename Profile>
size_t hcde::gpu_encoder<Profile>::compressed_size_bound(const extent<dimensions> &size) const {
    detail::file<Profile> file(size);
    size_t bound = file.combined_length_of_all_headers();
    bound += file.num_hypercubes() * Profile::compressed_block_size_bound;
    bound += detail::border_element_count(size, Profile::hypercube_side_length) * sizeof(data_type);
    return bound;
}


template<typename Profile>
size_t hcde::gpu_encoder<Profile>::compress(const slice<const data_type, dimensions> &data, void *stream) const {
    using bits_type = typename Profile::bits_type;
    using hypercube_offset_type = typename Profile::hypercube_offset_type;

    constexpr static auto side_length = Profile::hypercube_side_length;
    const auto hc_size = detail::ipow(side_length, Profile::dimensions);

    sycl::buffer<data_type> data_buffer(data.data(), sycl::range<1>(num_elements(data.size())));

    const detail::file<Profile> file(data.size());
    const auto superblocks = detail::collect_superblocks(file);
    sycl::buffer<detail::superblock<Profile>> superblock_buffer(superblocks.data(), sycl::range<1>(superblocks.size()));

    std::vector<uint64_t> sb_initial_offset_data;
    sb_initial_offset_data.reserve(file.num_superblocks());
    uint64_t offset = file.file_header_length();
    for (auto &sb: superblocks) {
        sb_initial_offset_data.push_back(offset);
        offset += file.superblock_header_length() + sb.num_hypercubes() * Profile::compressed_block_size_bound;
    }

    sycl::buffer<uint64_t> sb_initial_offsets(static_cast<const uint64_t*>(sb_initial_offset_data.data()),
        sycl::range<1>{sb_initial_offset_data.size()});
    sycl::buffer<std::byte> stream_buffer{sycl::range<1>(compressed_size_bound(data.size()))};
    sycl::buffer<uint64_t> sb_lengths(sycl::range<1>{superblocks.size()});

    static_assert(std::is_trivially_copyable_v<hcde::detail::superblock<Profile>>);
    static_assert(std::is_trivially_copyable_v<hcde::detail::file<Profile>>);

    auto event = _pimpl->q.submit([&](sycl::handler &cgh) {
        auto data_access = data_buffer.template get_access<sycl::access::mode::read>(cgh);
        auto superblock_access = superblock_buffer.template get_access<sycl::access::mode::read>(cgh);
        auto stream_access = stream_buffer.get_access<sycl::access::mode::discard_write>(cgh);
        auto sb_initial_offset_access = sb_initial_offsets.get_access<sycl::access::mode::read>(cgh);
        auto sb_length_access = sb_lengths.get_access<sycl::access::mode::discard_write>(cgh);
        auto bits_access = sycl::accessor<bits_type, 1, sycl::access::mode::read_write,
            sycl::access::target::local>(sycl::range<1>{file.max_num_hypercubes_per_superblock() * hc_size}, cgh);
        auto hc_length_access = sycl::accessor<hypercube_offset_type, 1, sycl::access::mode::read_write,
            sycl::access::target::local>(sycl::range<1>{file.max_num_hypercubes_per_superblock()}, cgh);
        auto hc_offset_access = sycl::accessor<hypercube_offset_type, 1, sycl::access::mode::read_write,
            sycl::access::target::local>(sycl::range<1>{file.max_num_hypercubes_per_superblock()}, cgh);

        cgh.parallel_for<detail::encode_kernel<Profile>>(sycl::nd_range<1>{
            sycl::range<1>{superblocks.size() * file.max_num_hypercubes_per_superblock()},
            sycl::range<1>{file.max_num_hypercubes_per_superblock()}},
            [file, superblock_access, data_access, stream_access, bits_access, sb_length_access,
                sb_initial_offset_access, hc_size, hc_length_access, hc_offset_access, data_size = data.size()](
                    sycl::nd_item<1> nd_item) {
                const auto sb_id = nd_item.get_group().get_id();
                const auto sb = superblock_access[sb_id];
                const auto hc_id = nd_item.get_local_id();
                const auto hc_index = hc_id[0];
                const auto n_hcs = sb.num_hypercubes();

                slice<const data_type, dimensions> device_data(data_access.get_pointer(), data_size);
                bits_type *bits = bits_access.get_pointer();

                Profile p;
                detail::load_superblock_warp(hc_index, sb, device_data, bits, p);

                nd_item.barrier();

                std::byte hc_stream[Profile::compressed_block_size_bound + sizeof(bits_type)] = {};
                if (hc_index < n_hcs) {
                    hc_length_access[hc_id] = p.encode_block(bits + hc_size * hc_index, hc_stream);
                }

                nd_item.barrier();

                if (hc_index == 0) {
                    auto offset = file.superblock_header_length();
                    for (size_t i = 0; i < sb.num_hypercubes(); ++i) {
                        hc_offset_access[sycl::id<1>{i}] = offset;
                        offset += static_cast<hypercube_offset_type>(hc_length_access[sycl::id<1>{i}]);
                    }

                    sb_length_access[sb_id] = static_cast<uint64_t>(offset);
                }

                nd_item.barrier();

                if (hc_index < n_hcs) {
                    std::byte *sb_stream = stream_access.get_pointer() + sb_initial_offset_access[sb_id];
                    if (hc_index > 0) {
                        detail::store_unaligned(sb_stream + sizeof(hypercube_offset_type) * (hc_index - 1),
                            detail::endian_transform(hc_offset_access[hc_id]));
                    }
                    memcpy(sb_stream + hc_offset_access[hc_id], hc_stream, hc_length_access[hc_id]);
                }
            });
    });

    auto file_pos = static_cast<uint64_t>(file.file_header_length());
    auto sb_stream_access = stream_buffer.get_access<sycl::access::mode::read>();
    auto sb_length_access = sb_lengths.get_access<sycl::access::mode::read>();
    for (size_t i = 0; i < superblocks.size(); ++i) {
        auto length = sb_length_access[sycl::id<1>{i}];
        if (i > 0) {
            auto file_offset_address = static_cast<char *>(stream) + (i - 1) * sizeof(uint64_t);
            detail::store_unaligned(file_offset_address, detail::endian_transform(file_pos));
        }

        memcpy(static_cast<char *>(stream) + file_pos,
            sb_stream_access.get_pointer() + sb_initial_offset_data[i], length);
        file_pos += length;
    }

    event.wait_and_throw();

    auto border_offset_address =
        static_cast<char *>(stream) + (file.num_superblocks() - 1) * sizeof(uint64_t);
    detail::store_unaligned(border_offset_address, detail::endian_transform(file_pos));
    file_pos += detail::pack_border(static_cast<char *>(stream) + file_pos, data, side_length);
    return file_pos;
}


template<typename Profile>
size_t hcde::gpu_encoder<Profile>::decompress(const void *stream, size_t bytes,
    const slice<data_type, dimensions> &data) const
{
    using bits_type = typename Profile::bits_type;
    using hypercube_offset_type = typename Profile::hypercube_offset_type;
    constexpr static auto side_length = Profile::hypercube_side_length;

    const detail::file<Profile> file(data.size());
    const auto superblocks = detail::collect_superblocks(file);
    sycl::buffer<detail::superblock<Profile>> superblock_buffer(superblocks.data(), sycl::range<1>(superblocks.size()));

    sycl::buffer<std::byte> stream_buffer(static_cast<const std::byte*>(stream), sycl::range<1>(bytes));
    sycl::buffer<data_type> data_buffer(sycl::range<1>(num_elements(data.size())));

    auto event = _pimpl->q.submit([&](sycl::handler &cgh) {
        auto stream_access = stream_buffer.get_access<sycl::access::mode::read>(cgh);
        auto data_access = data_buffer.template get_access<sycl::access::mode::discard_write>(cgh);
        auto superblock_access = superblock_buffer.template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for<detail::decode_kernel<Profile>>(
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

                Profile p;
                bits_type cube[detail::ipow(side_length, Profile::dimensions)] = {};
                p.decode_block(device_stream + sb_offset + hc_offset, cube);
                bits_type *cube_ptr = cube;
                sb.hypercube_at(hc_index).for_each_cell(device_data,
                    [&](auto *cell) { p.store_value(cell, *cube_ptr++); });
        });
    });

    event.wait_and_throw();

    auto host_data = data_buffer.template get_access<sycl::access::mode::read>();
    memcpy(data.data(), host_data.get_pointer(), num_elements(data.size()) * sizeof(data_type));

    auto border_pos_address = static_cast<const char *>(stream)
        + (file.num_superblocks() - 1) * sizeof(uint64_t);
    auto border_pos = static_cast<size_t>(detail::endian_transform(
        detail::load_unaligned<uint64_t>(border_pos_address)));
    auto border_length = detail::unpack_border(data, static_cast<const char *>(stream) + border_pos,
        side_length);
    return border_pos + border_length;
}


namespace hcde {
    extern template class mt_cpu_encoder<fast_profile<float, 1>>;
    extern template class mt_cpu_encoder<fast_profile<float, 2>>;
    extern template class mt_cpu_encoder<fast_profile<float, 3>>;
    extern template class mt_cpu_encoder<strong_profile<float, 1>>;
    extern template class mt_cpu_encoder<strong_profile<float, 2>>;
    extern template class mt_cpu_encoder<strong_profile<float, 3>>;
}
