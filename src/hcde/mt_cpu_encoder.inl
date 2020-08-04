#pragma once

#include "common.hh"

#include <cstddef>
#include <condition_variable>
#include <thread>
#include <vector>


namespace hcde::detail {

template<typename Profile>
using superblock_group = std::vector<std::tuple<size_t, detail::superblock<Profile>>>;

template<typename Profile>
std::vector<superblock_group<Profile>> collect_superblock_groups(
        const file<Profile> &f, size_t n_groups)
{
    std::vector<superblock_group<Profile>> groups(n_groups);
    size_t count = 0;
    f.for_each_superblock([&](auto sb) {
        groups[count % n_groups].emplace_back(count, sb);
        ++count;
    });
    return groups;
}

}


template<typename Profile>
hcde::mt_cpu_encoder<Profile>::mt_cpu_encoder()
    : mt_cpu_encoder(std::thread::hardware_concurrency())
{
}


template<typename Profile>
hcde::mt_cpu_encoder<Profile>::mt_cpu_encoder(size_t num_threads)
    : _num_threads(num_threads) {
}


template<typename Profile>
size_t hcde::mt_cpu_encoder<Profile>::compressed_size_bound(
        const extent<dimensions> &size) const {
    detail::file<Profile> file(size);
    size_t bound = file.combined_length_of_all_headers();
    bound += file.num_hypercubes() * Profile::compressed_block_size_bound;
    bound += detail::border_element_count(size, Profile::hypercube_side_length) * sizeof(data_type);
    return bound;
}


template<typename Profile>
size_t hcde::mt_cpu_encoder<Profile>::compress(
        const slice<const data_type, dimensions> &data, void *stream) const {
    using bits_type = typename Profile::bits_type;

    constexpr static auto side_length = Profile::hypercube_side_length;
    detail::file<Profile> file(data.size());

    auto num_sbs = file.num_superblocks();
    auto sb_groups = detail::collect_superblock_groups(file, _num_threads);

    std::vector<size_t> initial_sb_offsets(num_sbs);
    std::vector<size_t> sb_lengths(num_sbs);

    {
        size_t offset = file.file_header_length();
        file.for_each_superblock([&](auto sb, auto sb_index) {
            initial_sb_offsets[sb_index] = offset;
            offset += file.superblock_header_length()
                + sb.num_hypercubes() * Profile::compressed_block_size_bound;
        });
    }

    std::vector<std::thread> threads;
    threads.reserve(_num_threads);

    for (auto &group: sb_groups) {
        threads.emplace_back([&group, &data, &initial_sb_offsets, &sb_lengths, stream, file] {
            std::vector<bits_type> cube;

            for (auto &[index, sb]: group) {
                auto sb_stream = static_cast<std::byte*>(stream) + initial_sb_offsets[index];
                size_t sb_stream_pos = file.superblock_header_length();

                Profile p;
                cube.resize(detail::ipow(side_length, Profile::dimensions)
                        * sb.num_hypercubes());
                bits_type *cube_pos = cube.data();
                sb.for_each_hypercube([&](auto hc) {
                    hc.for_each_cell(data, [&](auto *element) { *cube_pos++ = p.load_value(element); });
                });

                cube_pos = cube.data();
                sb.for_each_hypercube([&](auto, auto hc_index) {
                    if (hc_index > 0) {
                        auto offset_address = sb_stream + (hc_index - 1)
                                * sizeof(typename Profile::hypercube_offset_type);
                        auto offset = static_cast<typename Profile::hypercube_offset_type>(sb_stream_pos);
                        detail::store_unaligned(offset_address, detail::endian_transform(offset));
                    }
                    sb_stream_pos += p.encode_block(cube_pos, sb_stream + sb_stream_pos);
                    cube_pos += detail::ipow(side_length, Profile::dimensions);
                });

                sb_lengths[index] = sb_stream_pos;
            }
        });
    }

    for (auto &t: threads) {
        t.join();
    }

    size_t file_pos = file.file_header_length() + sb_lengths[0];
    for (size_t sb_index = 1; sb_index < num_sbs; ++sb_index) {
        auto file_offset_address = static_cast<char *>(stream)
                + (sb_index - 1) * sizeof(uint64_t);
        auto file_offset = static_cast<uint64_t>(file_pos);
        detail::store_unaligned(file_offset_address, detail::endian_transform(file_offset));
        assert(file_pos <= initial_sb_offsets[sb_index]);
        memmove(static_cast<std::byte *>(stream) + file_pos,
            static_cast<const std::byte *>(stream) + initial_sb_offsets[sb_index], sb_lengths[sb_index]);
        file_pos += sb_lengths[sb_index];
    }

    auto border_offset_address =
            static_cast<char *>(stream) + (file.num_superblocks() - 1) * sizeof(uint64_t);
    detail::store_unaligned(border_offset_address, detail::endian_transform(file_pos));
    file_pos += detail::pack_border(static_cast<char *>(stream) + file_pos, data, side_length);
    return file_pos;
}


template<typename Profile>
size_t hcde::mt_cpu_encoder<Profile>::decompress(const void *stream, size_t bytes,
        const slice<data_type, dimensions> &data) const {
    using bits_type = typename Profile::bits_type;
    constexpr static auto side_length = Profile::hypercube_side_length;

    detail::file<Profile> file(data.size());
    auto sb_groups = detail::collect_superblock_groups(file, _num_threads);

    std::vector<std::thread> threads;
    threads.reserve(_num_threads);
    for (auto &group: sb_groups) {
        threads.emplace_back([&group, stream, &data, file] {
            for (auto &[index, sb]: group) {
                size_t stream_pos;
                if (index > 0) {
                    auto offset_address = static_cast<const char *>(stream)
                            + (index - 1) * sizeof(uint64_t);
                    stream_pos = static_cast<size_t>(detail::endian_transform(
                            detail::load_unaligned<uint64_t>(offset_address)));
                } else {
                    stream_pos = file.file_header_length();
                }
                stream_pos += file.superblock_header_length(); // simply skip the header

                sb.for_each_hypercube([&](auto hc) {
                    Profile p;
                    bits_type cube[detail::ipow(side_length, Profile::dimensions)] = {};
                    stream_pos += p.decode_block(
                            static_cast<const char *>(stream) + stream_pos, cube);
                    bits_type *cube_ptr = cube;
                    hc.for_each_cell(data, [&](auto *cell) { p.store_value(cell, *cube_ptr++); });
                });
            }
        });
    }

    auto border_pos_address = static_cast<const char *>(stream)
            + (file.num_superblocks() - 1) * sizeof(uint64_t);
    auto border_pos = static_cast<size_t>(detail::endian_transform(
            detail::load_unaligned<uint64_t>(border_pos_address)));
    auto border_length = detail::unpack_border(data, static_cast<const char *>(stream) + border_pos,
            side_length);

    for (auto &t: threads) {
        t.join();
    }

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
