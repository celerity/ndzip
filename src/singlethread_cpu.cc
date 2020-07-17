#include "common.hh"


template<typename Profile>
size_t hcde::singlethread_cpu_encoder<Profile>::compressed_size_bound(
        const extent<dimensions> &e) const {
    size_t n_cubes = 1;
    for (unsigned d = 0; d < Profile::dimensions; ++d) {
        n_cubes *= e[d] / Profile::hypercube_side_length;
    }
    size_t bound = n_cubes * Profile::compressed_block_size_bound;
    bound += detail::border_element_count(e, Profile::hypercube_side_length) * sizeof(data_type);
    return bound;
}


template<typename Profile>
size_t hcde::singlethread_cpu_encoder<Profile>::compress(
        const slice<const data_type, dimensions> &data, void *stream) const
{
    using bits_type = typename Profile::bits_type;
    constexpr static auto side_length = Profile::hypercube_side_length;
    size_t stream_pos = 0;
    detail::for_each_hypercube_offset(data.extent(), side_length, [&](auto offset) {
        Profile p;
        bits_type cube[detail::ipow(side_length, Profile::dimensions)] = {};
        bits_type *cube_ptr = cube;
        detail::for_each_in_hypercube<Profile::dimensions>(side_length, [&](auto element) {
            *cube_ptr++ = p.load_value(&data[offset + element]);
        });
        stream_pos += p.encode_block(cube, static_cast<char *>(stream) + stream_pos);
    });
    stream_pos += detail::pack_border(static_cast<char *>(stream) + stream_pos, data, side_length);
    return stream_pos;
}


template<typename Profile>
size_t hcde::singlethread_cpu_encoder<Profile>::decompress(const void *stream, size_t bytes,
        const slice<data_type, dimensions> &data) const
{
    using bits_type = typename Profile::bits_type;
    constexpr static auto side_length = Profile::hypercube_side_length;
    size_t stream_pos = 0;
    detail::for_each_hypercube_offset(data.extent(), side_length, [&](auto offset) {
        Profile p;
        bits_type cube[detail::ipow(side_length, Profile::dimensions)] = {};
        stream_pos += p.decode_block(static_cast<const char *>(stream) + stream_pos, cube);
        bits_type *cube_ptr = cube;
        detail::for_each_in_hypercube<Profile::dimensions>(side_length, [&](auto element) {
            p.store_value(&data[offset + element], *cube_ptr++);
        });
    });
    stream_pos += detail::unpack_border(data, static_cast<const char *>(stream) + stream_pos,
            side_length);
    return stream_pos;
}


namespace hcde {

template class singlethread_cpu_encoder<fast_profile<float, 1>>;
template class singlethread_cpu_encoder<fast_profile<float, 2>>;
template class singlethread_cpu_encoder<fast_profile<float, 3>>;

}
