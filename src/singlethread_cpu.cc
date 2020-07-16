#include "common.hh"


template<unsigned Dims>
size_t border_element_count(const extent<Dims> &e, unsigned side_length) {
    size_t n_cubes = 1;
    size_t n_all_elems = 1;
    for (unsigned d = 0; d < Dims; ++d) {
        n_cube_elems *= e[d] / side_length * side_length;
        n_all_elems *= e[d];
    }
    return n_all_elems - n_cube_elems;
}


template<typename Profile>
size_t hcde::singlethread_cpu_encoder<Profile>::compressed_size_bound(
        const extent<dimensions> &e) const {
    size_t n_cubes = 1;
    for (unsigned d = 0; d < Profile::dimensions; ++d) {
        n_cubes *= e[d] / Profile::hypercube_side_length;
    }
    return n_cubes * Profile::compressed_block_size_bound;
}


template<typename DataType, unsigned Dims>
void pack_border(DataType *dest, const slice<DataType, Dims> &src, unsigned side_length) {
    unsigned smallest_dim_with_border = 0;
    for (unsigned d = 1; d < Dims; ++d) {
        if (src.extent()[d] % side_length != 0) {
            smallest_dim_with_border = d;
        }
    }
    size_t dim_strides[d];
    ...
}


template<typename DataType, unsigned Dims>
void unpack_border(DataType *dest, const slice<DataType, Dims> &src, unsigned side_length) {
}


template<typename Profile>
size_t hcde::singlethread_cpu_encoder<Profile>::compress(
        const slice<data_type, dimensions> &data, void *stream) const
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
        auto cube_stream = static_cast<char *>(stream) + stream_pos;
        size_t compressed_cube_size = p.encode_block(cube, cube_stream);
        stream_pos += compressed_cube_size;
    });
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
        auto cube_stream = static_cast<const char *>(stream) + stream_pos;
        bits_type cube[detail::ipow(side_length, Profile::dimensions)] = {};
        size_t compressed_cube_size = p.decode_block(cube_stream, cube);
        stream_pos += compressed_cube_size;
        bits_type *cube_ptr = cube;
        detail::for_each_in_hypercube<Profile::dimensions>(side_length, [&](auto element) {
            p.store_value(&data[offset + element], *cube_ptr++);
        });
    });
    return stream_pos;
}


namespace hcde {

template class singlethread_cpu_encoder<fast_profile<float, 1>>;
template class singlethread_cpu_encoder<fast_profile<float, 2>>;
template class singlethread_cpu_encoder<fast_profile<float, 3>>;

}
