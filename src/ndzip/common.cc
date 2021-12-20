#include "common.hh"

namespace ndzip {

template<int Dims>
void compressor_requirements<Dims>::include(const extent<Dims> &data_size) {
    using profile = detail::profile<float, Dims>;  // TODO value_type does not matter here, refactor
    const auto file = detail::file<profile>(data_size);
    _max_num_hypercubes = std::max(_max_num_hypercubes, file.num_hypercubes());
}

template<int Dims>
compressor_requirements<Dims>::compressor_requirements(const ndzip::extent<Dims> &single_data_size) {
    include(single_data_size);
}

template<int Dims>
compressor_requirements<Dims>::compressor_requirements(std::initializer_list<extent<Dims>> data_sizes) {
    for (auto ds : data_sizes) {
        include(ds);
    }
}

template class compressor_requirements<1>;
template class compressor_requirements<2>;
template class compressor_requirements<3>;


template<typename T, ndzip::dim_type Dims>
index_type compressed_length_bound(const extent<Dims> &size) {
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;

    detail::file<profile> file(size);
    const auto header_length
            = detail::div_ceil(file.num_hypercubes(), detail::bits_of<bits_type> / detail::bits_of<index_type>);
    const auto compressed_length_bound = file.num_hypercubes() * profile::compressed_block_length_bound;
    const auto border_length = detail::border_element_count(size, profile::hypercube_side_length);
    return header_length + compressed_length_bound + border_length;
}

template<typename T>
index_type compressed_length_bound(const dynamic_extent &size) {
    switch (size.dimensions()) {
        case 1: return compressed_length_bound<T, 1>(extent<1>{size});
        case 2: return compressed_length_bound<T, 2>(extent<2>{size});
        case 3: return compressed_length_bound<T, 3>(extent<3>{size});
        default: abort();
    }
}

template index_type compressed_length_bound<float>(const dynamic_extent &);
template index_type compressed_length_bound<double>(const dynamic_extent &);

}  // namespace ndzip