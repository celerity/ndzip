#include "common.hh"

namespace ndzip {

template<typename T, ndzip::dim_type Dims>
size_t compressed_size_bound(const extent<Dims> &size) {
    using profile = detail::profile<T, Dims>;
    detail::file<profile> file(size);
    size_t bound = file.file_header_length();
    bound += file.num_hypercubes() * profile::compressed_block_size_bound;
    bound += detail::border_element_count(size, profile::hypercube_side_length) * sizeof(T);
    return bound;
}

template size_t compressed_size_bound<float>(const extent<1> &e);
template size_t compressed_size_bound<float>(const extent<2> &e);
template size_t compressed_size_bound<float>(const extent<3> &e);
template size_t compressed_size_bound<double>(const extent<1> &e);
template size_t compressed_size_bound<double>(const extent<2> &e);
template size_t compressed_size_bound<double>(const extent<3> &e);

}  // namespace ndzip