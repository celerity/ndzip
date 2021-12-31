#include "sycl_codec.inl"


template<typename T>
std::unique_ptr<ndzip::sycl_compressor<T>>
ndzip::make_sycl_compressor(sycl::queue &q, const compressor_requirements &req) {
    return detail::make_with_profile<sycl_compressor, detail::gpu_sycl::sycl_compressor_impl, T>(
            detail::get_dimensionality(req), q, req);
}

template<typename T, ndzip::dim_type Dims>
std::unique_ptr<ndzip::sycl_buffer_compressor<T, Dims>>
ndzip::make_sycl_buffer_compressor(sycl::queue &q, const compressor_requirements &req) {
    return std::make_unique<detail::gpu_sycl::sycl_compressor_impl<detail::profile<T, Dims>>>(q, req);
}

template<typename T>
std::unique_ptr<ndzip::sycl_decompressor<T>> ndzip::make_sycl_decompressor(sycl::queue &q, dim_type dims) {
    return detail::make_with_profile<sycl_decompressor, detail::gpu_sycl::sycl_decompressor_impl, T>(dims, q);
}

template<typename T, ndzip::dim_type Dims>
std::unique_ptr<ndzip::sycl_buffer_decompressor<T, Dims>> ndzip::make_sycl_buffer_decompressor(sycl::queue &q) {
    return std::make_unique<detail::gpu_sycl::sycl_decompressor_impl<detail::profile<T, Dims>>>(q);
}

template<typename T>
std::unique_ptr<ndzip::offloader<T>> ndzip::make_sycl_offloader(dim_type dimensions, bool enable_profiling) {
    return detail::make_with_profile<offloader, detail::gpu_sycl::sycl_offloader, T>(
            dimensions, enable_profiling, detail::verbose());
}

namespace ndzip {

template std::unique_ptr<sycl_compressor<float>> make_sycl_compressor<float>(
        sycl::queue &, const compressor_requirements &);
template std::unique_ptr<sycl_buffer_compressor<float, 1>> make_sycl_buffer_compressor<float, 1>(
        sycl::queue &, const compressor_requirements &);
template std::unique_ptr<sycl_buffer_compressor<float, 2>> make_sycl_buffer_compressor<float, 2>(
        sycl::queue &, const compressor_requirements &);
template std::unique_ptr<sycl_buffer_compressor<float, 3>> make_sycl_buffer_compressor<float, 3>(
        sycl::queue &, const compressor_requirements &);

template std::unique_ptr<sycl_compressor<double>> make_sycl_compressor<double>(
        sycl::queue &, const compressor_requirements &);
template std::unique_ptr<sycl_buffer_compressor<double, 1>> make_sycl_buffer_compressor<double, 1>(
        sycl::queue &, const compressor_requirements &);
template std::unique_ptr<sycl_buffer_compressor<double, 2>> make_sycl_buffer_compressor<double, 2>(
        sycl::queue &, const compressor_requirements &);
template std::unique_ptr<sycl_buffer_compressor<double, 3>> make_sycl_buffer_compressor<double, 3>(
        sycl::queue &, const compressor_requirements &);

template std::unique_ptr<sycl_decompressor<float>> make_sycl_decompressor<float>(sycl::queue &, dim_type);
template std::unique_ptr<sycl_buffer_decompressor<float, 1>> make_sycl_buffer_decompressor<float, 1>(sycl::queue &);
template std::unique_ptr<sycl_buffer_decompressor<float, 2>> make_sycl_buffer_decompressor<float, 2>(sycl::queue &);
template std::unique_ptr<sycl_buffer_decompressor<float, 3>> make_sycl_buffer_decompressor<float, 3>(sycl::queue &);

template std::unique_ptr<sycl_decompressor<double>> make_sycl_decompressor<double>(sycl::queue &, dim_type);
template std::unique_ptr<sycl_buffer_decompressor<double, 1>> make_sycl_buffer_decompressor<double, 1>(sycl::queue &);
template std::unique_ptr<sycl_buffer_decompressor<double, 2>> make_sycl_buffer_decompressor<double, 2>(sycl::queue &);
template std::unique_ptr<sycl_buffer_decompressor<double, 3>> make_sycl_buffer_decompressor<double, 3>(sycl::queue &);

template std::unique_ptr<offloader<float>> make_sycl_offloader<float>(dim_type, bool);
template std::unique_ptr<offloader<double>> make_sycl_offloader<double>(dim_type, bool);

}  // namespace ndzip
