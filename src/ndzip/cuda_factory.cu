#include "cuda_codec.inl"


template<typename T>
std::unique_ptr<ndzip::cuda_compressor<T>>
ndzip::make_cuda_compressor(const compressor_requirements &reqs, cudaStream_t stream) {
    return detail::make_with_profile<cuda_compressor, detail::gpu_cuda::cuda_compressor_impl, T>(
            detail::get_dimensionality(reqs), stream, reqs);
}

template<typename T>
std::unique_ptr<ndzip::cuda_decompressor<T>> ndzip::make_cuda_decompressor(dim_type dims, cudaStream_t stream) {
    return detail::make_with_profile<cuda_decompressor, detail::gpu_cuda::cuda_decompressor_impl, T>(dims, stream);
}

template<typename T>
std::unique_ptr<ndzip::offloader<T>> ndzip::make_cuda_offloader(dim_type dimensions) {
    return detail::make_with_profile<offloader, detail::gpu_cuda::cuda_offloader, T>(dimensions);
}

namespace ndzip {

template std::unique_ptr<ndzip::cuda_compressor<float>> make_cuda_compressor<float>(
        const compressor_requirements &, cudaStream_t);
template std::unique_ptr<ndzip::cuda_decompressor<float>> make_cuda_decompressor<float>(dim_type, cudaStream_t);
template std::unique_ptr<ndzip::cuda_compressor<double>> make_cuda_compressor<double>(
        const compressor_requirements &, cudaStream_t);
template std::unique_ptr<ndzip::cuda_decompressor<double>> make_cuda_decompressor<double>(dim_type, cudaStream_t);
template std::unique_ptr<offloader<float>> make_cuda_offloader<float>(dim_type);
template std::unique_ptr<offloader<double>> make_cuda_offloader<double>(dim_type);

}  // namespace ndzip
