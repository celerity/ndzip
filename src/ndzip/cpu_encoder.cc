#include "cpu_encoder.inl"

namespace ndzip {

template class cpu_encoder<float, 1>;
template class cpu_encoder<float, 2>;
template class cpu_encoder<float, 3>;
template class cpu_encoder<double, 1>;
template class cpu_encoder<double, 2>;
template class cpu_encoder<double, 3>;

#if NDZIP_OPENMP_SUPPORT
template class mt_cpu_encoder<float, 1>;
template class mt_cpu_encoder<float, 2>;
template class mt_cpu_encoder<float, 3>;
template class mt_cpu_encoder<double, 1>;
template class mt_cpu_encoder<double, 2>;
template class mt_cpu_encoder<double, 3>;
#endif  // NDZIP_OPENMP_SUPPORT

}  // namespace ndzip
