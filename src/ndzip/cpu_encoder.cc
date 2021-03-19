#ifdef SPLIT_CONFIGURATION

#include "cpu_encoder.inl"

namespace ndzip {

template class cpu_encoder<DATA_TYPE, DIMENSIONS>;

#if NDZIP_OPENMP_SUPPORT
template class mt_cpu_encoder<DATA_TYPE, DIMENSIONS>;
#endif  // NDZIP_OPENMP_SUPPORT

}  // namespace ndzip

#endif
