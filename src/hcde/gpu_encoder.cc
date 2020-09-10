//
#ifndef NDEBUG
#   define NDEBUG 1
#endif

#include "gpu_encoder.inl"

namespace hcde {
    template class gpu_encoder<float, 1>;
    template class gpu_encoder<float, 2>;
    template class gpu_encoder<float, 3>;
    template class gpu_encoder<double, 1>;
    template class gpu_encoder<double, 2>;
    template class gpu_encoder<double, 3>;
}
