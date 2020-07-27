#include "gpu_encoder.inl"
#include "fast_profile.inl"
#include "strong_profile.inl"

namespace hcde {
    template class gpu_encoder<fast_profile<float, 2>>;
    template class gpu_encoder<fast_profile<float, 3>>;
    template class gpu_encoder<strong_profile<float, 2>>;
    template class gpu_encoder<strong_profile<float, 3>>;
}
