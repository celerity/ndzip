#include "cpu_encoder.inl"
#include "fast_profile.inl"
#include "strong_profile.inl"

namespace hcde {
    template class cpu_encoder<fast_profile<float, 2>>;
    template class cpu_encoder<fast_profile<float, 3>>;
    template class cpu_encoder<strong_profile<float, 2>>;
    template class cpu_encoder<strong_profile<float, 3>>;
}
