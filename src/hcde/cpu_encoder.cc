#include "cpu_encoder.inl"
#include "fast_profile.inl"
#include "strong_profile.inl"
#include "xt_profile.inl"

namespace hcde {
    template class cpu_encoder<fast_profile<float, 1>>;
    template class cpu_encoder<fast_profile<float, 2>>;
    template class cpu_encoder<fast_profile<float, 3>>;
    template class cpu_encoder<fast_profile<double, 1>>;
    template class cpu_encoder<fast_profile<double, 2>>;
    template class cpu_encoder<fast_profile<double, 3>>;
    template class cpu_encoder<strong_profile<float, 1>>;
    template class cpu_encoder<strong_profile<float, 2>>;
    template class cpu_encoder<strong_profile<float, 3>>;
    template class cpu_encoder<strong_profile<double, 1>>;
    template class cpu_encoder<strong_profile<double, 2>>;
    template class cpu_encoder<strong_profile<double, 3>>;
    template class cpu_encoder<xt_profile<float, 1>>;
    template class cpu_encoder<xt_profile<float, 2>>;
    template class cpu_encoder<xt_profile<float, 3>>;
    template class cpu_encoder<xt_profile<double, 1>>;
    template class cpu_encoder<xt_profile<double, 2>>;
    template class cpu_encoder<xt_profile<double, 3>>;
}
