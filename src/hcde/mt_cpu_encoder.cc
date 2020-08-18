#include "mt_cpu_encoder.inl"
#include "fast_profile.inl"
#include "strong_profile.inl"
#include "xt_profile.inl"

namespace hcde {
    template class mt_cpu_encoder<fast_profile<float, 1>>;
    template class mt_cpu_encoder<fast_profile<float, 2>>;
    template class mt_cpu_encoder<fast_profile<float, 3>>;
    template class mt_cpu_encoder<strong_profile<float, 1>>;
    template class mt_cpu_encoder<strong_profile<float, 2>>;
    template class mt_cpu_encoder<strong_profile<float, 3>>;
    template class mt_cpu_encoder<xt_profile<float, 1>>;
    template class mt_cpu_encoder<xt_profile<float, 2>>;
    template class mt_cpu_encoder<xt_profile<float, 3>>;
}
