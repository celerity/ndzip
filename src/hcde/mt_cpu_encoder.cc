#include "mt_cpu_encoder.inl"

namespace hcde {
    template class mt_cpu_encoder<float, 1>;
    template class mt_cpu_encoder<float, 2>;
    template class mt_cpu_encoder<float, 3>;
    template class mt_cpu_encoder<double, 1>;
    template class mt_cpu_encoder<double, 2>;
    template class mt_cpu_encoder<double, 3>;
}
