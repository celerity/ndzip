#include "cpu_encoder.inl"

namespace hcde {
    template class cpu_encoder<float, 1>;
    template class cpu_encoder<float, 2>;
    template class cpu_encoder<float, 3>;
    template class cpu_encoder<double, 1>;
    template class cpu_encoder<double, 2>;
    template class cpu_encoder<double, 3>;
}
