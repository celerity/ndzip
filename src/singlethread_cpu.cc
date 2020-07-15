#include "common.hh"


template<typename Profile>
size_t hcde::singlethread_cpu_encoder<Profile>::compressed_size_bound(
        const extent<dimensions> &e) const {
    return 0;
}


template<typename Profile>
size_t hcde::singlethread_cpu_encoder<Profile>::compress(
        const slice<data_type, dimensions> &data, void *stream) const
{
    return 0;
}


template<typename Profile>
void hcde::singlethread_cpu_encoder<Profile>::decompress(const void *stream, size_t bytes,
        const extent<dimensions> &size) const
{
}


namespace hcde {

template class fast_profile<float, 2>;

}
