#pragma once

#include "array.hh"

#include <memory>

#include <SYCL/sycl.hpp>

namespace ndzip_sycl {

template<typename T>
struct compressor_scratch_buffer;

struct compress_events {
    sycl::event start;
    std::vector<sycl::event> stream_available;
    sycl::event stream_length_available;

    void wait() {
        sycl::event::wait(stream_available);
        stream_length_available.wait();
    }
};

struct decompress_events {
    sycl::event start;
    std::vector<sycl::event> data_available;

    void wait() { sycl::event::wait(data_available); }
};

template<typename T, unsigned Dims>
std::unique_ptr<compressor_scratch_buffer<T>> allocate_compressor_scratch_buffer(ndzip::extent<Dims> data_size);

template<typename T, int Dims>
compress_events compress_async(sycl::buffer<T, Dims> &in_data, sycl::buffer<ndzip::detail::bits_type<T>> &out_stream,
        sycl::buffer<ndzip::index_type> *out_stream_length, compressor_scratch_buffer<T> &scratch, sycl::queue &q);

template<typename T, int Dims>
decompress_events
decompress_async(sycl::buffer<ndzip::detail::bits_type<T>> &in_stream, sycl::buffer<T, Dims> &out_data, sycl::queue &q);

}  // namespace ndzip_sycl


namespace std {

template<typename T>
struct default_delete<ndzip_sycl::compressor_scratch_buffer<T>> {
    void operator()(ndzip_sycl::compressor_scratch_buffer<T> *m) const;
};

}  // namespace std
