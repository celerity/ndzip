#pragma once

#include "ndzip.hh"

#include <SYCL/sycl.hpp>


namespace ndzip {

struct sycl_compress_events {
    sycl::event start;
    std::vector<sycl::event> stream_available;
    sycl::event stream_length_available;

    void wait() {
        sycl::event::wait(stream_available);
        stream_length_available.wait();
    }
};

struct sycl_decompress_events {
    sycl::event start;
    std::vector<sycl::event> data_available;

    void wait() {  // NOLINT(readability-make-member-function-const)
        sycl::event::wait(data_available);
    }
};

template<typename T>
class sycl_compressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~sycl_compressor() = 0;

    // TODO can we have a generic base class for the interface even though buffers are explicitly dimensioned?
    // TODO USM variant
};

template<typename T>
sycl_compressor<T>::~sycl_compressor() = default;

template<typename T, dim_type Dims>
class sycl_buffer_compressor : public sycl_compressor<T> {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    constexpr static dim_type dimensions = Dims;

    sycl_compress_events compress(sycl::buffer<value_type, Dims> &in_data, sycl::buffer<compressed_type> &out_stream,
            sycl::buffer<index_type> *out_stream_length = nullptr) {
        return do_compress(in_data, out_stream, out_stream_length);
    }

  protected:
    virtual sycl_compress_events do_compress(sycl::buffer<value_type, Dims> &in_data,
            sycl::buffer<compressed_type> &out_stream, sycl::buffer<index_type> *out_stream_length)
            = 0;
};

template<typename T>
class sycl_decompressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~sycl_decompressor() = 0;

    // TODO can we have a generic base class for the interface even though buffers are explicitly dimensioned?
    // TODO USM variant
};

template<typename T>
sycl_decompressor<T>::~sycl_decompressor() = default;

template<typename T, dim_type Dims>
class sycl_buffer_decompressor : public sycl_decompressor<T> {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    sycl_decompress_events decompress(
            sycl::buffer<compressed_type> &in_stream, sycl::buffer<value_type, Dims> &out_data) {
        return do_decompress(in_stream, out_data);
    }

  protected:
    virtual sycl_decompress_events
    do_decompress(sycl::buffer<compressed_type> &in_stream, sycl::buffer<value_type, Dims> &out_data)
            = 0;
};


template<typename T>
std::unique_ptr<sycl_compressor<T>> make_sycl_compressor(sycl::queue &q, const compressor_requirements &req);

template<typename T, dim_type Dims>
std::unique_ptr<sycl_buffer_compressor<T, Dims>>
make_sycl_buffer_compressor(sycl::queue &q, const compressor_requirements &req);

template<typename T>
std::unique_ptr<sycl_decompressor<T>> make_sycl_decompressor(sycl::queue &q, dim_type dims);

template<typename T, dim_type Dims>
std::unique_ptr<sycl_buffer_decompressor<T, Dims>> make_sycl_buffer_decompressor(sycl::queue &q);

}  // namespace ndzip
