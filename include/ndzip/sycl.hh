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

template<int Dims>
class sycl_compressor_requirements {
  public:
    sycl_compressor_requirements() = default;

    sycl_compressor_requirements(ndzip::extent<Dims> single_data_size) {  // NOLINT(google-explicit-constructor)
        include(single_data_size);
    }

    sycl_compressor_requirements(std::initializer_list<extent<Dims>> data_sizes) {
        for (auto ds : data_sizes) {
            include(ds);
        }
    }

    void include(extent<Dims> data_size);

  private:
    template<typename, int>
    friend class sycl_compressor;

    index_type _max_num_hypercubes = 0;
};

template<typename T>
class basic_sycl_compressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~basic_sycl_compressor() = default;

    // TODO can we have a generic base class for the interface even though buffers are explicitly dimensioned?
    // TODO USM variant
};

template<typename T, int Dims>
class sycl_compressor : public basic_sycl_compressor<T> {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    explicit sycl_compressor(sycl::queue &q, sycl_compressor_requirements<Dims> req);

    sycl_compress_events compress(sycl::buffer<value_type, Dims> &in_data, sycl::buffer<compressed_type> &out_stream,
            sycl::buffer<index_type> *out_stream_length = nullptr);

    // TODO USM variant

  private:
    template<typename, dim_type>
    friend class sycl_offloader;

    sycl::queue *_q;
    sycl::buffer<compressed_type> _chunks_buf;
    sycl::buffer<index_type> _chunk_lengths_buf;
    std::vector<sycl::buffer<index_type>> _hierarchical_scan_bufs;

    explicit sycl_compressor(sycl::queue &q, std::pair<index_type, index_type> chunks_and_length_buf_sizes);
};

template<typename T>
class basic_sycl_decompressor {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    virtual ~basic_sycl_decompressor() = default;

    // TODO can we have a generic base class for the interface even though buffers are explicitly dimensioned?
    // TODO USM variant
};

template<typename T, int Dims>
class sycl_decompressor : public basic_sycl_decompressor<T> {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    explicit sycl_decompressor(sycl::queue &q) : _q{&q} {}

    sycl_decompress_events decompress(
            sycl::buffer<compressed_type> &in_stream, sycl::buffer<value_type, Dims> &out_data);

    // TODO USM variant

  private:
    sycl::queue *_q;
};

}  // namespace ndzip
