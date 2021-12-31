#include "cpu_codec.inl"


namespace ndzip::detail::cpu {

inline unsigned get_final_num_threads(const unsigned user_preference) {
#if NDZIP_OPENMP_SUPPORT
    if (user_preference == 0) {
        return boost::thread::physical_concurrency();
    } else {
        return user_preference;
    }
#else
    if (user_preference > 1) {
        throw std::invalid_argument{"ndzip was built without multithreading support, num_threads must be 1"};
    }
    return 1;
#endif
}

}  // namespace ndzip::detail::cpu

namespace ndzip {

template<typename T>
std::unique_ptr<compressor<T>> make_compressor(dim_type dims, unsigned num_threads) {
    num_threads = detail::cpu::get_final_num_threads(num_threads);
    if (num_threads == 1) {
        return detail::make_with_profile<compressor, detail::cpu::serial_compressor, T>(dims);
    } else {
#if NDZIP_OPENMP_SUPPORT
        return detail::make_with_profile<compressor, detail::cpu::openmp_compressor, T>(dims, num_threads);
#else
        abort();  // unreachable
#endif
    }
}

template<typename T>
std::unique_ptr<decompressor<T>> make_decompressor(dim_type dims, unsigned num_threads) {
    num_threads = detail::cpu::get_final_num_threads(num_threads);
    if (num_threads == 1) {
        return detail::make_with_profile<decompressor, detail::cpu::serial_decompressor, T>(dims);
    } else {
#if NDZIP_OPENMP_SUPPORT
        return detail::make_with_profile<decompressor, detail::cpu::openmp_decompressor, T>(dims, num_threads);
#else
        abort();  // unreachable
#endif
    }
}

}  // namespace ndzip
namespace ndzip::detail::cpu {

template<typename T>
class cpu_offloader final : public offloader<T> {
  public:
    using value_type = T;
    using compressed_type = detail::bits_type<T>;

    cpu_offloader() = default;

    explicit cpu_offloader(dim_type dims, unsigned num_threads)
        : _co{make_compressor<T>(dims, num_threads)}, _de{make_decompressor<T>(dims, num_threads)} {}

  protected:
    index_type do_compress(const value_type *data, const extent &data_size, compressed_type *stream,
            kernel_duration *duration) override {
        // TODO duration
        return _co->compress(data, data_size, stream);
    }

    index_type do_decompress(const compressed_type *stream, [[maybe_unused]] index_type stream_length, value_type *data,
            const extent &data_size, kernel_duration *duration) override {
        // TODO duration
        return _de->decompress(stream, data, data_size);
    }

  private:
    std::unique_ptr<compressor<T>> _co;
    std::unique_ptr<decompressor<T>> _de;
};

}  // namespace ndzip::detail::cpu

namespace ndzip {

template<typename T>
std::unique_ptr<offloader<T>> make_cpu_offloader(dim_type dims, unsigned num_threads) {
    return std::make_unique<detail::cpu::cpu_offloader<T>>(dims, num_threads);
}

template std::unique_ptr<offloader<float>> make_cpu_offloader<float>(dim_type, unsigned);
template std::unique_ptr<offloader<double>> make_cpu_offloader<double>(dim_type, unsigned);

}  // namespace ndzip
