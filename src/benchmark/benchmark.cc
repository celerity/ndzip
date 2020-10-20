#include <hcde/hcde.hh>
#include <io/io.hh>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#if HCDE_BENCHMARK_HAVE_ZLIB
#    define ZLIB_CONST
#    include <zlib.h>
#endif
#if HCDE_BENCHMARK_HAVE_LZ4
#    include <lz4.h>
#endif
#if HCDE_BENCHMARK_HAVE_LZMA
#    include <lzma.h>
#endif
#if HCDE_BENCHMARK_HAVE_ZSTD
#   include <zstd.h>
#endif
#if HCDE_BENCHMARK_HAVE_FPZIP
#   include <fpzip.h>
#endif
#include <fpc.h>
#include <SPDP_11.h>
#if HCDE_BENCHMARK_HAVE_GFC
#   include <GFC_22.h>
#endif


enum class data_type {
    t_float,
    t_double,
};

enum class tuning {
    min,
    min_max,
    full,
};

struct metadata {
    std::filesystem::path path;
    ::data_type data_type;
    std::vector<size_t> extent;

    metadata(std::filesystem::path path, ::data_type type, std::vector<size_t> extent)
        : path(std::move(path)), data_type(type), extent(std::move(extent)) {
    }

    size_t size_in_bytes() const {
        size_t size = data_type == data_type::t_float ? 4 : 8;
        for (auto e : extent) {
            size *= e;
        }
        return size;
    }
};


template<typename F>
class defer {
    public:
        [[nodiscard]] explicit defer(F &&f) : _f(std::move(f)) {}
        defer(const defer &) = delete;
        defer &operator=(const defer &) = delete;
        ~defer() { std::move(_f)(); }

    private:
        F _f;
};


static std::vector<metadata> load_metadata_file(const std::filesystem::path &path) {
    using namespace std::string_view_literals;

    std::ifstream ifs;
    ifs.exceptions(std::ios::badbit);
    ifs.open(path);

    std::vector<metadata> metadata;
    for (std::string line; std::getline(ifs, line);) {
        char data_file_name[100];
        char type_string[10];
        size_t extent[3];
        auto n_tokens = sscanf(line.c_str(), "%99[^;];%9[^;];%zu %zu %zu", data_file_name, type_string,
                extent, extent + 1, extent + 2);
        if (n_tokens >= 3 && n_tokens <= 5 && (type_string == "float"sv || type_string == "double"sv)) {
            metadata.emplace_back(path.parent_path()  / data_file_name,
                    type_string == "float"sv ? data_type::t_float : data_type::t_double,
                    std::vector<size_t>(extent, extent + n_tokens - 2));
        } else if (n_tokens != 0) {
            throw std::runtime_error(path.string() + ": Invalid line: " + std::move(line));
        }
    }
    return metadata;
}


struct benchmark_params {
    int tunable = 1;
    std::chrono::microseconds min_time = std::chrono::seconds(1);
    unsigned min_reps = 1;
};


struct benchmark_result {
    std::vector<std::chrono::microseconds> compression_times;
    std::vector<std::chrono::microseconds> decompression_times;
    uint64_t uncompressed_bytes = 0;
    uint64_t compressed_bytes = 0;
};


class benchmark {
public:
    explicit benchmark(const benchmark_params &params) : _params(params) {}

    std::chrono::steady_clock::time_point start() const { // NOLINT(readability-convert-member-functions-to-static)
        return std::chrono::steady_clock::now();
    }

    bool compress_more() const {
        return _compression.more(_params);
    }

    template<typename F>
    void time_compression(const F &f) {
        _compression.time(f);
    }

    void record_compression(std::chrono::microseconds time) {
        _compression.record(time);
    }

    bool decompress_more() const {
        return _decompression.more(_params);
    }

    template<typename F>
    void time_decompression(const F &f) {
        _decompression.time(f);
    }

    void record_decompression(std::chrono::microseconds time) {
        _decompression.record(time);
    }

    benchmark_result result(size_t uncompressed_bytes, size_t compressed_bytes) && {
        assert(_compression.reps > 0);
        assert(_decompression.reps > 0);
        return benchmark_result{std::move(_compression.times), std::move(_decompression.times),
                                uncompressed_bytes, compressed_bytes};
    }

private:
    struct accumulator {
        std::vector<std::chrono::microseconds> times;
        std::chrono::microseconds total_time{};
        unsigned reps{};

        bool more(const benchmark_params &params) const {
            return total_time < params.min_time || reps < std::max(1u, params.min_reps);
        }

        template<typename F>
        void time(const F &f) {
            auto start = std::chrono::steady_clock::now();
            run(f);
            auto finish = std::chrono::steady_clock::now();
            record(std::chrono::duration_cast<std::chrono::microseconds>(finish - start));
        }

        template<typename F>
        [[gnu::noinline]] void run(const F &f) {
            f();
        }

        void record(std::chrono::microseconds time) {
            times.push_back(time);
            total_time += time;
            ++reps;
        }
    };

    benchmark_params _params;
    accumulator _compression;
    accumulator _decompression;
};


template<typename T = std::byte>
class scratch_buffer {
public:
    [[gnu::noinline]] explicit scratch_buffer(size_t size)
    {
        _mem = malloc(size * sizeof(T));
        _size = size;
        // memset the to ensure all pages of the allocated buffer have been mapped by the OS
        memset(_mem, 0, size * sizeof(T));
    }

    scratch_buffer(const scratch_buffer &) = delete;
    scratch_buffer &operator=(const scratch_buffer &) = delete;

    ~scratch_buffer() {
        free(_mem);
    }

    size_t size() const {
        return _size;
    }

    const T *data() const {
        return static_cast<const T*>(_mem);
    }

    T *data() {
        return static_cast<T*>(_mem);
    }

private:
    void *_mem;
    size_t _size;
};


class buffer_mismatch: public std::exception {};

void assert_buffer_equality(const void *left, const void *right, size_t size) {
    if (memcmp(left, right, size) != 0) {
        throw buffer_mismatch();
    }
}


class not_implemented: public std::exception {};


template<template<typename, unsigned> typename Encoder, typename Data, unsigned Dims>
static benchmark_result benchmark_hcde_3(const Data *input_buffer, const hcde::extent<Dims> &size,
                                         const benchmark_params &params) {
    const auto uncompressed_size = hcde::num_elements(size) * sizeof(Data);
    const auto input_slice = hcde::slice{input_buffer, size};
    auto bench = benchmark{params};
    auto encoder = Encoder<Data, Dims>{};

    auto compress_buffer = scratch_buffer{hcde::compressed_size_bound<Data>(size)};
    size_t compressed_size;
    while (bench.compress_more()) {
        bench.time_compression([&] {
            compressed_size = encoder.compress(input_slice, compress_buffer.data());
        });
    }

    auto decompress_buffer = scratch_buffer<Data>(hcde::num_elements(size));
    const auto decompress_slice = hcde::slice{decompress_buffer.data(), size};
    while (bench.decompress_more()) {
        bench.time_decompression([&] {
            encoder.decompress(compress_buffer.data(), compressed_size, decompress_slice);
        });
    }

    assert_buffer_equality(input_buffer, decompress_buffer.data(), uncompressed_size);
    return std::move(bench).result(uncompressed_size, compressed_size);
}


template<template<typename, unsigned> typename Encoder, typename Data>
static benchmark_result benchmark_hcde_2(const Data *input_buffer, const metadata &meta,
                                         const benchmark_params &params) {
    auto &e = meta.extent;
    if (e.size() == 1) {
        return benchmark_hcde_3<Encoder, Data, 1>(input_buffer, hcde::extent{e[0]}, params);
    } else if (e.size() == 2) {
        return benchmark_hcde_3<Encoder, Data, 2>(input_buffer, hcde::extent{e[0], e[1]}, params);
    } else if (e.size() == 3) {
        return benchmark_hcde_3<Encoder, Data, 3>(input_buffer, hcde::extent{e[0], e[1], e[2]}, params);
    } else {
        throw not_implemented{};
    }
}


template<template<typename, unsigned> typename Encoder>
static benchmark_result benchmark_hcde(const void *input_buffer, const metadata &meta,
                                       const benchmark_params &params) {
    if (meta.data_type == data_type::t_float) {
        return benchmark_hcde_2<Encoder, float>(static_cast<const float*>(input_buffer), meta, params);
    } else {
        return benchmark_hcde_2<Encoder, double>(static_cast<const double*>(input_buffer), meta, params);
    }
}


#if HCDE_BENCHMARK_HAVE_FPZIP
static benchmark_result benchmark_fpzip(const void *input_buffer, const metadata &meta,
                                        const benchmark_params &params) {
    const auto uncompressed_size = meta.size_in_bytes();
    auto bench = benchmark{params};

    const auto fpz_set = [](FPZ *fpz, const metadata &meta) {
        fpz->type = meta.data_type == data_type::t_float ? 0 : 1;
        fpz->prec = 0; // lossless
        auto &e = meta.extent;
        fpz->nx = e.size() >= 1 ? static_cast<int>(e[e.size() - 1]) : 1; // NOLINT(readability-container-size-empty)
        fpz->ny = e.size() >= 2 ? static_cast<int>(e[e.size() - 2]) : 1;
        fpz->nz = e.size() >= 3 ? static_cast<int>(e[e.size() - 3]) : 1;
        fpz->nf = e.size() >= 4 ? static_cast<int>(e[e.size() - 4]) : 1;
    };

    auto compress_buffer = scratch_buffer{2*uncompressed_size + 1000}; // no bound function, just guess large enough
    size_t compressed_size;
    while (bench.compress_more()) {
        auto fpz = fpzip_write_to_buffer(compress_buffer.data(), compress_buffer.size());
        auto close_fpz = defer([&] { fpzip_write_close(fpz); });

        fpz_set(fpz, meta);
        bench.time_compression([&] {
            compressed_size = fpzip_write(fpz, input_buffer);
        });
        if (compressed_size == 0) {
            throw std::runtime_error("fpzip_write");
        }
    }

    auto decompress_buffer = scratch_buffer{uncompressed_size}; // no bound function, just guess large enough
    while (bench.decompress_more()) {
        auto fpz = fpzip_read_from_buffer(compress_buffer.data());
        auto close_fpz = defer([&] { fpzip_read_close(fpz); });
        fpz_set(fpz, meta);
        size_t result;
        bench.time_decompression([&] {
            result = fpzip_read(fpz, decompress_buffer.data());
        });
        if (result == 0) {
            throw std::runtime_error("fpzip_read");
        }
    }

    assert_buffer_equality(input_buffer, decompress_buffer.data(), uncompressed_size);
    return std::move(bench).result(uncompressed_size, compressed_size);
}
#endif


static benchmark_result benchmark_fpc(const void *input_buffer, const metadata &metadata,
    const benchmark_params &params) {
    if (metadata.data_type != data_type::t_double) {
        throw not_implemented{};
    }

    const auto uncompressed_size = metadata.size_in_bytes();
    const int pred_size = params.tunable;
    auto bench = benchmark{params};

    auto compress_buffer = scratch_buffer{2 * uncompressed_size + 1000}; // no bound function, just guess large enough
    size_t compressed_size;
    while (bench.compress_more()) {
        bench.time_compression([&] {
            compressed_size = FPC_Compress_Memory(input_buffer, uncompressed_size, compress_buffer.data(), pred_size);
        });
        if (compressed_size == 0) {
            throw std::runtime_error("FPC_Compress_Memory");
        }
    }

    auto decompress_buffer = scratch_buffer{uncompressed_size};
    while (bench.decompress_more()) {
        size_t result;
        bench.time_decompression([&] {
            result = FPC_Decompress_Memory(compress_buffer.data(), compressed_size, decompress_buffer.data());
        });
        if (result == 0) {
            throw std::runtime_error("FPC_Decompress_Memory");
        }
    }

    assert_buffer_equality(input_buffer, decompress_buffer.data(), uncompressed_size);
    return std::move(bench).result(uncompressed_size, compressed_size);
}


static benchmark_result benchmark_spdp(const void *input_buffer, const metadata &metadata,
        const benchmark_params &params) {
    const auto uncompressed_size = metadata.size_in_bytes();
    const int pred_size = params.tunable;
    auto bench = benchmark{params};

    auto compress_buffer = scratch_buffer{2 * uncompressed_size + 1000}; // no bound function, just guess large enough
    size_t compressed_size;
    while (bench.compress_more()) {
        bench.time_compression([&] {
            compressed_size = SPDP_Compress_Memory(input_buffer, metadata.size_in_bytes(),
                                                   compress_buffer.data(), pred_size);
        });
        if (compressed_size == 0) {
            throw std::runtime_error("SPDP_Compress_Memory");
        }
    }

    auto decompress_buffer = scratch_buffer{uncompressed_size};
    while (bench.decompress_more()) {
        size_t result;
        bench.time_decompression([&] {
            result = SPDP_Decompress_Memory(compress_buffer.data(), compressed_size, decompress_buffer.data());
        });
        if (result == 0) {
            throw std::runtime_error("SPDP_Decompress_Memory");
        }
    }

    assert_buffer_equality(input_buffer, decompress_buffer.data(), uncompressed_size);
    return std::move(bench).result(uncompressed_size, compressed_size);
}


#if HCDE_BENCHMARK_HAVE_GFC
static benchmark_result benchmark_gfc(const void *input_buffer, const metadata &metadata,
                                      const benchmark_params &params) {
    if (metadata.data_type != data_type::t_double) {
        throw not_implemented{};
    }

    const auto uncompressed_size = metadata.size_in_bytes();
    auto bench = benchmark{params};

    const int blocks = 28;
    const int warps_per_block = 18;
    const int dimensionality = 1;
    GFC_Init();

    auto compress_buffer = scratch_buffer{2 * uncompressed_size + 1000}; // no bound function, just guess large enough
    size_t compressed_size;
    while (bench.compress_more()) {
        uint64_t kernel_time_us;
        compressed_size = GFC_Compress_Memory(input_buffer, metadata.size_in_bytes(), compress_buffer.data(), blocks,
                                              warps_per_block, dimensionality , &kernel_time_us);
        bench.record_compression(std::chrono::microseconds(kernel_time_us));
    }

    auto decompress_buffer = scratch_buffer{uncompressed_size};
    while (bench.decompress_more()) {
        uint64_t kernel_time_us;
        GFC_Decompress_Memory(compress_buffer.data(), compressed_size, decompress_buffer.data(), &kernel_time_us);
        bench.record_decompression(std::chrono::microseconds(kernel_time_us));
    }

    assert_buffer_equality(input_buffer, decompress_buffer.data(), uncompressed_size);
    return std::move(bench).result(uncompressed_size, compressed_size);
}
#endif


#if HCDE_BENCHMARK_HAVE_ZLIB
static benchmark_result benchmark_deflate(const void *input_buffer, const metadata &metadata,
        const benchmark_params &params) {
    const auto uncompressed_size = metadata.size_in_bytes();
    const int level = params.tunable;
    auto bench = benchmark{params};

    auto strm_init = z_stream{};
    strm_init.next_in = static_cast<const Bytef*>(input_buffer);
    strm_init.avail_in = static_cast<uInt>(uncompressed_size);

    size_t output_buffer_size;
    {
        auto strm = strm_init;
        if (deflateInit(&strm, level) != Z_OK) {
            throw std::runtime_error("deflateInit");
        }
        output_buffer_size = deflateBound(&strm_init, strm_init.avail_in);
        deflateEnd(&strm_init);
    }

    auto compress_buffer = scratch_buffer<Bytef>{output_buffer_size}; // no bound function, just guess large enough
    strm_init.next_out = compress_buffer.data();
    strm_init.avail_out = static_cast<uInt>(output_buffer_size);

    size_t compressed_size;
    while (bench.compress_more()) {
        auto strm = strm_init;
        if (deflateInit(&strm, level) != Z_OK) {
            throw std::runtime_error("deflateInit");
        }
        auto end_deflate = defer([&] { deflateEnd(&strm); });

        int result;
        bench.time_compression([&] {
            result = deflate(&strm, Z_SYNC_FLUSH);
        });

        if (result != Z_OK) {
            throw std::runtime_error("deflate");
        }
        compressed_size = strm.total_out;
    }

    auto decompress_buffer = scratch_buffer<Bytef>{uncompressed_size};
    strm_init.next_in = compress_buffer.data();
    strm_init.avail_in = static_cast<uInt>(compressed_size);
    strm_init.next_out = decompress_buffer.data();
    strm_init.avail_out = static_cast<uInt>(uncompressed_size);

    while (bench.decompress_more()) {
        auto strm = strm_init;
        if (inflateInit(&strm) != Z_OK) {
            throw std::runtime_error("inflateInit");
        }
        auto end_inflate = defer([&] { inflateEnd(&strm); });

        int result;
        bench.time_decompression([&] {
            result = inflate(&strm, Z_SYNC_FLUSH);
        });

        if (result != Z_OK) {
            throw std::runtime_error("inflate");
        }
    }

    assert_buffer_equality(input_buffer, decompress_buffer.data(), uncompressed_size);
    return std::move(bench).result(uncompressed_size, compressed_size);
}
#endif


#if HCDE_BENCHMARK_HAVE_LZ4
static benchmark_result benchmark_lz4(const void *input_buffer, const metadata &metadata,
                                      const benchmark_params &params) {
    const auto uncompressed_size = metadata.size_in_bytes();
    auto bench = benchmark{params};

    auto compress_buffer = scratch_buffer<char>{
            static_cast<size_t>(LZ4_compressBound(static_cast<int>(uncompressed_size)))};
    size_t compressed_size;
    while (bench.compress_more()) {
        int result;
        bench.time_compression([&] {
            result = LZ4_compress_default(static_cast<const char *>(input_buffer), compress_buffer.data(),
                                          static_cast<int>(uncompressed_size),
                                          static_cast<int>(compress_buffer.size()));
        });
        if (result == 0) {
            throw std::runtime_error("LZ4_compress_default");
        }
        compressed_size = static_cast<size_t>(result);
    }

    auto decompress_buffer = scratch_buffer<char>{uncompressed_size};
    while (bench.decompress_more()) {
        int result;
        bench.time_decompression([&] {
            result = LZ4_decompress_safe(compress_buffer.data(), decompress_buffer.data(),
                                          static_cast<int>(compressed_size),
                                          static_cast<int>(decompress_buffer.size()));
        });
        if (result == 0) {
            throw std::runtime_error("LZ4_decompress_safe");
        }
    }

    assert_buffer_equality(input_buffer, decompress_buffer.data(), uncompressed_size);
    return std::move(bench).result(uncompressed_size, compressed_size);
}
#endif


#if HCDE_BENCHMARK_HAVE_LZMA
static benchmark_result benchmark_lzma(const void *input_buffer, const metadata &metadata,
                                       const benchmark_params &params) {
    const auto uncompressed_size = metadata.size_in_bytes();
    const int level = params.tunable;
    auto bench = benchmark{params};

    lzma_options_lzma opts;
    lzma_lzma_preset(&opts, static_cast<uint32_t>(level));

    auto compress_buffer = scratch_buffer<uint8_t>{lzma_stream_buffer_bound(uncompressed_size)};
    size_t compressed_size;
    while (bench.compress_more()) {
        lzma_stream strm = LZMA_STREAM_INIT;
        strm.next_in = static_cast<const uint8_t*>(input_buffer);
        strm.avail_in = uncompressed_size;
        strm.next_out = compress_buffer.data();
        strm.avail_out = compress_buffer.size();

        if (lzma_alone_encoder(&strm, &opts) != LZMA_OK) {
            throw std::runtime_error("lzma_alone_encoder");
        }
        auto end_lzma = defer([&] { lzma_end(&strm); });

        bench.time_compression([&] {
            lzma_ret ret = lzma_code(&strm, LZMA_RUN);
            if (ret != LZMA_OK && ret != LZMA_STREAM_END) {
                throw std::runtime_error("llzma_code(LZMA_RUN)");
            }
            for (;;) {
                ret = lzma_code(&strm, LZMA_FINISH);
                if (ret == LZMA_STREAM_END) {
                    break;
                }
                if (ret != LZMA_OK) {
                    throw std::runtime_error("llzma_code(LZMA_FINISH)");
                }
            }
        });

        compressed_size = strm.total_out;
    }

    auto decompress_buffer = scratch_buffer<uint8_t>{uncompressed_size};
    while (bench.decompress_more()) {
        lzma_stream strm = LZMA_STREAM_INIT;
        strm.next_in = compress_buffer.data();
        strm.avail_in = compressed_size;
        strm.next_out = decompress_buffer.data();
        strm.avail_out = decompress_buffer.size();

        if (lzma_alone_decoder(&strm, /* 10GiB mem limit */ 10 * (1ull << 30u)) != LZMA_OK) {
            throw std::runtime_error("lzma_alone_decoder");
        }
        auto end_lzma = defer([&] { lzma_end(&strm); });

        bench.time_decompression([&] {
            lzma_ret ret = lzma_code(&strm, LZMA_RUN);
            if (ret != LZMA_OK && ret != LZMA_STREAM_END) {
                throw std::runtime_error("llzma_code(LZMA_RUN)");
            }
        });
    }

    assert_buffer_equality(input_buffer, decompress_buffer.data(), uncompressed_size);
    return std::move(bench).result(uncompressed_size, compressed_size);
}
#endif


#if HCDE_BENCHMARK_HAVE_ZSTD
static benchmark_result benchmark_zstd(const void *input_buffer, const metadata &metadata,
                                       const benchmark_params &params) {
    const auto uncompressed_size = metadata.size_in_bytes();
    const int level = params.tunable;
    auto bench = benchmark{params};

    auto compress_buffer = scratch_buffer{ZSTD_compressBound(uncompressed_size)};
    size_t compressed_size;
    while (bench.compress_more()) {
        bench.time_compression([&] {
            compressed_size = ZSTD_compress(compress_buffer.data(), compress_buffer.size(), input_buffer,
                                            uncompressed_size, level);
        });
        if (ZSTD_isError(compressed_size)) {
            throw std::runtime_error(std::string{"ZSTD_compress: "} + ZSTD_getErrorName(compressed_size));
        }
    }

    auto decompress_buffer = scratch_buffer{uncompressed_size};
    while (bench.decompress_more()) {
        size_t result;
        bench.time_decompression([&] {
            result = ZSTD_decompress(decompress_buffer.data(), decompress_buffer.size(), compress_buffer.data(),
                            compressed_size);
        });
        if (ZSTD_isError(result)) {
            throw std::runtime_error(std::string{"ZSTD_decompress: "} + ZSTD_getErrorName(result));
        }
    }

    assert_buffer_equality(input_buffer, decompress_buffer.data(), uncompressed_size);
    return std::move(bench).result(uncompressed_size, compressed_size);
}
#endif


struct algorithm {
    std::function<benchmark_result(const void *, const metadata&, benchmark_params)> benchmark;
    int tunable_min = 1;
    int tunable_max = 1;
};

using algorithm_map = std::unordered_map<std::string, algorithm>;

const algorithm_map &available_algorithms() {
    using namespace std::placeholders;

    static const algorithm_map algorithms{
        {"hcde", {benchmark_hcde<hcde::cpu_encoder>}},
#if HCDE_OPENMP_SUPPORT
        {"hcde-mt", {benchmark_hcde<hcde::mt_cpu_encoder>}},
#endif
#if HCDE_GPU_SUPPORT
        // {"hcde-gpu", benchmark_hcde<hcde::gpu_encoder>},
#endif
#if HCDE_BENCHMARK_HAVE_FPZIP
        {"fpzip", {benchmark_fpzip}},
#endif
        {"fpc", {benchmark_fpc, 10, 20}},
        {"spdp", {benchmark_spdp, 1, 9}},
#if HCDE_BENCHMARK_HAVE_GFC
        {"gfc", {benchmark_gfc}},
#endif
#if HCDE_BENCHMARK_HAVE_ZLIB
        {"deflate", {benchmark_deflate, 1, 9}},
#endif
#if HCDE_BENCHMARK_HAVE_LZ4
        {"lz4", {benchmark_lz4}},
#endif
#if HCDE_BENCHMARK_HAVE_ZSTD
        {"zstd", {benchmark_zstd, 1, 19}},
#endif
#if HCDE_BENCHMARK_HAVE_LZMA
        {"lzma", {benchmark_lzma, 1, 9}},
#endif
    };

    return algorithms;
}


struct identity {
    template<typename T>
    decltype(auto) operator()(T &&v) const { return std::forward<T>(v); }
};


template<typename Seq, typename Map = identity>
struct join {
    const char *joiner;
    Seq &seq;
    Map map;

    join(const char *joiner, Seq &seq, Map map = identity{})
        : joiner(joiner), seq(seq), map(map) {}

    friend std::ostream &operator<<(std::ostream &os, const join &j) {
        size_t i = 0;
        for (auto &v: j.seq) {
            if (i > 0) {
                os << j.joiner;
            }
            os << j.map(v);
            ++i;
        }
        return os;
    }
};


static void benchmark_file(const metadata &metadata, const algorithm_map &algorithms,
     std::chrono::microseconds min_time, unsigned min_reps, tuning tunables,
     const hcde::detail::io_factory &io_factory) {

    auto input_stream = io_factory.create_input_stream(metadata.path.string(), metadata.size_in_bytes());
    auto input_buffer = input_stream->read_exact();

    for (auto &[name, algo]: algorithms) {
        std::vector<int> tunable_values;
        if (algo.tunable_min == algo.tunable_max || tunables == tuning::min) {
            tunable_values = {algo.tunable_min};
        } else if (tunables == tuning::min_max) {
            tunable_values = {algo.tunable_min, algo.tunable_max};
        } else {
            tunable_values.resize(algo.tunable_max - algo.tunable_min + 1);
            std::iota(tunable_values.begin(), tunable_values.end(), algo.tunable_min);
        }

        for (auto tunable: tunable_values) {
            auto params = benchmark_params{tunable, min_time, min_reps};

            benchmark_result result;
            try {
                result = algo.benchmark(input_buffer, metadata, params);
            } catch (not_implemented &) {
                continue;
            } catch (buffer_mismatch &) {
                std::ostringstream msg;
                msg << "mismatch between input and decompressed buffer for " << metadata.path.filename().string()
                << " with " <<  name << " (tunable=" << tunable << ")";
                throw std::logic_error(msg.str());
            }
            std::cout << metadata.path.filename().string() << ";"
                      << (metadata.data_type == data_type::t_float ? "float" : "double") << ";"
                      << metadata.extent.size() << ";"
                      << name << ";"
                      << tunable << ";"
                      << join(",", result.compression_times, [](auto d) { return d.count(); }) << ";"
                      << join(",", result.decompression_times, [](auto d) { return d.count(); }) << ";"
                      << result.uncompressed_bytes << ";"
                      << result.compressed_bytes << "\n";
        }
    }
}


static std::string available_algorithms_string() {
    std::string algos;
    for (auto &[name, _]: available_algorithms()) {
        if (!algos.empty()) {
            algos.push_back(' ');
        }
        algos += name;
    }
    return algos;
}


static void print_library_versions() {
#if HCDE_BENCHMARK_HAVE_ZLIB
    printf("zlib version %s\n", zlibVersion());
#endif
#if HCDE_BENCHMARK_HAVE_LZ4
    printf("LZ4 version %s\n", LZ4_versionString());
#endif
#if HCDE_BENCHMARK_HAVE_LZMA
    printf("LZMA version %s\n", lzma_version_string());
#endif
#if HCDE_BENCHMARK_HAVE_FPZIP
    printf("%s\n", fpzip_version_string);
#endif
#if HCDE_BENCHMARK_HAVE_GFC
    printf("GFC %s\n", GFC_Version_String);
#endif
}


int main(int argc, char **argv) {
    namespace opts = boost::program_options;
    using namespace std::string_literals;

    std::string metadata_csv_file;
    std::vector<std::string> include_algorithms;
    std::vector<std::string> exclude_algorithms;
    tuning tunables = tuning::min;
    unsigned benchmark_ms = 1000;
    unsigned benchmark_reps = 3;
    bool no_mmap = false;

    auto usage = "Usage: "s + argv[0] + " [options] csv-file\n\n";

    opts::options_description desc("Options");
    desc.add_options()
            ("help", "show this help")
            ("version", "show library versions")
            ("csv-file", opts::value(&metadata_csv_file)->required(), "csv file with benchmark file metadata")
            ("algorithms,a", opts::value(&include_algorithms)->multitoken(), "algorithms to evaluate (see --help)")
            ("skip-algorithms,A", opts::value(&exclude_algorithms)->multitoken(),
                "algorithms to NOT evaluate (see --help)")
            ("time-each,t", opts::value(&benchmark_ms), "repeat each for at least t ms (default 1000)")
            ("reps-each,r", opts::value(&benchmark_reps), "repeat each at least n times (default 3)")
            ("tunables", opts::value<std::string>(), "tunables min|minmax|full (default min)")
            ("no-mmap", opts::bool_switch(&no_mmap), "do not use memory-mapped I/O");
    opts::positional_options_description pos_desc;
    pos_desc.add("csv-file", 1);

    try {
        auto parsed = opts::command_line_parser(argc, argv).options(desc).positional(pos_desc).run();

        opts::variables_map vars;
        opts::store(parsed, vars);

        if (vars.count("help")) {
            std::cout << "Benchmark compression algorithms on float data\n\n" << usage << desc
                      << "\nAvailable algorithms: " << available_algorithms_string() << "\n";
            return EXIT_SUCCESS;
        }

        if (vars.count("version")) {
            print_library_versions();
            return EXIT_SUCCESS;
        }

        if (vars.count("tunables")) {
            auto &str = vars["tunables"].as<std::string>();
            if (str == "min") {
                tunables = tuning::min;
            } else if (str == "minmax") {
                tunables = tuning::min_max;
            } else if (str == "full") {
                tunables = tuning::full;
            } else {
                throw boost::program_options::invalid_option_value("--tunables must be min, minmax or full");
            }
        }

        opts::notify(vars);
    } catch (opts::error &e) {
        std::cerr << e.what() << "\n\n" << usage << desc;
        return EXIT_FAILURE;
    }

    algorithm_map selected_algorithms;
    if (!include_algorithms.empty()) {
        for (auto &name: include_algorithms) {
            if (auto iter = available_algorithms().find(name); iter != available_algorithms().end()) {
                selected_algorithms.insert(*iter);
            } else {
                std::cerr << "Unknown algorithm \"" << name << "\".\nAvailable algorithms are: "
                        << available_algorithms_string() << "\n";
                return EXIT_FAILURE;
            }
        }
    } else {
        selected_algorithms = available_algorithms();
    }
    for (auto &name: exclude_algorithms) {
        if (auto iter = selected_algorithms.find(name); iter != selected_algorithms.end()) {
            selected_algorithms.erase(iter);
        }
    }

    std::unique_ptr<hcde::detail::io_factory> io_factory;
#if HCDE_SUPPORT_MMAP
    if (!no_mmap) {
        io_factory = std::make_unique<hcde::detail::mmap_io_factory>();
    }
#endif
    if (!io_factory) {
        io_factory = std::make_unique<hcde::detail::stdio_io_factory>();
    }

    try {
        std::cout << "dataset;data type;dimensions;algorithm;tunable;"
                     "compression times (microseconds);"
                     "decompression times (microseconds);"
                     "uncompressed bytes;compressed bytes\n";
        for (auto &metadata : load_metadata_file(metadata_csv_file)) {
            benchmark_file(metadata, selected_algorithms, std::chrono::milliseconds(benchmark_ms),
                benchmark_reps, tunables, *io_factory);
        }
        return EXIT_SUCCESS;
    } catch (std::exception &e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
