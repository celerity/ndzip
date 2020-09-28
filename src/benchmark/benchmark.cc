#include <hcde/hcde.hh>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

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
#if HCDE_BENCHMARK_HAVE_FPZIP
#   include <fpzip.h>
#endif
#include <fpc.h>
#include <SPDP_11.h>
#if HCDE_BENCHMARK_HAVE_GFC
#   include <GFC_22.h>
#endif

#include <boost/program_options.hpp>


enum class data_type {
    t_float,
    t_double,
};

struct metadata {
    std::filesystem::path path;
    ::data_type data_type;
    std::vector<size_t> extent;

    metadata(std::filesystem::path path, ::data_type type, std::vector<size_t> extent)
        : path(std::move(path)), data_type(type), extent(std::move(extent)) {
    }

    size_t size_in_bytes() const {
        auto size = data_type == data_type::t_float ? 4 : 8;
        for (auto e : extent) {
            size *= e;
        }
        return size;
    }
};


struct malloc_deleter {
    void operator()(void *p) const {
        free(p);
    }
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


// memset the output buffer before calling the compression function to ensure all pages of the allocated buffer
// have been mapped by the OS.
[[gnu::noinline]]
static void memzero_noinline(void *mem, size_t n_bytes) {
    memset(mem, 0, n_bytes);
}


struct benchmark_result {
    std::chrono::microseconds duration;
    uint64_t uncompressed_bytes;
    uint64_t compressed_bytes;
};


class not_implemented: std::exception {};


template<template<typename, unsigned> typename Encoder, typename Data, unsigned Dims>
static benchmark_result benchmark_hcde_3(const Data *input_buffer, const hcde::extent<Dims> &size,
        std::chrono::milliseconds benchmark_time, unsigned benchmark_reps) {
    auto input_size = hcde::num_elements(size) * sizeof(Data);
    auto input_slice = hcde::slice<const Data, Dims>(input_buffer, size);

    Encoder<Data, Dims> e;
    auto output_buffer_size = hcde::compressed_size_bound<Data>(size);
    auto output_buffer = std::unique_ptr<std::byte, malloc_deleter>(
            static_cast<std::byte*>(malloc(output_buffer_size)));
    memzero_noinline(output_buffer.get(), output_buffer_size);

    std::chrono::steady_clock::duration min_time{std::chrono::hours(1000)};
    std::chrono::steady_clock::duration cum_time{};
    size_t compressed_size;
    for (unsigned i = 0; cum_time < benchmark_time || i < benchmark_reps; ++i) {
        auto run_start = std::chrono::steady_clock::now();
        compressed_size = e.compress(input_slice, output_buffer.get());
        auto run_time = std::chrono::steady_clock::now() - run_start;
        min_time = std::min(min_time, run_time);
        cum_time += run_time;
    }

    return {std::chrono::duration_cast<std::chrono::microseconds>(min_time), input_size, compressed_size};
}


template<template<typename, unsigned> typename Encoder, typename Data>
static benchmark_result benchmark_hcde_2(const Data *input_buffer, const metadata &metadata,
        std::chrono::milliseconds benchmark_time, unsigned benchmark_reps) {
    auto &e = metadata.extent;
    if (e.size() == 1) {
        return benchmark_hcde_3<Encoder, Data, 1>(input_buffer, hcde::extent{e[0]}, benchmark_time, benchmark_reps);
    } else if (e.size() == 2) {
        return benchmark_hcde_3<Encoder, Data, 2>(input_buffer, hcde::extent{e[0], e[1]}, benchmark_time,
            benchmark_reps);
    } else if (e.size() == 3) {
        return benchmark_hcde_3<Encoder, Data, 3>(input_buffer, hcde::extent{e[0], e[1], e[2]}, benchmark_time,
            benchmark_reps);
    } else {
        throw not_implemented{};
    }
}


template<template<typename, unsigned> typename Encoder>
static benchmark_result benchmark_hcde(const void *input_buffer, const metadata &metadata,
        std::chrono::milliseconds benchmark_time, unsigned benchmark_reps) {
    if (metadata.data_type == data_type::t_float) {
        return benchmark_hcde_2<Encoder, float>(static_cast<const float*>(input_buffer), metadata, benchmark_time,
            benchmark_reps);
    } else {
        return benchmark_hcde_2<Encoder, double>(static_cast<const double*>(input_buffer), metadata, benchmark_time,
            benchmark_reps);
    }
}


#if HCDE_BENCHMARK_HAVE_FPZIP
static benchmark_result benchmark_fpzip(const void *input_buffer, const metadata &metadata,
        std::chrono::milliseconds benchmark_time, unsigned benchmark_reps) {
    auto input_size = metadata.size_in_bytes();

    auto output_buffer_size = 2*input_size + 1000; // fpzip has no bound function, just guess large enough
    auto output_buffer = std::unique_ptr<Bytef, malloc_deleter>(
        static_cast<Bytef*>(malloc(output_buffer_size)));
    memzero_noinline(output_buffer.get(), output_buffer_size);

    std::chrono::steady_clock::duration cum_time{};
    std::chrono::steady_clock::duration min_time{std::chrono::hours(1000)};
    size_t compressed_size;
    for (unsigned i = 0; cum_time < benchmark_time || i < benchmark_reps; ++i) {
        std::unique_ptr<FPZ, decltype((fpzip_write_close))> fpz(
            fpzip_write_to_buffer(output_buffer.get(), output_buffer_size), fpzip_write_close);

        fpz->type = metadata.data_type == data_type::t_float ? 0 : 1;
        fpz->prec = 0; // lossless
        auto &e = metadata.extent;
        fpz->nx = e.size() >= 1 ? static_cast<int>(e[e.size() - 1]) : 1;
        fpz->ny = e.size() >= 2 ? static_cast<int>(e[e.size() - 2]) : 1;
        fpz->nz = e.size() >= 3 ? static_cast<int>(e[e.size() - 3]) : 1;
        fpz->nf = e.size() >= 4 ? static_cast<int>(e[e.size() - 4]) : 1;

        auto run_start = std::chrono::steady_clock::now();
        auto result = fpzip_write(fpz.get(), input_buffer);
        auto run_time = std::chrono::steady_clock::now() - run_start;

        if (result == 0) {
            throw std::runtime_error("fpzip_write");
        }
        compressed_size = result;
        min_time = std::min(min_time, run_time);
        cum_time += run_time;
    }

    return { std::chrono::duration_cast<std::chrono::microseconds>(min_time), input_size, compressed_size};
}
#endif


static benchmark_result benchmark_fpc(const void *input_buffer, const metadata &metadata, int pred_size,
        std::chrono::milliseconds benchmark_time, unsigned benchmark_reps) {
    if (metadata.data_type != data_type::t_double) {
        throw not_implemented{};
    }

    auto input_size = metadata.size_in_bytes();

    auto output_buffer_size = 2*input_size + 1000; // fpc has no bound function, just guess large enough
    auto output_buffer = std::unique_ptr<Bytef, malloc_deleter>(
        static_cast<Bytef*>(malloc(output_buffer_size)));
    memzero_noinline(output_buffer.get(), output_buffer_size);

    std::chrono::steady_clock::duration cum_time{};
    std::chrono::steady_clock::duration min_time{std::chrono::hours(1000)};
    size_t compressed_size;
    for (unsigned i = 0; cum_time < benchmark_time || i < benchmark_reps; ++i) {
        auto run_start = std::chrono::steady_clock::now();
        auto result = FPC_Compress_Memory(input_buffer, metadata.size_in_bytes(), output_buffer.get(), pred_size);
        auto run_time = std::chrono::steady_clock::now() - run_start;

        if (result == 0) {
            throw std::runtime_error("fpzip_write");
        }
        compressed_size = result;
        min_time = std::min(min_time, run_time);
        cum_time += run_time;
    }

    return { std::chrono::duration_cast<std::chrono::microseconds>(min_time), input_size, compressed_size };
}


static benchmark_result benchmark_spdp(const void *input_buffer, const metadata &metadata, int pred_size,
        std::chrono::milliseconds benchmark_time, unsigned benchmark_reps) {
    auto input_size = metadata.size_in_bytes();

    auto output_buffer_size = 2*input_size + 1000; // spdp has no bound function, just guess large enough
    auto output_buffer = std::unique_ptr<Bytef, malloc_deleter>(
        static_cast<Bytef*>(malloc(output_buffer_size)));
    memzero_noinline(output_buffer.get(), output_buffer_size);

    std::chrono::steady_clock::duration cum_time{};
    std::chrono::steady_clock::duration min_time{std::chrono::hours(1000)};
    size_t compressed_size;
    for (unsigned i = 0; cum_time < benchmark_time || i < benchmark_reps; ++i) {
        auto run_start = std::chrono::steady_clock::now();
        auto result = SPDP_Compress_Memory(input_buffer, metadata.size_in_bytes(), output_buffer.get(), pred_size);
        auto run_time = std::chrono::steady_clock::now() - run_start;

        if (result == 0) {
            throw std::runtime_error("fpzip_write");
        }
        compressed_size = result;
        min_time = std::min(min_time, run_time);
        cum_time += run_time;
    }

    return { std::chrono::duration_cast<std::chrono::microseconds>(min_time), input_size, compressed_size };
}


#if HCDE_BENCHMARK_HAVE_GFC
static benchmark_result benchmark_gfc(const void *input_buffer, const metadata &metadata,
        std::chrono::milliseconds benchmark_time, unsigned benchmark_reps) {
    if (metadata.data_type != data_type::t_double) {
        throw not_implemented{};
    }

    auto input_size = metadata.size_in_bytes();

    auto output_buffer_size = 2*input_size + 1000; // spdp has no bound function, just guess large enough
    auto output_buffer = std::unique_ptr<Bytef, malloc_deleter>(
            static_cast<Bytef*>(malloc(output_buffer_size)));
    memzero_noinline(output_buffer.get(), output_buffer_size);

    GFC_Init();

    std::chrono::microseconds cum_time{};
    std::chrono::microseconds min_time{std::chrono::hours(1000)};
    size_t compressed_size;
    for (unsigned i = 0; cum_time < benchmark_time || i < benchmark_reps; ++i) {
        uint64_t kernel_time_us;
        int blocks = 28;
        int warps_per_block = 18;
        int dimensionality = 1;

        compressed_size = GFC_Compress_Memory(input_buffer, metadata.size_in_bytes(), output_buffer.get(), blocks,
            warps_per_block, dimensionality , &kernel_time_us);

        auto run_time = std::chrono::microseconds(kernel_time_us);
        min_time = std::min(min_time, run_time);
        cum_time += run_time;
    }

    return { min_time, input_size, compressed_size };
}
#endif


#if HCDE_BENCHMARK_HAVE_ZLIB
static benchmark_result benchmark_deflate(const void *input_buffer, const metadata &metadata, int level,
        std::chrono::milliseconds benchmark_time, unsigned benchmark_reps) {
    class deflate_end_guard {
        public:
            explicit deflate_end_guard(z_streamp p) : _p(p) {}

            ~deflate_end_guard() {
                deflateEnd(_p);
            }

        private:
            z_streamp _p;
    };

    auto input_size = metadata.size_in_bytes();

    z_stream strm_init {
        .next_in = static_cast<const Bytef*>(input_buffer),
        .avail_in = static_cast<uInt>(input_size),
        .zalloc = nullptr,
        .zfree = nullptr,
        .opaque = nullptr,
        .data_type = Z_BINARY,
    };

    size_t output_buffer_size;
    {
        auto strm = strm_init;
        if (deflateInit(&strm, level) != Z_OK) {
            throw std::runtime_error("deflateInit");
        }
        deflate_end_guard guard(&strm);

        output_buffer_size = deflateBound(&strm_init, strm_init.avail_in);
    }

    auto output_buffer = std::unique_ptr<Bytef, malloc_deleter>(
            static_cast<Bytef*>(malloc(output_buffer_size)));
    memzero_noinline(output_buffer.get(), output_buffer_size);
    strm_init.next_out = output_buffer.get(),
    strm_init.avail_out = static_cast<uInt>(output_buffer_size);

    std::chrono::steady_clock::duration cum_time{};
    std::chrono::steady_clock::duration min_time{std::chrono::hours(1000)};
    size_t compressed_size;
    for (unsigned i = 0; cum_time < benchmark_time || i < benchmark_reps; ++i) {
        auto strm = strm_init;
        if (deflateInit(&strm, level) != Z_OK) {
            throw std::runtime_error("deflateInit");
        }
        deflate_end_guard guard(&strm);

        auto run_start = std::chrono::steady_clock::now();
        if (deflate(&strm, Z_SYNC_FLUSH) != Z_OK) {
            throw std::runtime_error("deflate");
        }
        auto run_time = std::chrono::steady_clock::now() - run_start;

        compressed_size = strm.total_out;
        min_time = std::min(min_time, run_time);
        cum_time += run_time;
    }

    return {std::chrono::duration_cast<std::chrono::microseconds>(min_time), input_size, compressed_size};
}
#endif


#if HCDE_BENCHMARK_HAVE_LZ4
static benchmark_result benchmark_lz4(const void *input_buffer, const metadata &metadata,
        std::chrono::milliseconds benchmark_time, unsigned benchmark_reps) {
    auto input_size = metadata.size_in_bytes();
    auto output_buffer_size = static_cast<size_t>(LZ4_compressBound(static_cast<int>(input_size)));
    auto output_buffer = std::unique_ptr<char, malloc_deleter>(
            static_cast<char*>(malloc(output_buffer_size)));
    memzero_noinline(output_buffer.get(), output_buffer_size);

    std::chrono::steady_clock::duration min_time{std::chrono::hours(1000)};
    std::chrono::steady_clock::duration cum_time{};
    size_t compressed_size;
    for (unsigned i = 0; cum_time < benchmark_time || i < benchmark_reps; ++i) {
        auto run_start = std::chrono::steady_clock::now();
        int result = LZ4_compress_default(static_cast<const char*>(input_buffer), output_buffer.get(),
                static_cast<int>(input_size), static_cast<int>(output_buffer_size));
        auto run_time = std::chrono::steady_clock::now() - run_start;

        if (result == 0) {
            throw std::runtime_error("LZ4_compress_default");
        }
        compressed_size = static_cast<size_t>(result);
        min_time = std::min(min_time, run_time);
        cum_time += run_time;
    }

    return { std::chrono::duration_cast<std::chrono::microseconds>(min_time), input_size, compressed_size};
}
#endif


#if HCDE_BENCHMARK_HAVE_LZMA
static benchmark_result benchmark_lzma(const void *input_buffer, const metadata &metadata, int level,
        std::chrono::milliseconds benchmark_time, unsigned benchmark_reps) {
    class lzma_end_guard {
       public:
        explicit lzma_end_guard(lzma_stream *strm) : _strm(strm) {}

        ~lzma_end_guard() {
            lzma_end(_strm);
        }

       private:
        lzma_stream *_strm;
    };

    auto input_size = metadata.size_in_bytes();
    auto output_buffer_size = static_cast<size_t>(lzma_stream_buffer_bound(input_size));
    auto output_buffer = std::unique_ptr<uint8_t, malloc_deleter>(
            static_cast<uint8_t*>(malloc(output_buffer_size)));
    memzero_noinline(output_buffer.get(), output_buffer_size);

    lzma_options_lzma opts;
    lzma_lzma_preset(&opts, static_cast<uint32_t>(level));

    lzma_stream strm_template = LZMA_STREAM_INIT;
    strm_template.next_in = static_cast<const uint8_t*>(input_buffer);
    strm_template.avail_in = input_size;
    strm_template.next_out = output_buffer.get();
    strm_template.avail_out = output_buffer_size;

    std::chrono::steady_clock::duration min_time{std::chrono::hours(1000)};
    std::chrono::steady_clock::duration cum_time{};
    size_t compressed_size;
    for (unsigned i = 0; cum_time < benchmark_time || i < benchmark_reps; ++i) {
        auto strm = strm_template;
        if (lzma_alone_encoder(&strm, &opts) != LZMA_OK) {
            throw std::runtime_error("lzma_alone_encoder");
        }
        lzma_end_guard guard(&strm);

        auto run_start = std::chrono::steady_clock::now();
        if (lzma_code(&strm, LZMA_RUN) != LZMA_OK) {
            throw std::runtime_error("llzma_code(LZMA_RUN)");
        }
        for (;;) {
            auto result = lzma_code(&strm, LZMA_FINISH);
            if (result == LZMA_STREAM_END) {
                break;
            }
            if (result != LZMA_OK) {
                throw std::runtime_error("llzma_code(LZMA_FINISH)");
            }
        }
        auto run_time = std::chrono::steady_clock::now() - run_start;

        compressed_size = strm.total_out;
        min_time = std::min(min_time, run_time);
        cum_time += run_time;
    }

    return {std::chrono::duration_cast<std::chrono::microseconds>(min_time), input_size, compressed_size};
}
#endif

using algorithm =  std::function<benchmark_result(const void *, const metadata&, std::chrono::milliseconds, unsigned)>;
using algorithm_map = std::unordered_map<std::string, algorithm>;

const algorithm_map &available_algorithms() {
    using namespace std::placeholders;

    static const algorithm_map algorithms{
        {"hcde/cpu", benchmark_hcde<hcde::cpu_encoder>},
#if HCDE_OPENMP_SUPPORT
        {"hcde/cpu-mt", benchmark_hcde<hcde::mt_cpu_encoder>},
#endif
#if HCDE_GPU_SUPPORT
        {"hcde/gpu", benchmark_hcde<hcde::gpu_encoder>},
#endif
#if HCDE_BENCHMARK_HAVE_FPZIP
        {"fpzip", benchmark_fpzip},
#endif
        {"fpc/10", std::bind(benchmark_fpc, _1, _2, 10, _3, _4)},
        {"fpc/20", std::bind(benchmark_fpc, _1, _2, 20, _3, _4)},
        {"spdp/1", std::bind(benchmark_spdp, _1, _2, 1, _3, _4)},
        {"spdp/9", std::bind(benchmark_spdp, _1, _2, 9, _3, _4)},
#if HCDE_BENCHMARK_HAVE_GFC
        {"gfc", benchmark_gfc},
#endif
#if HCDE_BENCHMARK_HAVE_ZLIB
        {"deflate/1", std::bind(benchmark_deflate, _1, _2, 1, _3, _4)},
        {"deflate/9", std::bind(benchmark_deflate, _1, _2, 9, _3, _4)},
#endif
#if HCDE_BENCHMARK_HAVE_LZ4
        {"lz4", benchmark_lz4},
#endif
#if HCDE_BENCHMARK_HAVE_LZMA
        {"lzma/1", std::bind(benchmark_lzma, _1, _2, 1, _3, _4)},
        {"lzma/9", std::bind(benchmark_lzma, _1, _2, 9, _3, _4)},
#endif
    };

    return algorithms;
}


static void benchmark_file(const metadata &metadata, const algorithm_map &algorithms,
        std::chrono::milliseconds benchmark_time, unsigned benchmark_reps) {
    auto input_buffer_size = metadata.size_in_bytes();
    auto input_buffer = std::unique_ptr<std::byte, malloc_deleter>(
            static_cast<std::byte*>(malloc(input_buffer_size)));

    auto file_name = metadata.path.string();
    auto input_file = std::unique_ptr<FILE, decltype((fclose))>(fopen(file_name.c_str(), "rb"), fclose);
    if (!input_file) {
        throw std::runtime_error(file_name + ": " + strerror(errno));
    }

    if (fread(input_buffer.get(), input_buffer_size, 1, input_file.get()) != 1) {
        throw std::runtime_error(file_name + ": " + strerror(errno));
    }

    for (auto &[algo_name, benchmark_algo]: algorithms) {
        benchmark_result result;
        try {
            result = benchmark_algo(input_buffer.get(), metadata, benchmark_time, benchmark_reps);
        } catch (not_implemented &) {
            continue;
        }
        std::cout << metadata.path.filename().string() << ";"
                << algo_name << ";"
                << std::chrono::duration_cast<std::chrono::duration<double>>(result.duration).count() << ";"
                << result.uncompressed_bytes << ";"
                << result.compressed_bytes << "\n";
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
    unsigned benchmark_ms = 1000;
    unsigned benchmark_reps = 2;

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
            ("reps-each,r", opts::value(&benchmark_reps), "repeat each at least n times (default 3)");
    opts::positional_options_description pos_desc;
    pos_desc.add("csv-file", 1);

    opts::variables_map vars;
    try {
        auto parsed = opts::command_line_parser(argc, argv).options(desc).positional(pos_desc).run();
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

    try {
        std::cout << "dataset;algorithm;fastest time (seconds);uncompressed bytes;compressed bytes\n";
        std::cout.precision(9);
        std::cout.setf(std::ios::fixed);
        for (auto &metadata : load_metadata_file(metadata_csv_file)) {
            benchmark_file(metadata, selected_algorithms, std::chrono::milliseconds(benchmark_ms), benchmark_reps);
        }
        return EXIT_SUCCESS;
    } catch (std::exception &e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
