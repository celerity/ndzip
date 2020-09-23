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
        if (n_tokens >= 3 && n_tokens <= 5 && type_string == "float"sv || type_string == "double"sv) {
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
void memzero_noinline(void *mem, size_t n_bytes) {
    memset(mem, 0, n_bytes);
}


struct benchmark_result {
    std::chrono::microseconds duration;
    double compression_ratio;
};


template<template<typename, unsigned> typename Encoder, typename Data, unsigned Dims>
static benchmark_result benchmark_hcde_3(const Data *input_buffer, const hcde::extent<Dims> &size,
        std::chrono::milliseconds benchmark_time) {
    auto input_size = hcde::num_elements(size) * sizeof(Data);
    auto input_slice = hcde::slice<const Data, Dims>(input_buffer, size);

    Encoder<Data, Dims> e;
    auto output_buffer_size = e.compressed_size_bound(size);
    auto output_buffer = std::unique_ptr<std::byte, malloc_deleter>(
            static_cast<std::byte*>(malloc(output_buffer_size)));

    std::chrono::microseconds cum_time{};
    auto n_samples = size_t{0};
    size_t compressed_size;
    while (cum_time < benchmark_time) {
        memzero_noinline(output_buffer.get(), output_buffer_size);
        auto start = std::chrono::steady_clock::now();
        compressed_size = e.compress(input_slice, output_buffer.get());
        auto end = std::chrono::steady_clock::now();
        cum_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        ++n_samples;
    }
    return {cum_time / n_samples, static_cast<double>(compressed_size) / input_size};
}


template<template<typename, unsigned> typename Encoder, typename Data>
static benchmark_result benchmark_hcde_2(const Data *input_buffer, const metadata &metadata,
        std::chrono::milliseconds benchmark_time) {
    auto &e = metadata.extent;
    if (e.size() == 1) {
        return benchmark_hcde_3<Encoder, Data, 1>(input_buffer, hcde::extent{e[0]}, benchmark_time);
    } else if (e.size() == 2) {
        return benchmark_hcde_3<Encoder, Data, 2>(input_buffer, hcde::extent{e[0], e[1]}, benchmark_time);
    } else if (e.size() == 3) {
        return benchmark_hcde_3<Encoder, Data, 3>(input_buffer, hcde::extent{e[0], e[1], e[2]}, benchmark_time);
    } else {
        std::abort(); // should be unreachable
    }
}


template<template<typename, unsigned> typename Encoder>
static benchmark_result benchmark_hcde(const void *input_buffer, const metadata &metadata,
        std::chrono::milliseconds benchmark_time) {
    if (metadata.data_type == data_type::t_float) {
        return benchmark_hcde_2<Encoder, float>(static_cast<const float*>(input_buffer), metadata, benchmark_time);
    } else {
        return benchmark_hcde_2<Encoder, double>(static_cast<const double*>(input_buffer), metadata, benchmark_time);
    }
}


#if HCDE_BENCHMARK_HAVE_FPZIP
static benchmark_result benchmark_fpzip(const void *input_buffer, const metadata &metadata,
        std::chrono::milliseconds benchmark_time) {
    auto input_size = metadata.size_in_bytes();

    auto output_buffer_size = 2*input_size; // fpzip has no bound function, just guess large enough
    auto output_buffer = std::unique_ptr<Bytef, malloc_deleter>(
            static_cast<Bytef*>(malloc(output_buffer_size)));
    memzero_noinline(output_buffer.get(), output_buffer_size);

    std::chrono::microseconds cum_time{};
    auto n_samples = size_t{0};
    size_t compressed_size;
    while (cum_time < benchmark_time) {
        std::unique_ptr<FPZ, decltype((fpzip_write_close))> fpz(
                fpzip_write_to_buffer(output_buffer.get(), output_buffer_size), fpzip_write_close);

        fpz->type = metadata.data_type == data_type::t_float ? 0 : 1;
        fpz->prec = 0; // lossless
        auto &e = metadata.extent;
        fpz->nx = e.size() >= 1 ? static_cast<int>(e[e.size() - 1]) : 1;
        fpz->ny = e.size() >= 2 ? static_cast<int>(e[e.size() - 2]) : 1;
        fpz->nz = e.size() >= 3 ? static_cast<int>(e[e.size() - 3]) : 1;
        fpz->nf = e.size() >= 4 ? static_cast<int>(e[e.size() - 4]) : 1;

        auto start = std::chrono::steady_clock::now();
        auto result = fpzip_write(fpz.get(), input_buffer);
        auto end = std::chrono::steady_clock::now();

        if (result == 0) {
            throw std::runtime_error("fpzip_write");
        }
        compressed_size = result;
        cum_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        ++n_samples;
    }

    return {cum_time / n_samples, static_cast<double>(compressed_size) / input_size};
}
#endif


#if HCDE_BENCHMARK_HAVE_ZLIB
static benchmark_result benchmark_deflate(const void *input_buffer, const metadata &metadata, int level,
        std::chrono::milliseconds benchmark_time) {
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

    std::chrono::microseconds cum_time{};
    auto n_samples = size_t{0};
    size_t compressed_size;
    while (cum_time < benchmark_time) {
        auto strm = strm_init;
        if (deflateInit(&strm, /* level */ 9) != Z_OK) {
            throw std::runtime_error("deflateInit");
        }
        deflate_end_guard guard(&strm);

        auto start = std::chrono::steady_clock::now();
        if (deflate(&strm, Z_SYNC_FLUSH) != Z_OK) {
            throw std::runtime_error("deflate");
        }
        auto end = std::chrono::steady_clock::now();

        compressed_size = strm.total_out;
        cum_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        ++n_samples;
    }

    return {cum_time / n_samples, static_cast<double>(compressed_size) / input_size};
}
#endif


#if HCDE_BENCHMARK_HAVE_LZ4
static benchmark_result benchmark_lz4(const void *input_buffer, const metadata &metadata,
        std::chrono::milliseconds benchmark_time) {
    auto input_size = metadata.size_in_bytes();
    auto output_buffer_size = static_cast<size_t>(LZ4_compressBound(static_cast<int>(input_size)));
    auto output_buffer = std::unique_ptr<char, malloc_deleter>(
            static_cast<char*>(malloc(output_buffer_size)));
    memzero_noinline(output_buffer.get(), output_buffer_size);

    std::chrono::microseconds cum_time{};
    auto n_samples = size_t{0};
    size_t compressed_size;
    while (cum_time < benchmark_time) {
        auto start = std::chrono::steady_clock::now();
        int result = LZ4_compress_default(static_cast<const char*>(input_buffer), output_buffer.get(),
                static_cast<int>(input_size), static_cast<int>(output_buffer_size));
        auto end = std::chrono::steady_clock::now();

        if (result == 0) {
            throw std::runtime_error("LZ4_compress_default");
        }
        compressed_size = static_cast<size_t>(result);
        cum_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        ++n_samples;
    }

    return {cum_time / n_samples, static_cast<double>(compressed_size) / input_size};
}
#endif


#if HCDE_BENCHMARK_HAVE_LZMA
static benchmark_result benchmark_lzma(const void *input_buffer, const metadata &metadata, int level,
        std::chrono::milliseconds benchmark_time) {
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

    std::chrono::microseconds cum_time{};
    auto n_samples = size_t{0};
    size_t compressed_size;
    while (cum_time < benchmark_time) {
        auto strm = strm_template;
        if (lzma_alone_encoder(&strm, &opts) != LZMA_OK) {
            throw std::runtime_error("lzma_alone_encoder");
        }
        lzma_end_guard guard(&strm);

        auto start = std::chrono::steady_clock::now();
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
        auto end = std::chrono::steady_clock::now();

        compressed_size = strm.total_out;
        cum_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        ++n_samples;
    }

    return {cum_time / n_samples, static_cast<double>(compressed_size) / input_size};
}
#endif

using algorithm_map = std::unordered_map<std::string, std::function<benchmark_result(const void *, const metadata&,
        std::chrono::milliseconds)>>;

static const algorithm_map available_algorithms {
        {"hcde/cpu", benchmark_hcde<hcde::cpu_encoder>},
        {"hcde/cpu-mt", benchmark_hcde<hcde::mt_cpu_encoder>},
#if HCDE_GPU_SUPPORT
        {"hcde/gpu", benchmark_hcde<hcde::gpu_encoder>},
#endif
#if HCDE_BENCHMARK_HAVE_FPZIP
        {"fpzip", benchmark_fpzip},
#endif
#if HCDE_BENCHMARK_HAVE_ZLIB
        {"deflate/1", [](auto *i, auto &m, auto t) { return benchmark_deflate(i, m, 1, t); }},
        {"deflate/9", [](auto *i, auto &m, auto t) { return benchmark_deflate(i, m, 9, t); }},
#endif
#if HCDE_BENCHMARK_HAVE_LZ4
        {"lz4", benchmark_lz4},
#endif
#if HCDE_BENCHMARK_HAVE_LZMA
        {"lzma/1", [](auto *i, auto &m, auto t) { return benchmark_lzma(i, m, 1, t); }},
        {"lzma/9", [](auto *i, auto &m, auto t) { return benchmark_lzma(i, m, 9, t); }},
#endif
};


static void benchmark_file(const metadata &metadata, const algorithm_map &algorithms,
        std::chrono::milliseconds benchmark_time) {
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
        auto result = benchmark_algo(input_buffer.get(), metadata, benchmark_time);
        std::cout << metadata.path.filename().string() << ";"
                << algo_name << ";"
                << std::chrono::duration_cast<std::chrono::duration<double>>(result.duration).count() << ";"
                << result.compression_ratio << "\n";
    }
}


static std::string available_algorithms_string() {
    std::string algos;
    for (auto &[name, _]: available_algorithms) {
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
}


int main(int argc, char **argv) {
    namespace opts = boost::program_options;
    using namespace std::string_literals;

    std::string metadata_csv_file;
    std::vector<std::string> algorithm_names;
    unsigned benchmark_ms = 1000;

    auto usage = "Usage: "s + argv[0] + " [options] csv-file\n\n";

    opts::options_description desc("Options");
    desc.add_options()
            ("help", "show this help")
            ("version", "show library versions")
            ("csv-file", opts::value(&metadata_csv_file)->required(), "csv file with benchmark file metadata")
            ("algorithms,a", opts::value(&algorithm_names)->multitoken(), "algorithms to evaluate (see --help)")
            ("time-each,t", opts::value(&benchmark_ms), "repeat each measurement for at least t ms");
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
    if (!algorithm_names.empty()) {
        for (auto &name: algorithm_names) {
            auto iter = available_algorithms.find(name);
            if (iter == available_algorithms.end()) {
                std::cerr << "Unknown algorithm \"" << name << "\".\nAvailable algorithms are: "
                        << available_algorithms_string() << "\n";
            }
            selected_algorithms.insert(*iter);
        }
    } else {
        selected_algorithms = available_algorithms;
    }

    try {
        std::cout << "dataset;algorithm;time;ratio\n";
        std::cout.precision(9);
        std::cout.setf(std::ios::fixed);
        for (auto &metadata : load_metadata_file(metadata_csv_file)) {
            benchmark_file(metadata, selected_algorithms, std::chrono::milliseconds(benchmark_ms));
        }
        return EXIT_SUCCESS;
    } catch (std::exception &e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
