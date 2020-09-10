#include <hcde/hcde.hh>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
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
#include <fpzip.h>

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


struct benchmark_result {
    std::chrono::microseconds duration;
    double compression_ratio;
};


template<template<typename, unsigned> typename Encoder, typename Data, unsigned Dims>
static benchmark_result benchmark_hcde_3(const Data *input_buffer, const hcde::extent<Dims> &size) {
    auto input_size = hcde::num_elements(size) * sizeof(Data);
    auto input_slice = hcde::slice<const Data, Dims>(input_buffer, size);

    Encoder<Data, Dims> e;
    auto output_buffer_size = e.compressed_size_bound(size);
    auto output_buffer = std::unique_ptr<std::byte, decltype((free))>(
            static_cast<std::byte*>(malloc(output_buffer_size)), free);

    std::chrono::microseconds cum_time{};
    auto n_samples = size_t{0};
    size_t compressed_size;
    while (cum_time < std::chrono::seconds(1)) {
        memset(output_buffer.get(), 0, output_buffer_size);
        auto start = std::chrono::steady_clock::now();
        compressed_size = e.compress(input_slice, output_buffer.get());
        auto end = std::chrono::steady_clock::now();
        cum_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        ++n_samples;
    }
    return {cum_time / n_samples, static_cast<double>(compressed_size) / input_size};
}


template<template<typename, unsigned> typename Encoder, typename Data>
static benchmark_result benchmark_hcde_2(const Data *input_buffer, const metadata &metadata) {
    auto &e = metadata.extent;
    if (e.size() == 1) {
        return benchmark_hcde_3<Encoder, Data, 1>(input_buffer, hcde::extent{e[0]});
    } else if (e.size() == 2) {
        return benchmark_hcde_3<Encoder, Data, 2>(input_buffer, hcde::extent{e[0], e[1]});
    } else if (e.size() == 3) {
        return benchmark_hcde_3<Encoder, Data, 3>(input_buffer, hcde::extent{e[0], e[1], e[2]});
    } else {
        std::abort(); // should be unreachable
    }
}


template<template<typename, unsigned> typename Encoder>
static benchmark_result benchmark_hcde(const void *input_buffer, const metadata &metadata) {
    if (metadata.data_type == data_type::t_float) {
        return benchmark_hcde_2<Encoder, float>(static_cast<const float*>(input_buffer), metadata);
    } else {
        return benchmark_hcde_2<Encoder, double>(static_cast<const double*>(input_buffer), metadata);
    }
}


static benchmark_result benchmark_fpzip(const void *input_buffer, const metadata &metadata) {
    return {};
}


#if HCDE_BENCHMARK_HAVE_ZLIB
static benchmark_result benchmark_deflate(const void *input_buffer, const metadata &metadata) {
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
        if (deflateInit(&strm, /* level */ 9) != Z_OK) {
            throw std::runtime_error("deflateInit");
        }
        deflate_end_guard guard(&strm);

        output_buffer_size = deflateBound(&strm_init, strm_init.avail_in);
    }

    auto output_buffer = std::unique_ptr<Bytef, decltype((free))>(
            static_cast<Bytef*>(malloc(output_buffer_size)), free);
    strm_init.next_out = output_buffer.get(),
    strm_init.avail_out = static_cast<uInt>(output_buffer_size);

    std::chrono::microseconds cum_time{};
    auto n_samples = size_t{0};
    size_t compressed_size;
    while (cum_time < std::chrono::seconds(1)) {
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
static benchmark_result benchmark_lz4(const void *input_buffer, const metadata &metadata) {
    // unimplemented
    return {};
}
#endif


#if HCDE_BENCHMARK_HAVE_LZMA
static benchmark_result benchmark_lzma(const void *input_buffer, const metadata &metadata) {
    // unimplemented
    return {};
}
#endif


static std::unordered_map<std::string, benchmark_result> benchmark_file(const metadata &metadata) {
    auto input_buffer_size = metadata.size_in_bytes();
    auto input_buffer = std::unique_ptr<std::byte, decltype((free))>(
            static_cast<std::byte*>(malloc(input_buffer_size)), free);

    auto file_name = metadata.path.string();
    auto input_file = std::unique_ptr<FILE, decltype((fclose))>(fopen(file_name.c_str(), "rb"), fclose);
    if (!input_file) {
        throw std::runtime_error(file_name + ": " + strerror(errno));
    }

    if (fread(input_buffer.get(), input_buffer_size, 1, input_file.get()) != 1) {
        throw std::runtime_error(file_name + ": " + strerror(errno));
    }

    return {
        {"hcde/cpu", benchmark_hcde<hcde::cpu_encoder>(input_buffer.get(), metadata)},
        {"hcde/cpu-mt", benchmark_hcde<hcde::mt_cpu_encoder>(input_buffer.get(), metadata)},
#if HCDE_GPU_SUPPORT
        {"hcde/gpu", benchmark_hcde<hcde::gpu_encoder>(input_buffer.get(), metadata)},
#endif
        {"fpzip", benchmark_fpzip(input_buffer.get(), metadata)},
#if HCDE_BENCHMARK_HAVE_ZLIB
        {"deflate", benchmark_deflate(input_buffer.get(), metadata)},
#endif
#if HCDE_BENCHMARK_HAVE_LZ4
        {"lz4", benchmark_lz4(input_buffer.get(), metadata)},
#endif
#if HCDE_BENCHMARK_HAVE_LZMA
        {"lzma", benchmark_lzma(input_buffer.get(), metadata)},
#endif
    };
}


int main(int argc, char **argv) {
    namespace opts = boost::program_options;
    using namespace std::string_literals;

    std::string metadata_csv_file;

    auto usage = "Usage: "s + argv[0] + " [options] csv-file\n\n";

    opts::options_description desc("Options");
    desc.add_options()
            ("help", "show this help")
            ("version", "show library versions")
            ("csv-file", opts::value(&metadata_csv_file)->required(), "csv file with benchmark file metadata");
    opts::positional_options_description pos_desc;
    pos_desc.add("csv-file", 1);

    opts::variables_map vars;
    try {
        auto parsed = opts::command_line_parser(argc, argv).options(desc).positional(pos_desc).run();
        opts::store(parsed, vars);

        if (vars.count("help")) {
            std::cout << "Compress or decompress binary float dump\n\n" << usage << desc << "\n";
            return EXIT_SUCCESS;
        }

        if (vars.count("version")) {
            printf("zlib version %s\n", zlibVersion());
            printf("LZ4 version %s\n", LZ4_versionString());
            printf("LZMA version %s\n", lzma_version_string());
            printf("%s\n", fpzip_version_string);
            return EXIT_SUCCESS;
        }

        opts::notify(vars);
    } catch (opts::error &e) {
        std::cerr << e.what() << "\n\n" << usage << desc;
        return EXIT_FAILURE;
    }

    try {
        std::cout << "dataset;algorithm;time;ratio\n";
        std::cout.precision(9);
        std::cout.setf(std::ios::fixed);
        for (auto &metadata : load_metadata_file(metadata_csv_file)) {
            for (auto &[algo, result] : benchmark_file(metadata)) {
                std::cout << metadata.path.filename().string() << ";"
                          << algo << ";"
                          << std::chrono::duration_cast<std::chrono::duration<double>>(result.duration).count() << ";"
                          << result.compression_ratio << "\n";
            }
        }
        return EXIT_SUCCESS;
    } catch (std::exception &e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
