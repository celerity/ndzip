#include <hcde.hh>

#include <cstdlib>
#include <iostream>
#include <iomanip>

#include <boost/program_options.hpp>


namespace opts = boost::program_options;


template<typename Encoder>
static bool compress_stream(FILE *in, FILE *out, const hcde::extent<Encoder::dimensions> &size,
        const Encoder &encoder) {
    using data_type = typename Encoder::data_type;
    const auto compressed_buffer_size = encoder.compressed_size_bound(size);
    const auto raw_bytes_per_chunk = static_cast<size_t>(size.linear_offset() * sizeof(data_type));
    std::unique_ptr<data_type, decltype((free))> array(
            static_cast<data_type *>(malloc(raw_bytes_per_chunk)), free);
    std::unique_ptr<char, decltype((free))> compressed(
            static_cast<char *>(calloc(1, compressed_buffer_size)), free);

    size_t compressed_size = 0;
    unsigned n_chunks = 0;
    for (;; ++n_chunks) {
        auto bytes_read = fread(array.get(), 1, raw_bytes_per_chunk, in);
        if (bytes_read < raw_bytes_per_chunk) {
            if (ferror(in)) {
                perror("fread");
                return false;
            }
            if (bytes_read == 0 && n_chunks > 0) {
                break;
            }
            std::cerr << "compress: Input file size is not a multiple of the array size\n";
            return false;
        }

        if (n_chunks > 0) {
            memset(compressed.get(), 0, compressed_buffer_size);
        }
        auto compressed_chunk_size = encoder.compress(
                hcde::slice<data_type, Encoder::dimensions>(array.get(), size),
                compressed.get());
        assert(compressed_chunk_size <= compressed_buffer_size);

        if (fwrite(compressed.get(), compressed_chunk_size, 1, out) < 1) {
            perror("fwrite");
            return false;
        }
        compressed_size += compressed_chunk_size;
    }

    std::cerr << "raw = " << n_chunks * raw_bytes_per_chunk << " bytes";
    if (n_chunks > 1) {
        std::cerr << " (" << n_chunks << " chunks à " << raw_bytes_per_chunk << " bytes)";
    }
    std::cerr << ", compressed = " << compressed_size << " bytes";
    std::cerr << ", ratio = " << std::fixed << std::setprecision(4) <<
            (static_cast<double>(n_chunks * raw_bytes_per_chunk) / compressed_size) << "\n";
    return true;
}


template<typename Encoder>
static bool decompress_stream(FILE *in, FILE *out, const hcde::extent<Encoder::dimensions> &size,
        const Encoder &encoder) {
    using data_type = typename Encoder::data_type;
    const auto compressed_buffer_size = encoder.compressed_size_bound(size);
    const auto raw_bytes_per_chunk = static_cast<size_t>(size.linear_offset() * sizeof(data_type));
    std::unique_ptr<data_type, decltype((free))> array(
            static_cast<data_type *>(malloc(raw_bytes_per_chunk)), free);
    std::unique_ptr<char, decltype((free))> compressed(
            static_cast<char *>(calloc(1, compressed_buffer_size)), free);

    size_t compressed_bytes_left = 0;
    for (;;) {
        auto bytes_to_read = compressed_buffer_size - compressed_bytes_left;
        auto bytes_read = fread(compressed.get() + compressed_bytes_left, 1, bytes_to_read, in);
        if (bytes_read < bytes_to_read && ferror(in)) {
            perror("fread");
            return false;
        }

        auto compressed_bytes_in_buffer = compressed_bytes_left + bytes_read;
        if (compressed_bytes_in_buffer == 0) {
            return true;
        }

        auto compressed_chunk_size = encoder.decompress(compressed.get(),
                compressed_bytes_in_buffer,
                hcde::slice<data_type, Encoder::dimensions>(array.get(), size));
        assert(compressed_chunk_size <= compressed_bytes_in_buffer);

        if (fwrite(array.get(), raw_bytes_per_chunk, 1, out) < 1) {
            perror("fwrite");
            return false;
        }

        compressed_bytes_left = compressed_bytes_in_buffer - compressed_chunk_size;
        memmove(compressed.get(), compressed.get() + compressed_chunk_size,
                compressed_bytes_left);
    }
}


template<typename Encoder>
static bool process_stream(bool decompress, FILE *in, FILE *out, const hcde::extent<Encoder::dimensions> &size,
        const Encoder &encoder) {
    if (decompress) {
        return decompress_stream(in, out, size, encoder);
    } else {
        return compress_stream(in, out, size, encoder);
    }
}


template<template<typename Profile> typename Encoder, template<typename Data, unsigned Dims> typename Profile,
        typename Data>
static bool process_stream(bool decompress, const std::vector<size_t> &size_components, FILE *in, FILE *out) {
    switch (size_components.size()) {
        // case 1:
        //     return process_stream(decompress, in, out, hcde::extent{size_components[0]},
        //             Encoder<Profile<Data, 1>>{});
        case 2:
            return process_stream(decompress, in, out, hcde::extent{size_components[0], size_components[1]},
                    Encoder<Profile<Data, 2>>{});
        case 3:
            return process_stream(decompress, in, out, hcde::extent{size_components[0], size_components[1],
                    size_components[2]}, Encoder<Profile<Data, 3>>{});
        // case 4:
        //     return process_stream(decompress, in, out, hcde::extent{size_components[0], size_components[1],
        //             size_components[2], size_components[3]}, Encoder<Profile<Data, 4>>{});
        default:
            throw opts::error("Invalid number of dimensions in -n / --array-size");
    }
}


template<template<typename Profile> typename Encoder, typename Data>
static bool process_stream(bool decompress, const std::vector<size_t> &size_components, const std::string &profile,
        FILE *in, FILE *out) {
    if (profile == "fast") {
        return process_stream<Encoder, hcde::fast_profile, Data>(decompress, size_components, in, out);
    } else if (profile == "strong") {
        return process_stream<Encoder, hcde::strong_profile, Data>(decompress, size_components, in, out);
    } else {
        throw opts::error("Invalid profile \"" + profile + "\" in option -p / --profile");
    }
}


template<typename Data>
static bool process_stream(bool decompress, const std::vector<size_t> &size_components, const std::string &profile,
        const std::string &encoder, FILE *in, FILE *out) {
    if (encoder == "cpu") {
        return process_stream<hcde::cpu_encoder, Data>(decompress, size_components, profile, in, out);
    } else if (encoder == "cpu-mt") {
        return process_stream<hcde::mt_cpu_encoder, Data>(decompress, size_components, profile, in, out);
#if HCDE_GPU_SUPPORT
    } else if (encoder == "gpu") {
        return process_stream<hcde::gpu_encoder, Data>(decompress, size_components, profile, in, out);
#endif
    } else {
        throw opts::error("Invalid encoder \"" + encoder + "\" in option -e / --encoder");
    }
}


static bool process_stream(bool decompress, const std::vector<size_t> &size_components, const std::string &profile,
        const std::string &encoder, const std::string &data_type, FILE *in, FILE *out) {
    if (data_type == "float") {
        return process_stream<float>(decompress, size_components, profile, encoder, in, out);
    // } else if (data_type == "double") {
    //     return process_stream<double>(decompress, size_components, profile, encoder, in, out):
    } else {
        throw opts::error("Invalid option \"" + data_type + "\" in option -t / --data-type");
    }
}


int main(int argc, char **argv) {
    using namespace std::string_literals;

    bool decompress = false;
    std::vector<size_t> size_components{};
    std::string input = "-";
    std::string output = "-";
    std::string data_type = "float";
    std::string profile = "strong";
    std::string encoder = "cpu-mt";

    auto usage = "Usage: "s + argv[0] + " [options]\n\n";

    opts::options_description desc("Options");
    desc.add_options()
            ("help", "show this help")
            ("decompress,d", opts::bool_switch(&decompress), "decompress (default compress)")
            ("array-size,n", opts::value(&size_components)->required()->multitoken(),
                    "array size (one value per dimension, first-major)")
            ("data-type,t", opts::value(&data_type), "float|double (default float)")
            ("profile,p", opts::value(&profile), "fast|strong (default strong)")
#if HCDE_GPU_SUPPORT
            ("encoder,e", opts::value(&encoder), "cpu|cpu-mt|gpu (default cpu-mt)")
#else
            ("encoder,e", opts::value(&encoder), "cpu|cpu-mt (default cpu-mt)")
#endif
            ("input,i", opts::value(&input), "input file (default '-' is stdin)")
            ("output,o", opts::value(&output), "output file (default '-' is stdout)");

    opts::variables_map vars;
    try {
        auto parsed = opts::command_line_parser(argc, argv)
                .options(desc)
                .run();
        opts::store(parsed, vars);

        if (vars.count("help")) {
            std::cout << "Compress or decompress binary float dump\n\n" << usage << desc << "\n";
            return EXIT_SUCCESS;
        }

        opts::notify(vars);
    }
    catch (opts::error &e) {
        std::cerr << e.what() << "\n\n" << usage << desc;
        return EXIT_FAILURE;
    }

    std::unique_ptr<FILE, decltype((fclose))> in_stream(nullptr, fclose);
    if (input == "-") {
        freopen(NULL, "rb", stdin);
        in_stream.reset(stdin);
    } else {
        in_stream.reset(fopen(input.c_str(), "rb"));
        if (!in_stream) {
            fprintf(stderr, "fopen: %s: %s\n", input.c_str(), strerror(errno));
            return EXIT_FAILURE;
        }
    }

    std::unique_ptr<FILE, decltype((fclose))> out_stream(nullptr, fclose);
    if (output == "-") {
        freopen(NULL, "wb", stdout);
        out_stream.reset(stdout);
    } else {
        out_stream.reset(fopen(output.c_str(), "wb"));
        if (!out_stream) {
            fprintf(stderr, "fopen: %s: %s\n", output.c_str(), strerror(errno));
            return EXIT_FAILURE;
        }
    }

    try {
        auto ok = process_stream(decompress, size_components, profile, encoder, data_type,
                in_stream.get(), out_stream.get());
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (opts::error &e) {
        std::cerr << e.what() << "\n\n" << usage << desc;
        return EXIT_FAILURE;
    }
}

