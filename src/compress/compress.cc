#include "io.hh"
#include <hcde.hh>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iomanip>

#include <boost/program_options.hpp>


namespace opts = boost::program_options;

namespace hcde::detail {

using duration = std::chrono::system_clock::duration;

template<typename Encoder>
void compress_stream(const std::string &in, const std::string &out, const hcde::extent<Encoder::dimensions> &size,
    const Encoder &encoder, const io_factory &io) {
    using data_type = typename Encoder::data_type;

    const auto array_chunk_length = static_cast<size_t>(num_elements(size) * sizeof(data_type));
    const auto max_compressed_chunk_length = encoder.compressed_size_bound(size);

    auto in_stream = io.create_input_stream(in, array_chunk_length);
    auto out_stream = io.create_output_stream(out, max_compressed_chunk_length);

    size_t compressed_length = 0;
    size_t n_chunks = 0;
    while (auto *chunk = in_stream->read_exact()) {
        auto input_buffer = static_cast<const data_type *>(chunk);
        auto compressed_chunk_length = encoder.compress(
            hcde::slice<const data_type, Encoder::dimensions>(input_buffer, size),
            out_stream->get_write_buffer());
        assert(compressed_chunk_length <= max_compressed_chunk_length);
        out_stream->commit_chunk(compressed_chunk_length);
        compressed_length += compressed_chunk_length;
        ++n_chunks;
    }

    const auto in_file_size = n_chunks * array_chunk_length;
    std::cerr << "raw = " << n_chunks * in_file_size << " bytes";
    if (n_chunks > 1) {
        std::cerr << " (" << n_chunks << " chunks à " << array_chunk_length << " bytes)";
    }
    std::cerr << ", compressed = " << compressed_length << " bytes";
    std::cerr << ", ratio = " << std::fixed << std::setprecision(4)
        << (static_cast<double>(in_file_size) / compressed_length) << "\n";
}

template<typename Encoder>
void decompress_stream(const std::string &in, const std::string &out, const hcde::extent<Encoder::dimensions> &size,
    const Encoder &encoder, const io_factory &io) {
    using data_type = typename Encoder::data_type;
    const auto max_compressed_chunk_length = encoder.compressed_size_bound(size);
    const auto array_chunk_length = static_cast<size_t>(num_elements(size) * sizeof(data_type));

    const auto in_stream = io.create_input_stream(in, max_compressed_chunk_length);
    const auto out_stream = io.create_output_stream(out, array_chunk_length);

    size_t compressed_bytes_left = 0;
    for (;;) {
        auto[chunk, bytes_in_chunk] = in_stream->read_some(compressed_bytes_left);
        if (bytes_in_chunk == 0) {
            break;
        }

        auto output_buffer = static_cast<data_type *>(out_stream->get_write_buffer());
        auto compressed_size = encoder.decompress(chunk, bytes_in_chunk,
            hcde::slice<data_type, Encoder::dimensions>(output_buffer, size));
        assert(compressed_size <= bytes_in_chunk);
        out_stream->commit_chunk(array_chunk_length);
        compressed_bytes_left = bytes_in_chunk - compressed_size;
    }
}

template<typename Encoder>
duration process_stream(bool decompress, const std::string &in, const std::string &out,
        const hcde::extent<Encoder::dimensions> &size, const Encoder &encoder, const io_factory &io)
{
    auto start = std::chrono::system_clock::now();
    if (decompress) {
        decompress_stream(in, out, size, encoder, io);
    } else {
        compress_stream(in, out, size, encoder, io);
    }
    auto end = std::chrono::system_clock::now();
    return std::chrono::duration_cast<duration>(end - start);
}

template<template<typename> typename Encoder,
    template<typename, unsigned> typename Profile, typename Data>
duration process_stream(bool decompress, const std::vector<size_t> &size_components, const std::string &in,
    const std::string &out, const io_factory &io) {
    switch (size_components.size()) {
        case 1:
            return process_stream(decompress, in, out, hcde::extent{size_components[0]},
                Encoder<Profile<Data, 1>>{}, io);
        case 2:
            return process_stream(decompress, in, out, hcde::extent{size_components[0], size_components[1]},
                Encoder<Profile<Data, 2>>{}, io);
        case 3:
            return process_stream(decompress, in, out, hcde::extent{size_components[0], size_components[1],
                size_components[2]}, Encoder<Profile<Data, 3>>{}, io);
            // case 4:
            //     return process_stream(decompress, in, out, hcde::extent{size_components[0], size_components[1],
            //             size_components[2], size_components[3]}, Encoder<Profile<Data, 4>>{}, io);
        default:
            throw opts::error("Invalid number of dimensions in -n / --array-size");
    }
}

template<template<typename> typename Encoder, typename Data>
duration process_stream(bool decompress, const std::vector<size_t> &size_components, const std::string &profile,
    const std::string &in, const std::string &out, const io_factory &io) {
    if (profile == "fast") {
        return process_stream<Encoder, hcde::fast_profile, Data>(decompress, size_components, in, out, io);
    } else if (profile == "strong") {
        return process_stream<Encoder, hcde::strong_profile, Data>(decompress, size_components, in, out, io);
    } else if (profile == "xt") {
        return process_stream<Encoder, hcde::xt_profile, Data>(decompress, size_components, in, out, io);
    } else {
        throw opts::error("Invalid profile \"" + profile + "\" in option -p / --profile");
    }
}

template<typename Data>
duration process_stream(bool decompress, const std::vector<size_t> &size_components, const std::string &profile,
    const std::string &encoder, const std::string &in, const std::string &out, const io_factory &io) {
    if (encoder == "cpu") {
        return process_stream<hcde::cpu_encoder, Data>(decompress, size_components, profile, in, out, io);
    } else if (encoder == "cpu-mt") {
        return process_stream<hcde::mt_cpu_encoder, Data>(decompress, size_components, profile, in, out, io);
#if HCDE_GPU_SUPPORT
    } else if (encoder == "gpu") {
        return process_stream<hcde::gpu_encoder, Data>(decompress, size_components, profile, in, out, io);
#endif
    } else {
        throw opts::error("Invalid encoder \"" + encoder + "\" in option -e / --encoder");
    }
}


duration process_stream(bool decompress, const std::vector<size_t> &size_components, const std::string &profile,
    const std::string &encoder, const std::string &data_type, const std::string &in, const std::string &out,
    const io_factory &io) {
    if (data_type == "float") {
        return process_stream<float>(decompress, size_components, profile, encoder, in, out, io);
        // } else if (data_type == "double") {
        //     return process_stream<double>(decompress, size_components, profile, encoder, in, out, io):
    } else {
        throw opts::error("Invalid option \"" + data_type + "\" in option -t / --data-type");
    }
}

}

int main(int argc, char **argv) {
    using namespace std::string_literals;

    bool decompress = false;
    bool no_mmap = false;
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
        ("output,o", opts::value(&output), "output file (default '-' is stdout)")
        ("no-mmap", opts::bool_switch(&no_mmap), "do not use memory-mapped I/O");

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

    std::unique_ptr<hcde::detail::io_factory> io_factory;
#if HCDE_MMAP_SUPPORT
    if (!no_mmap) {
        io_factory = std::make_unique<hcde::detail::mmap_io_factory>();
    }
#endif
    if (!io_factory) {
        io_factory = std::make_unique<hcde::detail::stdio_io_factory>();
    }

    try {
        auto duration = hcde::detail::process_stream(decompress, size_components, profile, encoder,
                data_type, input, output, *io_factory);
        std::cerr << "finished in " << std::setprecision(3) << std::fixed
            << std::chrono::duration_cast<std::chrono::duration<double>>(duration).count() << "s\n";
        return EXIT_SUCCESS;
    } catch (opts::error &e) {
        std::cerr << e.what() << "\n\n" << usage << desc;
        return EXIT_FAILURE;
    } catch (std::exception &e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}

