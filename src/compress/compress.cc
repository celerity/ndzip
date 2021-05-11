#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <boost/program_options.hpp>
#include <io/io.hh>
#include <ndzip/ndzip.hh>


namespace opts = boost::program_options;

namespace ndzip::detail {

using duration = std::chrono::system_clock::duration;

template<typename Encoder>
void compress_stream(const std::string &in, const std::string &out,
        const ndzip::extent<Encoder::dimensions> &size, const Encoder &encoder,
        const ndzip::detail::io_factory &io) {
    using data_type = typename Encoder::data_type;

    const auto array_chunk_length = static_cast<size_t>(num_elements(size) * sizeof(data_type));
    const auto max_compressed_chunk_length = ndzip::compressed_size_bound<data_type>(size);

    size_t compressed_length = 0;
    size_t n_chunks = 0;
    auto start = std::chrono::steady_clock::now();
    {
        auto in_stream = io.create_input_stream(in, array_chunk_length);
        auto out_stream = io.create_output_stream(out, max_compressed_chunk_length);

        while (auto *chunk = in_stream->read_exact()) {
            auto input_buffer = static_cast<const data_type *>(chunk);
            auto compressed_chunk_length = encoder.compress(
                    ndzip::slice<const data_type, Encoder::dimensions>(input_buffer, size),
                    out_stream->get_write_buffer());
            assert(compressed_chunk_length <= max_compressed_chunk_length);
            out_stream->commit_chunk(compressed_chunk_length);
            compressed_length += compressed_chunk_length;
            ++n_chunks;
        }
    }
    auto duration = std::chrono::steady_clock::now() - start;

    const auto in_file_size = n_chunks * array_chunk_length;
    std::cerr << "raw = " << n_chunks * in_file_size << " bytes";
    if (n_chunks > 1) {
        std::cerr << " (" << n_chunks << " chunks à " << array_chunk_length << " bytes)";
    }
    std::cerr << ", compressed = " << compressed_length << " bytes";
    std::cerr << ", ratio = " << std::fixed << std::setprecision(4)
              << (static_cast<double>(compressed_length) / in_file_size);
    std::cerr << ", time = " << std::setprecision(3) << std::fixed
              << std::chrono::duration_cast<std::chrono::duration<double>>(duration).count()
              << "s\n";
}

template<typename Encoder>
void decompress_stream(const std::string &in, const std::string &out,
        const ndzip::extent<Encoder::dimensions> &size, const Encoder &encoder,
        const ndzip::detail::io_factory &io) {
    using data_type = typename Encoder::data_type;
    const auto max_compressed_chunk_length = ndzip::compressed_size_bound<data_type>(size);
    const auto array_chunk_length = static_cast<size_t>(num_elements(size) * sizeof(data_type));

    const auto in_stream = io.create_input_stream(in, max_compressed_chunk_length);
    const auto out_stream = io.create_output_stream(out, array_chunk_length);

    size_t compressed_bytes_left = 0;
    for (;;) {
        auto [chunk, bytes_in_chunk] = in_stream->read_some(compressed_bytes_left);
        if (bytes_in_chunk == 0) { break; }

        auto output_buffer = static_cast<data_type *>(out_stream->get_write_buffer());
        auto compressed_size = encoder.decompress(chunk, bytes_in_chunk,
                ndzip::slice<data_type, Encoder::dimensions>(output_buffer, size));
        assert(compressed_size <= bytes_in_chunk);
        out_stream->commit_chunk(array_chunk_length);
        compressed_bytes_left = bytes_in_chunk - compressed_size;
    }
}

template<typename Encoder>
void process_stream(bool decompress, const std::string &in, const std::string &out,
        const ndzip::extent<Encoder::dimensions> &size, const Encoder &encoder,
        const ndzip::detail::io_factory &io) {
    if (decompress) {
        decompress_stream(in, out, size, encoder, io);
    } else {
        compress_stream(in, out, size, encoder, io);
    }
}

template<template<typename, unsigned> typename Encoder, typename Data>
void process_stream(bool decompress, const std::vector<size_t> &size_components,
        const std::string &in, const std::string &out, const ndzip::detail::io_factory &io) {
    switch (size_components.size()) {
        case 1:
            return process_stream(
                    decompress, in, out, ndzip::extent{size_components[0]}, Encoder<Data, 1>{}, io);
        case 2:
            return process_stream(decompress, in, out,
                    ndzip::extent{size_components[0], size_components[1]}, Encoder<Data, 2>{}, io);
        case 3:
            return process_stream(decompress, in, out,
                    ndzip::extent{size_components[0], size_components[1], size_components[2]},
                    Encoder<Data, 3>{}, io);
        // case 4:
        //     return process_stream(decompress, in, out, ndzip::extent{size_components[0],
        //     size_components[1],
        //             size_components[2], size_components[3]}, Encoder<Profile<Data, 4>>{}, io);
        default: throw opts::error("Invalid number of dimensions in -n / --array-size");
    }
}

template<typename Data>
void process_stream(bool decompress, const std::vector<size_t> &size_components,
        const std::string &encoder, const std::string &in, const std::string &out,
        const ndzip::detail::io_factory &io) {
    if (encoder == "cpu") {
        process_stream<ndzip::cpu_encoder, Data>(decompress, size_components, in, out, io);
#if NDZIP_OPENMP_SUPPORT
    } else if (encoder == "cpu-mt") {
        process_stream<ndzip::mt_cpu_encoder, Data>(decompress, size_components, in, out, io);
#endif
#if NDZIP_HIPSYCL_SUPPORT
    } else if (encoder == "sycl") {
        process_stream<ndzip::sycl_encoder, Data>(decompress, size_components, in, out, io);
#endif
    } else {
        throw opts::error("Invalid encoder \"" + encoder + "\" in option -e / --encoder");
    }
}


void process_stream(bool decompress, const std::vector<size_t> &size_components,
        const std::string &encoder, const std::string &data_type, const std::string &in,
        const std::string &out, const ndzip::detail::io_factory &io) {
    if (data_type == "float") {
        process_stream<float>(decompress, size_components, encoder, in, out, io);
    } else if (data_type == "double") {
        process_stream<double>(decompress, size_components, encoder, in, out, io);
    } else {
        throw opts::error("Invalid option \"" + data_type + "\" in option -t / --data-type");
    }
}

}  // namespace ndzip::detail

int main(int argc, char **argv) {
    using namespace std::string_literals;

    bool decompress = false;
    bool no_mmap = false;
    std::vector<size_t> size_components{};
    std::string input = "-";
    std::string output = "-";
    std::string data_type = "float";
    std::string encoder = "cpu";

    auto usage = "Usage: "s + argv[0] + " [options]\n\n";

    opts::options_description desc("Options");
    // clang-format off
    desc.add_options()
        ("help", "show this help")
        ("decompress,d", opts::bool_switch(&decompress), "decompress (default compress)")
        ("array-size,n", opts::value(&size_components)->required()->multitoken(),
                "array size (one value per dimension, first-major)")
        ("data-type,t", opts::value(&data_type), "float|double (default float)")
        ("encoder,e", opts::value(&encoder), "cpu"
#if NDZIP_OPENMP_SUPPORT
                                             "|cpu-mt"
#endif
#if NDZIP_HIPSYCL_SUPPORT
                                             "|sycl"
#endif
                                             " (default cpu)")
        ("input,i", opts::value(&input), "input file (default '-' is stdin)")
        ("output,o", opts::value(&output), "output file (default '-' is stdout)")
        ("no-mmap", opts::bool_switch(&no_mmap), "do not use memory-mapped I/O");
    // clang-format on

    opts::variables_map vars;
    try {
        auto parsed = opts::command_line_parser(argc, argv).options(desc).run();
        opts::store(parsed, vars);

        if (vars.count("help")) {
            std::cout << "Compress or decompress binary float dump\n\n" << usage << desc << "\n";
            return EXIT_SUCCESS;
        }

        opts::notify(vars);
    } catch (opts::error &e) {
        std::cerr << e.what() << "\n\n" << usage << desc;
        return EXIT_FAILURE;
    }

    std::unique_ptr<ndzip::detail::io_factory> io_factory;
#if NDZIP_SUPPORT_MMAP
    if (!no_mmap) { io_factory = std::make_unique<ndzip::detail::mmap_io_factory>(); }
#endif
    if (!io_factory) { io_factory = std::make_unique<ndzip::detail::stdio_io_factory>(); }

    try {
        ndzip::detail::process_stream(
                decompress, size_components, encoder, data_type, input, output, *io_factory);
        return EXIT_SUCCESS;
    } catch (opts::error &e) {
        std::cerr << e.what() << "\n\n" << usage << desc;
        return EXIT_FAILURE;
    } catch (std::exception &e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
