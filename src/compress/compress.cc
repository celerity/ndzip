#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <boost/program_options.hpp>
#include <io/io.hh>
#include <ndzip/offload.hh>


namespace opts = boost::program_options;

namespace ndzip::detail {

enum class data_type { t_float, t_double };

template<typename T>
void compress_stream(const std::string &in, const std::string &out, const ndzip::extent &size,
        ndzip::offloader<T> &offloader, const ndzip::detail::io_factory &io) {
    using compressed_type = ndzip::compressed_type<T>;

    const auto array_chunk_length = static_cast<size_t>(num_elements(size));
    const auto array_chunk_size = array_chunk_length * sizeof(T);
    const auto max_compressed_chunk_length = ndzip::compressed_length_bound<T>(size);
    const auto max_compressed_chunk_size = max_compressed_chunk_length * sizeof(compressed_type);

    size_t compressed_length = 0;
    size_t n_chunks = 0;
    kernel_duration total_duration{};
    {
        auto in_stream = io.create_input_stream(in, array_chunk_size);
        auto out_stream = io.create_output_stream(out, max_compressed_chunk_size);

        while (auto *chunk = in_stream->read_exact()) {
            const auto input_buffer = static_cast<const T *>(chunk);
            const auto write_buffer = static_cast<compressed_type *>(out_stream->get_write_buffer());
            kernel_duration chunk_duration;
            const auto compressed_chunk_length = offloader.compress(input_buffer, size, write_buffer, &chunk_duration);
            const auto compressed_chunk_size = compressed_chunk_length * sizeof(compressed_type);
            assert(compressed_chunk_length <= max_compressed_chunk_length);
            out_stream->commit_chunk(compressed_chunk_size);
            compressed_length += compressed_chunk_length;
            total_duration += chunk_duration;
            ++n_chunks;
        }
    }

    const auto in_file_size = n_chunks * array_chunk_size;
    const auto compressed_size = compressed_length * sizeof(compressed_type);
    std::cerr << "raw = " << n_chunks * in_file_size << " bytes";
    if (n_chunks > 1) { std::cerr << " (" << n_chunks << " chunks à " << array_chunk_size << " bytes)"; }
    std::cerr << ", compressed = " << compressed_size << " bytes";
    std::cerr << ", ratio = " << std::fixed << std::setprecision(4)
              << (static_cast<double>(compressed_size) / in_file_size);
    std::cerr << ", time = " << std::setprecision(3) << std::fixed
              << std::chrono::duration_cast<std::chrono::duration<double>>(total_duration).count() << "s\n";
}

template<typename T>
void decompress_stream(const std::string &in, const std::string &out, const ndzip::extent &size,
        ndzip::offloader<T> &offloader, const ndzip::detail::io_factory &io) {
    using compressed_type = ndzip::compressed_type<T>;

    const auto array_chunk_length = static_cast<size_t>(num_elements(size));
    const auto array_chunk_size = array_chunk_length * sizeof(T);
    const auto max_compressed_chunk_length = ndzip::compressed_length_bound<T>(size);
    const auto max_compressed_chunk_size = max_compressed_chunk_length * sizeof(compressed_type);

    const auto in_stream = io.create_input_stream(in, max_compressed_chunk_size);
    const auto out_stream = io.create_output_stream(out, array_chunk_size);

    size_t compressed_bytes_left = 0;
    for (;;) {
        const auto [chunk, bytes_in_chunk] = in_stream->read_some(compressed_bytes_left);
        if (bytes_in_chunk == 0) { break; }

        const auto chunk_buffer = static_cast<const compressed_type *>(chunk);
        const auto chunk_buffer_length = bytes_in_chunk / sizeof(compressed_type);  // floor division!
        const auto output_buffer = static_cast<T *>(out_stream->get_write_buffer());
        const auto compressed_length = offloader.decompress(chunk_buffer, chunk_buffer_length, output_buffer, size);
        const auto compressed_size = compressed_length * sizeof(compressed_type);
        assert(compressed_length <= chunk_buffer_length);
        out_stream->commit_chunk(array_chunk_size);
        compressed_bytes_left = bytes_in_chunk - compressed_size;
    }
}

template<typename T>
void process_stream(bool decompress, const std::string &in, const std::string &out, const ndzip::extent &size,
        ndzip::offloader<T> &offloader, const ndzip::detail::io_factory &io) {
    if (decompress) {
        decompress_stream(in, out, size, offloader, io);
    } else {
        compress_stream(in, out, size, offloader, io);
    }
}

template<typename T>
void process_stream(bool decompress, const ndzip::extent &size, ndzip::target target,
        std::optional<size_t> num_cpu_threads, const std::string &in, const std::string &out,
        const ndzip::detail::io_factory &io) {
    std::unique_ptr<ndzip::offloader<T>> offloader;
    if (target == ndzip::target::cpu && num_cpu_threads.has_value()) {
        offloader = ndzip::make_cpu_offloader<T>(size.dimensions(), *num_cpu_threads);
    } else {
        offloader = ndzip::make_offloader<T>(target, size.dimensions(), true /* enable_profiling */);
    }
    process_stream(decompress, in, out, size, *offloader, io);
}

void process_stream(bool decompress, const ndzip::extent &size, ndzip::target target,
        std::optional<size_t> num_cpu_threads, const data_type &data_type, const std::string &in,
        const std::string &out, const ndzip::detail::io_factory &io) {
    switch (data_type) {
        case detail::data_type::t_float:
            return process_stream<float>(decompress, size, target, num_cpu_threads, in, out, io);
        case detail::data_type::t_double:
            return process_stream<double>(decompress, size, target, num_cpu_threads, in, out, io);
        default: std::terminate();
    }
}

}  // namespace ndzip::detail

int main(int argc, char **argv) {
    using namespace std::string_literals;

    bool decompress = false;
    bool no_mmap = false;
    std::vector<ndzip::index_type> size_components;
    std::string input = "-";
    std::string output = "-";
    std::string data_type_str = "float";
    std::string target_str = "cpu";
    size_t num_threads_or_0 = 0;

    auto usage = "Usage: "s + argv[0] + " [options]\n\n";

    opts::options_description desc("Options");
    // clang-format off
    desc.add_options()
        ("help", "show this help")
        ("decompress,d", opts::bool_switch(&decompress), "decompress (default compress)")
        ("array-size,n", opts::value(&size_components)->required()->multitoken(),
                "array size (one value per dimension, first-major)")
        ("data-type,t", opts::value(&data_type_str), "float|double (default float)")
        ("target_str,e", opts::value(&target_str), "cpu"
#if NDZIP_HIPSYCL_SUPPORT
                                             "|sycl"
#endif
#if NDZIP_CUDA_SUPPORT
                                             "|cuda"
#endif
                                             " (default cpu)")
        ("threads,T", opts::value(&num_threads_or_0), "number of CPU threads")
        ("input,i", opts::value(&input), "input file (default '-' is stdin)")
        ("output,o", opts::value(&output), "output file (default '-' is stdout)")
        ("no-mmap", opts::bool_switch(&no_mmap), "do not use memory-mapped I/O");
    // clang-format on

    opts::variables_map vars;
    ndzip::target target;
    ndzip::extent size;
    ndzip::detail::data_type data_type;
    std::optional<size_t> opt_num_threads;
    try {
        auto parsed = opts::command_line_parser(argc, argv).options(desc).run();
        opts::store(parsed, vars);

        if (vars.count("help")) {
            std::cout << "Compress or decompress binary float dump\n\n" << usage << desc << "\n";
            return EXIT_SUCCESS;
        }

        opts::notify(vars);

        if (target_str == "cpu") {
            target = ndzip::target::cpu;
#if NDZIP_HIPSYCL_SUPPORT
        } else if (target_str == "sycl") {
            target = ndzip::target::sycl;
#endif
#if NDZIP_CUDA_SUPPORT
        } else if (target_str == "cuda") {
            target = ndzip::target::cuda;
#endif
        } else {
            throw opts::error{"Unimplemented target " + target_str};
        }

        if (size_components.empty() || size_components.size() > 3) {
            throw opts::error{"Expected between 1 and 3 dimensions, got " + std::to_string(size_components.size())};
        }
        size = ndzip::extent{static_cast<ndzip::dim_type>(size_components.size())};
        for (ndzip::dim_type d = 0; d < size.dimensions(); ++d) {
            size[d] = size_components[d];
        }

        if (data_type_str == "float") {
            data_type = ndzip::detail::data_type::t_float;
        } else if (data_type_str == "double") {
            data_type = ndzip::detail::data_type::t_double;
        } else {
            throw opts::error{"Invalid data type " + data_type_str};
        }

        if (num_threads_or_0 != 0) { opt_num_threads = num_threads_or_0; }

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
        ndzip::detail::process_stream(decompress, size, target, opt_num_threads, data_type, input, output, *io_factory);
        return EXIT_SUCCESS;
    } catch (opts::error &e) {
        std::cerr << e.what() << "\n\n" << usage << desc;
        return EXIT_FAILURE;
    } catch (std::exception &e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
