
#include <hcde.hh>

#include <cstdlib>
#include <iostream>
#include <iomanip>

#include <boost/program_options.hpp>


template<typename Encoder>
static bool compress_stream(FILE *in, FILE *out, const hcde::extent<Encoder::dimensions> &size,
        const Encoder &encoder)
{
    using data_type = typename Encoder::data_type;
    std::vector<data_type> array(size.linear_offset());
    std::vector<char> compressed(encoder.compressed_size_bound(size));

    const auto raw_bytes_per_chunk = static_cast<size_t>(array.size() * sizeof(data_type));
    size_t compressed_size = 0;
    unsigned n_chunks = 0;
    for (;; ++n_chunks) {
        auto bytes_read = fread(array.data(), 1, raw_bytes_per_chunk, in);
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

        memset(compressed.data(), 0, compressed.size());
        auto compressed_chunk_size = encoder.compress(
                hcde::slice<data_type, Encoder::dimensions>(array.data(), size),
                compressed.data());
        assert(compressed_chunk_size <= compressed.size());

        if (fwrite(compressed.data(), compressed_chunk_size, 1, out) < 1) {
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
        const Encoder &encoder)
{
    using data_type = typename Encoder::data_type;
    std::vector<data_type> array(size.linear_offset());
    std::vector<char> compressed(encoder.compressed_size_bound(size));

    const auto raw_bytes_per_chunk = static_cast<size_t>(array.size() * sizeof(data_type));
    size_t compressed_bytes_left = 0;
    for (;;) {
        auto bytes_to_read = compressed.size() - compressed_bytes_left;
        auto bytes_read = fread(compressed.data() + compressed_bytes_left, 1, bytes_to_read, in);
        if (bytes_read < bytes_to_read && ferror(in)) {
            perror("fread");
            return false;
        }

        auto compressed_bytes_in_buffer = compressed_bytes_left + bytes_read;
        if (compressed_bytes_in_buffer == 0) {
            return true;
        }

        auto compressed_chunk_size = encoder.decompress(compressed.data(),
                compressed_bytes_in_buffer,
                hcde::slice<data_type, Encoder::dimensions>(array.data(), size));
        assert(compressed_chunk_size <= compressed_bytes_in_buffer);

        if (fwrite(array.data(), raw_bytes_per_chunk, 1, out) < 1) {
            perror("fwrite");
            return false;
        }

        compressed_bytes_left = compressed_bytes_in_buffer - compressed_chunk_size;
        memmove(compressed.data(), compressed.data() + compressed_chunk_size,
                compressed_bytes_left);
    }
}


int main(int argc, char **argv) {
    namespace opts = boost::program_options;
    using namespace std::string_literals;

    bool decompress;
    std::vector<size_t> size_components{};
    std::string input = "-";
    std::string output = "-";

	auto usage = "Usage: "s + argv[0] + " [options]\n\n";

	opts::options_description desc("Options");
	desc.add_options()
			("help", "show this help")
			("decompress,d", opts::bool_switch(&decompress), "decompress (default compress)")
			("array-size,n", opts::value(&size_components)->required()->multitoken(),
                "array size (one value per dimension, first-major)")
            ("input,i", opts::value(&input), "input file (default '-' is stdin)")
            ("output,o", opts::value(&output), "output file (default '-' is stdout)");

	opts::variables_map vars;
	try
	{
		auto parsed = opts::command_line_parser(argc, argv)
				.options(desc)
				.run();
		opts::store(parsed, vars);

		if (vars.count("help"))
		{
			std::cout << "Compress or decompress binary float dump\n\n" << usage << desc << "\n";
			return EXIT_SUCCESS;
		}

		opts::notify(vars);

        auto dim = size_components.size();
        if (dim < 1 || dim > 4) {
            throw opts::error("Invalid number of dimensions " + std::to_string(dim) + " for -d");
        }
	}
	catch (opts::error &e)
	{
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

    bool ok = false;
    if (size_components.size() == 2) {
        hcde::singlethread_cpu_encoder<hcde::fast_profile<float, 2>> encoder;
        auto size = hcde::extent<2>{size_components[0], size_components[1]};
        if (decompress) {
            ok = decompress_stream(in_stream.get(), out_stream.get(), size, encoder);
        } else {
            ok = compress_stream(in_stream.get(), out_stream.get(), size, encoder);
        }
    } else if (size_components.size() == 3) {
        hcde::singlethread_cpu_encoder<hcde::fast_profile<float, 3>> encoder;
        auto size = hcde::extent<3>{size_components[0], size_components[1], size_components[2]};
        if (decompress) {
            ok = decompress_stream(in_stream.get(), out_stream.get(), size, encoder);
        } else {
            ok = compress_stream(in_stream.get(), out_stream.get(), size, encoder);
        }
    }

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

