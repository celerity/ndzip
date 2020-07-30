#pragma once

#include <utility>
#include <memory>
#include <cstdlib>
#include <stdexcept>


#ifdef __unix__
#   define HCDE_MMAP_SUPPORT 1
#else
#   define HCDE_MMAP_SUPPORT 0
#endif


namespace hcde::detail {

class io_error
    : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
};

class input_stream {
    public:
        virtual ~input_stream() noexcept(false) = default;
        virtual std::pair<const void *, size_t> read_some(size_t remainder_from_last_chunk) = 0;

        std::pair<const void *, size_t> read_some() { return read_some(0); }

        virtual const void *read_exact() = 0;
};

class output_stream {
    public:
        virtual ~output_stream() noexcept(false) = default;
        virtual void *get_write_buffer() = 0;
        virtual void commit_chunk(size_t length) = 0;
};

class io_factory {
    public:
        virtual ~io_factory() = default;
        virtual std::unique_ptr<input_stream> create_input_stream(const std::string &file_name,
            size_t chunk_length) const = 0;
        virtual std::unique_ptr<output_stream> create_output_stream(const std::string &file_name,
            size_t max_chunk_length) const = 0;
};

class stdio_io_factory
    : public io_factory {
    public:
        std::unique_ptr<input_stream> create_input_stream(const std::string &file_name,
            size_t chunk_length) const override;

        std::unique_ptr<output_stream>
        create_output_stream(const std::string &file_name, size_t max_chunk_length) const override;
};

#if HCDE_MMAP_SUPPORT

class mmap_io_factory
    : public io_factory {
    public:
        std::unique_ptr<input_stream> create_input_stream(const std::string &file_name,
            size_t chunk_length) const override;

        std::unique_ptr<output_stream> create_output_stream(const std::string &file_name,
            size_t max_chunk_length) const override;
};

#endif

}
