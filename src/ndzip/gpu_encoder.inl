#pragma once

#include "common.hh"

#include <numeric>
#include <stdexcept>
#include <vector>

#include <SYCL/sycl.hpp>


namespace ndzip::detail::gpu {

template<typename T>
using local_accessor
        = sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local>;

template<unsigned Dims, typename U, typename T>
U extent_cast(const T &e) {
    U v;
    for (unsigned i = 0; i < Dims; ++i) {
        v[i] = e[i];
    }
    return v;
}

template<typename U, unsigned Dims>
U extent_cast(const extent<Dims> &e) {
    return extent_cast<Dims, U>(e);
}

template<typename T, int Dims>
T extent_cast(const sycl::range<Dims> &r) {
    return extent_cast<static_cast<unsigned>(Dims), T>(r);
}

template<typename T, int Dims>
T extent_cast(const sycl::id<Dims> &r) {
    return extent_cast<static_cast<unsigned>(Dims), T>(r);
}

template<typename U, typename T>
[[gnu::always_inline]] U bit_cast(T v) {
    static_assert(std::is_pod_v<T> && std::is_pod_v<U> && sizeof(U) == sizeof(T));
    U cast;
    __builtin_memcpy(&cast, &v, sizeof cast);
    return cast;
}


template<int Dims>
class work_item : public sycl::nd_item<Dims> {
  public:
    work_item(const sycl::nd_item<Dims> &nd_item)
        : sycl::nd_item<Dims>(nd_item) {}  // NOLINT(google-explicit-constructor)

    size_t num_threads() const { return this->get_local_range(Dims - 1); }
    size_t thread_id() const { return this->get_local_id(Dims - 1); }
    void local_memory_barrier() const { this->barrier(sycl::access::fence_space::local_space); }
};

template<int Dims>
work_item(const sycl::nd_item<Dims> &) -> work_item<Dims>;


constexpr size_t num_threads_per_hypercube = 32;

class hypercube_item : public work_item<2> {
  public:
    hypercube_item(const sycl::nd_item<2> &nd_item)  // NOLINT(google-explicit-constructor)
        : work_item<2>(nd_item) {}

    constexpr size_t
    num_threads() const {  // NOLINT(readability-convert-member-functions-to-static)
        return num_threads_per_hypercube;
    }

    size_t hc_index() { return this->get_global_id(0); }
    size_t num_hypercubes() { return this->get_global_range(0); }
};

class hypercube_range : public sycl::nd_range<2> {
  public:
    explicit hypercube_range(size_t num_hypercubes, size_t first_hc_index = 0)
        : sycl::nd_range<2>{sycl::range<2>{num_hypercubes, num_threads_per_hypercube},
                sycl::range<2>{1, num_threads_per_hypercube}, sycl::id<2>{first_hc_index, 0}} {}
};


template<typename DestAccessor, typename SourceAccessor, int ItemDims>
void nd_memcpy(const DestAccessor &dest, const SourceAccessor &source, size_t count,
        work_item<ItemDims> item) {
    for (size_t i = item.thread_id(); i < count; i += item.num_threads()) {
        dest[i] = source[i];
    }
}


template<typename Scalar, int ItemDims>
void local_inclusive_prefix_sum(
        /* local */ Scalar *scratch, size_t n_elements, work_item<ItemDims> item) {
    // Hillis-Steele (short-span) prefix sum
    size_t pout = 0;
    size_t pin;
    for (size_t offset = 1; offset < n_elements || pout > 0; offset <<= 1u) {
        pout = 1 - pout;
        pin = 1 - pout;
        for (size_t i = item.thread_id(); i < n_elements; i += item.num_threads()) {
            if (i >= offset) {
                scratch[pout * n_elements + i]
                        = scratch[pin * n_elements + i] + scratch[pin * n_elements + i - offset];
            } else {
                scratch[pout * n_elements + i] = scratch[pin * n_elements + i];
            }
        }
        item.local_memory_barrier();
    }
}


template<typename Scalar>
void global_inclusive_prefix_sum_reduce(
        /* global */ Scalar *__restrict big, /* global */ Scalar *__restrict small,
        /* local */ Scalar *__restrict scratch, size_t count, work_item<1> item) {
    const size_t global_id = item.get_global_id(0);
    const size_t local_id = item.get_local_id(0);
    const size_t local_size = item.get_local_range(0);
    const size_t group_id = item.get_group(0);

    scratch[local_id] = global_id < count ? big[global_id] : 0;
    item.local_memory_barrier();

    local_inclusive_prefix_sum(scratch, local_size, work_item{item});

    const Scalar local_result = scratch[local_id];
    if (global_id < count) { big[global_id] = local_result; }
    if (local_id == local_size - 1) { small[group_id] = local_result; }
}


template<typename Scalar>
void global_inclusive_prefix_sum_expand(/* global */ Scalar *__restrict small,
        /* global */ Scalar *__restrict big, size_t count, work_item<1> item) {
    // TODO range check (necessary? or can it be eliminated by increasing buffer size?)
    // `+ get_local_size`: We skip the first WG since `small` orginates from an inclusive, not an
    // exclusive scan
    const size_t global_id = item.get_global_id(0) + item.get_local_range(0);
    if (global_id < count) { big[global_id] += small[item.get_group(0)]; }
}


template<typename Scalar>
class hierarchical_inclusive_prefix_sum {
  public:
    hierarchical_inclusive_prefix_sum(size_t n_elems, size_t local_size)
        : _n_elems(n_elems), _local_size(local_size) {
        while (n_elems > 1) {
            n_elems = (n_elems + local_size - 1) / local_size;
            _intermediate_buffers.emplace_back(n_elems);
        }
    }

    void operator()(sycl::queue &queue, sycl::buffer<Scalar> &in_out_buffer) const {
        using sam = sycl::access::mode;

        for (size_t i = 0; i < _intermediate_buffers.size(); ++i) {
            auto &big_buffer = i ? _intermediate_buffers[i - 1] : in_out_buffer;
            auto count = i ? big_buffer.get_count() : _n_elems;
            auto &small_buffer = _intermediate_buffers[i];
            const auto global_range = sycl::range<1>{
                    (big_buffer.get_count() + _local_size - 1) / _local_size * _local_size};
            const auto local_range = sycl::range<1>{_local_size};
            const auto scratch_range = sycl::range<1>{_local_size * 2};
            queue.submit([&](sycl::handler &cgh) {
                auto big_acc = big_buffer.template get_access<sam::read_write>(cgh);
                auto small_acc = small_buffer.template get_access<sam::discard_write>(cgh);
                auto scratch_acc = local_accessor<Scalar>{scratch_range, cgh};
                cgh.parallel_for<reduction_kernel>(sycl::nd_range<1>(global_range, local_range),
                        [big_acc, small_acc, scratch_acc, count](sycl::nd_item<1> item) {
                            global_inclusive_prefix_sum_reduce<Scalar>(big_acc.get_pointer(),
                                    small_acc.get_pointer(), scratch_acc.get_pointer(), count,
                                    item);
                        });
            });
        }

        for (size_t i = 1; i < _intermediate_buffers.size(); ++i) {
            auto ii = _intermediate_buffers.size() - 1 - i;
            auto &small_buffer = _intermediate_buffers[ii];
            auto &big_buffer = ii > 0 ? _intermediate_buffers[ii - 1] : in_out_buffer;
            auto count = i ? big_buffer.get_count() : _n_elems;
            const auto global_range = sycl::range<1>{
                    (big_buffer.get_count() + _local_size - 1) / _local_size * _local_size};
            const auto local_range = sycl::range<1>{_local_size};
            queue.submit([&](sycl::handler &cgh) {
                auto small_acc = small_buffer.template get_access<sam::read_write>(cgh);
                auto big_acc = big_buffer.template get_access<sam::discard_write>(cgh);
                cgh.parallel_for<expansion_kernel>(sycl::nd_range<1>(global_range, local_range),
                        [small_acc, big_acc, count](sycl::nd_item<1> item) {
                            global_inclusive_prefix_sum_expand<Scalar>(
                                    small_acc.get_pointer(), big_acc.get_pointer(), count, item);
                        });
            });
        }
    }

  private:
    class reduction_kernel;
    class expansion_kernel;

    size_t _n_elems;
    size_t _local_size;
    mutable std::vector<sycl::buffer<Scalar>> _intermediate_buffers;
};


template<typename Profile>
size_t global_offset(size_t local_offset, extent<Profile::dimensions> global_size) {
    size_t global_offset = 0;
    size_t global_stride = 1;
    for (unsigned d = 0; d < Profile::dimensions; ++d) {
        global_offset += global_stride * (local_offset % Profile::hypercube_side_length);
        local_offset /= Profile::hypercube_side_length;
        global_stride *= global_size[Profile::dimensions - 1 - d];
    }
    return global_offset;
}


template<typename Profile, typename F>
void distribute_for_hypercube_indices(extent<Profile::dimensions> data_size, hypercube_item item,
        F &&f) {
    auto side_length = Profile::hypercube_side_length;
    auto hc_size = ipow(side_length, Profile::dimensions);
    auto hc_offset
            = detail::extent_from_linear_id(item.hc_index(), data_size / side_length) * side_length;

    size_t initial_offset = linear_offset(hc_offset, data_size);
    for (size_t local_idx = item.thread_id(); local_idx < hc_size;
         local_idx += item.num_threads()) {
        // TODO re-calculating the GO every iteration is probably painfully slow
        size_t global_idx = initial_offset + global_offset<Profile>(local_idx, data_size);
        f(global_idx, local_idx);
    }
}


template<typename Profile>
void load_hypercube(/* global */ const typename Profile::data_type *__restrict data,
        /* local */ typename Profile::bits_type *__restrict cube,
                                 extent<Profile::dimensions> data_size, hypercube_item item) {
    distribute_for_hypercube_indices<Profile>(data_size, item, [&](size_t global_idx, size_t local_idx) {
      cube[local_idx] = bit_cast<typename Profile::bits_type>(data[global_idx]);
    });
}


template<typename Profile>
void store_hypercube(/* local */ const typename Profile::bits_type *__restrict cube,
        /* global */ typename Profile::data_type *__restrict data,
        extent<Profile::dimensions> data_size, hypercube_item item) {
    distribute_for_hypercube_indices<Profile>(data_size, item, [&](size_t global_idx, size_t local_idx) {
      data[global_idx] = bit_cast<typename Profile::bits_type>(cube[local_idx]);
    });
}


template<typename Profile>
void block_transform(/* local */ typename Profile::bits_type *x, hypercube_item item) {
    constexpr auto n = Profile::hypercube_side_length;
    constexpr auto dims = Profile::dimensions;
    constexpr auto hc_size = ipow(n, dims);

    for (size_t i = item.thread_id(); i < hc_size; i += item.num_threads()) {
        x[i] = rotate_left_1(x[i]);
    }

    item.local_memory_barrier();

    if constexpr (dims == 1) {
        if (item.thread_id() == 0) { block_transform_step(x, n, 1); }
    } else if constexpr (dims == 2) {
        for (size_t i = item.thread_id(); i < n; i += item.num_threads()) {
            const auto ii = n * i;
            block_transform_step(x + ii, n, 1);
        }
        item.local_memory_barrier();
        for (size_t i = item.thread_id(); i < n; i += item.num_threads()) {
            block_transform_step(x + i, n, n);
        }
    } else if constexpr (dims == 3) {
        for (size_t i = item.thread_id(); i < n; i += item.num_threads()) {
            const auto ii = n * n * i;
            for (size_t j = 0; j < n; ++j) {
                block_transform_step(x + ii + j, n, n);
            }
        }
        item.local_memory_barrier();
        for (size_t i = item.thread_id(); i < n * n; i += item.num_threads()) {
            const auto ii = n * i;
            block_transform_step(x + ii, n, 1);
        }
        item.local_memory_barrier();
        for (size_t i = item.thread_id(); i < n * n; i += item.num_threads()) {
            block_transform_step(x + i, n, n * n);
        }
    }

    item.local_memory_barrier();

    for (size_t i = item.thread_id(); i < hc_size; i += item.num_threads()) {
        x[i] = complement_negative(x[i]);
    }
}


template<typename Profile>
void inverse_block_transform(/* local */ typename Profile::bits_type *x, hypercube_item item) {
    constexpr auto n = Profile::hypercube_side_length;
    constexpr auto dims = Profile::dimensions;
    constexpr auto hc_size = ipow(n, dims);

    for (size_t i = item.thread_id(); i < hc_size; i += item.num_threads()) {
        x[i] = complement_negative(x[i]);
    }

    item.local_memory_barrier();

    if (dims == 1) {
        if (item.thread_id() == 0) { inverse_block_transform_step(x, n, 1); }
    } else if (dims == 2) {
        for (size_t i = item.thread_id(); i < n; i += item.num_threads()) {
            inverse_block_transform_step(x + i, n, n);
        }
        item.local_memory_barrier();
        for (size_t i = item.thread_id(); i < n; i += item.num_threads()) {
            auto ii = i * n;
            inverse_block_transform_step(x + ii, n, 1);
        }
    } else if (dims == 3) {
        for (size_t i = item.thread_id(); i < n * n; i += item.num_threads()) {
            inverse_block_transform_step(x + i, n, n * n);
        }
        item.local_memory_barrier();
        for (size_t i = item.thread_id(); i < n * n; i += item.num_threads()) {
            auto ii = i * n;
            inverse_block_transform_step(x + ii, n, 1);
        }
        item.local_memory_barrier();
        for (size_t i = item.thread_id(); i < n; i += item.num_threads()) {
            auto ii = i * n * n;
            for (size_t j = 0; j < n; ++j) {
                inverse_block_transform_step(x + ii + j, n, n);
            }
        }
    }

    item.local_memory_barrier();

    for (size_t i = item.thread_id(); i < hc_size; i += item.num_threads()) {
        x[i] = rotate_right_1(x[i]);
    }
}


template<typename Bits>
void transpose_bits(/* local */ Bits *cube, hypercube_item item) {
    constexpr auto cols_per_thread = bitsof<Bits> / item.num_threads();

    Bits columns[cols_per_thread] = {0};
    for (size_t k = 0; k < bitsof<Bits>; ++k) {
        auto row = cube[k];
        for (size_t c = 0; c < cols_per_thread; ++c) {
            size_t i = c * item.num_threads() + item.thread_id();
            columns[c] |= ((row >> (bitsof<Bits> - 1 - i)) & Bits{1}) << (bitsof<Bits> - 1 - k);
        }
    }

    item.local_memory_barrier();

    for (size_t c = 0; c < cols_per_thread; ++c) {
        size_t i = c * item.num_threads() + item.thread_id();
        cube[i] = columns[c];
    }
}


template<typename Bits>
size_t compact_zero_words(/* global */ Bits *__restrict out, /* local */ const Bits *__restrict in,
        /* local */ Bits *__restrict scratch, hypercube_item item) {
    constexpr auto n_columns = bitsof<Bits>;

    for (size_t i = item.thread_id(); i < n_columns; i += item.num_threads()) {
        scratch[i] = in[i] != 0;
        // TODO this is dumb, we can just OR values before transposition
        scratch[2 * n_columns + i] = Bits{in[i] != 0} << i;
    }

    item.local_memory_barrier();

    local_inclusive_prefix_sum(scratch, n_columns, item);

    // Header reduction - TODO merge with prefix sum loop, or better yet, move into a all_zero check
    for (size_t offset = n_columns / 2; offset > 0; offset /= 2) {
        for (size_t i = item.thread_id(); i < n_columns; i += item.num_threads()) {
            if (i < offset) { scratch[2 * n_columns + i] |= scratch[2 * n_columns + i + offset]; }
        }
        item.local_memory_barrier();
    }

    if (item.thread_id() == 0) { out[0] = scratch[2 * n_columns]; }

    for (size_t i = item.thread_id(); i < n_columns; i += item.num_threads()) {
        if (in[i] != 0) {
            size_t offset = i ? scratch[i - 1] : 0;
            out[1 + offset] = in[i];
        }
    }

    return 1 + scratch[n_columns - 1];
}


template<typename Bits>
size_t zero_bit_encode(/* local */ Bits *__restrict cube, /* global */ Bits *__restrict stream,
        /* local */ Bits *__restrict scratch, size_t hc_size, hypercube_item item) {
    for (size_t offset = 0; offset < hc_size; offset += bitsof<Bits>) {
        transpose_bits(cube + offset, item);
    }

    auto out = stream;
    for (size_t offset = 0; offset < hc_size; offset += bitsof<Bits>) {
        item.local_memory_barrier();
        out += compact_zero_words(out, cube + offset, scratch, item);
    }

    return out - stream;
}


template<typename Bits>
size_t expand_zero_words(/* local */ Bits *__restrict out, /* global */ const Bits *__restrict in,
        /* local */ Bits *__restrict scratch, hypercube_item item) {
    constexpr auto n_columns = bitsof<Bits>;

    auto head = in[0];
    for (size_t i = item.thread_id(); i < n_columns; i += item.num_threads()) {
        scratch[i] = (head >> i) & Bits{1};
    }

    item.local_memory_barrier();

    local_inclusive_prefix_sum(scratch, n_columns, item);

    for (size_t i = item.thread_id(); i < n_columns; i += item.num_threads()) {
        if (((head >> i) & Bits{1}) != 0) {
            out[i] = in[scratch[i]];  // +1 from header offset and -1 from inclusive scan cancel
        } else {
            out[i] = 0;
        }
    }

    return 1 + scratch[n_columns - 1];
}


template<typename Bits>
void zero_bit_decode(/* global */ const Bits *__restrict stream, /* local */ Bits *__restrict cube,
        /* local */ Bits *__restrict scratch, size_t hc_size, hypercube_item item) {
    auto in = stream;
    for (size_t offset = 0; offset < hc_size; offset += bitsof<Bits>) {
        in += expand_zero_words(cube + offset, in, scratch, item);
        item.local_memory_barrier();
    }

    for (size_t offset = 0; offset < hc_size; offset += bitsof<Bits>) {
        transpose_bits(cube + offset, item);
    }
}


template<typename Profile>
void compress_hypercubes(/* global */ const typename Profile::data_type *__restrict data,
        /* global */ typename Profile::bits_type *__restrict stream_chunks,
        /* global */ detail::file_offset_type *__restrict stream_chunk_lengths,
        /* local */ typename Profile::bits_type *__restrict local_memory,
        extent<Profile::dimensions> data_size, hypercube_item item) {
    using bits_type = typename Profile::bits_type;

    const auto side_length = Profile::hypercube_side_length;
    const auto hc_size = detail::ipow(side_length, Profile::dimensions);
    const auto cube = local_memory;
    const auto scratch = local_memory + hc_size;
    const auto max_chunk_size
            = (Profile::compressed_block_size_bound + sizeof(bits_type) - 1) / sizeof(bits_type);

    load_hypercube<Profile>(data, cube, data_size, item);
    item.local_memory_barrier();
    block_transform<Profile>(cube, item);
    item.local_memory_barrier();
    stream_chunk_lengths[item.hc_index()] = zero_bit_encode<bits_type>(
            cube, stream_chunks + item.hc_index() * max_chunk_size, scratch, hc_size, item);
}


template<typename Profile>
void compact_stream(/* global */ typename Profile::bits_type *stream,
        /* global */ typename Profile::bits_type *stream_chunks,
        /* global */ const file_offset_type *stream_chunk_offsets,
        /* global */ const file_offset_type *stream_chunk_lengths, hypercube_item item) {
    using bits_type = typename Profile::bits_type;

    const auto max_chunk_size
            = (Profile::compressed_block_size_bound + sizeof(bits_type) - 1) / sizeof(bits_type);
    const auto this_chunk_size = stream_chunk_lengths[item.hc_index()];
    const auto this_chunk_offset = item.hc_index() ? stream_chunk_offsets[item.hc_index() - 1] : 0;
    const auto source = stream_chunks + item.hc_index() * max_chunk_size;
    const auto dest = stream + this_chunk_offset;
    detail::gpu::nd_memcpy(dest, source, this_chunk_size, item);
}


template<typename Profile>
void decompress_hypercubes(/* global */ const typename Profile::bits_type *__restrict stream,
        /* global */ const typename Profile::data_type *__restrict data,
        /* local */ typename Profile::bits_type *__restrict local_memory,
                                        extent<Profile::dimensions> data_size, hypercube_item item) {
    using bits_type = typename Profile::bits_type;

    const auto side_length = Profile::hypercube_side_length;
    const auto hc_size = detail::ipow(side_length, Profile::dimensions);
    const auto cube = local_memory;
    const auto scratch = local_memory + hc_size;

    // TODO is this reinterpret_cast sane?
    const auto offset_table = reinterpret_cast<const file_offset_type*>(stream);
    const auto chunk_offset = item.hc_index() ? offset_table[item.hc_index() - 1] : 0;

    zero_bit_decode<bits_type>(stream + chunk_offset / 4, cube, hc_size, item);
    item.local_memory_barrier();
    inverse_block_transform<Profile>(cube, item);
    item.local_memory_barrier();
    store_hypercube<Profile>(cube, data, data_size, item);
}


// SYCL kernel names
template<typename, unsigned>
class block_compression_kernel;

template<typename, unsigned>
class block_compaction_kernel;

}  // namespace ndzip::detail::gpu


template<typename T, unsigned Dims>
struct ndzip::gpu_encoder<T, Dims>::impl {
    sycl::queue q;
};

template<typename T, unsigned Dims>
ndzip::gpu_encoder<T, Dims>::gpu_encoder() : _pimpl(std::make_unique<impl>()) {
}

template<typename T, unsigned Dims>
ndzip::gpu_encoder<T, Dims>::~gpu_encoder() = default;


template<typename T, unsigned Dims>
size_t ndzip::gpu_encoder<T, Dims>::compress(
        const slice<const data_type, dimensions> &data, void *stream) const {
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;
    using sam = sycl::access::mode;

    constexpr auto side_length = profile::hypercube_side_length;
    constexpr auto hc_size = detail::ipow(side_length, profile::dimensions);
    const auto max_chunk_size
            = (profile::compressed_block_size_bound + sizeof(bits_type) - 1) / sizeof(bits_type);

    detail::file<profile> file(data.size());

    sycl::buffer<data_type, dimensions> data_buffer{
            data.data(), detail::gpu::extent_cast<sycl::range<dimensions>>(data.size())};
    sycl::buffer<bits_type, 1> stream_chunks_buffer{
            sycl::range<1>{file.num_hypercubes() * max_chunk_size}};
    sycl::buffer<detail::file_offset_type, 1> stream_chunk_lengths_buffer{
            sycl::range<1>{file.num_hypercubes()}};

    _pimpl->q.submit([&](sycl::handler &cgh) {
        auto data_acc = data_buffer.template get_access<sam::read>(cgh);
        auto stream_chunks_acc
                = stream_chunks_buffer.template get_access<sam::discard_read_write>(cgh);
        auto stream_chunk_lengths_acc
                = stream_chunk_lengths_buffer.get_access<sam::discard_write>(cgh);
        auto local_memory_acc = detail::gpu::local_accessor<bits_type>{
                hc_size + 3 * detail::bitsof<bits_type>, cgh};
        auto data_size = data.size();
        cgh.parallel_for<detail::gpu::block_compression_kernel<T, Dims>>(
                detail::gpu::hypercube_range{file.num_hypercubes()},
                [=](detail::gpu::hypercube_item item) {
                    detail::gpu::compress_hypercubes<profile>(data_acc.get_pointer(),
                            stream_chunks_acc.get_pointer(), stream_chunk_lengths_acc.get_pointer(),
                            local_memory_acc.get_pointer(), data_size, item);
                });
    });

    sycl::buffer<detail::file_offset_type> stream_chunk_offsets_buffer{
            sycl::range<1>{file.num_hypercubes()}};
    _pimpl->q.submit([&](sycl::handler &cgh) {
        cgh.copy(stream_chunk_lengths_buffer.get_access<sam::read>(cgh),
                stream_chunk_offsets_buffer.get_access<sam::discard_write>(cgh));
    });

    detail::gpu::hierarchical_inclusive_prefix_sum<detail::file_offset_type> prefix_sum(
            file.num_hypercubes(), 256 /* local size */);
    prefix_sum(_pimpl->q, stream_chunk_offsets_buffer);

    detail::file_offset_type num_compressed_words;
    _pimpl->q.submit([&](sycl::handler &cgh) {
        cgh.copy(stream_chunk_offsets_buffer.get_access<sam::read>(
                         cgh, sycl::range<1>{1}, sycl::id<1>{file.num_hypercubes() - 1}),
                &num_compressed_words);
    });

    sycl::buffer<bits_type> stream_buffer(stream_chunks_buffer.get_count());
    _pimpl->q.submit([&](sycl::handler &cgh) {
        // TODO is the cast sane?
        cgh.copy(stream_buffer.template get_access<sam::read>(cgh),
                static_cast<bits_type *>(stream));
    });

    _pimpl->q.submit([&](sycl::handler &cgh) {
        auto stream_chunks_acc = stream_chunks_buffer.template get_access<sam::read>(cgh);
        auto stream_chunk_lengths_acc = stream_chunk_lengths_buffer.get_access<sam::read>(cgh);
        auto stream_chunk_offsets_acc = stream_chunk_offsets_buffer.get_access<sam::read>(cgh);
        auto stream_acc = stream_buffer.template get_access<sam::discard_write>(cgh);
        cgh.parallel_for<detail::gpu::block_compaction_kernel<T, Dims>>(
                // TODO hypercube_range / hypercube_item might not be optimal here, experiment
                // with other local sizes
                detail::gpu::hypercube_range{file.num_hypercubes()}, [=](sycl::nd_item<2> item) {
                    detail::gpu::compact_stream<profile>(stream_acc.get_pointer(),
                            stream_chunks_acc.get_pointer(), stream_chunk_offsets_acc.get_pointer(),
                            stream_chunk_lengths_acc.get_pointer(), item);
                });
    });

    _pimpl->q.submit([&](sycl::handler &cgh) {
        cgh.copy(stream_buffer.template get_access<sam::read>(cgh),
                // TODO is this reinterpret_cast sane?
                reinterpret_cast<bits_type *>(
                        static_cast<std::byte *>(stream) + file.file_header_length()));
    });

    _pimpl->q.wait();

    size_t stream_pos = file.file_header_length() + num_compressed_words * sizeof(bits_type);
    stream_pos += detail::pack_border(
            static_cast<char *>(stream) + stream_pos, data, profile::hypercube_side_length);
    return stream_pos;
}


template<typename T, unsigned Dims>
size_t ndzip::gpu_encoder<T, Dims>::decompress(
        const void *stream, size_t bytes, const slice<data_type, dimensions> &data) const {
    return 0;
}


namespace ndzip {

extern template class gpu_encoder<float, 1>;
extern template class gpu_encoder<float, 2>;
extern template class gpu_encoder<float, 3>;
extern template class gpu_encoder<double, 1>;
extern template class gpu_encoder<double, 2>;
extern template class gpu_encoder<double, 3>;

}  // namespace ndzip
