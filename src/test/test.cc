#include <ndzip/common.hh>
#include <ndzip/cpu_encoder.inl>

#if NDZIP_GPU_SUPPORT
#include <ndzip/gpu_encoder.inl>
#endif

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>


using namespace ndzip;
using namespace ndzip::detail;


template<typename Arithmetic>
static std::vector<Arithmetic> make_random_vector(size_t size) {
    std::vector<Arithmetic> vector(size);
    auto gen = std::minstd_rand();
    if constexpr (std::is_floating_point_v<Arithmetic>) {
        auto dist = std::uniform_real_distribution<Arithmetic>();
        std::generate(vector.begin(), vector.end(), [&] { return dist(gen); });
    } else {
        auto dist = std::uniform_int_distribution<Arithmetic>();
        std::generate(vector.begin(), vector.end(), [&] { return dist(gen); });
    }
    return vector;
}


TEMPLATE_TEST_CASE("block transform is reversible", "[profile]", (profile<float, 1>),
        (profile<float, 2>), (profile<float, 3>), (profile<double, 1>), (profile<double, 2>),
        (profile<double, 3>) ) {
    using bits_type = typename TestType::bits_type;

    const auto input = make_random_vector<bits_type>(
            ipow(TestType::hypercube_side_length, TestType::dimensions));

    auto transformed = input;
    detail::block_transform(
            transformed.data(), TestType::dimensions, TestType::hypercube_side_length);

    detail::inverse_block_transform(
            transformed.data(), TestType::dimensions, TestType::hypercube_side_length);

    CHECK(input == transformed);
}


TEMPLATE_TEST_CASE("CPU zero-word compaction is reversible", "[cpu]", uint32_t, uint64_t) {
    std::vector<TestType> input(bitsof<TestType>);
    auto gen = std::minstd_rand();  // NOLINT(cert-msc51-cpp)
    auto dist = std::uniform_int_distribution<TestType>();
    size_t zeroes = 0;
    for (auto &v : input) {
        auto r = dist(gen);
        if (r % 5 == 2) {
            v = 0;
            zeroes++;
        } else {
            v = r;
        }
    }

    std::vector<std::byte> compact((bitsof<TestType> + 1) * sizeof(TestType));
    auto bytes_written = detail::cpu::compact_zero_words(input.data(), compact.data());
    CHECK(bytes_written == (bitsof<TestType> + 1 - zeroes) * sizeof(TestType));

    std::vector<TestType> output(bitsof<TestType>);
    auto bytes_read = detail::cpu::expand_zero_words(compact.data(), output.data());
    CHECK(bytes_read == bytes_written);
    CHECK(output == input);
}


TEMPLATE_TEST_CASE("CPU bit transposition is reversible", "[cpu]", uint32_t, uint64_t) {
    alignas(cpu::simd_width_bytes) TestType input[bitsof<TestType>];
    auto rng = std::minstd_rand(1);
    auto bit_dist = std::uniform_int_distribution<TestType>();
    auto shift_dist = std::uniform_int_distribution<unsigned>(0, bitsof<TestType> - 1);
    for (auto &value : input) {
        value = bit_dist(rng) >> shift_dist(rng);
    }

    alignas(cpu::simd_width_bytes) TestType transposed[bitsof<TestType>];
    detail::cpu::transpose_bits(input, transposed);

    alignas(cpu::simd_width_bytes) TestType output[bitsof<TestType>];
    detail::cpu::transpose_bits(transposed, output);

    CHECK(memcmp(input, output, sizeof input) == 0);
}


using border_slice = std::pair<size_t, size_t>;
using slice_vec = std::vector<border_slice>;

namespace std {
ostream &operator<<(ostream &os, const border_slice &s) {
    return os << "(" << s.first << ", " << s.second << ")";
}
}  // namespace std

template<unsigned Dims>
static auto dump_border_slices(const extent<Dims> &size, unsigned side_length) {
    slice_vec v;
    for_each_border_slice(
            size, side_length, [&](size_t offset, size_t count) { v.emplace_back(offset, count); });
    return v;
}


TEST_CASE("for_each_border_slice iterates correctly") {
    CHECK(dump_border_slices(extent<2>{4, 4}, 4) == slice_vec{});
    CHECK(dump_border_slices(extent<2>{4, 6}, 2) == slice_vec{});
    CHECK(dump_border_slices(extent<2>{5, 4}, 4) == slice_vec{{16, 4}});
    CHECK(dump_border_slices(extent<2>{4, 5}, 4) == slice_vec{{4, 1}, {9, 1}, {14, 1}, {19, 1}});
    CHECK(dump_border_slices(extent<2>{4, 5}, 2) == slice_vec{{4, 1}, {9, 1}, {14, 1}, {19, 1}});
    CHECK(dump_border_slices(extent<2>{4, 6}, 4) == slice_vec{{4, 2}, {10, 2}, {16, 2}, {22, 2}});
    CHECK(dump_border_slices(extent<2>{4, 6}, 5) == slice_vec{{0, 24}});
    CHECK(dump_border_slices(extent<2>{6, 4}, 5) == slice_vec{{0, 24}});
}


TEMPLATE_TEST_CASE("file produces a sane hypercube / header layout", "[file]",
        (std::integral_constant<unsigned, 1>), (std::integral_constant<unsigned, 2>),
        (std::integral_constant<unsigned, 3>), (std::integral_constant<unsigned, 4>) ) {
    constexpr unsigned dims = TestType::value;
    using profile = detail::profile<float, dims>;
    const size_t n = 100;
    const auto n_hypercubes_per_dim = n / profile::hypercube_side_length;
    const auto side_length = profile::hypercube_side_length;

    extent<dims> size;
    for (unsigned d = 0; d < dims; ++d) {
        size[d] = n;
    }

    std::vector<std::vector<extent<dims>>> superblocks;
    std::vector<bool> visited(ipow(n_hypercubes_per_dim, dims));

    file<profile> f(size);
    std::vector<extent<dims>> blocks;
    size_t hypercube_index = 0;
    f.for_each_hypercube([&](auto hc_offset, auto hc_index) {
        CHECK(hc_index == hypercube_index);

        auto off = hc_offset;
        for (unsigned d = 0; d < dims; ++d) {
            CHECK(off[d] < n);
            CHECK(off[d] % side_length == 0);
        }

        auto cell_index = off[0] / side_length;
        for (unsigned d = 1; d < dims; ++d) {
            cell_index = cell_index * n_hypercubes_per_dim + off[d] / side_length;
        }
        CHECK(!visited[cell_index]);
        visited[cell_index] = true;

        blocks.push_back(off);
        ++hypercube_index;
    });
    CHECK(blocks.size() == f.num_hypercubes());

    CHECK(std::all_of(visited.begin(), visited.end(), [](auto b) { return b; }));

    CHECK(f.file_header_length() == f.num_hypercubes() * sizeof(detail::file_offset_type));
    CHECK(f.num_hypercubes() == ipow(n_hypercubes_per_dim, dims));
}


/* requires Profile parameters
TEMPLATE_TEST_CASE("encoder produces the expected bit stream", "[encoder]",
    (cpu_encoder<float, 2>), (cpu_encoder<float, 3>),
    (mt_cpu_encoder<float, 2>), (mt_cpu_encoder<float, 3>)
) {
    using profile = detail::profile<typename TestType::data_type, TestType::dimensions>;
    using bits_type = typename profile::bits_type;
    using hc_offset_type = typename profile::hypercube_offset_type;

    const size_t n = 199;
    const auto cell = 3.141592f;
    const auto border = 2.71828f;
    constexpr auto dims = profile::dimensions;

    std::vector<float> data(ipow(n, dims));
    slice<float, dims> array(data.data(), extent<dims>::broadcast(n));

    const auto border_start = n / profile::hypercube_side_length * profile::hypercube_side_length;
    for_each_in_hyperslab(array.size(), [=](auto index) {
        if (std::all_of(index.begin(), index.end(), [=](auto s) { return s < border_start; })) {
            array[index] = cell;
        } else {
            array[index] = border;
        }
    });

    file<profile> f(array.size());
    REQUIRE(f.num_hypercubes() > 1);

    TestType encoder;
    std::vector<std::byte> stream(ndzip::compressed_size_bound<data_type>(array.size()));
    size_t size = encoder.compress(array, stream.data());

    CHECK(size <= stream.size());
    stream.resize(size);

    const size_t hc_size = sizeof(float) * ipow(profile::hypercube_side_length, dims);

    const auto *file_header = stream.data();
    f.for_each_hypercube([&](auto hc, auto hc_index) {
        const auto hc_offset = hc_index * hc_size;
        if (hc_index > 0) {
            const void *hc_offset_address = file_header + (hc_index - 1) * sizeof(hc_offset_type);
            CHECK(endian_transform(load_unaligned<hc_offset_type>(hc_offset_address))
                == hc_offset);
        }
        for (size_t i = 0; i < ipow(profile::hypercube_side_length, dims); ++i) {
            float value;
            const void *value_offset_address = file_header + hc_offset + i * sizeof value;
            detail::store_value<profile>(&value, load_unaligned<bits_type>(value_offset_address));
            CHECK(memcmp(&value, &cell, sizeof value) == 0);
        }
    });

    const auto border_offset = f.file_header_length() + f.num_hypercubes() * hc_size;
    const void *border_offset_address = file_header + (f.num_hypercubes() - 1) * sizeof(uint64_t);
    CHECK(endian_transform(load_unaligned<uint64_t>(border_offset_address)) == border_offset);
    size_t n_border_elems = 0;
    for_each_border_slice(array.size(), profile::hypercube_side_length, [&](auto, auto count) {
        for (unsigned i = 0; i < count; ++i) {
            float value;
            const void *value_offset_address = stream.data() + border_offset
                + (n_border_elems + i) * sizeof value;
            detail::store_value<profile>(&value, load_unaligned<bits_type>(value_offset_address));
            CHECK(memcmp(&value, &border, sizeof value) == 0);
        }
        n_border_elems += count;
    });
    CHECK(n_border_elems == num_elements(array.size()) - ipow(border_start, dims));
}
*/


TEMPLATE_TEST_CASE("encoder reproduces the bit-identical array", "[encoder]",
        (cpu_encoder<float, 1>), (cpu_encoder<float, 2>), (cpu_encoder<float, 3>),
        (cpu_encoder<double, 1>), (cpu_encoder<double, 2>), (cpu_encoder<double, 3>)
#if NDZIP_OPENMP_SUPPORT
                                                                    ,
        (mt_cpu_encoder<float, 1>), (mt_cpu_encoder<float, 2>), (mt_cpu_encoder<float, 3>),
        (mt_cpu_encoder<double, 1>), (mt_cpu_encoder<double, 2>), (mt_cpu_encoder<double, 3>)
#endif
#if NDZIP_GPU_SUPPORT
                                                                          ,
        (gpu_encoder<float, 1>), (gpu_encoder<float, 2>), (gpu_encoder<float, 3>),
        (gpu_encoder<double, 1>), (gpu_encoder<double, 2>), (gpu_encoder<double, 3>)
#endif
) {
    using data_type = typename TestType::data_type;
    using profile = detail::profile<data_type, TestType::dimensions>;

    constexpr auto dims = profile::dimensions;
    constexpr auto side_length = profile::hypercube_side_length;
    const size_t n = side_length * 4 - 1;

    auto input_data = make_random_vector<data_type>(ipow(n, dims));

    // Regression test: trigger bug in decoder optimization by ensuring first chunk = 0
    std::fill(input_data.begin(), input_data.begin() + bitsof<data_type>, data_type{});

    slice<const data_type, dims> input(input_data.data(), extent<dims>::broadcast(n));

    TestType encoder;
    std::vector<std::byte> stream(
            ndzip::compressed_size_bound<typename TestType::data_type>(input.size()));
    stream.resize(encoder.compress(input, stream.data()));

    cpu_encoder<data_type, dims> decoder;
    std::vector<data_type> output_data(ipow(n, dims));
    slice<data_type, dims> output(output_data.data(), extent<dims>::broadcast(n));
    decoder.decompress(stream.data(), stream.size(), output);

    CHECK(memcmp(input_data.data(), output_data.data(), input_data.size() * sizeof(float)) == 0);
}


#if NDZIP_GPU_SUPPORT

using sam = sycl::access::mode;
using sat = sycl::access::target;
using sycl::accessor, sycl::nd_range, sycl::buffer, sycl::nd_item, sycl::range, sycl::id,
        sycl::handler;

template<typename T, unsigned Dims>
struct mock_profile {
    using data_type = T;
    using bits_type = T;
    constexpr static unsigned dimensions = Dims;
    constexpr static unsigned hypercube_side_length = 2;
    constexpr static unsigned compressed_block_size_bound
            = sizeof(T) * (ipow(hypercube_side_length, dimensions) + 1);
};


template<typename Profile>
class load_and_dump_hypercube_kernel;

template<typename Profile>
static std::vector<typename Profile::bits_type>
load_and_dump_hypercube(const slice<const typename Profile::data_type, Profile::dimensions> &data,
        size_t hc_index, sycl::queue &q) {
    using data_type = typename Profile::data_type;
    using bits_type = typename Profile::bits_type;

    auto hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    buffer<data_type> data_buf{data.data(), range<1>{num_elements(data.size())}};
    std::vector<bits_type> result(hc_size * 2);
    buffer<bits_type> result_buf{result.size()};
    detail::file<Profile> file(data.size());

    q.submit([&](handler &cgh) {
        cgh.fill(result_buf.template get_access<sam::discard_write>(cgh), bits_type{0});
    });
    q.submit([&](handler &cgh) {
        auto data_acc = data_buf.template get_access<sam::read>(cgh);
        auto local_acc = accessor<bits_type, 1, sam::read_write, sat::local>(hc_size, cgh);
        auto result_acc = result_buf.template get_access<sam ::discard_write>(cgh);
        cgh.parallel_for<load_and_dump_hypercube_kernel<Profile>>(
                detail::gpu::hypercube_range{1, hc_index},
                [data_acc, local_acc, result_acc, data_size = data.size(), hc_size](
                        detail::gpu::hypercube_item item) {
                    detail::gpu::load_hypercube<Profile>(
                            data_acc.get_pointer(), local_acc.get_pointer(), data_size, item);
                    item.local_memory_barrier();
                    detail::gpu::nd_memcpy(result_acc, local_acc, hc_size, item);
                });
    });
    q.submit([&](handler &cgh) {
        cgh.copy(result_buf.template get_access<sam::read>(cgh), result.data());
    });
    q.wait();
    return result;
}


TEMPLATE_TEST_CASE(
        "correctly load small hypercubes into local memory", "[gpu]", uint32_t, uint64_t) {
    sycl::queue q;

    SECTION("1d") {
        using profile = mock_profile<TestType, 1>;
        std::vector<TestType> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        auto result = load_and_dump_hypercube<profile>(
                slice<TestType, 1>{data.data(), data.size()}, 1 /* hc_index */, q);
        CHECK(result == std::vector<TestType>{3, 4, 0, 0});
    }

    SECTION("2d") {
        using profile = mock_profile<TestType, 2>;
        // clang-format off
        std::vector<TestType> data{
            10, 20, 30, 40, 50, 60, 70, 80, 90,
            11, 21, 31, 41, 51, 61, 71, 81, 91,
            12, 22, 32, 42, 52, 62, 72, 82, 92,
            13, 23, 33, 43, 53, 63, 73, 83, 93,
            14, 24, 34, 44, 54, 64, 74, 84, 94,
            15, 25, 35, 45, 55, 65, 75, 85, 95,
            16, 26, 36, 46, 56, 66, 76, 86, 96,
            17, 27, 37, 47, 57, 67, 77, 87, 97,
        };
        // clang-format on
        auto result = load_and_dump_hypercube<profile>(
                slice<TestType, 2>{data.data(), extent{8, 9}}, 6 /* hc_index */, q);
        CHECK(result == std::vector<TestType>{52, 62, 53, 63, 0, 0, 0, 0});
    }

    SECTION("3d") {
        using profile = mock_profile<TestType, 3>;
        // clang-format off
        std::vector<TestType> data{
            111, 211, 311, 411, 511,
            121, 221, 321, 421, 521,
            131, 231, 331, 431, 531,
            141, 241, 341, 441, 541,
            151, 251, 351, 451, 551,

            112, 212, 312, 412, 512,
            122, 222, 322, 422, 522,
            132, 232, 332, 432, 532,
            142, 242, 342, 442, 542,
            152, 252, 352, 452, 552,

            113, 213, 313, 413, 513,
            123, 223, 323, 423, 523,
            133, 233, 333, 433, 533,
            143, 243, 343, 443, 543,
            153, 253, 353, 453, 553,

            114, 214, 314, 414, 514,
            124, 224, 324, 424, 524,
            134, 234, 334, 434, 534,
            144, 244, 344, 444, 544,
            154, 254, 354, 454, 554,

            115, 215, 315, 415, 515,
            125, 225, 325, 425, 525,
            135, 235, 335, 435, 535,
            145, 245, 345, 445, 545,
            155, 255, 355, 455, 555,
        };
        auto result = load_and_dump_hypercube<profile>(
                slice<TestType, 3>{data.data(), extent{5, 5, 5}}, 3 /* hc_index */, q);
        CHECK(result == std::vector<TestType>{331, 431, 341, 441, 332, 432, 342, 442,
                      0, 0, 0, 0, 0, 0, 0, 0});
        // clang-format on
    }
}


TEMPLATE_TEST_CASE("flattening of larger hypercubes is identical between CPU and GPU", "[gpu]",
        (gpu_encoder<float, 1>), (gpu_encoder<float, 2>), (gpu_encoder<float, 3>),
        (gpu_encoder<double, 1>), (gpu_encoder<double, 2>), (gpu_encoder<double, 3>) ) {
    using data_type = typename TestType::data_type;
    using profile = detail::profile<data_type, TestType::dimensions>;

    constexpr auto dims = profile::dimensions;
    constexpr auto side_length = profile::hypercube_side_length;
    const size_t n = side_length * 4 - 1;

    auto input_data = make_random_vector<data_type>(ipow(n, dims));
    slice<const data_type, dims> input(input_data.data(), extent<dims>::broadcast(n));

    extent<dims> hc_offset;
    hc_offset[dims - 1] = side_length;
    size_t hc_index = 1;

    sycl::queue q;
    auto gpu_dump = load_and_dump_hypercube<profile>(input, hc_index, q);

    detail::cpu::simd_aligned_buffer<typename profile::bits_type> cpu_dump(ipow(side_length, dims));
    detail::cpu::load_hypercube<profile>(hc_offset, input, cpu_dump.data());

    CHECK(memcmp(gpu_dump.data(), cpu_dump.data(),
                  sizeof(typename profile::bits_type) * detail::ipow(side_length, dims))
            == 0);
}


template<typename TransformName, typename Bits, typename CPUTransform, typename GPUTransform>
static void test_cpu_gpu_transform_equality(
        size_t hc_size, const CPUTransform &cpu_transform, const GPUTransform &gpu_transform) {
    const auto input = make_random_vector<Bits>(hc_size);

    auto cpu_transformed = input;
    cpu_transform(cpu_transformed.data());

    sycl::queue q;
    buffer<Bits> global_buf{range<1>{hc_size}};

    q.submit([&](handler &cgh) {
        cgh.copy(input.data(), global_buf.template get_access<sam::discard_write>(cgh));
    });
    q.submit([&](handler &cgh) {
        auto global_acc = global_buf.template get_access<sam::read_write>(cgh);
        auto local_acc = accessor<Bits, 1, sam::read_write, sat::local>(hc_size, cgh);
        cgh.parallel_for<TransformName>(
                detail::gpu::hypercube_range{1},
                [global_acc, local_acc, hc_size = hc_size, gpu_transform](detail::gpu::hypercube_item item) {
                    detail::gpu::nd_memcpy(local_acc, global_acc, hc_size, item);
                    item.local_memory_barrier();
                    gpu_transform(local_acc.get_pointer(), item);
                    item.local_memory_barrier();
                    detail::gpu::nd_memcpy(global_acc, local_acc, hc_size, item);
                });
    });

    std::vector<Bits> gpu_transformed(hc_size);
    q.submit([&](handler &cgh) {
        cgh.copy(global_buf.template get_access<sam::read>(cgh), gpu_transformed.data());
    });
    q.wait();

    CHECK(gpu_transformed == cpu_transformed);
}

template<typename>
class gpu_forward_transform_test_kernel;

TEMPLATE_TEST_CASE("CPU and GPU forward block transforms are identical", "[gpu]",
        (profile<float, 1>), (profile<float, 2>), (profile<float, 3>), (profile<double, 1>),
        (profile<double, 2>), (profile<double, 3>) ) {
    using bits_type = typename TestType::bits_type;
    test_cpu_gpu_transform_equality<gpu_forward_transform_test_kernel<TestType>, bits_type>(
            ipow(TestType::hypercube_side_length, TestType::dimensions),
            [](bits_type *block) {
                detail::block_transform(
                        block, TestType::dimensions, TestType::hypercube_side_length);
            },
            // Uses lambda instead of the function name, otherwise a host function pointer will
            // be passed into the device kernel
            [](bits_type *block, detail::gpu::hypercube_item item) {
                detail::gpu::block_transform<TestType>(block, item);
            });
}

template<typename>
class gpu_inverse_transform_test_kernel;

TEMPLATE_TEST_CASE("CPU and GPU inverse block transforms are identical", "[gpu]",
        (profile<float, 1>), (profile<float, 2>), (profile<float, 3>), (profile<double, 1>),
        (profile<double, 2>), (profile<double, 3>) ) {
    using bits_type = typename TestType::bits_type;
    test_cpu_gpu_transform_equality<gpu_inverse_transform_test_kernel<TestType>, bits_type>(
            ipow(TestType::hypercube_side_length, TestType::dimensions),
            [](bits_type *block) {
                detail::inverse_block_transform(
                        block, TestType::dimensions, TestType::hypercube_side_length);
            },
            // Uses lambda instead of the function name, otherwise a host function pointer will
            // be passed into the device kernel
            [](bits_type *block, detail::gpu::hypercube_item item) {
                detail::gpu::inverse_block_transform<TestType>(block, item);
            });
}

template<typename>
class gpu_transpose_bits_test_kernel;

TEMPLATE_TEST_CASE("CPU and GPU bit transposition are identical", "[gpu]", uint32_t, uint64_t) {
    test_cpu_gpu_transform_equality<gpu_transpose_bits_test_kernel<TestType>, TestType>(
            bitsof<TestType>,
            [](TestType *chunk) {
                TestType tmp[bitsof<TestType>];
                detail::cpu::transpose_bits_trivial(chunk, tmp);
                memcpy(chunk, tmp, sizeof tmp);
            },
            // Uses lambda instead of the function name, otherwise a host function pointer will
            // be passed into the device kernel
            [](TestType *chunk, detail::gpu::hypercube_item item) {
                detail::gpu::transpose_bits(chunk, item);
            });
}

template<typename>
class gpu_compact_zero_words_test_kernel;

template<typename>
class gpu_expand_zero_words_test_kernel;

TEMPLATE_TEST_CASE("CPU and GPU zero-word compaction is identical", "[gpu]", uint32_t, uint64_t) {
    auto input = make_random_vector<TestType>(bitsof<TestType>);
    for (auto idx : {0, 12, 13, 29, static_cast<int>(bitsof<TestType> - 2)}) {
        input[idx] = 0;
    }

    std::vector<TestType> cpu_compacted(bitsof<TestType> * 2);
    detail::cpu::compact_zero_words(
            input.data(), reinterpret_cast<std::byte *>(cpu_compacted.data()));

    sycl::queue q;

    buffer<TestType> input_buf{range<1>{input.size()}};
    q.submit([&](handler &cgh) {
        cgh.copy(input.data(), input_buf.template get_access<sam::discard_write>(cgh));
    });

    std::vector<TestType> gpu_compacted(bitsof<TestType> * 2);
    buffer<TestType> compacted_buf{range<1>{gpu_compacted.size()}};
    q.submit([&](handler &cgh) {
        cgh.fill(compacted_buf.template get_access<sam::discard_write>(cgh), TestType{0});
    });

    const auto scratch_size = 3 * bitsof<TestType>;
    std::vector<TestType> scratch_dump(scratch_size);
    q.submit([&](handler &cgh) {
        auto input_acc = input_buf.template get_access<sam::read>(cgh);
        auto output_acc = compacted_buf.template get_access<sam::write>(cgh);
        auto local_in_acc
                = accessor<TestType, 1, sam::read_write, sat::local>(bitsof<TestType>, cgh);
        auto scratch_acc = accessor<TestType, 1, sam::read_write, sat::local>(scratch_size, cgh);
        cgh.parallel_for<gpu_compact_zero_words_test_kernel<TestType>>(
                detail::gpu::hypercube_range{1},
                [input_acc, output_acc, local_in_acc, scratch_acc](
                        detail::gpu::hypercube_item item) {
                    detail::gpu::nd_memcpy(local_in_acc, input_acc, input_acc.get_range()[0], item);
                    item.local_memory_barrier();
                    detail::gpu::compact_zero_words<TestType>(output_acc.get_pointer(),
                            local_in_acc.get_pointer(), scratch_acc.get_pointer(), item);
                });
    });

    q.submit([&](handler &cgh) {
        cgh.copy(compacted_buf.template get_access<sam::read>(cgh), gpu_compacted.data());
    });
    q.wait();

    CHECK(gpu_compacted == cpu_compacted);
}


TEMPLATE_TEST_CASE("GPU zero-word expansion works", "[gpu]", uint32_t, uint64_t) {
    auto input = make_random_vector<TestType>(bitsof<TestType>);
    for (auto idx : {0, 12, 13, 29, static_cast<int>(bitsof<TestType> - 2)}) {
        input[idx] = 0;
    }

    std::vector<TestType> cpu_compacted(bitsof<TestType>);
    detail::cpu::compact_zero_words(
            input.data(), reinterpret_cast<std::byte *>(cpu_compacted.data()));

    sycl::queue q;

    buffer<TestType> compacted_buf{range<1>{cpu_compacted.size()}};
    q.submit([&](handler &cgh) {
      cgh.copy(cpu_compacted.data(), compacted_buf.template get_access<sam::discard_write>());
    });

    std::vector<TestType> gpu_expanded(bitsof<TestType>);
    buffer<TestType> expanded_buf{range<1>{gpu_expanded.size()}};
    const auto scratch_size = 2 * bitsof<TestType>;
    q.submit([&](handler &cgh) {
      auto input_acc = compacted_buf.template get_access<sam::read>(cgh);
      auto output_acc = expanded_buf.template get_access<sam::write>(cgh);
      auto local_out_acc
              = accessor<TestType, 1, sam::read_write, sat::local>(bitsof<TestType>, cgh);
      auto scratch_acc = accessor<TestType, 1, sam::read_write, sat::local>(scratch_size, cgh);
      cgh.parallel_for<gpu_expand_zero_words_test_kernel<TestType>>(
              detail::gpu::hypercube_range{1},
              [input_acc, output_acc, local_out_acc, scratch_acc](
                      detail::gpu::hypercube_item item) {
                detail::gpu::expand_zero_words<TestType>(local_out_acc.get_pointer(),
                                                          input_acc.get_pointer(), scratch_acc.get_pointer(), item);
                item.local_memory_barrier();
                detail::gpu::nd_memcpy(output_acc, local_out_acc, output_acc.get_range()[0], item);
              });
    });

    q.submit([&](handler &cgh) {
      cgh.copy(expanded_buf.template get_access<sam::read>(cgh), gpu_expanded.data());
    });
    q.wait();

    CHECK(gpu_expanded == input);
}

template<typename Bits>
class gpu_hypercube_encoding_test_kernel;

TEMPLATE_TEST_CASE("CPU and GPU hypercube encodings are equivalent", "[gpu]", uint32_t, uint64_t) {
    const auto hc_size = 256;

    auto input = make_random_vector<TestType>(hc_size);
    for (size_t i = 0; i < hc_size; ++i) {
        for (auto idx : {0, 12, 13, 29, static_cast<int>(bitsof<TestType> - 2)}) {
            input[i] &= ~(TestType{1} << ((static_cast<unsigned>(idx) * (i / bitsof<TestType>) )
                                  % bitsof<TestType>) );
        }
    }

    detail::cpu::simd_aligned_buffer<TestType> cpu_cube(input.size());
    memcpy(cpu_cube.data(), input.data(), input.size() * sizeof(TestType));
    std::vector<TestType> cpu_stream(hc_size * 2);
    auto cpu_length = detail::cpu::zero_bit_encode(
            cpu_cube.data(), reinterpret_cast<std::byte *>(cpu_stream.data()), hc_size);

    sycl::queue q;

    buffer<TestType> input_buf{range<1>{hc_size}};
    q.submit([&](handler &cgh) {
        cgh.copy(input.data(), input_buf.template get_access<sam::discard_write>(cgh));
    });

    buffer<TestType> stream_buf(range<1>{hc_size * 2});
    q.submit([&](handler &cgh) {
        cgh.fill(stream_buf.template get_access<sam::discard_write>(cgh), TestType{0});
    });

    const auto scratch_size = 3 * bitsof<TestType>;
    buffer<size_t> length_buf{range<1>{1}};
    q.submit([&](handler &cgh) {
        auto input_acc = input_buf.template get_access<sam::read>(cgh);
        auto stream_acc = stream_buf.template get_access<sam::discard_write>(cgh);
        auto cube_acc = accessor<TestType, 1, sam::read_write, sat::local>(hc_size, cgh);
        auto scratch_acc = accessor<TestType, 1, sam::read_write, sat::local>(scratch_size, cgh);
        auto length_acc = length_buf.get_access<sam::discard_write>(cgh);
        cgh.parallel_for<gpu_hypercube_encoding_test_kernel<TestType>>(
                detail::gpu::hypercube_range{1},
                [input_acc, stream_acc, cube_acc, hc_size=hc_size, scratch_acc, length_acc](
                        detail::gpu::hypercube_item item) {
                    detail::gpu::nd_memcpy(cube_acc, input_acc, hc_size, item);
                    item.local_memory_barrier();
                    auto l = detail::gpu::zero_bit_encode<TestType>(cube_acc.get_pointer(),
                            stream_acc.get_pointer(), scratch_acc.get_pointer(), hc_size, item);
                    item.local_memory_barrier();
                    if (item.get_global_id() == sycl::id<2>{0, 0}) { length_acc[0] = l; }
                });
    });

    std::vector<TestType> gpu_stream(stream_buf.get_range()[0]);
    q.submit([&](handler &cgh) {
        cgh.copy(stream_buf.template get_access<sam::read>(cgh), gpu_stream.data());
    });
    size_t gpu_length;
    q.submit([&](handler &cgh) {
        cgh.copy(length_buf.template get_access<sam::read>(cgh), &gpu_length);
    });
    q.wait();

    CHECK(sizeof(TestType) * gpu_length == cpu_length);
    CHECK(gpu_stream == cpu_stream);
}


TEST_CASE("hierarchical_inclusive_prefix_sum produces the expected results", "[gpu]") {
    std::vector<size_t> input(1'000'000);
    std::iota(input.begin(), input.end(), size_t{});

    std::vector<size_t> cpu_prefix_sum(input.size());
    std::inclusive_scan(input.begin(), input.end(), cpu_prefix_sum.begin());

    sycl::buffer<size_t> prefix_sum_buffer(sycl::range<1>(input.size()));
    detail::gpu::hierarchical_inclusive_prefix_sum<size_t> gpu_prefix_sum_operator(input.size(), 256);
    sycl::queue q;
    q.submit([&](sycl::handler &cgh) {
        cgh.copy(input.data(), prefix_sum_buffer.get_access<sam::discard_write>(cgh));
    });

    gpu_prefix_sum_operator(q, prefix_sum_buffer);

    std::vector<size_t> gpu_prefix_sum(input.size());
    q.submit([&](sycl::handler &cgh) {
        cgh.copy(prefix_sum_buffer.get_access<sam::read>(cgh), gpu_prefix_sum.data());
    });
    q.wait();

    CHECK(gpu_prefix_sum == cpu_prefix_sum);
}

#endif
