#define CATCH_CONFIG_MAIN
#include "test_utils.hh"

#include <ndzip/common.hh>
#include <ndzip/cpu_encoder.inl>

#if NDZIP_GPU_SUPPORT
#include <ndzip/gpu_encoder.inl>
#endif

#include <iostream>

#define ALL_PROFILES \
    (profile<float, 1>), (profile<float, 2>), (profile<float, 3>), (profile<double, 1>), \
            (profile<double, 2>), (profile<double, 3>)


using namespace ndzip;
using namespace ndzip::detail;


template<typename Bits>
Bits zero_map_from_transposed(const Bits *transposed) {
    Bits zero_map = 0;
    for (size_t i = 0; i < bitsof<Bits>; ++i) {
        zero_map = (zero_map << 1u) | (transposed[i] != 0);
    }
    return zero_map;
}


TEMPLATE_TEST_CASE("block transform is reversible", "[profile]", ALL_PROFILES) {
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
    cpu::simd_aligned_buffer<TestType> input(bitsof<TestType>);
    auto gen = std::minstd_rand();  // NOLINT(cert-msc51-cpp)
    auto dist = std::uniform_int_distribution<TestType>();
    size_t zeroes = 0;
    for (size_t i = 0; i < bitsof<TestType>; ++i) {
        auto r = dist(gen);
        if (r % 5 == 2) {
            input[i] = 0;
            zeroes++;
        } else {
            input[i] = r;
        }
    }

    std::vector<std::byte> compact((bitsof<TestType> + 1) * sizeof(TestType));
    auto head = zero_map_from_transposed(input.data());
    auto bytes_written = cpu::compact_zero_words(input.data(), compact.data());
    CHECK(bytes_written == (bitsof<TestType> - zeroes) * sizeof(TestType));

    std::vector<TestType> output(bitsof<TestType>);
    auto bytes_read = cpu::expand_zero_words(compact.data(), output.data(), head);
    CHECK(bytes_read == bytes_written);
    CHECK(output == std::vector<TestType>(input.data(), input.data() + bitsof<TestType>));
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
    cpu::transpose_bits(input, transposed);

    alignas(cpu::simd_width_bytes) TestType output[bitsof<TestType>];
    cpu::transpose_bits(transposed, output);

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
    using value_type = typename profile::value_type;
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
            detail::store_value<profile>(&value, load_unaligned<value_type>(value_offset_address));
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
            detail::store_value<profile>(&value, load_unaligned<value_type>(value_offset_address));
            CHECK(memcmp(&value, &border, sizeof value) == 0);
        }
        n_border_elems += count;
    });
    CHECK(n_border_elems == num_elements(array.size()) - ipow(border_start, dims));
}
*/


TEMPLATE_TEST_CASE("decode(encode(input)) reproduces the input", "[encoder][de]", ALL_PROFILES) {
    using profile = TestType;
    using data_type = typename profile::data_type;

    constexpr auto dims = profile::dimensions;
    constexpr auto side_length = profile::hypercube_side_length;
    const size_t n = side_length * 4 - 1;

    auto input_data = make_random_vector<data_type>(ipow(n, dims));

    // Regression test: trigger bug in decoder optimization by ensuring first chunk = 0
    std::fill(input_data.begin(), input_data.begin() + bitsof<data_type>, data_type{});

    auto test_encoder_decoder_pair = [&](auto &&encoder, auto &&decoder) {
        slice<const data_type, dims> input(input_data.data(), extent<dims>::broadcast(n));
        std::vector<std::byte> stream(
                ndzip::compressed_size_bound<typename TestType::data_type>(input.size()));
        stream.resize(encoder.compress(input, stream.data()));

        std::vector<data_type> output_data(input_data.size());
        slice<data_type, dims> output(output_data.data(), extent<dims>::broadcast(n));
        auto stream_bytes_read = decoder.decompress(stream.data(), stream.size(), output);

        CHECK(stream_bytes_read == stream.size());
        check_for_vector_equality(input_data, output_data);
    };

    SECTION("cpu_encoder::encode() => cpu_encoder::decode()") {
        test_encoder_decoder_pair(cpu_encoder<data_type, dims>{}, cpu_encoder<data_type, dims>{});
    }

#if NDZIP_OPENMP_SUPPORT
    SECTION("cpu_encoder::encode() => mt_cpu_encoder::decode()") {
        test_encoder_decoder_pair(
                cpu_encoder<data_type, dims>{}, mt_cpu_encoder<data_type, dims>{});
    }

    SECTION("mt_cpu_encoder::encode() => cpu_encoder::decode()") {
        test_encoder_decoder_pair(
                mt_cpu_encoder<data_type, dims>{}, cpu_encoder<data_type, dims>{});
    }
#endif

#if NDZIP_GPU_SUPPORT
    SECTION("cpu_encoder::encode() => gpu_encoder::decode()") {
        test_encoder_decoder_pair(cpu_encoder<data_type, dims>{}, gpu_encoder<data_type, dims>{});
    }

    SECTION("gpu_encoder::encode() => cpu_encoder::decode()") {
        test_encoder_decoder_pair(gpu_encoder<data_type, dims>{}, cpu_encoder<data_type, dims>{});
    }
#endif
}


#if NDZIP_OPENMP_SUPPORT || NDZIP_GPU_SUPPORT
TEMPLATE_TEST_CASE("file headers from different encoders are identical", "[encoder][header]",
#if NDZIP_OPENMP_SUPPORT
        (mt_cpu_encoder<float, 1>), (mt_cpu_encoder<float, 2>), (mt_cpu_encoder<float, 3>),
        (mt_cpu_encoder<double, 1>), (mt_cpu_encoder<double, 2>), (mt_cpu_encoder<double, 3>)
#if NDZIP_GPU_SUPPORT
                                                                          ,
#endif
#endif
#if NDZIP_GPU_SUPPORT
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
    slice<const data_type, dims> input(input_data.data(), extent<dims>::broadcast(n));

    const auto file = detail::file<profile>(input.size());
    const auto aligned_stream_size_bound
            = compressed_size_bound<typename TestType::data_type>(input.size())
                    / sizeof(file_offset_type)
            + 1;

    cpu_encoder<data_type, dims> reference_encoder;
    std::vector<file_offset_type> reference_stream(aligned_stream_size_bound);
    auto reference_stream_length = reference_encoder.compress(input, reference_stream.data());
    reference_stream.resize(file.num_hypercubes());

    TestType test_encoder;
    std::vector<file_offset_type> test_stream(aligned_stream_size_bound);
    auto test_stream_length = test_encoder.compress(input, test_stream.data());
    test_stream.resize(file.num_hypercubes());

    check_for_vector_equality(reference_stream, test_stream);
    CHECK(reference_stream_length == test_stream_length);
}
#endif


#if NDZIP_GPU_SUPPORT

using sam = sycl::access::mode;
using sat = sycl::access::target;
using sycl::accessor, sycl::nd_range, sycl::buffer, sycl::nd_item, sycl::range, sycl::id,
        sycl::handler, sycl::group, sycl::physical_item, sycl::logical_item, sycl::sub_group,
        sycl::local_memory;

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
static std::vector<typename Profile::bits_type>
load_and_dump_hypercube(const slice<const typename Profile::data_type, Profile::dimensions> &in,
        size_t hc_index, sycl::queue &q) {
    using data_type = typename Profile::data_type;
    using bits_type = typename Profile::bits_type;
    using hc_layout = gpu::hypercube_layout<Profile::dimensions, gpu::forward_transform_tag>;

    auto hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    buffer<data_type> load_buf{in.data(), range<1>{num_elements(in.size())}};
    std::vector<bits_type> out(hc_size * 2);
    buffer<data_type> store_buf{out.size()};
    detail::file<Profile> file(in.size());

    q.submit([&](handler &cgh) {
        cgh.fill(store_buf.template get_access<sam::discard_write>(cgh), data_type{0});
    });
    q.submit([&](handler &cgh) {
        auto data_acc = load_buf.template get_access<sam::read>(cgh);
        auto result_acc = store_buf.template get_access<sam::discard_write>(cgh);
        const auto data_size = in.size();
        cgh.parallel(range<1>{1}, range<1>{gpu::hypercube_group_size},
                [=](gpu::hypercube_group grp, physical_item<1>) {
                    slice<const data_type, Profile::dimensions> data_in{
                            data_acc.get_pointer(), data_size};
                    gpu::hypercube_memory<bits_type, hc_layout> lm{grp};
                    gpu::hypercube_ptr<Profile, gpu::forward_transform_tag> hc{lm()};
                    gpu::load_hypercube(grp, hc_index, data_in, hc);
                    // TODO rotate should probaly happen during CPU load_hypercube as well to hide
                    //  memory access latencies
                    grp.distribute_for(hc_layout::hc_size, [&](index_type item) {
                        result_acc[item] = gpu::bit_cast<data_type>(rotate_right_1(hc.load(item)));
                    });
                });
    });
    q.submit([&](handler &cgh) {
        cgh.copy(store_buf.template get_access<sam::read>(cgh),
                reinterpret_cast<data_type *>(out.data()));
    });
    q.wait();
    return out;
}


#if 0  // gpu::hypercube_ptr assumes 4096 elements per hc
TEMPLATE_TEST_CASE(
        "correctly load small hypercubes into local memory", "[gpu]", uint32_t, uint64_t) {
    sycl::queue q{sycl::gpu_selector{}};

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
#endif


TEMPLATE_TEST_CASE("Flattening of hypercubes is identical between CPU and GPU", "[gpu][load]",
        (gpu_encoder<float, 1>), (gpu_encoder<float, 2>), (gpu_encoder<float, 3>),
        (gpu_encoder<double, 1>), (gpu_encoder<double, 2>), (gpu_encoder<double, 3>) ) {
    using data_type = typename TestType::data_type;
    using profile = detail::profile<data_type, TestType::dimensions>;
    using bits_type = typename profile::bits_type;

    constexpr auto dims = profile::dimensions;
    constexpr auto side_length = profile::hypercube_side_length;
    const size_t hc_size = ipow(side_length, dims);
    const size_t n = side_length * 4 - 1;

    auto input_data = make_random_vector<data_type>(ipow(n, dims));
    slice<const data_type, dims> input(input_data.data(), extent<dims>::broadcast(n));

    extent<dims> hc_offset;
    hc_offset[dims - 1] = side_length;
    size_t hc_index = 1;

    sycl::queue q{sycl::gpu_selector{}};
    auto gpu_dump = load_and_dump_hypercube<profile>(input, hc_index, q);

    cpu::simd_aligned_buffer<bits_type> cpu_dump(hc_size);
    cpu::load_hypercube<profile>(hc_offset, input, cpu_dump.data());

    check_for_vector_equality(gpu_dump.data(), cpu_dump.data(), hc_size);
}


TEMPLATE_TEST_CASE(
        "GPU store_hypercube is the inverse of load_hypercube", "[gpu][load]", ALL_PROFILES) {
    using data_type = typename TestType::data_type;
    using bits_type = typename TestType::bits_type;

    constexpr auto dims = TestType::dimensions;
    constexpr auto side_length = TestType::hypercube_side_length;
    const size_t hc_size = ipow(side_length, dims);
    const size_t n = side_length * 3;

    auto input_data = make_random_vector<data_type>(ipow(n, dims));
    slice<const data_type, dims> input(input_data.data(), extent<dims>::broadcast(n));

    buffer<data_type> input_buf{input.data(), range<1>{num_elements(input.size())}};
    // buffer needed for hypercube_ptr forward_transform_tag => inverse_transform_tag translation
    buffer<bits_type> temp_buf{input_buf.get_range()};
    buffer<data_type> output_buf{input_buf.get_range()};
    detail::file<TestType> file(input.size());

    sycl::queue q{sycl::gpu_selector{}};
    q.submit([&](handler &cgh) {
        cgh.fill(output_buf.template get_access<sam::discard_write>(cgh), data_type{0});
    });
    q.submit([&](handler &cgh) {
        using hc_layout = gpu::hypercube_layout<TestType::dimensions, gpu::forward_transform_tag>;
        auto input_acc = input_buf.template get_access<sam::read>(cgh);
        auto temp_acc = temp_buf.template get_access<sam::discard_write>(cgh);
        const auto data_size = input.size();
        cgh.parallel(range<1>{file.num_hypercubes()}, range<1>{gpu::hypercube_group_size},
                [=](gpu::hypercube_group grp, physical_item<1>) {
                    auto hc_index = grp.get_id(0);
                    slice<const data_type, TestType::dimensions> input{
                            input_acc.get_pointer(), data_size};
                    gpu::hypercube_memory<bits_type, hc_layout> lm{grp};
                    gpu::hypercube_ptr<TestType, gpu::forward_transform_tag> hc{lm()};
                    gpu::load_hypercube(grp, hc_index, input, hc);
                    grp.distribute_for(hc_size,
                            [&](index_type i) { temp_acc[hc_index * hc_size + i] = hc.load(i); });
                });
    });
    q.submit([&](handler &cgh) {
        using hc_layout = gpu::hypercube_layout<TestType::dimensions, gpu::inverse_transform_tag>;
        auto temp_acc = temp_buf.template get_access<sam::read>(cgh);
        auto output_acc = output_buf.template get_access<sam::discard_write>(cgh);
        const auto data_size = input.size();
        cgh.parallel(range<1>{file.num_hypercubes()}, range<1>{gpu::hypercube_group_size},
                [=](gpu::hypercube_group grp, physical_item<1>) {
                    auto hc_index = grp.get_id(0);
                    slice<data_type, TestType::dimensions> output{
                            output_acc.get_pointer(), data_size};
                    gpu::hypercube_memory<bits_type, hc_layout> lm{grp};
                    gpu::hypercube_ptr<TestType, gpu::inverse_transform_tag> hc{lm()};
                    grp.distribute_for(hc_size,
                            [&](index_type i) { hc.store(i, temp_acc[hc_index * hc_size + i]); });
                    gpu::store_hypercube(grp, hc_index, output, hc);
                });
    });
    std::vector<data_type> output_data(input_data.size());
    q.submit([&](handler &cgh) {
        cgh.copy(output_buf.template get_access<sam::read>(cgh), output_data.data());
    });
    q.wait();

    check_for_vector_equality(input_data, output_data);
}


template<typename, typename>
class gpu_transform_test_kernel;

template<typename Profile, typename Tag, typename CPUTransform, typename GPUTransform>
static void test_cpu_gpu_transform_equality(
        const CPUTransform &cpu_transform, const GPUTransform &gpu_transform) {
    using bits_type = typename Profile::bits_type;
    using hc_layout = gpu::hypercube_layout<Profile::dimensions, Tag>;
    constexpr auto hc_size
            = static_cast<index_type>(ipow(Profile::hypercube_side_length, Profile::dimensions));

    const auto input = make_random_vector<bits_type>(hc_size);

    auto cpu_transformed = input;
    cpu_transform(cpu_transformed.data());

    sycl::queue q{sycl::gpu_selector{}};
    buffer<bits_type> io_buf{range<1>{hc_size}};

    q.submit([&](handler &cgh) {
        cgh.copy(input.data(), io_buf.template get_access<sam::discard_write>(cgh));
    });
    q.submit([&](handler &cgh) {
        auto global_acc = io_buf.template get_access<sam::read_write>(cgh);
        cgh.parallel<gpu_transform_test_kernel<Profile, Tag>>(range<1>{1},
                range<1>{gpu::hypercube_group_size},
                [global_acc, hc_size = hc_size, gpu_transform](
                        gpu::hypercube_group grp, physical_item<1>) {
                    gpu::hypercube_memory<bits_type, hc_layout> lm{grp};
                    gpu::hypercube_ptr<Profile, Tag> hc{lm()};
                    grp.distribute_for(hc_size, [&](index_type i) { hc.store(i, global_acc[i]); });
                    gpu_transform(grp, hc);
                    grp.distribute_for(hc_size, [&](index_type i) { global_acc[i] = hc.load(i); });
                });
    });

    std::vector<bits_type> gpu_transformed(hc_size);
    q.submit([&](handler &cgh) {
        cgh.copy(io_buf.template get_access<sam::read>(cgh), gpu_transformed.data());
    });
    q.wait();

    check_for_vector_equality(gpu_transformed, cpu_transformed);
}

TEMPLATE_TEST_CASE("CPU and GPU forward block transforms are identical", "[gpu]", ALL_PROFILES) {
    using bits_type = typename TestType::bits_type;
    test_cpu_gpu_transform_equality<TestType, gpu::forward_transform_tag>(
            [](bits_type *block) {
                detail::block_transform(
                        block, TestType::dimensions, TestType::hypercube_side_length);
            },
            // Use lambda instead of the function name, otherwise a host function pointer will
            // be passed into the device kernel
            [](gpu::hypercube_group grp,
                    gpu::hypercube_ptr<TestType, gpu::forward_transform_tag> hc) {
                const auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);
                grp.distribute_for(
                        hc_size, [&](index_type i) { hc.store(i, rotate_left_1(hc.load(i))); });
                gpu::forward_block_transform(grp, hc);
            });
}

TEMPLATE_TEST_CASE("CPU and GPU inverse block transforms are identical", "[gpu]", ALL_PROFILES) {
    using bits_type = typename TestType::bits_type;
    test_cpu_gpu_transform_equality<TestType, gpu::inverse_transform_tag>(
            [](bits_type *block) {
                detail::inverse_block_transform(
                        block, TestType::dimensions, TestType::hypercube_side_length);
            },
            // Use lambda instead of the function name, otherwise a host function pointer will
            // be passed into the device kernel
            [](gpu::hypercube_group grp,
                    gpu::hypercube_ptr<TestType, gpu::inverse_transform_tag> hc) {
                gpu::inverse_block_transform<TestType>(grp, hc);
                const auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);
                grp.distribute_for(
                        hc_size, [&](index_type i) { hc.store(i, rotate_right_1(hc.load(i))); });
            });
}


template<typename>
class gpu_hypercube_decode_test_kernel;

TEMPLATE_TEST_CASE("GPU hypercube decoding works", "[gpu]", ALL_PROFILES) {
    using bits_type = typename TestType::bits_type;
    const auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);
    using hc_layout = gpu::hypercube_layout<TestType::dimensions, gpu::inverse_transform_tag>;

    auto input = make_random_vector<bits_type>(hc_size);
    for (size_t i = 0; i < hc_size; ++i) {
        for (auto idx : {0, 12, 13, 29, static_cast<int>(bitsof<bits_type> - 2)}) {
            input[i] &= ~(bits_type{1} << ((static_cast<unsigned>(idx) * (i / bitsof<bits_type>) )
                                  % bitsof<bits_type>) );
            input[floor(i, bitsof<bits_type>) + idx] = 0;
        }
    }

    cpu::simd_aligned_buffer<bits_type> cpu_cube(input.size());
    memcpy(cpu_cube.data(), input.data(), input.size() * sizeof(bits_type));
    std::vector<bits_type> stream(hc_size * 2);
    auto cpu_length = cpu::zero_bit_encode(
            cpu_cube.data(), reinterpret_cast<std::byte *>(stream.data()), hc_size);

    sycl::queue q{sycl::gpu_selector{}};

    buffer<bits_type> stream_buf{range<1>{cpu_length}};
    q.submit([&](handler &cgh) {
        cgh.copy(stream.data(), stream_buf.template get_access<sam::discard_write>());
    });

    buffer<bits_type> output_buf{range<1>{hc_size}};
    q.submit([&](handler &cgh) {
        auto stream_acc = stream_buf.template get_access<sam::read>(cgh);
        auto output_acc = output_buf.template get_access<sam::discard_write>(cgh);
        cgh.parallel<gpu_hypercube_decode_test_kernel<TestType>>(sycl::range{1},
                sycl::range<1>{gpu::hypercube_group_size},
                [stream_acc, output_acc](gpu::hypercube_group grp, sycl::physical_item<1>) {
                    gpu::hypercube_memory<bits_type, hc_layout> lm{grp};
                    gpu::hypercube_ptr<TestType, gpu::inverse_transform_tag> hc{lm()};
                    gpu::read_transposed_chunks<TestType>(grp, hc, stream_acc.get_pointer());
                    grp.distribute_for(hc_size, [&](index_type i) { output_acc[i] = hc.load(i); });
                });
    });

    std::vector<bits_type> output(hc_size);
    q.submit([&](handler &cgh) {
        cgh.copy(output_buf.template get_access<sam::read>(cgh), output.data());
    });
    q.wait();

    check_for_vector_equality(output, input);
}


template<typename>
class gpu_hypercube_transpose_test_kernel;
template<typename>
class gpu_hypercube_compact_test_kernel;

TEMPLATE_TEST_CASE("CPU and GPU hypercube encodings are equivalent", "[gpu]", ALL_PROFILES) {
    using bits_type = typename TestType::bits_type;
    const auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);
    using hc_layout = gpu::hypercube_layout<TestType::dimensions, gpu::forward_transform_tag>;

    constexpr index_type col_chunk_size = detail::bitsof<bits_type>;
    constexpr index_type header_chunk_size = hc_size / col_chunk_size;
    constexpr index_type hc_total_chunks_size = hc_size + header_chunk_size;
    constexpr index_type chunks_per_hc = 1 /* header */ + hc_size / col_chunk_size;

    auto input = make_random_vector<bits_type>(hc_size);
    for (size_t i = 0; i < hc_size; ++i) {
        for (auto idx : {0, 12, 13, 29, static_cast<int>(bitsof<bits_type> - 2)}) {
            input[i] &= ~(bits_type{1} << ((static_cast<unsigned>(idx) * (i / bitsof<bits_type>) )
                                  % bitsof<bits_type>) );
            input[floor(i, bitsof<bits_type>) + idx] = 0;
        }
    }

    cpu::simd_aligned_buffer<bits_type> cpu_cube(input.size());
    memcpy(cpu_cube.data(), input.data(), input.size() * sizeof(bits_type));
    std::vector<bits_type> cpu_stream(hc_size * 2);
    auto cpu_length = cpu::zero_bit_encode(
            cpu_cube.data(), reinterpret_cast<std::byte *>(cpu_stream.data()), hc_size);

    sycl::queue q{sycl::gpu_selector{}};

    buffer<bits_type> input_buf{range<1>{hc_size}};
    q.submit([&](handler &cgh) {
        cgh.copy(input.data(), input_buf.template get_access<sam::discard_write>(cgh));
    });

    buffer<bits_type> chunks_buf{hc_total_chunks_size};
    const auto num_chunks = 1 + hc_size / col_chunk_size;
    buffer<index_type> chunk_lengths_buf{
            range<1>{ceil(1 + num_chunks, gpu::hierarchical_inclusive_scan_granularity)}};

    q.submit([&](handler &cgh) {
        auto input_acc = input_buf.template get_access<sam::read>(cgh);
        auto columns_acc = chunks_buf.template get_access<sam::discard_write>(cgh);
        auto chunk_lengths_acc = chunk_lengths_buf.get_access<sam::discard_write>(cgh);
        cgh.parallel<gpu_hypercube_transpose_test_kernel<TestType>>(sycl::range<1>{1},
                sycl::range<1>{gpu::hypercube_group_size},
                [=](gpu::hypercube_group grp, sycl::physical_item<1> phys_idx) {
                    gpu::hypercube_memory<bits_type, hc_layout> lm{grp};
                    gpu::hypercube_ptr<TestType, gpu::forward_transform_tag> hc{lm()};
                    grp.distribute_for(hc_size, [&](index_type i) { hc.store(i, input_acc[i]); });
                    gpu::write_transposed_chunks(grp, hc, &columns_acc[0], &chunk_lengths_acc[1]);
                    // hack
                    if (phys_idx.get_global_linear_id() == 0) {
                        grp.single_item([&] { chunk_lengths_acc[0] = 0; });
                    }
                });
    });

    std::vector<index_type> chunk_lengths(chunk_lengths_buf.get_range()[0]);
    q.submit([&](handler &cgh) {
         cgh.copy(chunk_lengths_buf.template get_access<sam::read>(cgh), chunk_lengths.data());
     }).wait();

    gpu::hierarchical_inclusive_scan(q, chunk_lengths_buf, sycl::plus<index_type>{});

    buffer<bits_type> stream_buf(range<1>{hc_size * 2});
    q.submit([&](handler &cgh) {
        cgh.fill(stream_buf.template get_access<sam::discard_write>(cgh), bits_type{0});
    });

    index_type gpu_num_words;
    q.submit([&](handler &cgh) {
         cgh.copy(chunk_lengths_buf.template get_access<sam::read>(
                          cgh, sycl::range<1>{1}, sycl::id<1>{num_chunks}),
                 &gpu_num_words);
     }).wait();
    auto gpu_length = gpu_num_words * sizeof(bits_type);

    buffer<file_offset_type> length_buf{range<1>{1}};
    q.submit([&](sycl::handler &cgh) {
        auto chunks_acc = chunks_buf.template get_access<sam::read>(cgh);
        auto chunk_offsets_acc = chunk_lengths_buf.template get_access<sam::read>(cgh);
        auto stream_acc = stream_buf.template get_access<sam::discard_write>(cgh);
        auto length_acc = length_buf.template get_access<sam::discard_write>(cgh);
        cgh.parallel<gpu_hypercube_compact_test_kernel<TestType>>(
                sycl::range<1>{1 /* num_hypercubes */}, sycl::range<1>{gpu::hypercube_group_size},
                [=](gpu::hypercube_group grp, sycl::physical_item<1> phys_idx) {
                    const auto hc_index = grp.get_id(0);
                    gpu::compact_chunks<TestType>(grp,
                            &chunks_acc.get_pointer()[hc_index * hc_total_chunks_size],
                            &chunk_offsets_acc.get_pointer()[hc_index * chunks_per_hc],
                            &stream_acc.get_pointer()[0]);
                    // hack
                    if (phys_idx.get_global_linear_id() == 0) {
                        grp.single_item([&] {
                            length_acc[0] = sizeof(bits_type)
                                    * chunk_offsets_acc[chunk_offsets_acc.get_count() - 1];
                        });
                    }
                });
    });

    std::vector<bits_type> gpu_stream(stream_buf.get_range()[0]);
    q.submit([&](handler &cgh) {
         cgh.copy(stream_buf.template get_access<sam::read>(cgh), gpu_stream.data());
     }).wait();

    CHECK(gpu_length == cpu_length);
    check_for_vector_equality(gpu_stream, cpu_stream);
}

#endif
