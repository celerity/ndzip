#include "test_utils.hh"

#include <ndzip/common.hh>
#include <ndzip/cpu_encoder.inl>

#if NDZIP_HIPSYCL_SUPPORT
#include <ndzip/sycl_encoder.inl>
#endif

#if NDZIP_CUDA_SUPPORT
#include <ndzip/cuda_encoder.inl>
#endif

#include <iostream>

#define ALL_PROFILES (profile<DATA_TYPE, DIMENSIONS>)


using namespace ndzip;
using namespace ndzip::detail;


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


TEMPLATE_TEST_CASE("decode(encode(input)) reproduces the input", "[encoder][de]", ALL_PROFILES) {
    using profile = TestType;
    using data_type = typename profile::data_type;

    constexpr auto dims = profile::dimensions;
    constexpr auto side_length = profile::hypercube_side_length;
    const index_type n = side_length * 4 - 1;

    auto input_data = make_random_vector<data_type>(ipow(n, dims));

    // Regression test: trigger bug in decoder optimization by ensuring first chunk = 0
    std::fill(input_data.begin(), input_data.begin() + bits_of<data_type>, data_type{});

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

#if NDZIP_HIPSYCL_SUPPORT
    SECTION("cpu_encoder::encode() => sycl_encoder::decode()") {
        test_encoder_decoder_pair(cpu_encoder<data_type, dims>{}, sycl_encoder<data_type, dims>{});
    }

    SECTION("sycl_encoder::encode() => cpu_encoder::decode()") {
        test_encoder_decoder_pair(sycl_encoder<data_type, dims>{}, cpu_encoder<data_type, dims>{});
    }
#endif

#if NDZIP_CUDA_SUPPORT
    SECTION("cpu_encoder::encode() => cuda_encoder::decode()") {
        test_encoder_decoder_pair(cpu_encoder<data_type, dims>{}, cuda_encoder<data_type, dims>{});
    }

    SECTION("cuda_encoder::encode() => cpu_encoder::decode()") {
        test_encoder_decoder_pair(cuda_encoder<data_type, dims>{}, cpu_encoder<data_type, dims>{});
    }
#endif
}


#if NDZIP_OPENMP_SUPPORT || NDZIP_HIPSYCL_SUPPORT
TEMPLATE_TEST_CASE("file headers from different encoders are identical", "[header]"
#if NDZIP_OPENMP_SUPPORT
        ,
        (mt_cpu_encoder<DATA_TYPE, DIMENSIONS>)
#endif
#if NDZIP_HIPSYCL_SUPPORT
                ,
        (sycl_encoder<DATA_TYPE, DIMENSIONS>)
#endif
#if NDZIP_CUDA_SUPPORT
                ,
        (cuda_encoder<DATA_TYPE, DIMENSIONS>)
#endif
) {
    using data_type = typename TestType::data_type;
    using profile = detail::profile<data_type, TestType::dimensions>;

    constexpr auto dims = profile::dimensions;
    constexpr auto side_length = profile::hypercube_side_length;
    const index_type n = side_length * 4 - 1;

    auto input_data = make_random_vector<data_type>(ipow(n, dims));
    slice<const data_type, dims> input(input_data.data(), extent<dims>::broadcast(n));

    const auto file = detail::file<profile>(input.size());
    const auto aligned_stream_size_bound
            = compressed_size_bound<typename TestType::data_type>(input.size()) / sizeof(index_type)
            + 1;

    cpu_encoder<data_type, dims> reference_encoder;
    std::vector<index_type> reference_stream(aligned_stream_size_bound);
    auto reference_stream_length = reference_encoder.compress(input, reference_stream.data());
    reference_stream.resize(file.num_hypercubes());

    TestType test_encoder;
    std::vector<index_type> test_stream(aligned_stream_size_bound);
    auto test_stream_length = test_encoder.compress(input, test_stream.data());
    test_stream.resize(file.num_hypercubes());

    check_for_vector_equality(reference_stream, test_stream);
    CHECK(reference_stream_length == test_stream_length);
}
#endif

#if NDZIP_HIPSYCL_SUPPORT

using sam = sycl::access::mode;
using sycl::accessor, sycl::nd_range, sycl::buffer, sycl::nd_item, sycl::range, sycl::id,
        sycl::handler, sycl::group, sycl::physical_item, sycl::logical_item, sycl::sub_group,
        sycl::local_memory;

template<typename Profile>
static std::vector<typename Profile::bits_type> sycl_load_and_dump_hypercube(
        const slice<const typename Profile::data_type, Profile::dimensions> &in,
        index_type hc_index, sycl::queue &q) {
    using data_type = typename Profile::data_type;
    using bits_type = typename Profile::bits_type;

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
        cgh.parallel(range<1>{1}, range<1>{gpu::hypercube_group_size<Profile>},
                [=](gpu_sycl::hypercube_group<Profile> grp, physical_item<1>) {
                    slice<const data_type, Profile::dimensions> data_in{
                            data_acc.get_pointer(), data_size};
                    gpu_sycl::hypercube_memory<Profile, gpu::forward_transform_tag> lm{grp};
                    gpu_sycl::hypercube_ptr<Profile, gpu::forward_transform_tag> hc{lm()};
                    gpu_sycl::load_hypercube(grp, hc_index, data_in, hc);
                    // TODO rotate should probaly happen during CPU load_hypercube as well to hide
                    //  memory access latencies
                    grp.distribute_for(hc_size, [&](index_type item) {
                        result_acc[item] = bit_cast<data_type>(rotate_right_1(hc.load(item)));
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
template<typename T, unsigned Dims>
struct mock_profile {
    using data_type = T;
    using bits_type = T;
    constexpr static unsigned dimensions = Dims;
    constexpr static unsigned hypercube_side_length = 2;
    constexpr static unsigned compressed_block_size_bound
            = sizeof(T) * (ipow(hypercube_side_length, dimensions) + 1);
};


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


TEMPLATE_TEST_CASE(
        "SYCL store_hypercube is the inverse of load_hypercube", "[sycl][load]", ALL_PROFILES) {
    using data_type = typename TestType::data_type;
    using bits_type = typename TestType::bits_type;

    constexpr auto dims = TestType::dimensions;
    constexpr auto side_length = TestType::hypercube_side_length;
    const index_type hc_size = ipow(side_length, dims);
    const index_type n = side_length * 3;

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
        auto input_acc = input_buf.template get_access<sam::read>(cgh);
        auto temp_acc = temp_buf.template get_access<sam::discard_write>(cgh);
        const auto data_size = input.size();
        cgh.parallel(range<1>{file.num_hypercubes()}, range<1>{gpu::hypercube_group_size<TestType>},
                [=](gpu_sycl::hypercube_group<TestType> grp, physical_item<1>) {
                    auto hc_index = grp.get_id(0);
                    slice<const data_type, TestType::dimensions> input{
                            input_acc.get_pointer(), data_size};
                    gpu_sycl::hypercube_memory<TestType, gpu::forward_transform_tag> lm{grp};
                    gpu::hypercube_ptr<TestType, gpu::forward_transform_tag> hc{lm()};
                    gpu_sycl::load_hypercube(grp, hc_index, input, hc);
                    grp.distribute_for(hc_size,
                            [&](index_type i) { temp_acc[hc_index * hc_size + i] = hc.load(i); });
                });
    });
    q.submit([&](handler &cgh) {
        auto temp_acc = temp_buf.template get_access<sam::read>(cgh);
        auto output_acc = output_buf.template get_access<sam::discard_write>(cgh);
        const auto data_size = input.size();
        cgh.parallel(range<1>{file.num_hypercubes()}, range<1>{gpu::hypercube_group_size<TestType>},
                [=](gpu_sycl::hypercube_group<TestType> grp, physical_item<1>) {
                    auto hc_index = grp.get_id(0);
                    slice<data_type, TestType::dimensions> output{
                            output_acc.get_pointer(), data_size};
                    gpu_sycl::hypercube_memory<TestType, gpu::inverse_transform_tag> lm{grp};
                    gpu::hypercube_ptr<TestType, gpu::inverse_transform_tag> hc{lm()};
                    grp.distribute_for(hc_size,
                            [&](index_type i) { hc.store(i, temp_acc[hc_index * hc_size + i]); });
                    gpu_sycl::store_hypercube(grp, hc_index, output, hc);
                });
    });
    std::vector<data_type> output_data(input_data.size());
    q.submit([&](handler &cgh) {
        cgh.copy(output_buf.template get_access<sam::read>(cgh), output_data.data());
    });
    q.wait();

    check_for_vector_equality(input_data, output_data);
}

template<typename>
class gpu_hypercube_transpose_test_kernel;
template<typename>
class gpu_hypercube_compact_test_kernel;

template<typename, typename>
class sycl_transform_test_kernel;

#endif


#if NDZIP_CUDA_SUPPORT

template<typename T>
static __global__ void cuda_fill_kernel(T *dest, T value, index_type count) {
    const auto i = static_cast<index_type>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) { dest[i] = 0; }
}

template<typename T>
static void cuda_fill(T *dest, T value, index_type count) {
    constexpr index_type threads_per_block = 256;
    cuda_fill_kernel<<<div_ceil(count, threads_per_block), threads_per_block>>>(dest, value, count);
}

template<typename Profile>
static __global__ void
cuda_load_and_dump_kernel(slice<const typename Profile::data_type, Profile::dimensions> data,
        index_type hc_index, typename Profile::data_type *result) {
    using data_type = typename Profile::data_type;
    constexpr auto hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);

    gpu_cuda::hypercube_memory<Profile, gpu::forward_transform_tag> lm;
    auto *lmp = lm;  // workaround for https://bugs.llvm.org/show_bug.cgi?id=50316
    gpu_cuda::hypercube_ptr<Profile, gpu::forward_transform_tag> hc{lmp};

    auto block = gpu_cuda::hypercube_block<Profile>{};
    gpu_cuda::load_hypercube(block, hc_index, data, hc);
    __syncthreads();
    // TODO rotate should probaly happen during CPU load_hypercube as well to hide
    //  memory access latencies
    distribute_for(hc_size, block, [&](index_type item) {
        result[item] = bit_cast<data_type>(rotate_right_1(hc.load(item)));
    });
}

template<typename Profile>
static std::vector<typename Profile::bits_type> cuda_load_and_dump_hypercube(
        const slice<const typename Profile::data_type, Profile::dimensions> &in,
        index_type hc_index) {
    using data_type = typename Profile::data_type;
    using bits_type = typename Profile::bits_type;

    auto hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    gpu_cuda::cuda_buffer<data_type> load_buf(num_elements(in.size()));
    std::vector<bits_type> out(hc_size * 2);
    gpu_cuda::cuda_buffer<data_type> store_buf(out.size());
    detail::file<Profile> file(in.size());

    CHECKED_CUDA_CALL(cudaMemcpy, load_buf.get(), in.data(), load_buf.size() * sizeof(data_type),
            cudaMemcpyHostToDevice);
    cuda_fill(store_buf.get(), data_type{0}, store_buf.size());
    cuda_load_and_dump_kernel<Profile>
            <<<file.num_hypercubes(), (gpu::hypercube_group_size<Profile>)>>>(
                    slice{load_buf.get(), in.size()}, hc_index, store_buf.get());
    CHECKED_CUDA_CALL(cudaMemcpy, out.data(), store_buf.get(), out.size() * sizeof(data_type),
            cudaMemcpyDeviceToHost);
    return out;
}

template<typename Profile>
static __global__ void
cuda_load_hypercube_kernel(slice<const typename Profile::data_type, Profile::dimensions> input,
        typename Profile::bits_type *temp) {
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    auto hc_index = static_cast<index_type>(blockIdx.x);

    gpu_cuda::hypercube_memory<Profile, gpu::forward_transform_tag> lm;
    auto *lmp = lm;  // workaround for https://bugs.llvm.org/show_bug.cgi?id=50316
    gpu::hypercube_ptr<Profile, gpu::forward_transform_tag> hc{lmp};

    auto block = gpu_cuda::hypercube_block<Profile>{};
    gpu_cuda::load_hypercube(block, hc_index, input, hc);
    __syncthreads();
    distribute_for(
            hc_size, block, [&](index_type i) { temp[hc_index * hc_size + i] = hc.load(i); });
}

template<typename Profile>
static __global__ void cuda_store_hypercube_kernel(const typename Profile::bits_type *temp,
        slice<typename Profile::data_type, Profile::dimensions> output) {
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);
    auto hc_index = static_cast<index_type>(blockIdx.x);

    gpu_cuda::hypercube_memory<Profile, gpu::inverse_transform_tag> lm;
    auto *lmp = lm;  // workaround for https://bugs.llvm.org/show_bug.cgi?id=50316
    gpu::hypercube_ptr<Profile, gpu::inverse_transform_tag> hc{lmp};

    auto block = gpu_cuda::hypercube_block<Profile>{};
    distribute_for(
            hc_size, block, [&](index_type i) { hc.store(i, temp[hc_index * hc_size + i]); });
    __syncthreads();
    gpu_cuda::store_hypercube(block, hc_index, output, hc);
}

TEMPLATE_TEST_CASE(
        "CUDA store_hypercube is the inverse of load_hypercube", "[cuda][load]", ALL_PROFILES) {
    using data_type = typename TestType::data_type;
    using bits_type = typename TestType::bits_type;

    constexpr auto dims = TestType::dimensions;
    constexpr auto side_length = TestType::hypercube_side_length;
    const index_type n = side_length * 3;

    auto input_data = make_random_vector<data_type>(ipow(n, dims));
    slice<const data_type, dims> input(input_data.data(), extent<dims>::broadcast(n));

    gpu_cuda::cuda_buffer<data_type> input_buf(num_elements(input.size()));
    // buffer needed for hypercube_ptr forward_transform_tag => inverse_transform_tag translation
    gpu_cuda::cuda_buffer<bits_type> temp_buf(input_buf.size());
    gpu_cuda::cuda_buffer<data_type> output_buf(input_buf.size());
    detail::file<TestType> file(input.size());

    CHECKED_CUDA_CALL(cudaMemcpy, input_buf.get(), input.data(),
            input_buf.size() * sizeof(data_type), cudaMemcpyHostToDevice);

    cuda_fill(output_buf.get(), data_type{0}, output_buf.size());

    cuda_load_hypercube_kernel<TestType>
            <<<file.num_hypercubes(), (gpu_cuda::hypercube_group_size<TestType>)>>>(
                    slice{input_buf.get(), input.size()}, temp_buf.get());
    cuda_store_hypercube_kernel<TestType>
            <<<file.num_hypercubes(), (gpu_cuda::hypercube_group_size<TestType>)>>>(
                    temp_buf.get(), slice{output_buf.get(), input.size()});

    std::vector<data_type> output_data(input_data.size());
    CHECKED_CUDA_CALL(cudaMemcpy, output_data.data(), output_buf.get(),
            output_buf.size() * sizeof(data_type), cudaMemcpyDeviceToHost);

    check_for_vector_equality(input_data, output_data);
}

template<typename Profile>
__global__ void encode_hypercube_kernel(const typename Profile::bits_type *input,
        typename Profile::bits_type *chunks, index_type *chunk_lengths) {
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);

    __shared__ gpu_cuda::hypercube_memory<Profile, gpu::forward_transform_tag> lm;
    auto *lmp = lm;  // workaround for https://bugs.llvm.org/show_bug.cgi?id=50316
    gpu_cuda::hypercube_ptr<Profile, gpu::forward_transform_tag> hc{lmp};

    auto block = gpu_cuda::hypercube_block<Profile>{};
    distribute_for(hc_size, block, [&](index_type i) { hc.store(i, input[i]); });
    __syncthreads();
    write_transposed_chunks(block, hc, chunks, chunk_lengths + 1);
    // hack
    if (blockIdx.x == 0 && threadIdx.x == 0) { chunk_lengths[0] = 0; }
}

template<typename Profile>
__global__ void test_compact_chunks(const typename Profile::bits_type *chunks,
        const index_type *offsets, index_type *length, typename Profile::bits_type *stream) {
    auto block = gpu_cuda::hypercube_block<Profile>{};
    gpu_cuda::compact_chunks<Profile>(block, chunks, offsets, length, stream);
}

template<typename Profile>
__global__ void hypercube_decode_test_kernel(
        const typename Profile::bits_type *stream, typename Profile::bits_type *output) {
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);

    __shared__ gpu_cuda::hypercube_memory<Profile, gpu::inverse_transform_tag> lm;
    auto *lmp = lm;  // workaround for https://bugs.llvm.org/show_bug.cgi?id=50316
    gpu::hypercube_ptr<Profile, gpu::inverse_transform_tag> hc{lmp};

    auto block = gpu_cuda::hypercube_block<Profile>{};
    gpu_cuda::read_transposed_chunks<Profile>(block, hc, stream);
    __syncthreads();
    distribute_for(hc_size, block, [&](index_type i) { output[i] = hc.load(i); });
}

template<typename Profile, typename Tag, typename CudaTransform>
__global__ void
test_gpu_transform(typename Profile::bits_type *buffer, CudaTransform cuda_transform) {
    constexpr index_type hc_size = ipow(Profile::hypercube_side_length, Profile::dimensions);

    __shared__ gpu_cuda::hypercube_memory<Profile, Tag> lm;
    auto *lmp = lm;  // workaround for https://bugs.llvm.org/show_bug.cgi?id=50316
    gpu::hypercube_ptr<Profile, Tag> hc{lmp};

    auto block = gpu_cuda::hypercube_block<Profile>{};
    distribute_for(hc_size, block, [&](index_type i) { hc.store(i, buffer[i]); });
    __syncthreads();
    cuda_transform(block, hc);
    __syncthreads();
    distribute_for(hc_size, block, [&](index_type i) { buffer[i] = hc.load(i); });
}

#endif


TEMPLATE_TEST_CASE("Flattening of hypercubes is identical between encoders", "[cuda][sycl][load]",
        ALL_PROFILES) {
    using data_type = typename TestType::data_type;
    using profile = detail::profile<data_type, TestType::dimensions>;
    using bits_type = typename profile::bits_type;

    constexpr auto dims = profile::dimensions;
    constexpr auto side_length = profile::hypercube_side_length;
    const index_type hc_size = ipow(side_length, dims);
    const index_type n = side_length * 4 - 1;

    auto input_data = make_random_vector<data_type>(ipow(n, dims));
    slice<const data_type, dims> input(input_data.data(), extent<dims>::broadcast(n));

    extent<dims> hc_offset;
    hc_offset[dims - 1] = side_length;
    index_type hc_index = 1;

    cpu::simd_aligned_buffer<bits_type> cpu_dump(hc_size);
    cpu::load_hypercube<profile>(hc_offset, input, cpu_dump.data());

#if NDZIP_HIPSYCL_SUPPORT
    SECTION("SYCL vs CPU") {
        sycl::queue sycl_q{sycl::gpu_selector{}};
        auto sycl_dump = sycl_load_and_dump_hypercube<profile>(input, hc_index, sycl_q);
        check_for_vector_equality(sycl_dump.data(), cpu_dump.data(), hc_size);
    }
#endif

#if NDZIP_CUDA_SUPPORT
    SECTION("CUDA vs CPU") {
        auto cuda_dump = cuda_load_and_dump_hypercube<profile>(input, hc_index);
        check_for_vector_equality(cuda_dump.data(), cpu_dump.data(), hc_size);
    }
#endif
}


TEMPLATE_TEST_CASE("Residual encodings from different encoders are equivalent",
        "[sycl][cuda][residual]", ALL_PROFILES) {
    using bits_type = typename TestType::bits_type;
    const auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);

    constexpr index_type col_chunk_size = detail::bits_of<bits_type>;
    constexpr index_type header_chunk_size = hc_size / col_chunk_size;
    constexpr index_type hc_total_chunks_size = hc_size + header_chunk_size;
    constexpr index_type chunks_per_hc = 1 /* header */ + hc_size / col_chunk_size;

    auto input = make_random_vector<bits_type>(hc_size);
    for (index_type i = 0; i < hc_size; ++i) {
        for (auto idx : {0, 12, 13, 29, static_cast<int>(bits_of<bits_type> - 2)}) {
            input[i] &= ~(bits_type{1} << ((static_cast<unsigned>(idx) * (i / bits_of<bits_type>) )
                                  % bits_of<bits_type>) );
            input[floor(i, bits_of<bits_type>) + idx] = 0;
        }
    }

    cpu::simd_aligned_buffer<bits_type> cpu_cube(input.size());
    memcpy(cpu_cube.data(), input.data(), input.size() * sizeof(bits_type));
    std::vector<bits_type> cpu_stream(hc_size * 2);
    auto cpu_length_bytes = cpu::zero_bit_encode(
            cpu_cube.data(), reinterpret_cast<std::byte *>(cpu_stream.data()), hc_size);

    const auto num_chunks = 1 + hc_size / col_chunk_size;

#if NDZIP_HIPSYCL_SUPPORT
    SECTION("SYCL vs CPU") {
        sycl::queue q{sycl::gpu_selector{}};

        buffer<bits_type> input_buf{hc_size};
        q.submit([&](handler &cgh) {
            cgh.copy(input.data(), input_buf.template get_access<sam::discard_write>(cgh));
        });

        buffer<bits_type> chunks_buf{hc_total_chunks_size};
        buffer<index_type> chunk_lengths_buf{
                range<1>{ceil(1 + num_chunks, gpu_sycl::hierarchical_inclusive_scan_granularity)}};

        q.submit([&](handler &cgh) {
            auto input_acc = input_buf.template get_access<sam::read>(cgh);
            auto columns_acc = chunks_buf.template get_access<sam::discard_write>(cgh);
            auto chunk_lengths_acc = chunk_lengths_buf.get_access<sam::discard_write>(cgh);
            cgh.parallel<gpu_hypercube_transpose_test_kernel<TestType>>(sycl::range<1>{1},
                    sycl::range<1>{gpu::hypercube_group_size<TestType>},
                    [=](gpu_sycl::hypercube_group<TestType> grp, sycl::physical_item<1> phys_idx) {
                        gpu_sycl::hypercube_memory<TestType, gpu::forward_transform_tag> lm{grp};
                        gpu::hypercube_ptr<TestType, gpu::forward_transform_tag> hc{lm()};
                        grp.distribute_for(
                                hc_size, [&](index_type i) { hc.store(i, input_acc[i]); });
                        gpu_sycl::write_transposed_chunks(
                                grp, hc, &columns_acc[0], &chunk_lengths_acc[1]);
                        // hack
                        if (phys_idx.get_global_linear_id() == 0) {
                            grp.single_item([&] { chunk_lengths_acc[0] = 0; });
                        }
                    });
        });

        std::vector<bits_type> chunks(chunks_buf.get_range().get(0));
        q.submit([&](handler &cgh) {
             cgh.copy(chunks_buf.template get_access<sam::read>(cgh), chunks.data());
         }).wait();

        std::vector<index_type> chunk_lengths(chunk_lengths_buf.get_range().get(0));
        q.submit([&](handler &cgh) {
             cgh.copy(chunk_lengths_buf.get_access<sam::read>(cgh), chunk_lengths.data());
         }).wait();
        gpu_sycl::hierarchical_inclusive_scan(q, chunk_lengths_buf, sycl::plus<index_type>{});

        buffer<bits_type> stream_buf(range<1>{hc_size * 2});
        q.submit([&](handler &cgh) {
            cgh.fill(stream_buf.template get_access<sam::discard_write>(cgh), bits_type{0});
        });

        buffer<index_type> length_buf{range<1>{1}};
        q.submit([&](sycl::handler &cgh) {
            auto chunks_acc = chunks_buf.template get_access<sam::read>(cgh);
            auto chunk_offsets_acc = chunk_lengths_buf.template get_access<sam::read>(cgh);
            auto stream_acc = stream_buf.template get_access<sam::discard_write>(cgh);
            auto length_acc = length_buf.template get_access<sam::discard_write>(cgh);
            cgh.parallel<gpu_hypercube_compact_test_kernel<TestType>>(
                    sycl::range<1>{1 /* num_hypercubes */},
                    sycl::range<1>{gpu::hypercube_group_size<TestType>},
                    [=](gpu_sycl::hypercube_group<TestType> grp, sycl::physical_item<1> phys_idx) {
                        const auto hc_index = grp.get_id(0);
                        gpu_sycl::compact_chunks<TestType>(grp,
                                &chunks_acc.get_pointer()[hc_index * hc_total_chunks_size],
                                &chunk_offsets_acc.get_pointer()[hc_index * chunks_per_hc],
                                &length_acc[0], &stream_acc.get_pointer()[0]);
                    });
        });

        index_type gpu_num_words;
        q.submit([&](handler &cgh) {
             cgh.copy(length_buf.template get_access<sam::read>(cgh), &gpu_num_words);
         }).wait();
        auto gpu_length_bytes = gpu_num_words * sizeof(bits_type);

        std::vector<bits_type> gpu_stream(stream_buf.get_range()[0]);
        q.submit([&](handler &cgh) {
             cgh.copy(stream_buf.template get_access<sam::read>(cgh), gpu_stream.data());
         }).wait();

        CHECK(gpu_length_bytes == cpu_length_bytes);
        check_for_vector_equality(gpu_stream, cpu_stream);
    }
#endif

#if NDZIP_CUDA_SUPPORT
    SECTION("CUDA vs CPU") {
        (void) chunks_per_hc; // unused when compiling only for CUDA

        gpu_cuda::cuda_buffer<bits_type> input_buf(hc_size);
        gpu_cuda::cuda_buffer<bits_type> chunks_buf(hc_total_chunks_size);
        gpu_cuda::cuda_buffer<index_type> chunk_lengths_buf(
                ceil(1 + num_chunks, gpu_cuda::hierarchical_inclusive_scan_granularity));

        CHECKED_CUDA_CALL(cudaMemcpy, input_buf.get(), input.data(), hc_size * sizeof(bits_type),
                cudaMemcpyHostToDevice);

        cuda_fill(chunks_buf.get(), bits_type{}, chunks_buf.size());

        encode_hypercube_kernel<TestType><<<1, (gpu::hypercube_group_size<TestType>)>>>(
                input_buf.get(), chunks_buf.get(), chunk_lengths_buf.get());

        std::vector<bits_type> chunks(chunks_buf.size());
        CHECKED_CUDA_CALL(cudaMemcpy, chunks.data(), chunks_buf.get(),
                chunks_buf.size() * sizeof(bits_type), cudaMemcpyDeviceToHost);

        std::vector<index_type> chunk_lengths(chunk_lengths_buf.size());
        CHECKED_CUDA_CALL(cudaMemcpy, chunk_lengths.data(), chunk_lengths_buf.get(),
                chunk_lengths_buf.size() * sizeof(index_type), cudaMemcpyDeviceToHost);

        gpu_cuda::hierarchical_inclusive_scan(
                chunk_lengths_buf.get(), chunk_lengths_buf.size(), gpu_cuda::plus<index_type>{});

        gpu_cuda::cuda_buffer<index_type> stream_length_buf(1);
        gpu_cuda::cuda_buffer<bits_type> stream_buf(hc_size * 2);
        cuda_fill(stream_buf.get(), bits_type{}, stream_buf.size());

        test_compact_chunks<TestType>
                <<<1, (gpu::hypercube_group_size<TestType>)>>>(chunks_buf.get(),
                        chunk_lengths_buf.get(), stream_length_buf.get(), stream_buf.get());

        std::vector<bits_type> gpu_stream(stream_buf.size());
        CHECKED_CUDA_CALL(cudaMemcpy, gpu_stream.data(), stream_buf.get(),
                stream_buf.size() * sizeof(bits_type), cudaMemcpyDeviceToHost);

        index_type gpu_stream_length;
        CHECKED_CUDA_CALL(cudaMemcpy, &gpu_stream_length, stream_length_buf.get(),
                sizeof(index_type), cudaMemcpyDeviceToHost);
        auto gpu_length_bytes = gpu_stream_length * sizeof(bits_type);

        CHECK(gpu_length_bytes == cpu_length_bytes);
        check_for_vector_equality(gpu_stream, cpu_stream);
    }
#endif
}


template<typename>
class gpu_hypercube_decode_test_kernel;

TEMPLATE_TEST_CASE("GPU hypercube decoding works", "[sycl][cuda][decode]", ALL_PROFILES) {
    using bits_type = typename TestType::bits_type;
    const auto hc_size = ipow(TestType::hypercube_side_length, TestType::dimensions);

    auto input = make_random_vector<bits_type>(hc_size);
    for (index_type i = 0; i < hc_size; ++i) {
        for (auto idx : {0, 12, 13, 29, static_cast<int>(bits_of<bits_type> - 2)}) {
            input[i] &= ~(bits_type{1} << ((static_cast<unsigned>(idx) * (i / bits_of<bits_type>) )
                                  % bits_of<bits_type>) );
            input[floor(i, bits_of<bits_type>) + idx] = 0;
        }
    }

    cpu::simd_aligned_buffer<bits_type> cpu_cube(input.size());
    memcpy(cpu_cube.data(), input.data(), input.size() * sizeof(bits_type));
    std::vector<bits_type> stream(hc_size * 2);
    auto cpu_length_bytes = cpu::zero_bit_encode(
            cpu_cube.data(), reinterpret_cast<std::byte *>(stream.data()), hc_size);
    REQUIRE(cpu_length_bytes % sizeof(bits_type) == 0);

#if NDZIP_HIPSYCL_SUPPORT
    SECTION("Using SYCL") {
        sycl::queue q{sycl::gpu_selector{}};

        buffer<bits_type> stream_buf{stream.data(), range<1>{cpu_length_bytes / sizeof(bits_type)}};

        buffer<bits_type> output_buf{range<1>{hc_size}};
        q.submit([&](handler &cgh) {
            auto stream_acc = stream_buf.template get_access<sam::read>(cgh);
            auto output_acc = output_buf.template get_access<sam::discard_write>(cgh);
            cgh.parallel<gpu_hypercube_decode_test_kernel<TestType>>(sycl::range{1},
                    sycl::range<1>{gpu::hypercube_group_size<TestType>},
                    [stream_acc, output_acc](
                            gpu_sycl::hypercube_group<TestType> grp, sycl::physical_item<1>) {
                        gpu_sycl::hypercube_memory<TestType, gpu::inverse_transform_tag> lm{grp};
                        gpu::hypercube_ptr<TestType, gpu::inverse_transform_tag> hc{lm()};
                        gpu_sycl::read_transposed_chunks<TestType>(
                                grp, hc, stream_acc.get_pointer());
                        grp.distribute_for(
                                hc_size, [&](index_type i) { output_acc[i] = hc.load(i); });
                    });
        });

        std::vector<bits_type> output(hc_size);
        q.submit([&](handler &cgh) {
            cgh.copy(output_buf.template get_access<sam::read>(cgh), output.data());
        });
        q.wait();

        check_for_vector_equality(output, input);
    }
#endif

#if NDZIP_CUDA_SUPPORT
    SECTION("Using CUDA") {
        gpu_cuda::cuda_buffer<bits_type> stream_buf(stream.size());
        gpu_cuda::cuda_buffer<bits_type> output_buf(hc_size);

        CHECKED_CUDA_CALL(cudaMemcpy, stream_buf.get(), stream.data(),
                stream.size() * sizeof(bits_type), cudaMemcpyHostToDevice);

        hypercube_decode_test_kernel<TestType>
                <<<1, (gpu::hypercube_group_size<TestType>)>>>(stream_buf.get(), output_buf.get());

        std::vector<bits_type> output(hc_size);
        CHECKED_CUDA_CALL(cudaMemcpy, output.data(), output_buf.get(), hc_size * sizeof(bits_type),
                cudaMemcpyDeviceToHost);

        check_for_vector_equality(output, input);
    }
#endif
}


template<typename Profile, typename Tag, typename CpuTransform
#if NDZIP_HIPSYCL_SUPPORT
        ,
        typename SyclTransform
#endif
#if NDZIP_CUDA_SUPPORT
        ,
        typename CudaTransform
#endif
        >
static void test_cpu_gpu_transform_equality(const CpuTransform &cpu_transform
#if NDZIP_HIPSYCL_SUPPORT
        ,
        const SyclTransform &sycl_transform
#endif
#if NDZIP_CUDA_SUPPORT
        ,
        const CudaTransform &cuda_transform
#endif
) {
    using bits_type = typename Profile::bits_type;
    constexpr auto hc_size
            = static_cast<index_type>(ipow(Profile::hypercube_side_length, Profile::dimensions));

    const auto input = make_random_vector<bits_type>(hc_size);

    auto cpu_transformed = input;
    cpu_transform(cpu_transformed.data());

#if NDZIP_HIPSYCL_SUPPORT
    SECTION("with SYCL") {
        sycl::queue sycl_q{sycl::gpu_selector{}};
        buffer<bits_type> sycl_io_buf{range<1>{hc_size}};

        sycl_q.submit([&](handler &cgh) {
            cgh.copy(input.data(), sycl_io_buf.template get_access<sam::discard_write>(cgh));
        });
        sycl_q.submit([&](handler &cgh) {
            auto global_acc = sycl_io_buf.template get_access<sam::read_write>(cgh);
            cgh.parallel<sycl_transform_test_kernel<Profile, Tag>>(range<1>{1},
                    range<1>{gpu::hypercube_group_size<Profile>},
                    [global_acc, hc_size = hc_size, sycl_transform](
                            gpu_sycl::hypercube_group<Profile> grp, physical_item<1>) {
                        gpu_sycl::hypercube_memory<Profile, Tag> lm{grp};
                        gpu::hypercube_ptr<Profile, Tag> hc{lm()};
                        grp.distribute_for(
                                hc_size, [&](index_type i) { hc.store(i, global_acc[i]); });
                        sycl_transform(grp, hc);
                        grp.distribute_for(
                                hc_size, [&](index_type i) { global_acc[i] = hc.load(i); });
                    });
        });

        std::vector<bits_type> sycl_transformed(hc_size);
        sycl_q.submit([&](handler &cgh) {
            cgh.copy(sycl_io_buf.template get_access<sam::read>(cgh), sycl_transformed.data());
        });
        sycl_q.wait();

        check_for_vector_equality(sycl_transformed, cpu_transformed);
    }
#endif

#if NDZIP_CUDA_SUPPORT
    SECTION("with CUDA") {
        gpu_cuda::cuda_buffer<bits_type> cuda_io_buf(hc_size);
        CHECKED_CUDA_CALL(cudaMemcpy, cuda_io_buf.get(), input.data(),
                input.size() * sizeof(bits_type), cudaMemcpyHostToDevice);
        test_gpu_transform<Profile, Tag, CudaTransform>
                <<<1, (gpu::hypercube_group_size<Profile>)>>>(cuda_io_buf.get(), cuda_transform);

        std::vector<bits_type> cuda_transformed(hc_size);
        CHECKED_CUDA_CALL(cudaMemcpy, cuda_transformed.data(), cuda_io_buf.get(),
                cuda_transformed.size() * sizeof(bits_type), cudaMemcpyDeviceToHost);

        check_for_vector_equality(cuda_transformed, cpu_transformed);
    }
#endif
}

TEMPLATE_TEST_CASE("CPU and GPU forward block transforms are identical", "[transform][sycl][cuda]",
        ALL_PROFILES) {
    using bits_type = typename TestType::bits_type;
    test_cpu_gpu_transform_equality<TestType, gpu::forward_transform_tag>(
            [](bits_type *block) {
                detail::block_transform(
                        block, TestType::dimensions, TestType::hypercube_side_length);
            }
#if NDZIP_HIPSYCL_SUPPORT
            ,
            // Use lambda instead of the function name, otherwise a host function pointer will
            // be passed into the device kernel
            [](gpu_sycl::hypercube_group<TestType> grp,
                    gpu::hypercube_ptr<TestType, gpu::forward_transform_tag> hc) {
                constexpr auto hc_size
                        = ipow(TestType::hypercube_side_length, TestType::dimensions);
                grp.distribute_for(
                        hc_size, [&](index_type i) { hc.store(i, rotate_left_1(hc.load(i))); });
                gpu_sycl::forward_block_transform(grp, hc);
            }
#endif
#if NDZIP_CUDA_SUPPORT
            ,
            [] __device__(gpu_cuda::hypercube_block<TestType> block,
                    gpu::hypercube_ptr<TestType, gpu::forward_transform_tag> hc) {
                constexpr auto hc_size
                        = ipow(TestType::hypercube_side_length, TestType::dimensions);
                distribute_for(hc_size, block,
                        [&](index_type i) { hc.store(i, rotate_left_1(hc.load(i))); });
                __syncthreads();
                gpu_cuda::forward_block_transform(block, hc);
            }
#endif
    );
}

TEMPLATE_TEST_CASE("CPU and GPU inverse block transforms are identical", "[sycl][gpu][transform]",
        ALL_PROFILES) {
    using bits_type = typename TestType::bits_type;
    test_cpu_gpu_transform_equality<TestType, gpu::inverse_transform_tag>(
            [](bits_type *block) {
                detail::inverse_block_transform(
                        block, TestType::dimensions, TestType::hypercube_side_length);
            }
#if NDZIP_HIPSYCL_SUPPORT
            ,
            // Use lambda instead of the function name, otherwise a host function pointer will
            // be passed into the device kernel
            [](gpu_sycl::hypercube_group<TestType> grp,
                    gpu::hypercube_ptr<TestType, gpu::inverse_transform_tag> hc) {
                constexpr auto hc_size
                        = ipow(TestType::hypercube_side_length, TestType::dimensions);
                gpu_sycl::inverse_block_transform<TestType>(grp, hc);
                grp.distribute_for(
                        hc_size, [&](index_type i) { hc.store(i, rotate_right_1(hc.load(i))); });
            }
#endif
#if NDZIP_CUDA_SUPPORT
            ,
            [] __device__(gpu_cuda::hypercube_block<TestType> block,
                    gpu::hypercube_ptr<TestType, gpu::inverse_transform_tag> hc) {
                constexpr auto hc_size
                        = ipow(TestType::hypercube_side_length, TestType::dimensions);
                gpu_cuda::inverse_block_transform<TestType>(block, hc);
                __syncthreads();
                distribute_for(hc_size, block,
                        [&](index_type i) { hc.store(i, rotate_right_1(hc.load(i))); });
            });
#endif
}

TEMPLATE_TEST_CASE("Single block compresses identically on all encoders",
        "[omp][sycl][gpu][compress]", ALL_PROFILES) {
    using data_type = typename TestType::data_type;
    using bits_type = typename TestType::bits_type;
    constexpr auto dimensions = TestType::dimensions;

    const auto size = extent<dimensions>::broadcast(TestType::hypercube_side_length);
    const auto input = make_random_vector<data_type>(num_elements(size));
    const auto input_slice = slice{input.data(), size};
    const auto max_output_words
            = div_ceil(compressed_size_bound<data_type>(size), sizeof(data_type));

    std::vector<bits_type> cpu_output(max_output_words);
    auto cpu_output_bytes
            = cpu_encoder<data_type, dimensions>{}.compress(input_slice, cpu_output.data());
    cpu_output.resize(cpu_output_bytes / sizeof(data_type));

#if NDZIP_OPENMP_SUPPORT
    SECTION("Multi-threaded vs single-threaded CPU") {
        std::vector<bits_type> mt_cpu_output(max_output_words);
        auto mt_cpu_output_bytes = mt_cpu_encoder<data_type, dimensions>{}.compress(
                input_slice, mt_cpu_output.data());
        mt_cpu_output.resize(mt_cpu_output_bytes / sizeof(data_type));
        check_for_vector_equality(mt_cpu_output, cpu_output);
    }
#endif

#if NDZIP_HIPSYCL_SUPPORT
    SECTION("SYCL vs CPU") {
        std::vector<bits_type> sycl_output(max_output_words);
        auto sycl_output_bytes
                = sycl_encoder<data_type, dimensions>{}.compress(input_slice, sycl_output.data());
        sycl_output.resize(sycl_output_bytes / sizeof(data_type));
        check_for_vector_equality(sycl_output, cpu_output);
    }
#endif

#if NDZIP_CUDA_SUPPORT
    SECTION("CUDA vs CPU") {
        std::vector<bits_type> cuda_output(max_output_words);
        auto cuda_output_bytes
                = cuda_encoder<data_type, dimensions>{}.compress(input_slice, cuda_output.data());
        cuda_output.resize(cuda_output_bytes / sizeof(data_type));
        check_for_vector_equality(cuda_output, cpu_output);
    }
#endif
}

TEMPLATE_TEST_CASE("Single block decompresses correctly on all encoders",
        "[omp][sycl][gpu][decompress]", ALL_PROFILES) {
    using data_type = typename TestType::data_type;
    using bits_type = typename TestType::bits_type;
    constexpr auto dimensions = TestType::dimensions;

    const auto size = extent<dimensions>::broadcast(TestType::hypercube_side_length);
    const auto input = make_random_vector<data_type>(num_elements(size));
    const auto input_slice = slice{input.data(), size};
    const auto max_output_words
            = div_ceil(compressed_size_bound<data_type>(size), sizeof(data_type));

    std::vector<bits_type> compressed(max_output_words);
    auto compressed_bytes
            = cpu_encoder<data_type, dimensions>{}.compress(input_slice, compressed.data());
    compressed.resize(compressed_bytes / sizeof(data_type));

    SECTION("On single-threaded CPU") {
        std::vector<data_type> cpu_output(num_elements(size));
        cpu_encoder<data_type, dimensions>{}.decompress(
                compressed.data(), compressed.size(), slice{cpu_output.data(), size});
        check_for_vector_equality(cpu_output, input);
    }

#if NDZIP_OPENMP_SUPPORT
    SECTION("On multi-threaded CPU") {
        std::vector<data_type> mt_cpu_output(num_elements(size));
        mt_cpu_encoder<data_type, dimensions>{}.decompress(
                compressed.data(), compressed.size(), slice{mt_cpu_output.data(), size});
        check_for_vector_equality(mt_cpu_output, input);
    }
#endif

#if NDZIP_HIPSYCL_SUPPORT
    SECTION("On SYCL") {
        std::vector<data_type> sycl_output(num_elements(size));
        sycl_encoder<data_type, dimensions>{}.decompress(
                compressed.data(), compressed.size(), slice{sycl_output.data(), size});
        check_for_vector_equality(sycl_output, input);
    }
#endif

#if NDZIP_CUDA_SUPPORT
    SECTION("On CUDA") {
        std::vector<data_type> cuda_output(num_elements(size));
        cuda_encoder<data_type, dimensions>{}.decompress(
                compressed.data(), compressed.size(), slice{cuda_output.data(), size});
        check_for_vector_equality(cuda_output, input);
    }
#endif
}
