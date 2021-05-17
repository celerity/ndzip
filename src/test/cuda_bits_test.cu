#include "test_utils.hh"
#include <ndzip/cuda_bits.cuh>

using namespace ndzip;
using namespace ndzip::detail;
using namespace ndzip::detail::gpu_cuda;


template<typename T>
struct logical_or {
    __host__ __device__ T operator()(T a, T b) const { return a || b; }
};


__global__ void test_warp_reduce(index_type *out) {
    out[threadIdx.x] = warp_reduce(index_type{1}, plus<index_type>{});
}

__global__ void test_warp_inclusive_scan(index_type *out) {
    out[threadIdx.x]
            = warp_inclusive_scan(static_cast<index_type>(threadIdx.x), plus<index_type>{});
}

__global__ void test_warp_exclusive_scan(index_type *out) {
    out[threadIdx.x] = warp_exclusive_scan(
            static_cast<index_type>(threadIdx.x), index_type{0}, plus<index_type>{});
}

template<typename T, index_type Range, index_type GroupSize>
__global__ void test_inclusive_scan(T *out) {
    auto block = known_size_block<GroupSize>{};
    distribute_for(Range, block, [&](index_type item) { out[item] = 1; });
    __syncthreads();
    inclusive_scan<Range>(block, out, plus<T>{});
}


TEST_CASE("Warp-cooperative primitives work", "[cuda][scan]") {
    cuda_buffer<index_type> buf(warp_size);
    std::vector<index_type> out(warp_size);
    std::vector<index_type> iota(warp_size);
    std::vector<index_type> ref(warp_size);

    std::iota(iota.begin(), iota.end(), index_type{0});

    SECTION("warp_reduce") {
        test_warp_reduce<<<1, warp_size>>>(buf.get());
        CHECKED_CUDA_CALL(cudaMemcpy, out.data(), buf.get(), warp_size * sizeof(index_type),
                cudaMemcpyDeviceToHost);
        std::fill(ref.begin(), ref.end(), warp_size);
        check_for_vector_equality(out, ref);
    }

    SECTION("warp_inclusive_scan") {
        test_warp_inclusive_scan<<<1, warp_size>>>(buf.get());
        CHECKED_CUDA_CALL(cudaMemcpy, out.data(), buf.get(), warp_size * sizeof(index_type),
                cudaMemcpyDeviceToHost);
        iter_inclusive_scan(iota.begin(), iota.end(), ref.begin(), plus<index_type>{});
        check_for_vector_equality(out, ref);
    }

    SECTION("warp_exclusive_scan") {
        test_warp_exclusive_scan<<<1, warp_size>>>(buf.get());
        CHECKED_CUDA_CALL(cudaMemcpy, out.data(), buf.get(), warp_size * sizeof(index_type),
                cudaMemcpyDeviceToHost);
        iter_exclusive_scan(iota.begin(), iota.end(), ref.begin(), plus<index_type>{});
        check_for_vector_equality(out, ref);
    }
}


TEMPLATE_TEST_CASE(
        "Subgroup hierarchical inclusive scan works", "[cuda][scan]", uint32_t, uint64_t) {
    constexpr index_type group_size = 1024;
    constexpr index_type n_groups = 9;
    constexpr index_type range = group_size * n_groups;

    cuda_buffer<TestType> out(range);
    test_inclusive_scan<TestType, range, group_size><<<1, group_size>>>(out.get());

    std::vector<TestType> cpu_input(range, TestType{1});
    std::vector<TestType> cpu_result(range);
    iter_inclusive_scan(cpu_input.begin(), cpu_input.end(), cpu_result.begin());

    std::vector<TestType> gpu_result(range);
    CHECKED_CUDA_CALL(cudaMemcpy, gpu_result.data(), out.get(), out.size() * sizeof(TestType),
            cudaMemcpyDeviceToHost);

    check_for_vector_equality(cpu_result, gpu_result);
}


TEMPLATE_TEST_CASE("hierarchical_inclusive_scan produces the expected results", "[cuda][scan]",
        plus<uint32_t>, logical_or<uint32_t>) {
    std::vector<uint32_t> input(size_t{1} << 24u);
    std::iota(input.begin(), input.end(), uint32_t{});

    std::vector<uint32_t> cpu_prefix_sum(input.size());
    iter_inclusive_scan(input.begin(), input.end(), cpu_prefix_sum.begin(), TestType{});

    cuda_buffer<uint32_t> prefix_sum_buf(input.size());
    CHECKED_CUDA_CALL(cudaMemcpy, prefix_sum_buf.get(), input.data(),
            prefix_sum_buf.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    auto keepalive
            = hierarchical_inclusive_scan(prefix_sum_buf.get(), prefix_sum_buf.size(), TestType{});

    std::vector<uint32_t> gpu_prefix_sum(input.size());
    CHECKED_CUDA_CALL(cudaMemcpy, gpu_prefix_sum.data(), prefix_sum_buf.get(),
            prefix_sum_buf.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    check_for_vector_equality(gpu_prefix_sum, cpu_prefix_sum);
}
