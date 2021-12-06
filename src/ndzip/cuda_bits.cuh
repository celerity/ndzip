#pragma once

#include "gpu_common.hh"

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>


namespace ndzip::detail::gpu_cuda {

using namespace ndzip::detail::gpu;


template<typename T>
struct plus {
    __host__ __device__ T operator()(T a, T b) const { return a + b; }
};

template<typename T>
struct bit_or {
    __host__ __device__ T operator()(T a, T b) const { return a | b; }
};


template<index_type ThreadsPerBlock>
class known_size_block {};

template<typename F>
[[gnu::always_inline]] __device__ void distribute_for_invoke(F &&f, index_type item, index_type iteration) {
    if constexpr (std::is_invocable_v<F, index_type, index_type>) {
        f(item, iteration);
    } else {
        f(item);
    }
}

template<index_type ThreadsPerBlock, typename F>
[[gnu::always_inline]] __device__ void
distribute_for_partial_iteration(index_type range, known_size_block<ThreadsPerBlock>, F &&f) {
    const index_type num_full_iterations = range / ThreadsPerBlock;
    const index_type partial_iteration_length = range % ThreadsPerBlock;
    const auto tid = static_cast<index_type>(threadIdx.x);
    if (tid < partial_iteration_length) {
        auto iteration = num_full_iterations;
        auto item = iteration * ThreadsPerBlock + tid;
        distribute_for_invoke(f, item, iteration);
    }
}

template<index_type ThreadsPerBlock, typename F>
[[gnu::always_inline]] __device__ void
distribute_for(index_type range, known_size_block<ThreadsPerBlock> block, F &&f) {
    const index_type num_full_iterations = range / ThreadsPerBlock;
    const auto tid = static_cast<index_type>(threadIdx.x);

    for (index_type iteration = 0; iteration < num_full_iterations; ++iteration) {
        auto item = iteration * ThreadsPerBlock + tid;
        distribute_for_invoke(f, item, iteration);
    }
    distribute_for_partial_iteration(range, block, f);
}

template<index_type Range, index_type ThreadsPerBlock, typename F>
[[gnu::always_inline]] __device__ void distribute_for(known_size_block<ThreadsPerBlock> block, F &&f) {
    constexpr index_type num_full_iterations = Range / ThreadsPerBlock;
    const auto tid = static_cast<index_type>(threadIdx.x);

#pragma unroll
    for (index_type iteration = 0; iteration < num_full_iterations; ++iteration) {
        auto item = iteration * ThreadsPerBlock + tid;
        distribute_for_invoke(f, item, iteration);
    }
    distribute_for_partial_iteration(Range, block, f);
}


template<typename T, typename BinaryOperation>
__device__ T warp_reduce(T x, BinaryOperation op) {
    auto warp_thread_id = threadIdx.x % warp_size;

    for (size_t i = warp_size / 2; i > 0; i /= 2) {
        auto y = __shfl_sync(0xffff'ffff, x, warp_thread_id + i);
        if (warp_thread_id < i) { x = op(x, y); }
    }
    return __shfl_sync(0xffff'ffff, x, 0);
}


template<typename T, typename BinaryOp>
__device__ T warp_inclusive_scan(T x, BinaryOp op) {
    auto warp_thread_id = threadIdx.x % warp_size;

    for (size_t i = 1; i < warp_size; i *= 2) {
        size_t next_id = warp_thread_id - i;
        if (i > warp_thread_id) { next_id = 0; }

        auto y = __shfl_sync(0xffff'ffff, x, next_id);
        if (i <= warp_thread_id && warp_thread_id < warp_size) { x = op(x, y); }
    }

    return x;
}


template<typename T, typename BinaryOp>
__device__ T warp_exclusive_scan(T x, T init, BinaryOp binary_op) {
    auto warp_thread_id = threadIdx.x % warp_size;

    for (size_t i = 1; i < warp_size; i *= 2) {
        size_t next_id = warp_thread_id - i;
        if (i > warp_thread_id) { next_id = 0; }

        auto y = __shfl_sync(0xffff'ffff, x, next_id);
        if (i <= warp_thread_id && warp_thread_id < warp_size) { x = binary_op(x, y); }
    }

    auto scan = __shfl_sync(0xffff'ffff, x, (warp_thread_id - 1) % warp_size);
    return warp_thread_id == 0 ? init : binary_op(scan, init);
}


template<index_type Range, index_type ThreadsPerBlock, typename Accessor, typename BinaryOp>
__device__ std::enable_if_t<(Range <= warp_size)>
inclusive_scan(known_size_block<ThreadsPerBlock> block, Accessor acc, BinaryOp op) {
    static_assert(ThreadsPerBlock % warp_size == 0);
    distribute_for<ceil(Range, warp_size)>(block, [&](index_type item, index_type) {
        auto a = item < Range ? acc[item] : 0;
        auto b = warp_inclusive_scan(a, op);
        if (item < Range) { acc[item] = b; }
    });
}


template<index_type Range, index_type ThreadsPerBlock, typename Accessor, typename BinaryOp>
__device__ std::enable_if_t<(Range > warp_size)>
inclusive_scan(known_size_block<ThreadsPerBlock> block, Accessor acc, BinaryOp op) {
    static_assert(ThreadsPerBlock % warp_size == 0);
    using value_type = std::decay_t<decltype(acc[index_type{}])>;

    value_type fine[div_ceil(Range, ThreadsPerBlock)];
    __shared__ value_type coarse[div_ceil(Range, warp_size)];
    distribute_for<ceil(Range, warp_size)>(block, [&](index_type item, index_type iteration) {
        fine[iteration] = warp_inclusive_scan(item < Range ? acc[item] : 0, op);
        if (item % warp_size == warp_size - 1) { coarse[item / warp_size] = fine[iteration]; }
    });
    __syncthreads();

    inclusive_scan<div_ceil(Range, warp_size)>(block, coarse, op);
    __syncthreads();

    distribute_for<Range>(block, [&](index_type item, index_type iteration) {
        auto value = fine[iteration];
        if (item >= warp_size) { value = op(value, coarse[item / warp_size - 1]); }
        acc[item] = value;
    });
}

#define STRINGIZE2(x) #x
#define STRINGIZE(x) STRINGIZE2(x)

inline void cuda_check(cudaError_t err, const char *tag) {
    if (err != cudaSuccess) { throw std::runtime_error(std::string{tag} + cudaGetErrorString(err)); }
}

#define CHECKED_CUDA_CALL(f, ...) ::ndzip::detail::gpu_cuda::cuda_check(f(__VA_ARGS__), STRINGIZE(f) ": ")


template<typename T>
class cuda_buffer {
  public:
    cuda_buffer() = default;

    explicit cuda_buffer(index_type size) : _size(size) { CHECKED_CUDA_CALL(cudaMalloc, &_memory, size * sizeof(T)); }

    cuda_buffer(cuda_buffer &&other) noexcept {
        std::swap(_memory, other._memory);
        std::swap(_size, other._size);
    }

    cuda_buffer &operator=(cuda_buffer &&other) noexcept {
        reset();
        std::swap(_memory, other._memory);
        std::swap(_memory, other._size);
        return *this;
    }

    ~cuda_buffer() { reset(); }

    void allocate(index_type size) {
        reset();
        CHECKED_CUDA_CALL(cudaMalloc, &_memory, size * sizeof(T));
        _size = size;
    }

    void reset() {
        if (_memory) { CHECKED_CUDA_CALL(cudaFree, _memory); }
        _memory = nullptr;
        _size = 0;
    }

    index_type size() const { return _size; }

    T *get() { return _memory; }

  private:
    T *_memory = nullptr;
    index_type _size = 0;
};


class cuda_event {
  public:
    class allocate_t {
    } inline constexpr static allocate{};

    cuda_event() noexcept = default;

    cuda_event(allocate_t) { CHECKED_CUDA_CALL(cudaEventCreate, &_evt); }

    cuda_event(cuda_event &&other) noexcept {
        using std::swap;
        swap(_evt, other._evt);
    }

    cuda_event &operator=(cuda_event &&other) noexcept {
        using std::swap;
        reset();
        swap(_evt, other._evt);
        return *this;
    }

    ~cuda_event() { reset(); }

    explicit operator bool() const noexcept { return _evt != nullptr; }

    void reset() {
        if (_evt) {
            CHECKED_CUDA_CALL(cudaEventDestroy, _evt);
            _evt = nullptr;
        }
    }

    void record() {
        if (!_evt) { CHECKED_CUDA_CALL(cudaEventCreate, &_evt); }
        CHECKED_CUDA_CALL(cudaEventRecord, _evt);
    }

    friend kernel_duration operator-(const cuda_event &a, const cuda_event &b) {
        assert(a);
        assert(b);
        CHECKED_CUDA_CALL(cudaEventSynchronize, a._evt);
        float ms;
        CHECKED_CUDA_CALL(cudaEventElapsedTime, &ms, b._evt, a._evt);
        return std::chrono::duration_cast<kernel_duration>(std::chrono::duration<float, std::milli>(ms));
    }

  private:
    cudaEvent_t _evt = nullptr;
};


template<typename Scalar, typename BinaryOp>
__global__ void hierarchical_inclusive_scan_reduce(Scalar *big_buf, Scalar *small_buf, BinaryOp op) {
    constexpr index_type granularity = hierarchical_inclusive_scan_granularity;
    constexpr index_type threads_per_block = 256;
    auto block = known_size_block<threads_per_block>{};

    Scalar *big = &big_buf[blockIdx.x * granularity];
    Scalar &small = small_buf[blockIdx.x];
    inclusive_scan<granularity>(block, big, op);
    // TODO unnecessary GM read from big -- maybe return final sum
    //  in the last item? Or allow additional accessor in inclusive_scan?
    __syncthreads();
    if (threadIdx.x == 0) { small = big[granularity - 1]; }
}

template<typename Scalar, typename BinaryOp>
__global__ void hierarchical_inclusive_scan_expand(const Scalar *small_buf, Scalar *big_buf, BinaryOp op) {
    constexpr index_type granularity = hierarchical_inclusive_scan_granularity;
    constexpr index_type threads_per_block = 256;
    auto block = known_size_block<threads_per_block>{};

    Scalar *big = &big_buf[(blockIdx.x + 1) * granularity];
    Scalar small = small_buf[blockIdx.x];
    distribute_for(granularity, block, [&](index_type i) { big[i] = op(big[i], small); });
}


template<typename Scalar>
auto hierarchical_inclusive_scan_allocate(index_type n_elems) {
    constexpr index_type granularity = hierarchical_inclusive_scan_granularity;

    std::vector<cuda_buffer<Scalar>> intermediate_bufs;
    auto n = n_elems;
    assert(n % granularity == 0);  // otherwise we will overrun the in_out buffer bounds

    while (n > 1) {
        n = div_ceil(n, granularity);
        intermediate_bufs.emplace_back(ceil(n, granularity));
    }
    return intermediate_bufs;
}


template<typename Scalar, typename BinaryOp>
void hierarchical_inclusive_scan(Scalar *in_out_buf, std::vector<cuda_buffer<Scalar>> &intermediate_bufs,
        index_type n_elems, BinaryOp op = {}, cudaStream_t stream = nullptr) {
    constexpr index_type granularity = hierarchical_inclusive_scan_granularity;
    constexpr index_type threads_per_block = 256;

    for (index_type i = 0; i < intermediate_bufs.size(); ++i) {
        auto *big_buf = i > 0 ? intermediate_bufs[i - 1].get() : in_out_buf;
        auto big_buf_size = i > 0 ? intermediate_bufs[i - 1].size() : n_elems;
        auto *small_buf = intermediate_bufs[i].get();
        assert(big_buf_size > 0);
        const auto blocks = div_ceil(big_buf_size, granularity);
        hierarchical_inclusive_scan_reduce<<<blocks, threads_per_block, 0, stream>>>(big_buf, small_buf, op);
    }

    for (index_type i = 1; i < intermediate_bufs.size(); ++i) {
        auto ii = static_cast<index_type>(intermediate_bufs.size()) - 1 - i;
        auto *small_buf = intermediate_bufs[ii].get();
        auto *big_buf = ii > 0 ? intermediate_bufs[ii - 1].get() : in_out_buf;
        auto big_buf_size = ii > 0 ? intermediate_bufs[ii - 1].size() : n_elems;
        assert(big_buf_size > 0);
        const auto blocks = div_ceil(big_buf_size, granularity) - 1;
        hierarchical_inclusive_scan_expand<<<blocks, threads_per_block, 0, stream>>>(small_buf, big_buf, op);
    }
}


}  // namespace ndzip::detail::gpu_cuda
