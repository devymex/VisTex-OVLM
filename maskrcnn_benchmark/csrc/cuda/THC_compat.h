// Compatibility header for PyTorch 2.x+ (THC headers removed)
#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>

// Replace THCudaCheck with C10_CUDA_CHECK
#define THCudaCheck(err) C10_CUDA_CHECK(err)

// Replace THCCeilDiv
template <typename T>
__host__ __device__ inline T THCCeilDiv(T a, T b) {
    return (a + b - 1) / b;
}

// Replace atomicAdd for half types if needed
// gpuAtomicAdd is available in modern PyTorch
