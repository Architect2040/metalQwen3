#include <metal_stdlib>
using namespace metal;

// Matrix multiplication kernel with tiling optimization
kernel void matrix_mul_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& N [[buffer(3)]],    // Matrix dimension (N x N)
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    const uint row = gid.y;
    const uint col = gid.x;

    // Check bounds
    if (row >= N || col >= N) return;

    // Simple matrix multiplication (non-optimized)
    float result = 0.0f;
    for (uint k = 0; k < N; k++) {
        result += A[row * N + k] * B[k * N + col];
    }

    // Store result
    C[row * N + col] = result;
}