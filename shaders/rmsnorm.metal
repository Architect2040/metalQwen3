/**
 * MetalQwen3 - High-Performance Transformer Inference on Apple Silicon
 *
 * @file rmsnorm.metal
 * @brief RMS Normalization Metal compute shader
 * @author Shlomo Kashnai
 * @date 2024
 *
 * Metal compute shader for GPU-accelerated RMS normalization.
 * Implements parallel reduction optimized for Apple Silicon GPU.
 * Matches qwen3.c algorithm exactly with GPU acceleration.
 *
 * Based on qwen3.c implementation by Adrian Cable
 * https://github.com/adriancable/qwen3.c
 *
 * @license MIT License - See project root for full license text
 */

#include <metal_stdlib>
using namespace metal;

// RMS Normalization kernel - matches qwen3_original.c exactly
kernel void rmsnorm_kernel(
    device const float* input_data [[buffer(0)]],
    device const float* weight_data [[buffer(1)]],
    device float* output_data [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    threadgroup float* shared_memory [[threadgroup(0)]]
) {
    const uint threadgroup_size = 256; // Make sure this matches dispatch size

    // Step 1: Each thread computes partial sum of squares
    float local_sum = 0.0f;
    for (uint i = tid; i < size; i += threadgroup_size) {
        float val = input_data[i];
        local_sum += val * val;
    }

    shared_memory[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Parallel reduction to compute total sum of squares
    for (uint stride = threadgroup_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_memory[tid] += shared_memory[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 3: Thread 0 computes the normalization factor
    float ss_inv;
    if (tid == 0) {
        float ss = shared_memory[0];
        ss_inv = 1.0f / sqrt((ss / float(size)) + eps);
        shared_memory[0] = ss_inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss_inv = shared_memory[0];

    // Step 4: Apply normalization and scaling (matches ground truth: weight[j] * (ss * x[j]))
    for (uint i = tid; i < size; i += threadgroup_size) {
        output_data[i] = weight_data[i] * (ss_inv * input_data[i]);
    }
}