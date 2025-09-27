#include <metal_stdlib>
using namespace metal;

// Softmax kernel - matches qwen3_original.c exactly
// In-place softmax operation
kernel void softmax_kernel(
    device float* data [[buffer(0)]],              // Input and output buffer (in-place operation)
    constant uint& size [[buffer(1)]],             // Size of vector
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    threadgroup float* shared_data [[threadgroup(0)]]
) {
    const uint threadgroup_size = 256; // Make sure this matches dispatch size

    // Phase 1: Find maximum value for numerical stability (matches ground truth)
    float local_max = -1e9f;
    for (uint i = tid; i < size; i += threadgroup_size) {
        local_max = max(local_max, data[i]);
    }

    shared_data[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction to find global max
    for (uint stride = threadgroup_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float max_val = shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute exponentials and sum (matches ground truth: x[i] = expf(x[i] - max_val))
    float local_sum = 0.0f;
    for (uint i = tid; i < size; i += threadgroup_size) {
        float exp_val = exp(data[i] - max_val);
        data[i] = exp_val;
        local_sum += exp_val;
    }

    shared_data[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction to compute sum
    for (uint stride = threadgroup_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sum_exp = shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize (matches ground truth: x[i] /= sum)
    for (uint i = tid; i < size; i += threadgroup_size) {
        data[i] /= sum_exp;
    }
}