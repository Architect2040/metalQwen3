/**
 * MetalQwen3 - High-Performance Transformer Inference on Apple Silicon
 *
 * @file quantized_matmul.metal
 * @brief INT8 quantized matrix multiplication Metal compute shader
 * @author Shlomo Kashnai
 * @date 2024
 *
 * GPU-accelerated quantized matrix multiplication for Qwen3 transformer.
 * Implements Q8_0 quantization with group-wise scaling for maximum performance
 * while preserving numerical accuracy from original qwen3.c implementation.
 *
 * Based on qwen3.c matrix multiplication by Adrian Cable
 * https://github.com/adriancable/qwen3.c
 *
 * @license MIT License - See project root for full license text
 */

#include <metal_stdlib>
using namespace metal;

// Quantized matrix multiplication kernel - matches qwen3_original.c exactly
// W (d,n) @ x (n,) -> xout (d,) - the most computationally intensive function
kernel void quantized_matmul_kernel(
    device const int8_t* x_q [[buffer(0)]],          // x quantized (n,)
    device const int8_t* w_q [[buffer(1)]],          // W quantized (d, n)
    device const float* x_s [[buffer(2)]],           // x scales (n/group_size,)
    device const float* w_s [[buffer(3)]],           // W scales (d*n/group_size,)
    device float* output_data [[buffer(4)]],         // output (d,)
    constant uint& M [[buffer(5)]],                  // d (output dimension)
    constant uint& N [[buffer(6)]],                  // 1 (we process one vector at a time)
    constant uint& K [[buffer(7)]],                  // n (input dimension)
    constant uint& group_size [[buffer(8)]],         // GS (quantization group size)
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= M) return;  // Each thread handles one output element

    const uint i = gid;
    float val = 0.0f;
    const uint in = i * K;  // W row start index

    // Process in groups of GS (matches ground truth: for (int j = 0; j <= n - GS; j += GS))
    for (uint j = 0; j <= K - group_size; j += group_size) {
        // Accumulate INT8 products within this group
        int32_t ival = 0;
        for (uint k = 0; k < group_size; k++) {
            ival += int32_t(x_q[j + k]) * int32_t(w_q[in + j + k]);
        }

        // Apply scaling factors (matches ground truth exactly)
        val += float(ival) * w_s[(in + j) / group_size] * x_s[j / group_size];
    }

    output_data[i] = val;
}