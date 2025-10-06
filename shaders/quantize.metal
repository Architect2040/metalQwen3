/**
 * MetalQwen3 - High-Performance Transformer Inference on Apple Silicon
 *
 * @file quantize.metal
 * @brief GPU-accelerated quantization kernel
 * @author Shlomo Kashnai
 * @date 2024
 *
 * Quantize float32 tensors to int8 with group-wise scaling (GS=64)
 *
 * @license MIT License - See project root for full license text
 */

#include <metal_stdlib>
using namespace metal;

// Quantize kernel - converts fp32 to int8 with group-wise scaling
kernel void quantize_kernel(
    device const float* input [[buffer(0)]],      // input floats (n,)
    device int8_t* output_q [[buffer(1)]],        // output quantized (n,)
    device float* output_s [[buffer(2)]],         // output scales (n/GS,)
    constant uint& n [[buffer(3)]],               // input size
    constant uint& group_size [[buffer(4)]],      // GS (typically 64)
    uint gid [[thread_position_in_grid]]
) {
    uint group_idx = gid;
    if (group_idx >= (n / group_size)) return;

    uint group_start = group_idx * group_size;

    // Find max absolute value in group
    float wmax = 0.0f;
    for (uint i = 0; i < group_size; i++) {
        float val = abs(input[group_start + i]);
        wmax = max(wmax, val);
    }

    // Calculate scale
    float scale = wmax / 127.0f;
    output_s[group_idx] = scale;

    // Quantize values in group
    for (uint i = 0; i < group_size; i++) {
        float quant_value = input[group_start + i] / scale;
        output_q[group_start + i] = int8_t(round(quant_value));
    }
}
