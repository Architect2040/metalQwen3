/**
 * MetalQwen3 - High-Performance Transformer Inference on Apple Silicon
 *
 * @file swiglu.metal
 * @brief SwiGLU activation function Metal compute shader
 * @author Shlomo Kashnai
 * @date 2024
 *
 * Metal compute shader for SwiGLU (Swish-Gated Linear Unit) activation.
 * Implements element-wise SiLU activation with gating mechanism for
 * Qwen3 feed-forward network layers.
 *
 * Based on qwen3.c SwiGLU implementation by Adrian Cable
 * https://github.com/adriancable/qwen3.c
 *
 * @license MIT License - See project root for full license text
 */

#include <metal_stdlib>
using namespace metal;

// SwiGLU activation function kernel - matches qwen3_original.c exactly
// Ground truth: s->hb[i] *= s->hb2[i] * (1.0f / (1.0f + expf(-s->hb[i])));
// This is an in-place operation where hb is both input and output
kernel void swiglu_kernel(
    device float* hb [[buffer(0)]],              // W1(x) - input/output buffer (in-place)
    device const float* hb2 [[buffer(1)]],       // W3(x) - gate values
    constant uint& size [[buffer(2)]],           // hidden_dim size
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;

    float x1 = hb[gid];     // Current value from W1(x)
    float x3 = hb2[gid];    // Gate value from W3(x)

    // Compute SiLU activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float silu_x1 = x1 * (1.0f / (1.0f + exp(-x1)));

    // Apply gating and store back (matches ground truth exactly)
    hb[gid] = silu_x1 * x3;
}