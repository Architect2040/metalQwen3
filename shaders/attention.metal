#include <metal_stdlib>
using namespace metal;

// Multi-head attention kernel - matches qwen3_original.c exactly
// Each thread handles one attention head
kernel void attention_kernel(
    device const float* q [[buffer(0)]],              // Query (n_heads * head_dim)
    device float* att [[buffer(1)]],                  // Attention scores (n_heads * seq_len)
    device float* xb [[buffer(2)]],                   // Output (n_heads * head_dim)
    device const float* key_cache [[buffer(3)]],      // Key cache (seq_len * kv_dim)
    device const float* value_cache [[buffer(4)]],    // Value cache (seq_len * kv_dim)
    constant uint& pos [[buffer(5)]],                 // Current position
    constant uint& head_dim [[buffer(6)]],
    constant uint& n_heads [[buffer(7)]],
    constant uint& n_kv_heads [[buffer(8)]],
    constant uint& seq_len [[buffer(9)]],
    constant uint& kv_dim [[buffer(10)]],
    constant uint& kv_mul [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_heads) return;

    const uint h = gid;
    device const float* q_head = q + h * head_dim;
    device float* att_head = att + h * seq_len;

    // Step 1: Compute attention scores (matches ground truth)
    // for (int t = 0; t <= pos; t++)
    for (uint t = 0; t <= pos; t++) {
        device const float* k = key_cache + t * kv_dim + (h / kv_mul) * head_dim;

        // Calculate dot product: score = sum(q[i] * k[i])
        float score = 0.0f;
        for (uint i = 0; i < head_dim; i++) {
            score += q_head[i] * k[i];
        }

        // Scale by sqrt(head_dim) for numerical stability
        att_head[t] = score / sqrt(float(head_dim));
    }

    // Step 2: Apply softmax to attention scores (pos + 1 elements)
    // Find max for numerical stability
    float max_val = att_head[0];
    for (uint t = 1; t <= pos; t++) {
        max_val = max(max_val, att_head[t]);
    }

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (uint t = 0; t <= pos; t++) {
        att_head[t] = exp(att_head[t] - max_val);
        sum_exp += att_head[t];
    }

    // Normalize
    for (uint t = 0; t <= pos; t++) {
        att_head[t] /= sum_exp;
    }

    // Step 3: Weighted sum of values (matches ground truth)
    device float* xb_head = xb + h * head_dim;

    // Clear output
    for (uint i = 0; i < head_dim; i++) {
        xb_head[i] = 0.0f;
    }

    // Accumulate weighted values
    for (uint t = 0; t <= pos; t++) {
        device const float* v = value_cache + t * kv_dim + (h / kv_mul) * head_dim;
        float att_weight = att_head[t];

        for (uint i = 0; i < head_dim; i++) {
            xb_head[i] += att_weight * v[i];
        }
    }
}