#include <metal_stdlib>
using namespace metal;

// Rotary Positional Embedding (RoPE) kernel - matches qwen3_original.c exactly
// Ground truth applies RMSNorm + RoPE in sequence for both Q and K heads
kernel void rope_kernel(
    device float* q [[buffer(0)]],                    // Query heads (n_heads * head_dim)
    device float* k [[buffer(1)]],                    // Key heads (n_kv_heads * head_dim)
    device const float* q_norm_weights [[buffer(2)]], // Q RMSNorm weights (head_dim)
    device const float* k_norm_weights [[buffer(3)]], // K RMSNorm weights (head_dim)
    constant uint& head_dim [[buffer(4)]],
    constant uint& pos [[buffer(5)]],
    constant uint& n_heads [[buffer(6)]],
    constant uint& n_kv_heads [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const float eps = 1e-6f;

    // Process Q heads: RMSNorm + RoPE for each head
    if (gid < n_heads) {
        uint h = gid;
        device float* q_head = q + h * head_dim;

        // Q-RMSNorm (matches ground truth: rmsnorm(q, q, w->q_norm_weights + l * p->head_dim, p->head_dim))
        float ss = 0.0f;
        for (uint j = 0; j < head_dim; j++) {
            ss += q_head[j] * q_head[j];
        }
        ss = 1.0f / sqrt((ss / float(head_dim)) + eps);

        for (uint j = 0; j < head_dim; j++) {
            q_head[j] = q_norm_weights[j] * (ss * q_head[j]);
        }

        // RoPE for Q (matches ground truth exactly)
        for (uint j = 0; j < head_dim/2; j++) {
            float freq = pow(1e6f, -float(j) / float(head_dim/2));
            float cos_freq = cos(float(pos) * freq);
            float sin_freq = sin(float(pos) * freq);

            float x = q_head[j];                      // real part
            float y = q_head[j + head_dim/2];         // imag part

            q_head[j] = x * cos_freq - y * sin_freq;              // new real
            q_head[j + head_dim/2] = x * sin_freq + y * cos_freq; // new imag
        }
    }

    // Process K heads: RMSNorm + RoPE for each head
    if (gid < n_kv_heads) {
        uint h = gid;
        device float* k_head = k + h * head_dim;

        // K-RMSNorm (matches ground truth: rmsnorm(k, k, w->k_norm_weights + l * p->head_dim, p->head_dim))
        float ss = 0.0f;
        for (uint j = 0; j < head_dim; j++) {
            ss += k_head[j] * k_head[j];
        }
        ss = 1.0f / sqrt((ss / float(head_dim)) + eps);

        for (uint j = 0; j < head_dim; j++) {
            k_head[j] = k_norm_weights[j] * (ss * k_head[j]);
        }

        // RoPE for K (matches ground truth exactly)
        for (uint j = 0; j < head_dim/2; j++) {
            float freq = pow(1e6f, -float(j) / float(head_dim/2));
            float cos_freq = cos(float(pos) * freq);
            float sin_freq = sin(float(pos) * freq);

            float x = k_head[j];
            float y = k_head[j + head_dim/2];

            k_head[j] = x * cos_freq - y * sin_freq;
            k_head[j + head_dim/2] = x * sin_freq + y * cos_freq;
        }
    }
}