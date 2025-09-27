/**
 * MetalQwen3 - High-Performance Transformer Inference on Apple Silicon
 *
 * @file MetalQwen3.h
 * @brief Metal GPU-accelerated Qwen3 transformer inference engine header
 * @author Shlomo Kashnai
 * @date 2024
 *
 * Built upon Adrian Cable's qwen3.c educational implementation
 * https://github.com/adriancable/qwen3.c
 *
 * This implementation optimizes Qwen3 transformer inference using Apple Metal GPU
 * compute shaders, achieving 73.9 tokens/second average performance with
 * 2.8x speedup over CPU baseline.
 *
 * @license MIT License - See project root for full license text
 */

#pragma once

#include "MetalContext.h"
#include "Qwen3Original.h"

#include <memory>
#include <string>
#include <vector>

// Copy exact state structs from qwen3_original.c (renamed to avoid conflicts)
typedef struct {
    int8_t *q;    // quantized values
    float *s; // scaling factors
} MetalQuantizedTensor;

typedef struct {
    int magic_number; // checkpoint magic number
    int version; // file format version
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
    int head_dim; // head dimension
    int shared_classifier; // 1 if wcls == p_tokens
    int group_size; // quantization group size (export.py uses 64)
} MetalConfig;

typedef struct {
    // token embedding table
    MetalQuantizedTensor *q_tokens; // (vocab_size, dim)
    float *token_embedding_table; // same, but dequantized
    // weights for rmsnorms
    float *rms_att_weight; // (layer, dim) rmsnorm weights
    float *rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    MetalQuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    MetalQuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    MetalQuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    MetalQuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // QK-RMSNorm for Qwen3
    float *q_norm_weights;
    float *k_norm_weights;
    // weights for ffn
    MetalQuantizedTensor *w1; // (layer, hidden_dim, dim)
    MetalQuantizedTensor *w2; // (layer, dim, hidden_dim)
    MetalQuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float *rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    MetalQuantizedTensor *wcls;
} MetalTransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    MetalQuantizedTensor xq; // quantized x (dim,)
    MetalQuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float *key_cache;   // (layer, seq_len, dim)
    float *value_cache; // (layer, seq_len, dim)
} MetalRunState;

typedef struct {
    MetalConfig config; // the hyperparameters of the architecture (the blueprint)
    MetalTransformerWeights weights; // the weights of the model
    MetalRunState state; // buffers for the "wave" of activations in the forward pass
    float *data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} MetalTransformer;

class MetalQwen3 {
public:
    MetalQwen3();
    ~MetalQwen3();

    bool initialize();
    void cleanup();

    bool loadModel(const std::string& modelPath, int ctxLength = 0);

    const std::vector<float>& forward(const std::vector<int>& tokens);
    const std::vector<float>& forward(int token, int position);

    std::string generateText(const std::string& prompt, int maxTokens = 128);

    void resetState();

    int getVocabSize() const { return metalTransformer.config.vocab_size; }
    int getSeqLen() const { return metalTransformer.config.seq_len; }
    int getDim() const { return metalTransformer.config.dim; }

    unsigned int getBOSToken() const;
    unsigned int getEOSToken() const;

    double getLastInferenceTime() const { return lastInferenceTime; }
    size_t getGPUMemoryUsed() const { return gpuMemoryUsed; }

private:
    std::unique_ptr<MetalContext> metalContext;
    std::unique_ptr<Qwen3Original> cpuReference; // For tokenizer and sampling only

    std::vector<float> latestLogits;

    // Ground truth state from qwen3_original.c
    MetalTransformer metalTransformer;
    int GS = 0; // global group size for quantization

    double lastInferenceTime = 0.0;
    size_t gpuMemoryUsed = 0;

    // Metal implementation of forward pass - using original layer loop but Metal shaders
    float* metal_forward(int token, int pos);

    // Metal shader calls for hotspots
    void metal_rmsnorm(float *o, float *x, float *weight, int size);
    void metal_softmax(float *x, int size);
    void metal_matmul(float *xout, MetalQuantizedTensor *x, MetalQuantizedTensor *w, int n, int d);
    void metal_swiglu(float *hb, float *hb2, int hidden_dim);
    void metal_rope(float *q, float *k, int head_dim, int pos, int n_heads, int n_kv_heads, int layer);
    void metal_attention(float *xb, float *q, float *att, float *key_cache, float *value_cache,
                        int pos, int head_dim, int n_heads, int n_kv_heads, int seq_len, int kv_dim, uint64_t loff, int kv_mul);

    // Helper functions from original
    void dequantize(MetalQuantizedTensor *qx, float *x, int n);
    void quantize(MetalQuantizedTensor *qx, float *x, int n);
    MetalQuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each);
    void memory_map_weights(MetalTransformerWeights *w, MetalConfig *p, void *ptr);
    void read_checkpoint(const char *checkpoint, MetalConfig *config, MetalTransformerWeights* weights, float** data, ssize_t* file_size, int ctx_length);
    void malloc_run_state(MetalRunState* s, MetalConfig *p);
    void free_run_state(MetalRunState* s);
    void free_transformer_weights(MetalTransformerWeights *w);
};
