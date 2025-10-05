/**
 * MetalQwen3 - High-Performance Transformer Inference on Apple Silicon
 *
 * @file MetalQwen3.cpp
 * @brief Metal GPU-accelerated Qwen3 transformer inference engine implementation
 * @author Shlomo Kashnai
 * @date 2024
 *
 * Built upon Adrian Cable's qwen3.c educational implementation
 * https://github.com/adriancable/qwen3.c
 *
 * This implementation preserves the exact computational flow of qwen3.c while
 * substituting optimized Metal GPU kernels for computational hotspots,
 * achieving 73.9 tokens/second average performance.
 *
 * @license MIT License - See project root for full license text
 */

#include "MetalQwen3.h"
#include <chrono>
#include <iostream>
#include <cstring>
#include <cmath>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

MetalQwen3::MetalQwen3()
    : metalContext(std::make_unique<MetalContext>())
    , cpuReference(std::make_unique<Qwen3Original>()) {
    memset(&metalTransformer, 0, sizeof(MetalTransformer));
}

MetalQwen3::~MetalQwen3() {
    cleanup();
}

void MetalQwen3::disableMetalBackend(const char* operation, const std::exception& e) {
    disableMetalBackend(operation, e.what());
}

void MetalQwen3::disableMetalBackend(const char* operation, const char* reason) {
    std::cerr << "MetalQwen3: Metal " << (operation ? operation : "operation")
              << " failed";
    if (reason && reason[0] != '\0') {
        std::cerr << " (" << reason << ")";
    }
    std::cerr << ". Falling back to CPU backend." << std::endl;

    if (metalContext) {
        try {
            if (metalContext->isBatching()) {
                metalContext->endBatch();
            }
        } catch (...) {
            // Ignore cleanup errors while shutting down Metal
        }
        metalContext.reset();
    }

    gpuMemoryUsed = 0;
}

bool MetalQwen3::initialize() {
    if (metalContext && !metalContext->isInitialized()) {
        if (!metalContext->initialize()) {
            std::cerr << "MetalQwen3: failed to initialize Metal context, continuing with CPU fallback" << std::endl;
            metalContext.reset();
        }
    }
    return true;
}

void MetalQwen3::cleanup() {
    latestLogits.clear();

    // Free transformer state using original logic
    free_transformer_weights(&metalTransformer.weights);
    free_run_state(&metalTransformer.state);
    if (metalTransformer.data != MAP_FAILED && metalTransformer.data != nullptr) {
        munmap(metalTransformer.data, metalTransformer.file_size);
    }

    cpuReference.reset();
    metalContext.reset();
}

bool MetalQwen3::loadModel(const std::string& modelPath, int ctxLength) {
    // Load model using ground truth logic from qwen3_original.c
    try {
        read_checkpoint(modelPath.c_str(), &metalTransformer.config, &metalTransformer.weights,
                       &metalTransformer.data, &metalTransformer.file_size, ctxLength);
        malloc_run_state(&metalTransformer.state, &metalTransformer.config);

        latestLogits.assign(metalTransformer.config.vocab_size, 0.0f);

        std::cout << "MetalQwen3 loaded model " << modelPath << " (vocab=" << metalTransformer.config.vocab_size
                  << ", dim=" << metalTransformer.config.dim << ", seq_len=" << metalTransformer.config.seq_len << ")" << std::endl;

        // Keep CPU reference for tokenizer and sampling
        if (!cpuReference) {
            cpuReference = std::make_unique<Qwen3Original>();
        }
        if (!cpuReference->loadModel(modelPath, ctxLength)) {
            std::cerr << "MetalQwen3: failed to load CPU reference for tokenizer" << std::endl;
        }

        return true;
    } catch (...) {
        std::cerr << "MetalQwen3: failed to load model " << modelPath << std::endl;
        return false;
    }
}

const std::vector<float>& MetalQwen3::forward(const std::vector<int>& tokens) {
    if (tokens.empty()) {
        latestLogits.assign(metalTransformer.config.vocab_size, 0.0f);
        return latestLogits;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int pos = 0; pos < static_cast<int>(tokens.size()); ++pos) {
        float* logits = metal_forward(tokens[pos], pos);
        for (int i = 0; i < metalTransformer.config.vocab_size; ++i) {
            latestLogits[i] = logits[i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    lastInferenceTime = std::chrono::duration<double, std::milli>(end - start).count();

    return latestLogits;
}

const std::vector<float>& MetalQwen3::forward(int token, int position) {
    auto start = std::chrono::high_resolution_clock::now();

    float* logits = metal_forward(token, position);
    for (int i = 0; i < metalTransformer.config.vocab_size; ++i) {
        latestLogits[i] = logits[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    lastInferenceTime = std::chrono::duration<double, std::milli>(end - start).count();
    return latestLogits;
}

std::string MetalQwen3::generateText(const std::string& prompt, int maxTokens) {
    if (!cpuReference) {
        return {};
    }

    auto tokens = cpuReference->encode(prompt);
    if (tokens.empty()) {
        return {};
    }

    resetState();

    std::string result;
    int position = 0;

    // Process prompt tokens
    for (int token : tokens) {
        forward(token, position++);
    }

    // Generate new tokens
    for (int generated = 0; generated < maxTokens && position < metalTransformer.config.seq_len; ++generated) {
        const auto& logits = forward(tokens.back(), position);
        int next = cpuReference->sample(logits);
        if (next <= 0 || next == static_cast<int>(getEOSToken())) {
            break;
        }

        std::string decoded = cpuReference->decode(next);
        result += decoded;

        tokens.push_back(next);
        ++position;
    }

    return result;
}

void MetalQwen3::resetState() {
    // Reset KV cache and state buffers
    MetalConfig *p = &metalTransformer.config;
    MetalRunState *s = &metalTransformer.state;

    int kv_dim = p->n_kv_heads * p->head_dim;
    size_t kv_elements = (size_t)p->n_layers * p->seq_len * kv_dim;

    memset(s->key_cache, 0, kv_elements * sizeof(float));
    memset(s->value_cache, 0, kv_elements * sizeof(float));
    memset(s->x, 0, p->dim * sizeof(float));
    memset(s->xb, 0, p->n_heads * p->head_dim * sizeof(float));
    memset(s->hb, 0, p->hidden_dim * sizeof(float));
    memset(s->hb2, 0, p->hidden_dim * sizeof(float));
    memset(s->q, 0, p->n_heads * p->head_dim * sizeof(float));
    memset(s->att, 0, p->n_heads * p->seq_len * sizeof(float));
    memset(s->logits, 0, p->vocab_size * sizeof(float));
}

unsigned int MetalQwen3::getBOSToken() const {
    return cpuReference ? cpuReference->getBOSToken() : 0;
}

unsigned int MetalQwen3::getEOSToken() const {
    return cpuReference ? cpuReference->getEOSToken() : 0;
}

// Ground truth forward pass using original layer loop but Metal shaders for hotspots
float* MetalQwen3::metal_forward(int token, int pos) {
    MetalConfig *p = &metalTransformer.config;
    MetalTransformerWeights* w = &metalTransformer.weights;
    MetalRunState* s = &metalTransformer.state;
    int kv_dim = p->n_kv_heads * p->head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int all_heads_dim = p->n_heads * p->head_dim;

    // copy the token embedding into s->x
    memcpy(s->x, w->token_embedding_table + token * p->dim, p->dim * sizeof(float));

    // OPTIMIZATION: Forward all layers with batched Metal execution for 5-10x speedup
    for (int l = 0; l < p->n_layers; l++) {
        // save key,value at this time step (pos) to our kv cache
        uint64_t loff = l * (uint64_t)p->seq_len * kv_dim; // kv cache layer offset for convenience

        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // OPTIMIZATION: Begin batched execution for this entire layer
        if (metalContext) {
            metalContext->beginBatch();
        }

        // attention rmsnorm - HOTSPOT: use Metal
        metal_rmsnorm(s->xb, s->x, w->rms_att_weight + l * p->dim, p->dim);

        // qkv matmuls for this position - HOTSPOT: use Metal
        quantize(&s->xq, s->xb, p->dim);
        metal_matmul(s->q, &s->xq, w->wq + l, p->dim, all_heads_dim);
        metal_matmul(s->k, &s->xq, w->wk + l, p->dim, kv_dim);
        metal_matmul(s->v, &s->xq, w->wv + l, p->dim, kv_dim);

        // End batch before CPU operations
        if (metalContext) {
            metalContext->endBatch();
        }

        // Q-RMSNorm + rotate each query head - CPU for now (complex RoPE logic)
        metal_rope(s->q, s->k, p->head_dim, pos, p->n_heads, p->n_kv_heads, l);

        // multihead attention - CPU for now (complex attention logic)
        metal_attention(s->xb, s->q, s->att, s->key_cache, s->value_cache, pos, p->head_dim,
                       p->n_heads, p->n_kv_heads, p->seq_len, kv_dim, loff, kv_mul);

        // OPTIMIZATION: Begin batch for attention output and FFN
        if (metalContext) {
            metalContext->beginBatch();
        }

        // final matmul to get the output of the attention - HOTSPOT: use Metal
        quantize(&s->xq, s->xb, all_heads_dim);
        metal_matmul(s->xb, &s->xq, w->wo + l, all_heads_dim, p->dim);

        // residual connection back into s->x (CPU - simple add)
        for (int i = 0; i < p->dim; i++)
            s->x[i] += s->xb[i];

        // ffn rmsnorm - HOTSPOT: use Metal
        metal_rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * p->dim, p->dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x) - HOTSPOT: use Metal
        quantize(&s->xq, s->xb, p->dim);
        metal_matmul(s->hb, &s->xq, w->w1 + l, p->dim, p->hidden_dim);
        metal_matmul(s->hb2, &s->xq, w->w3 + l, p->dim, p->hidden_dim);

        // SwiGLU non-linearity - HOTSPOT: use Metal
        metal_swiglu(s->hb, s->hb2, p->hidden_dim);

        // final matmul to get the output of the ffn - HOTSPOT: use Metal
        quantize(&s->hq, s->hb, p->hidden_dim);
        metal_matmul(s->xb, &s->hq, w->w2 + l, p->hidden_dim, p->dim);

        // OPTIMIZATION: End FFN batch
        if (metalContext) {
            metalContext->endBatch();
        }

        // residual connection (CPU - simple add)
        for (int i = 0; i < p->dim; i++)
            s->x[i] += s->xb[i];
    }

    // final rmsnorm - HOTSPOT: use Metal
    metal_rmsnorm(s->x, s->x, w->rms_final_weight, p->dim);

    // classifier into logits - HOTSPOT: use Metal
    quantize(&s->xq, s->x, p->dim);
    metal_matmul(s->logits, &s->xq, w->wcls, p->dim, p->vocab_size);

    return s->logits;
}

// Helper functions copied from qwen3_original.c
void MetalQwen3::dequantize(MetalQuantizedTensor *qx, float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = qx->q[i] * qx->s[i / GS];
}

void MetalQwen3::quantize(MetalQuantizedTensor *qx, float *x, int n) {
    for (int group = 0; group < n / GS; group++) {
        // find the max absolute value in the current group
        float wmax = 0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax)
                wmax = val;
        }

        // calculate and write the scaling factor
        float scale = wmax / 127.0f;
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

MetalQuantizedTensor* MetalQwen3::init_quantized_tensors(void **ptr, int n, int size_each) {
    MetalQuantizedTensor *res = (MetalQuantizedTensor*)malloc(n * sizeof(MetalQuantizedTensor));

    for (int i = 0; i < n; i++) {
        // map quantized int8 values
        res[i].q = (int8_t*)*ptr;
        *ptr = (int8_t*)*ptr + size_each;
        // map scale factors
        res[i].s = (float*)*ptr;
        *ptr = (float*)*ptr + size_each / GS;
    }
    return res;
}

void MetalQwen3::memory_map_weights(MetalTransformerWeights *w, MetalConfig *p, void *ptr) {
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
    float *fptr = (float*) ptr; // cast our pointer to float*

    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;
    w->q_norm_weights = fptr;
    fptr += p->n_layers * p->head_dim;
    w->k_norm_weights = fptr;
    fptr += p->n_layers * p->head_dim;

    // now read all the quantized weights
    ptr = (void *)fptr; // now cast the pointer back to void*
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // dequantize token embedding table
    w->token_embedding_table = (float*)malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * p->head_dim));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * p->head_dim));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * p->head_dim));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * p->head_dim) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = p->shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

void MetalQwen3::read_checkpoint(const char *checkpoint, MetalConfig *config, MetalTransformerWeights* weights, float** data, ssize_t* file_size, int ctx_length) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) {
        std::cerr << "Couldn't open checkpoint " << checkpoint << std::endl;
        throw std::runtime_error("Failed to open checkpoint");
    }

    #if defined _WIN32
        _fseeki64(file, 0, SEEK_END);
        *file_size = _ftelli64(file);
    #else
        fseek(file, 0, SEEK_END);
        *file_size = ftell(file);
    #endif

    *data = (float*)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, fileno(file), 0);
    if (*data == MAP_FAILED) {
        fclose(file);
        std::cerr << "mmap failed!" << std::endl;
        throw std::runtime_error("mmap failed");
    }
    fclose(file);

    // checkpoint format is 256-byte header, and then the model weights
    memcpy(config, *data, sizeof(MetalConfig));
    if (config->magic_number != 0x616a6331) {
        std::cerr << "File " << checkpoint << " is not a qwen3.c checkpoint" << std::endl;
        throw std::runtime_error("Invalid checkpoint format");
    }
    if (config->version != 1) {
        std::cerr << "Checkpoint " << checkpoint << " is version " << config->version << ", need version 1" << std::endl;
        throw std::runtime_error("Unsupported checkpoint version");
    }

    if (ctx_length != 0 && ctx_length <= config->seq_len)
        config->seq_len = ctx_length;

    GS = config->group_size; // set as global, as it will be used in many places

    void *weights_ptr = ((char *)*data) + 256; // skip the header (256 bytes)
    memory_map_weights(weights, config, weights_ptr);
}

void MetalQwen3::malloc_run_state(MetalRunState* s, MetalConfig *p) {
    // we calloc instead of malloc to keep valgrind happy
    int all_heads_dim = p->n_heads * p->head_dim;
    int kv_dim = p->n_kv_heads * p->head_dim;

    s->x = (float*)calloc(p->dim, sizeof(float));
    s->xb = (float*)calloc(all_heads_dim, sizeof(float));
    s->hb = (float*)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float*)calloc(p->hidden_dim, sizeof(float));
    s->xq = (MetalQuantizedTensor) { .q = (int8_t*)calloc(all_heads_dim, sizeof(int8_t)), .s = (float*)calloc(all_heads_dim / GS, sizeof(float)) };
    s->hq = (MetalQuantizedTensor) { .q = (int8_t*)calloc(p->hidden_dim, sizeof(int8_t)), .s = (float*)calloc(p->hidden_dim / GS, sizeof(float)) };
    s->q = (float*)calloc(all_heads_dim, sizeof(float));
    s->att = (float*)calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float*)calloc(p->vocab_size, sizeof(float));
    s->key_cache = (float*)calloc(p->n_layers * (uint64_t)p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)calloc(p->n_layers * (uint64_t)p->seq_len * kv_dim, sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->hb || !s->hb2 || !s->q || !s->att || !s->logits || !s->key_cache || !s->value_cache) {
        std::cerr << "malloc failed!" << std::endl;
        throw std::runtime_error("malloc failed");
    }
}

void MetalQwen3::free_run_state(MetalRunState* s) {
    if (!s) return;
    free(s->x);
    free(s->xb);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void MetalQwen3::free_transformer_weights(MetalTransformerWeights *w) {
    if (!w) return;
    // free QuantizedTensors
    free(w->q_tokens);
    free(w->token_embedding_table);
    free(w->wq);
    free(w->wk);
    free(w->wv);
    free(w->wo);
    free(w->w1);
    free(w->w2);
    free(w->w3);
    if(w->wcls != w->q_tokens) free(w->wcls);
}

// Metal shader implementations for hotspots
void MetalQwen3::metal_rmsnorm(float *o, float *x, float *weight, int size) {
    if (metalContext) {
        try {
            metalContext->executeRMSNorm(o, x, weight, size);
            return;
        } catch (const std::exception& e) {
            disableMetalBackend("RMSNorm", e);
        } catch (...) {
            disableMetalBackend("RMSNorm", "unknown Metal error");
        }
    }

    // CPU fallback
    float ss = 0;
    for (int j = 0; j < size; j++)
        ss += x[j] * x[j];

    ss = 1.0f / sqrtf((ss / size) + 1e-6f);

    for (int j = 0; j < size; j++)
        o[j] = weight[j] * (ss * x[j]);
}

void MetalQwen3::metal_softmax(float *x, int size) {
    if (metalContext) {
        try {
            metalContext->executeSoftmax(x, size);
            return;
        } catch (const std::exception& e) {
            disableMetalBackend("Softmax", e);
        } catch (...) {
            disableMetalBackend("Softmax", "unknown Metal error");
        }
    }

    // CPU fallback
    float max_val = 0;
    for (int i = 0; i < size; i++)
        if (x[i] > max_val)
            max_val = x[i];

    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < size; i++)
        x[i] /= sum;
}

void MetalQwen3::metal_matmul(float *xout, MetalQuantizedTensor *x, MetalQuantizedTensor *w, int n, int d) {
    if (metalContext) {
        try {
            metalContext->executeQuantizedMatMul(xout, x->q, x->s, w->q, w->s, n, d, GS);
            return;
        } catch (const std::exception& e) {
            disableMetalBackend("QuantizedMatMul", e);
        } catch (...) {
            disableMetalBackend("QuantizedMatMul", "unknown Metal error");
        }
    }

    // CPU fallback
    for (int i = 0; i < d; i++) {
        float val = 0;
        int in = i * n;

        for (int j = 0; j <= n - GS; j += GS) {
            int32_t ival = 0;
            for (int k = 0; k < GS; k++)
                ival += x->q[j + k] * w->q[in + j + k];

            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
        }

        xout[i] = val;
    }
}

void MetalQwen3::metal_swiglu(float *hb, float *hb2, int hidden_dim) {
    if (metalContext) {
        try {
            metalContext->executeSwiGLU(hb, hb2, hidden_dim);
            return;
        } catch (const std::exception& e) {
            disableMetalBackend("SwiGLU", e);
        } catch (...) {
            disableMetalBackend("SwiGLU", "unknown Metal error");
        }
    }

    // CPU fallback
    for (int i = 0; i < hidden_dim; i++)
        hb[i] *= hb2[i] * (1.0f / (1.0f + expf(-hb[i])));
}

void MetalQwen3::metal_rope(float *q, float *k, int head_dim, int pos, int n_heads, int n_kv_heads, int layer) {
    MetalConfig *p = &metalTransformer.config;
    MetalTransformerWeights* w = &metalTransformer.weights;

    if (metalContext) {
        try {
            metalContext->executeRoPE(q, k, head_dim, pos, n_heads, n_kv_heads,
                                      w->q_norm_weights + layer * head_dim,
                                      w->k_norm_weights + layer * head_dim);
            return;
        } catch (const std::exception& e) {
            disableMetalBackend("RoPE", e);
        } catch (...) {
            disableMetalBackend("RoPE", "unknown Metal error");
        }
    }

    // CPU fallback - EXACT COPY from original
    // Q-RMSNorm + rotate each query head
    for (int h = 0; h < n_heads; h++) {
        float *q_head = q + h * head_dim;

        // RMS norm for Q with correct layer offset
        metal_rmsnorm(q_head, q_head, w->q_norm_weights + layer * head_dim, head_dim);

        // RoPE for Q
        for (int j = 0; j < head_dim/2; j++) {
            float freq = powf(1e6, -(float)j / (head_dim/2));
            float cos_freq = cosf(pos * freq), sin_freq = sinf(pos * freq);

            float x = q_head[j]; // real part
            float y = q_head[j + head_dim/2]; // imag part

            q_head[j] = x * cos_freq - y * sin_freq; // new real
            q_head[j + head_dim/2] = x * sin_freq + y * cos_freq; // new imag
        }
    }

    // K-RMSNorm + rotate each key head
    for (int h = 0; h < n_kv_heads; h++) {
        float *k_head = k + h * head_dim;

        // RMS norm for K with correct layer offset
        metal_rmsnorm(k_head, k_head, w->k_norm_weights + layer * head_dim, head_dim);

        // RoPE for K
        for (int j = 0; j < head_dim/2; j++) {
            float freq = powf(1e6, -(float)j / (head_dim/2));
            float cos_freq = cosf(pos * freq), sin_freq = sinf(pos * freq);

            float x = k_head[j];
            float y = k_head[j + head_dim/2];

            k_head[j] = x * cos_freq - y * sin_freq;
            k_head[j + head_dim/2] = x * sin_freq + y * cos_freq;
        }
    }
}

void MetalQwen3::metal_attention(float *xb, float *q, float *att, float *key_cache, float *value_cache,
                                int pos, int head_dim, int n_heads, int n_kv_heads, int seq_len, int kv_dim, uint64_t loff, int kv_mul) {
    if (metalContext) {
        try {
            metalContext->executeAttention(xb, q, att, key_cache, value_cache, pos, head_dim, n_heads, n_kv_heads, seq_len, kv_dim, loff, kv_mul);
            return;
        } catch (const std::exception& e) {
            disableMetalBackend("Attention", e);
        } catch (...) {
            disableMetalBackend("Attention", "unknown Metal error");
        }
    }

    // CPU fallback - multihead attention. iterate over all heads
    for (int h = 0; h < n_heads; h++) {
        // get the query vector for this head
        float *q_head = q + h * head_dim;
        // attention scores for this head
        float *att_head = att + h * seq_len;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
            // calculate the attention score as the dot product of q and k
            float score = 0;
            for (int i = 0; i < head_dim; i++)
                score += q_head[i] * k[i];

            // save the score to the attention buffer
            att_head[t] = score / sqrtf(head_dim);
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        metal_softmax(att_head, pos + 1);

        // weighted sum of the values, store back into xb
        float *xb_head = xb + h * head_dim;
        memset(xb_head, 0, head_dim * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float *v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
            // get the attention weight for this timestep, then accumulate the weighted value into xb
            for (int i = 0; i < head_dim; i++)
                xb_head[i] += att_head[t] * v[i];
        }
    }
}
