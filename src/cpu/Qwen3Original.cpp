#include "Qwen3Original.h"
#include "qwen3_c_api.h"

#include <iostream>
#include <chrono>
#include <random>
#include <cstring>

Qwen3Original::Qwen3Original()
    : transformer(nullptr)
    , tokenizer(nullptr)
    , sampler(nullptr)
    , initialized(false) {
}

Qwen3Original::~Qwen3Original() {
    cleanup();
}

bool Qwen3Original::loadModel(const std::string& checkpoint_path, int ctx_length, bool enable_thinking) {
    cleanup();

    transformer = qwen3_transformer_create(checkpoint_path.c_str(), ctx_length);
    if (!transformer) {
        std::cerr << "Failed to create Qwen3 transformer" << std::endl;
        return false;
    }

    tokenizer = qwen3_tokenizer_create(checkpoint_path.c_str(), qwen3_get_vocab_size(transformer), enable_thinking);
    if (!tokenizer) {
        std::cerr << "Failed to create Qwen3 tokenizer" << std::endl;
        cleanup();
        return false;
    }

    // Use deterministic seed for reproducibility per load
    sampler = qwen3_sampler_create(qwen3_get_vocab_size(transformer), 1.0f, 0.9f, 42);
    if (!sampler) {
        std::cerr << "Failed to create Qwen3 sampler" << std::endl;
        cleanup();
        return false;
    }

    initialized = true;
    return true;
}

void Qwen3Original::cleanup() {
    if (sampler) {
        qwen3_sampler_free(sampler);
        sampler = nullptr;
    }
    if (tokenizer) {
        qwen3_tokenizer_free(tokenizer);
        tokenizer = nullptr;
    }
    if (transformer) {
        qwen3_transformer_free(transformer);
        transformer = nullptr;
    }
    initialized = false;
}

std::vector<float> Qwen3Original::forward(int token, int position) {
    if (!initialized || !transformer) {
        return {};
    }

    float* logits = qwen3_forward(transformer, token, position);
    if (!logits) {
        return {};
    }

    int vocab = qwen3_get_vocab_size(transformer);
    return std::vector<float>(logits, logits + vocab);
}

const float* Qwen3Original::forwardRaw(int token, int position) {
    if (!initialized || !transformer) {
        return nullptr;
    }
    return qwen3_forward(transformer, token, position);
}

int Qwen3Original::sample(const std::vector<float>& logits) {
    if (!sampler || logits.empty()) {
        return -1;
    }

    // Create a modifiable copy for sampling
    std::vector<float> temp = logits;
    return qwen3_sample_token(sampler, temp.data());
}

int Qwen3Original::sample(const float* logits) {
    if (!sampler || !logits) {
        return -1;
    }
    // Create a temporary buffer because qwen3 sampler modifies logits in-place
    static thread_local std::vector<float> temp;
    int vocab = getVocabSize();
    temp.assign(logits, logits + vocab);
    return qwen3_sample_token(sampler, temp.data());
}

void Qwen3Original::setSamplingParams(float temperature, float top_p, unsigned long long seed) {
    if (!sampler) {
        return;
    }
    qwen3_sampler_set_params(sampler, temperature, top_p);
    if (seed == 0) {
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        seed = static_cast<unsigned long long>(now);
    }
    qwen3_sampler_set_seed(sampler, seed);
}

std::vector<int> Qwen3Original::encode(const std::string& text) {
    std::vector<int> tokens;
    if (!tokenizer || text.empty()) {
        return tokens;
    }

    int max_tokens = qwen3_get_seq_len(transformer);
    tokens.resize(max_tokens);
    int encoded = qwen3_encode_text(tokenizer, text.c_str(), tokens.data(), max_tokens);
    tokens.resize(encoded);
    return tokens;
}

std::string Qwen3Original::decode(int token) const {
    if (!tokenizer) {
        return {};
    }
    const char* text = qwen3_decode_token(tokenizer, token);
    return text ? std::string(text) : std::string();
}

void Qwen3Original::resetKVCache() {
    if (transformer) {
        qwen3_reset_kv_cache(transformer);
    }
}

int Qwen3Original::getVocabSize() const {
    return transformer ? qwen3_get_vocab_size(transformer) : 0;
}

int Qwen3Original::getSeqLen() const {
    return transformer ? qwen3_get_seq_len(transformer) : 0;
}

int Qwen3Original::getDim() const {
    return transformer ? qwen3_get_dim(transformer) : 0;
}

unsigned int Qwen3Original::getBOSToken() const {
    return tokenizer ? qwen3_get_bos_token(tokenizer) : 0;
}

unsigned int Qwen3Original::getEOSToken() const {
    return tokenizer ? qwen3_get_eos_token(tokenizer) : 0;
}
