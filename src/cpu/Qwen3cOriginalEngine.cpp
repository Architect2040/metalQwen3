#include "Qwen3cOriginalEngine.h"
#include <iostream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <sstream>

extern "C" {

// Simplified embedded version of key qwen3.c structures and functions
// In a full implementation, we would compile and link the original runq.c

typedef struct {
    int magic_number;
    int version;
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    int head_dim;
    int shared_classifier;
    int group_size;
} Config_Original;

typedef struct {
    int8_t *q;
    float *s;
} QuantizedTensor_Original;

typedef struct {
    char **vocab;
    float *merge_scores;
    int vocab_size;
    unsigned int max_token_length;
    unsigned int bos_token_id;
    unsigned int eos_token_id;
    char prompt_template[1024];
    char system_prompt_template[1024];
} Tokenizer_Original;

typedef struct {
    float *probabilities;
    int vocab_size;
    unsigned long long rng_state;
    float temperature;
    float topp;
} Sampler_Original;

typedef struct {
    QuantizedTensor_Original *q_tokens;
    float *token_embedding_table;
    float *rms_att_weight;
    float *rms_ffn_weight;
    // ... other weights (simplified for demo)
} TransformerWeights_Original;

typedef struct {
    float *x;
    float *xb;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
} RunState_Original;

typedef struct {
    Config_Original config;
    TransformerWeights_Original weights;
    RunState_Original state;
    float *data;
    size_t file_size;
} Transformer_Original;

// Stub implementations for demo purposes
// In real implementation, these would be the actual qwen3.c functions

int read_checkpoint_stub(char *checkpoint, Config_Original *config, TransformerWeights_Original *weights, float **data, size_t *file_size, int ctx_length) {
    // This is a stub - in real implementation would read the actual model file
    std::cout << "Reading checkpoint: " << checkpoint << std::endl;

    // Set some reasonable defaults for demo
    config->dim = 2048;
    config->hidden_dim = 5632;
    config->n_layers = 24;
    config->n_heads = 16;
    config->n_kv_heads = 16;
    config->vocab_size = 32768;
    config->seq_len = ctx_length > 0 ? ctx_length : 2048;
    config->head_dim = config->dim / config->n_heads;
    config->shared_classifier = 1;
    config->group_size = 64;

    // In real implementation, would allocate and load actual weights
    *data = nullptr;
    *file_size = 0;

    // For demo, just allocate minimal structures
    weights->q_tokens = (QuantizedTensor_Original*)calloc(1, sizeof(QuantizedTensor_Original));
    weights->token_embedding_table = (float*)calloc(config->vocab_size * config->dim, sizeof(float));

    return 1; // success
}

void build_transformer_stub(Transformer_Original *t, char *checkpoint_path, int ctx_length) {
    if (read_checkpoint_stub(checkpoint_path, &t->config, &t->weights, &t->data, &t->file_size, ctx_length)) {
        // Allocate minimal run state for demo
        t->state.x = (float*)calloc(t->config.dim, sizeof(float));
        t->state.xb = (float*)calloc(t->config.dim, sizeof(float));
        t->state.logits = (float*)calloc(t->config.vocab_size, sizeof(float));
        t->state.key_cache = (float*)calloc(t->config.n_layers * t->config.seq_len * t->config.dim, sizeof(float));
        t->state.value_cache = (float*)calloc(t->config.n_layers * t->config.seq_len * t->config.dim, sizeof(float));
    }
}

void build_tokenizer_stub(Tokenizer_Original *t, char *checkpoint_path, int vocab_size, int enable_thinking) {
    t->vocab_size = vocab_size;
    t->max_token_length = 64;
    t->bos_token_id = 1;
    t->eos_token_id = 2;

    // Allocate minimal vocab for demo
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->merge_scores = (float*)malloc(vocab_size * sizeof(float));

    for (int i = 0; i < vocab_size; i++) {
        t->vocab[i] = (char*)malloc(32);
        snprintf(t->vocab[i], 32, "token_%d", i);
        t->merge_scores[i] = 0.0f;
    }

    // Set basic tokens
    if (vocab_size > 0) strcpy(t->vocab[0], "<pad>");
    if (vocab_size > 1) strcpy(t->vocab[1], "<bos>");
    if (vocab_size > 2) strcpy(t->vocab[2], "<eos>");
    if (vocab_size > 3) strcpy(t->vocab[3], "<unk>");

    // Set default templates
    strcpy(t->prompt_template, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n");
    strcpy(t->system_prompt_template, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n");
}

void build_sampler_stub(Sampler_Original *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed > 0 ? rng_seed : (unsigned long long)time(NULL);
    sampler->probabilities = (float*)malloc(vocab_size * sizeof(float));
}

float* forward_stub(Transformer_Original *transformer, int token, int pos) {
    // This is a stub - in real implementation would do actual forward pass
    // For demo, just return random-ish logits
    for (int i = 0; i < transformer->config.vocab_size; i++) {
        transformer->state.logits[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    return transformer->state.logits;
}

int sample_stub(Sampler_Original *sampler, float *logits) {
    // Simple sampling - in real implementation would use proper temperature/top-p
    int best_token = 0;
    float best_score = logits[0];

    for (int i = 1; i < sampler->vocab_size; i++) {
        if (logits[i] > best_score) {
            best_score = logits[i];
            best_token = i;
        }
    }

    // Add some randomness
    if ((rand() % 100) < 30) { // 30% chance to pick a different token
        best_token = rand() % sampler->vocab_size;
    }

    return best_token;
}

void encode_stub(Tokenizer_Original *t, char *text, int *tokens, int *n_tokens) {
    // Simple word tokenization for demo
    *n_tokens = 0;
    if (!text || strlen(text) == 0) return;

    char *text_copy = strdup(text);
    char *word = strtok(text_copy, " \t\n\r");

    while (word && *n_tokens < 1024) {
        // For demo, hash the word to get a token ID
        int token_id = abs((int)(word[0] + strlen(word))) % t->vocab_size;
        if (token_id < 4) token_id = 4; // avoid special tokens
        tokens[(*n_tokens)++] = token_id;
        word = strtok(NULL, " \t\n\r");
    }

    free(text_copy);
}

char* decode_stub(Tokenizer_Original *t, int token) {
    if (token >= 0 && token < t->vocab_size) {
        return t->vocab[token];
    }
    return t->vocab[3]; // <unk>
}

void free_transformer_stub(Transformer_Original *t) {
    if (t->weights.q_tokens) free(t->weights.q_tokens);
    if (t->weights.token_embedding_table) free(t->weights.token_embedding_table);
    if (t->state.x) free(t->state.x);
    if (t->state.xb) free(t->state.xb);
    if (t->state.logits) free(t->state.logits);
    if (t->state.key_cache) free(t->state.key_cache);
    if (t->state.value_cache) free(t->state.value_cache);
}

void free_tokenizer_stub(Tokenizer_Original *t) {
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) {
            if (t->vocab[i]) free(t->vocab[i]);
        }
        free(t->vocab);
    }
    if (t->merge_scores) free(t->merge_scores);
}

void free_sampler_stub(Sampler_Original *s) {
    if (s->probabilities) free(s->probabilities);
}

} // extern "C"

// Internal implementation structures
struct Qwen3cTransformer_Internal {
    Transformer_Original transformer;
};

struct Qwen3cTokenizer_Internal {
    Tokenizer_Original tokenizer;
};

struct Qwen3cSampler_Internal {
    Sampler_Original sampler;
};

Qwen3cOriginalEngine::Qwen3cOriginalEngine() : initialized(false) {
    transformer = std::make_unique<Qwen3cTransformer_Internal>();
    tokenizer = std::make_unique<Qwen3cTokenizer_Internal>();
    sampler = std::make_unique<Qwen3cSampler_Internal>();

    memset(&transformer->transformer, 0, sizeof(transformer->transformer));
    memset(&tokenizer->tokenizer, 0, sizeof(tokenizer->tokenizer));
    memset(&sampler->sampler, 0, sizeof(sampler->sampler));
    memset(&last_metrics, 0, sizeof(last_metrics));
}

Qwen3cOriginalEngine::~Qwen3cOriginalEngine() {
    if (initialized) {
        free_transformer_stub(&transformer->transformer);
        free_tokenizer_stub(&tokenizer->tokenizer);
        free_sampler_stub(&sampler->sampler);
    }
}

bool Qwen3cOriginalEngine::initialize(const std::string& model_path) {
    this->model_path = model_path;

    std::cout << "Initializing Qwen3c Original Engine..." << std::endl;
    std::cout << "Model path: " << model_path << std::endl;

    // Build transformer
    char* path_str = const_cast<char*>(model_path.c_str());
    build_transformer_stub(&transformer->transformer, path_str, 2048);

    // Build tokenizer
    build_tokenizer_stub(&tokenizer->tokenizer, path_str,
                        transformer->transformer.config.vocab_size, 0);

    // Build sampler with default parameters
    build_sampler_stub(&sampler->sampler,
                      transformer->transformer.config.vocab_size,
                      0.7f, 0.9f, 12345);

    initialized = true;

    std::cout << "✓ Qwen3c Original Engine initialized successfully" << std::endl;
    std::cout << "  Model dimensions: " << getDim() << std::endl;
    std::cout << "  Vocabulary size: " << getVocabSize() << std::endl;
    std::cout << "  Sequence length: " << getSeqLen() << std::endl;

    return true;
}

std::string Qwen3cOriginalEngine::generate(const std::string& prompt,
                                          float temperature,
                                          float top_p,
                                          int max_tokens,
                                          int seed) {
    if (!initialized) return "";

    auto start_time = std::chrono::high_resolution_clock::now();
    auto first_token_time = start_time;
    bool first_token_recorded = false;

    // Update sampler parameters
    sampler->sampler.temperature = temperature;
    sampler->sampler.topp = top_p;
    if (seed > 0) {
        sampler->sampler.rng_state = seed;
    }

    // Encode prompt
    int prompt_tokens[1024];
    int num_prompt_tokens = 0;
    char* prompt_str = const_cast<char*>(prompt.c_str());
    encode_stub(&tokenizer->tokenizer, prompt_str, prompt_tokens, &num_prompt_tokens);

    if (num_prompt_tokens == 0) {
        std::cerr << "Warning: Empty prompt after tokenization" << std::endl;
        return "";
    }

    std::ostringstream result;
    int token = prompt_tokens[0];
    int tokens_generated = 0;

    // Generation loop (simplified version of original qwen3.c generate function)
    for (int pos = 0; pos < max_tokens && pos + num_prompt_tokens < getSeqLen(); pos++) {
        // Forward pass
        float* logits = forward_stub(&transformer->transformer, token, pos + num_prompt_tokens);

        // Sample next token
        int next = sample_stub(&sampler->sampler, logits);

        if (!first_token_recorded) {
            first_token_time = std::chrono::high_resolution_clock::now();
            first_token_recorded = true;
        }

        // Check for EOS
        if (next == tokenizer->tokenizer.eos_token_id) {
            break;
        }

        // Decode and append
        char* token_str = decode_stub(&tokenizer->tokenizer, next);
        if (token_str) {
            result << token_str;
            if (pos > 0) result << " "; // Add space between tokens for demo
        }

        token = next;
        tokens_generated++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate metrics
    last_metrics.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    last_metrics.first_token_time_ms = first_token_recorded
        ? std::chrono::duration<double, std::milli>(first_token_time - start_time).count()
        : last_metrics.total_time_ms;
    last_metrics.tokens_generated = tokens_generated;

    double generation_time_ms = last_metrics.total_time_ms - last_metrics.first_token_time_ms;
    last_metrics.tokens_per_second = generation_time_ms > 0
        ? (tokens_generated * 1000.0) / generation_time_ms
        : 0.0;

    return result.str();
}

int Qwen3cOriginalEngine::getDim() const {
    return initialized ? transformer->transformer.config.dim : 0;
}

int Qwen3cOriginalEngine::getSeqLen() const {
    return initialized ? transformer->transformer.config.seq_len : 0;
}

int Qwen3cOriginalEngine::getVocabSize() const {
    return initialized ? transformer->transformer.config.vocab_size : 0;
}

std::string Qwen3cOriginalEngine::getModelInfo() const {
    if (!initialized) return "Not initialized";

    std::ostringstream info;
    info << "Qwen3c Original Engine\n";
    info << "Model: " << model_path << "\n";
    info << "Dimensions: " << getDim() << "\n";
    info << "Layers: " << transformer->transformer.config.n_layers << "\n";
    info << "Heads: " << transformer->transformer.config.n_heads << "\n";
    info << "Vocab Size: " << getVocabSize() << "\n";
    info << "Sequence Length: " << getSeqLen();

    return info.str();
}

Qwen3cOriginalEngine::GenerationMetrics Qwen3cOriginalEngine::getLastGenerationMetrics() const {
    return last_metrics;
}