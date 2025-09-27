#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Transformer Transformer;
typedef struct Tokenizer Tokenizer;
typedef struct Sampler Sampler;

Transformer* qwen3_transformer_create(const char* checkpoint_path, int ctx_length);
void qwen3_transformer_free(Transformer* transformer);

Tokenizer* qwen3_tokenizer_create(const char* checkpoint_path, int vocab_size, int enable_thinking);
void qwen3_tokenizer_free(Tokenizer* tokenizer);

Sampler* qwen3_sampler_create(int vocab_size, float temperature, float topp, unsigned long long seed);
void qwen3_sampler_free(Sampler* sampler);
void qwen3_sampler_set_params(Sampler* sampler, float temperature, float topp);
void qwen3_sampler_set_seed(Sampler* sampler, unsigned long long seed);

float* qwen3_forward(Transformer* transformer, int token, int position);
int qwen3_sample_token(Sampler* sampler, float* logits);

int qwen3_encode_text(Tokenizer* tokenizer, const char* text, int* tokens, int max_tokens);
const char* qwen3_decode_token(Tokenizer* tokenizer, int token);

unsigned int qwen3_get_bos_token(const Tokenizer* tokenizer);
unsigned int qwen3_get_eos_token(const Tokenizer* tokenizer);

int qwen3_get_vocab_size(const Transformer* transformer);
int qwen3_get_seq_len(const Transformer* transformer);
int qwen3_get_dim(const Transformer* transformer);

void qwen3_reset_kv_cache(Transformer* transformer);

#ifdef __cplusplus
}
#endif
