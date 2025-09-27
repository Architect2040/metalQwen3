#pragma once

#include <vector>
#include <string>

struct Transformer;
struct Tokenizer;
struct Sampler;

class Qwen3Original {
public:
    Qwen3Original();
    ~Qwen3Original();

    bool loadModel(const std::string& checkpoint_path, int ctx_length = 0, bool enable_thinking = false);
    void cleanup();

    std::vector<float> forward(int token, int position);
    const float* forwardRaw(int token, int position);
    int sample(const std::vector<float>& logits);
    int sample(const float* logits);

    std::vector<int> encode(const std::string& text);
    std::string decode(int token) const;

    void setSamplingParams(float temperature, float top_p, unsigned long long seed);

    void resetKVCache();

    // Utility functions
    int getVocabSize() const;
    int getSeqLen() const;
    int getDim() const;
    unsigned int getBOSToken() const;
    unsigned int getEOSToken() const;

    Transformer* getTransformer() const { return transformer; }
    Tokenizer* getTokenizer() const { return tokenizer; }
    Sampler* getSampler() const { return sampler; }

private:
    Transformer* transformer;
    Tokenizer* tokenizer;
    Sampler* sampler;
    bool initialized;
};
