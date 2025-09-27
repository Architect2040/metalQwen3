#pragma once

#include <string>
#include <vector>
#include <memory>

// Forward declaration for original qwen3.c structures
struct Qwen3cTransformer_Internal;
struct Qwen3cTokenizer_Internal;
struct Qwen3cSampler_Internal;

/**
 * C++ wrapper for the original qwen3.c inference engine
 * This provides OpenAI-compatible inference using the original CPU implementation
 */
class Qwen3cOriginalEngine {
public:
    Qwen3cOriginalEngine();
    ~Qwen3cOriginalEngine();

    // Initialize the engine with model file
    bool initialize(const std::string& model_path);

    // Generate text using the original qwen3.c engine
    std::string generate(const std::string& prompt,
                        float temperature = 0.7f,
                        float top_p = 0.9f,
                        int max_tokens = 256,
                        int seed = -1);

    // Model information
    int getDim() const;
    int getSeqLen() const;
    int getVocabSize() const;
    std::string getModelInfo() const;

    // Performance metrics
    struct GenerationMetrics {
        double total_time_ms;
        double first_token_time_ms;
        int tokens_generated;
        double tokens_per_second;
    };

    GenerationMetrics getLastGenerationMetrics() const;

private:
    std::unique_ptr<Qwen3cTransformer_Internal> transformer;
    std::unique_ptr<Qwen3cTokenizer_Internal> tokenizer;
    std::unique_ptr<Qwen3cSampler_Internal> sampler;

    bool initialized;
    std::string model_path;
    GenerationMetrics last_metrics;
};