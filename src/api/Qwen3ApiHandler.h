/**
 * MetalQwen3 - High-Performance Transformer Inference on Apple Silicon
 *
 * @file Qwen3ApiHandler.h
 * @brief OpenAI-compatible API handler for MetalQwen3 inference engine
 * @author Shlomo Kashnai
 * @date 2024
 *
 * Implements OpenAI Chat Completions API compatible interface for MetalQwen3
 * transformer inference with streaming support and standardized metrics.
 *
 * Built upon Adrian Cable's qwen3.c educational implementation
 * https://github.com/adriancable/qwen3.c
 *
 * @license MIT License - See project root for full license text
 */

#pragma once

#include "MetalQwen3.h"
#include "Qwen3Original.h"
#include "Qwen3Tokenizer.h"
#include <nlohmann/json.hpp>
#include <httplib.h>
#include <memory>
#include <chrono>
#include <atomic>

struct ChatMessage {
    std::string role;
    std::string content;
};

struct CompletionRequest {
    std::vector<ChatMessage> messages;
    std::string model = "qwen3-metal";
    float temperature = 0.7f;
    int max_tokens = 2048;
    bool stream = false;
    std::vector<std::string> stop;
    float top_p = 0.9f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    std::string user = "";
    unsigned long long seed = 0;
    bool has_seed = false;
};

struct TextCompletionRequest {
    std::string prompt;
    std::string model = "qwen3-metal";
    float temperature = 0.7f;
    int max_tokens = 2048;
    bool stream = false;
    std::vector<std::string> stop;
    float top_p = 0.9f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    std::string user = "";
    unsigned long long seed = 0;
    bool has_seed = false;
    std::string suffix = "";
    bool echo = false;
    int best_of = 1;
    int n = 1;
};

struct EmbeddingRequest {
    std::vector<std::string> input;
    std::string model = "qwen3-metal";
    std::string encoding_format = "float";
    std::string user = "";
};

struct Usage {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;
};

struct Choice {
    int index = 0;
    ChatMessage message;
    std::string finish_reason = "stop";
};

struct CompletionResponse {
    std::string id;
    std::string object = "chat.completion";
    int64_t created;
    std::string model;
    std::vector<Choice> choices;
    Usage usage;
};

struct Delta {
    std::string role = "";
    std::string content = "";
};

struct StreamChoice {
    int index = 0;
    Delta delta;
    std::string finish_reason = "";
};

struct StreamChunk {
    std::string id;
    std::string object = "chat.completion.chunk";
    int64_t created;
    std::string model;
    std::vector<StreamChoice> choices;
};

class Qwen3ApiHandler {
public:
    Qwen3ApiHandler(MetalContext& metalContext);
    ~Qwen3ApiHandler();

    // Initialize with model
    bool initialize(const std::string& model_path);

    // HTTP handlers
    void handleChatCompletions(const httplib::Request& req, httplib::Response& res);
    void handleCompletions(const httplib::Request& req, httplib::Response& res);
    void handleEmbeddings(const httplib::Request& req, httplib::Response& res);
    void handleModels(const httplib::Request& req, httplib::Response& res);
    void handleHealth(const httplib::Request& req, httplib::Response& res);

    // Utility functions
    std::string generateCompletion(const CompletionRequest& request, int& prompt_tokens, int& completion_tokens);
    void streamCompletion(const CompletionRequest& request, httplib::Response& res);
    std::string generateTextCompletion(const TextCompletionRequest& request, int& prompt_tokens, int& completion_tokens);
    void streamTextCompletion(const TextCompletionRequest& request, httplib::Response& res);

private:
    MetalContext& metalContext;
    std::unique_ptr<MetalQwen3> metalModel;
    std::unique_ptr<Qwen3Original> cpuModel;
    std::unique_ptr<Qwen3Tokenizer> tokenizer;
    bool metalReady = false;

    std::atomic<unsigned long long> request_counter{0};

    // Helper functions
    CompletionRequest parseCompletionRequest(const nlohmann::json& json);
    TextCompletionRequest parseTextCompletionRequest(const nlohmann::json& json);
    EmbeddingRequest parseEmbeddingRequest(const nlohmann::json& json);
    nlohmann::json createCompletionResponse(
        const std::string& request_id,
        const std::string& model_id,
        const std::string& generated_text,
        int prompt_tokens,
        int completion_tokens
    );
    nlohmann::json createStreamChunk(
        const std::string& request_id,
        const std::string& model_id,
        const std::string& content,
        bool is_final = false
    );
    nlohmann::json createTextCompletionResponse(
        const std::string& request_id,
        const std::string& model_id,
        const std::string& generated_text,
        int prompt_tokens,
        int completion_tokens,
        const std::string& echo_text = ""
    );
    nlohmann::json createEmbeddingResponse(
        const std::string& model_id,
        const std::vector<std::string>& input,
        const std::string& encoding_format
    );
    nlohmann::json createTextCompletionStreamChunk(
        const std::string& request_id,
        const std::string& model_id,
        const std::string& content,
        bool is_final = false
    );

    std::string generateRequestId();
    int64_t getCurrentTimestamp();

    // Text generation
    struct GenerationResult {
        std::string text;
        int prompt_tokens = 0;
        int completion_tokens = 0;
        std::vector<int> generated_token_ids;
    };

    GenerationResult runGeneration(const std::string& prompt, const CompletionRequest& request);
    std::string buildPrompt(const CompletionRequest& request) const;
    bool shouldStop(const std::string& generated_text, const std::vector<std::string>& stop_tokens);

    // Performance metrics
    mutable std::chrono::high_resolution_clock::time_point start_time;
    mutable std::chrono::high_resolution_clock::time_point first_token_time;
    mutable bool first_token_recorded = false;
    // Last-request metrics exposed in JSON responses (seconds)
    mutable double last_ttft = 0.0;
    mutable double last_tokens_per_second = 0.0;
    mutable double last_total_time = 0.0;
};
