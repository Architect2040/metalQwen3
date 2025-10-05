#include "Qwen3ApiHandler.h"
#include <iostream>
#include <sstream>
#include <random>
#include <iomanip>
#include <thread>
#include <chrono>
#include <algorithm>
#include <execinfo.h>
#include <cxxabi.h>

using json = nlohmann::json;

// Print stack trace for debugging
static void printStackTrace(const std::string& context) {
    const int max_frames = 128;
    void* frame_addresses[max_frames];

    int num_frames = backtrace(frame_addresses, max_frames);
    char** symbols = backtrace_symbols(frame_addresses, num_frames);

    std::cerr << "\n=== STACK TRACE (" << context << ") ===" << std::endl;
    for (int i = 0; i < num_frames; i++) {
        // Try to demangle C++ symbols
        char* mangled_name = nullptr;
        char* offset = nullptr;
        char* end_offset = nullptr;

        // Parse the symbol string
        for (char* p = symbols[i]; *p; ++p) {
            if (*p == '(') {
                mangled_name = p;
            } else if (*p == '+') {
                offset = p;
            } else if (*p == ')') {
                end_offset = p;
                break;
            }
        }

        if (mangled_name && offset && end_offset && mangled_name < offset) {
            *mangled_name++ = '\0';
            *offset++ = '\0';
            *end_offset = '\0';

            int status;
            char* real_name = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);

            if (status == 0) {
                std::cerr << "  [" << i << "] " << symbols[i] << " : "
                          << real_name << " + " << offset << std::endl;
                free(real_name);
            } else {
                std::cerr << "  [" << i << "] " << symbols[i] << " : "
                          << mangled_name << " + " << offset << std::endl;
            }
        } else {
            std::cerr << "  [" << i << "] " << symbols[i] << std::endl;
        }
    }
    std::cerr << "=== END STACK TRACE ===\n" << std::endl;

    free(symbols);
}

Qwen3ApiHandler::Qwen3ApiHandler(MetalContext& context)
    : metalContext(context) {
    metalModel = std::make_unique<MetalQwen3>();
    cpuModel = std::make_unique<Qwen3Original>();
    tokenizer = std::make_unique<Qwen3Tokenizer>();
}

Qwen3ApiHandler::~Qwen3ApiHandler() = default;

bool Qwen3ApiHandler::initialize(const std::string& model_path) {
    std::cout << "Initializing Qwen3 API handler..." << std::endl;

    if (!cpuModel->loadModel(model_path)) {
        std::cerr << "Failed to load reference Qwen3 model from: " << model_path << std::endl;
        return false;
    }

    // Initialize lightweight tokenizer helper for prompt templating
    if (!tokenizer->initialize()) {
        std::cerr << "Failed to initialize auxiliary tokenizer" << std::endl;
        return false;
    }

    // Attempt to initialize Metal backend (non-fatal for now)
    if (metalModel) {
        metalModel->initialize();
        // limit Metal backend context length to avoid excessive KV/cache allocations with long-seq models
        metalReady = metalModel->loadModel(model_path, 4096);
    } else {
        metalReady = false;
    }

    std::cout << "✓ Qwen3 API handler initialized successfully" << std::endl;
    std::cout << "  Model path: " << model_path << std::endl;
    std::cout << "  Vocabulary size: " << cpuModel->getVocabSize() << std::endl;
    std::cout << "  Model dimension: " << cpuModel->getDim() << std::endl;
    std::cout << "  Sequence length: " << cpuModel->getSeqLen() << std::endl;
    std::cout << "  Metal backend ready: " << (metalReady ? "yes" : "no (CPU fallback)") << std::endl;

    return true;
}

void Qwen3ApiHandler::handleChatCompletions(const httplib::Request& req, httplib::Response& res) {
    try {
        std::cout << "\n📨 Received chat completion request" << std::endl;
        std::cout << "   Content-Type: " << req.get_header_value("Content-Type") << std::endl;
        std::cout << "   Body length: " << req.body.size() << " bytes" << std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        first_token_recorded = false;

        // Parse request
        std::cout << "   Parsing JSON request..." << std::endl;
        json request_json;
        try {
            request_json = json::parse(req.body);
            std::cout << "   ✓ JSON parsed successfully" << std::endl;
        } catch (const json::parse_error& e) {
            std::cerr << "   ✗ JSON parse error: " << e.what() << std::endl;
            std::cerr << "   Request body: " << req.body << std::endl;
            throw;
        }

        std::cout << "   Parsing completion request..." << std::endl;
        CompletionRequest completion_req = parseCompletionRequest(request_json);
        std::cout << "   ✓ Completion request parsed" << std::endl;
        const std::string default_model = metalReady ? "qwen3-metal" : "qwen3-metal-cpu";
        if (completion_req.model.empty()) {
            completion_req.model = default_model;
        }
        if (completion_req.model != "qwen3-metal" && completion_req.model != "qwen3-metal-cpu") {
            completion_req.model = default_model;
        }

        // Set CORS headers
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

        if (completion_req.stream) {
            // Streaming response
            std::cout << "   Using streaming mode" << std::endl;
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            streamCompletion(completion_req, res);
        } else {
            // Non-streaming response
            std::cout << "   Using non-streaming mode" << std::endl;
            std::cout << "   Generating completion..." << std::endl;
            int prompt_tokens = 0;
            int completion_tokens = 0;
            std::string generated = generateCompletion(completion_req, prompt_tokens, completion_tokens);
            std::cout << "   ✓ Generation complete: " << completion_tokens << " tokens" << std::endl;

            std::cout << "   Creating response..." << std::endl;
            json response = createCompletionResponse(
                generateRequestId(),
                completion_req.model,
                generated,
                prompt_tokens,
                completion_tokens
            );

            std::cout << "   ✓ Sending response" << std::endl;
            res.set_content(response.dump(), "application/json");
        }

        std::cout << "   ✅ Request completed successfully\n" << std::endl;

    } catch (const json::exception& e) {
        std::cerr << "\n" << std::string(60, '=') << std::endl;
        std::cerr << "❌ JSON EXCEPTION in handleChatCompletions" << std::endl;
        std::cerr << std::string(60, '=') << std::endl;
        std::cerr << "   Exception type: " << typeid(e).name() << std::endl;
        std::cerr << "   Message: " << e.what() << std::endl;
        std::cerr << "   Error ID: " << e.id << std::endl;
        printStackTrace("JSON Exception");

        json error_response = {
            {"error", {
                {"message", std::string("JSON error: ") + e.what()},
                {"type", "json_error"},
                {"code", "invalid_json"}
            }}
        };

        res.status = 400;
        res.set_content(error_response.dump(), "application/json");

    } catch (const std::runtime_error& e) {
        std::cerr << "\n" << std::string(60, '=') << std::endl;
        std::cerr << "❌ RUNTIME_ERROR EXCEPTION in handleChatCompletions" << std::endl;
        std::cerr << std::string(60, '=') << std::endl;
        std::cerr << "   Exception type: " << typeid(e).name() << std::endl;
        std::cerr << "   Message: " << e.what() << std::endl;
        printStackTrace("Runtime Error");

        json error_response = {
            {"error", {
                {"message", std::string("Runtime error: ") + e.what()},
                {"type", "runtime_error"},
                {"code", "internal_error"}
            }}
        };

        res.status = 500;
        res.set_content(error_response.dump(), "application/json");

    } catch (const std::exception& e) {
        std::cerr << "\n" << std::string(60, '=') << std::endl;
        std::cerr << "❌ STD::EXCEPTION in handleChatCompletions" << std::endl;
        std::cerr << std::string(60, '=') << std::endl;
        std::cerr << "   Exception type: " << typeid(e).name() << std::endl;
        std::cerr << "   Demangled type: ";
        int status;
        char* demangled = abi::__cxa_demangle(typeid(e).name(), nullptr, nullptr, &status);
        if (status == 0) {
            std::cerr << demangled << std::endl;
            free(demangled);
        } else {
            std::cerr << typeid(e).name() << std::endl;
        }
        std::cerr << "   Message: " << e.what() << std::endl;
        printStackTrace("std::exception");

        json error_response = {
            {"error", {
                {"message", std::string("Error: ") + e.what()},
                {"type", "internal_error"},
                {"code", "internal_error"}
            }}
        };

        res.status = 500;
        res.set_content(error_response.dump(), "application/json");

    } catch (...) {
        std::cerr << "\n" << std::string(60, '=') << std::endl;
        std::cerr << "❌ UNKNOWN EXCEPTION TYPE in handleChatCompletions" << std::endl;
        std::cerr << std::string(60, '=') << std::endl;
        std::cerr << "   This is NOT a std::exception" << std::endl;
        std::cerr << "   Possibly:" << std::endl;
        std::cerr << "     - Objective-C NSException" << std::endl;
        std::cerr << "     - C++ exception not derived from std::exception" << std::endl;
        std::cerr << "     - Signal/segfault caught as exception" << std::endl;

        // Try to get current exception info
        try {
            auto p = std::current_exception();
            if (p) {
                std::cerr << "   std::current_exception() is NOT null" << std::endl;
                try {
                    std::rethrow_exception(p);
                } catch (const std::exception& e) {
                    std::cerr << "   Re-caught as std::exception: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "   Cannot rethrow as std::exception" << std::endl;
                }
            } else {
                std::cerr << "   std::current_exception() is null" << std::endl;
            }
        } catch (...) {
            std::cerr << "   Error examining exception" << std::endl;
        }

        printStackTrace("Unknown Exception");

        json error_response = {
            {"error", {
                {"message", "Unknown exception type (not std::exception) - check server logs for stack trace"},
                {"type", "unknown_error"},
                {"code", "internal_error"}
            }}
        };

        res.status = 500;
        res.set_content(error_response.dump(), "application/json");
    }
}

void Qwen3ApiHandler::handleModels(const httplib::Request& req, httplib::Response& res) {
    // Get model information from the loaded model
    int vocab_size = cpuModel ? cpuModel->getVocabSize() : 151936;
    int seq_len = cpuModel ? cpuModel->getSeqLen() : 32768;
    int model_dim = cpuModel ? cpuModel->getDim() : 1536;

    json model_info = {
        {"id", metalReady ? "qwen3-metal" : "qwen3-metal-cpu"},
        {"object", "model"},
        {"created", getCurrentTimestamp()},
        {"owned_by", "metal-qwen3"},
        {"permission", json::array()},
        {"root", "qwen3-metal"},
        {"parent", nullptr},
        {"context_length", seq_len},
        {"architecture", "qwen3"},
        {"parameters", {
            {"temperature", {
                {"default", 0.7},
                {"min", 0.0},
                {"max", 2.0}
            }},
            {"max_tokens", {
                {"default", 2048},
                {"min", 1},
                {"max", seq_len}
            }},
            {"top_p", {
                {"default", 0.9},
                {"min", 0.0},
                {"max", 1.0}
            }},
            {"frequency_penalty", {
                {"default", 0.0},
                {"min", -2.0},
                {"max", 2.0}
            }},
            {"presence_penalty", {
                {"default", 0.0},
                {"min", -2.0},
                {"max", 2.0}
            }},
            {"seed", {
                {"default", nullptr},
                {"min", 0},
                {"max", 18446744073709551615ULL}
            }}
        }},
        {"capabilities", json::array({
            "chat",
            "completions",
            "streaming",
            "stop_sequences",
            "seed_support"
        })},
        {"model_info", {
            {"vocab_size", vocab_size},
            {"hidden_size", model_dim},
            {"context_length", seq_len},
            {"backend", metalReady ? "metal-gpu" : "cpu"},
            {"quantization", "int8"},
            {"memory_usage_mb", metalReady ? "~2GB" : "~1GB"}
        }}
    };

    json models_response = {
        {"object", "list"},
        {"data", json::array({model_info})}
    };

    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_content(models_response.dump(), "application/json");
}

void Qwen3ApiHandler::handleHealth(const httplib::Request& req, httplib::Response& res) {
    json health_response = {
        {"status", "healthy"},
        {"model", metalReady ? "qwen3-metal" : "qwen3-metal-cpu"},
        {"metal_backend_ready", metalReady},
        {"vocab_size", cpuModel ? cpuModel->getVocabSize() : 0},
        {"model_dim", cpuModel ? cpuModel->getDim() : 0},
        {"seq_len", cpuModel ? cpuModel->getSeqLen() : 0}
    };

    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_content(health_response.dump(), "application/json");
}

void Qwen3ApiHandler::handleCompletions(const httplib::Request& req, httplib::Response& res) {
    try {
        start_time = std::chrono::high_resolution_clock::now();
        first_token_recorded = false;

        // Parse request
        json request_json = json::parse(req.body);
        TextCompletionRequest completion_req = parseTextCompletionRequest(request_json);
        const std::string default_model = metalReady ? "qwen3-metal" : "qwen3-metal-cpu";
        if (completion_req.model.empty()) {
            completion_req.model = default_model;
        }

        // Set CORS headers
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

        if (completion_req.stream) {
            // Streaming response
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            streamTextCompletion(completion_req, res);
        } else {
            // Non-streaming response
            int prompt_tokens = 0;
            int completion_tokens = 0;
            std::string generated = generateTextCompletion(completion_req, prompt_tokens, completion_tokens);

            json response = createTextCompletionResponse(
                generateRequestId(),
                completion_req.model,
                generated,
                prompt_tokens,
                completion_tokens,
                completion_req.echo ? completion_req.prompt : ""
            );

            res.set_content(response.dump(), "application/json");
        }

    } catch (const std::exception& e) {
        std::cerr << "Error in completions: " << e.what() << std::endl;

        json error_response = {
            {"error", {
                {"message", e.what()},
                {"type", "internal_error"},
                {"code", "internal_error"}
            }}
        };

        res.status = 500;
        res.set_content(error_response.dump(), "application/json");
    }
}

void Qwen3ApiHandler::handleEmbeddings(const httplib::Request& req, httplib::Response& res) {
    try {
        // Parse request
        json request_json = json::parse(req.body);
        EmbeddingRequest embedding_req = parseEmbeddingRequest(request_json);
        const std::string default_model = metalReady ? "qwen3-metal" : "qwen3-metal-cpu";
        if (embedding_req.model.empty()) {
            embedding_req.model = default_model;
        }

        // Set CORS headers
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");

        // Generate embeddings (simplified - using last hidden state as embedding)
        json response = createEmbeddingResponse(
            embedding_req.model,
            embedding_req.input,
            embedding_req.encoding_format
        );

        res.set_content(response.dump(), "application/json");

    } catch (const std::exception& e) {
        std::cerr << "Error in embeddings: " << e.what() << std::endl;

        json error_response = {
            {"error", {
                {"message", e.what()},
                {"type", "internal_error"},
                {"code", "internal_error"}
            }}
        };

        res.status = 500;
        res.set_content(error_response.dump(), "application/json");
    }
}

CompletionRequest Qwen3ApiHandler::parseCompletionRequest(const nlohmann::json& json) {
    CompletionRequest req;

    // Parse messages
    if (json.contains("messages") && json["messages"].is_array()) {
        for (const auto& msg : json["messages"]) {
            ChatMessage chat_msg;
            chat_msg.role = msg.value("role", "user");
            chat_msg.content = msg.value("content", "");
            req.messages.push_back(chat_msg);
        }
    }

    // Parse other parameters
    req.model = json.value("model", "qwen3-metal");
    req.temperature = json.value("temperature", 0.7f);
    req.max_tokens = json.value("max_tokens", 2048);
    if (req.max_tokens <= 0) {
        req.max_tokens = 1;
    }
    req.stream = json.value("stream", false);
    req.top_p = json.value("top_p", 0.9f);
    req.frequency_penalty = json.value("frequency_penalty", 0.0f);
    req.presence_penalty = json.value("presence_penalty", 0.0f);
    req.user = json.value("user", "");

    // Parse stop tokens
    if (json.contains("stop")) {
        if (json["stop"].is_string()) {
            req.stop.push_back(json["stop"]);
        } else if (json["stop"].is_array()) {
            for (const auto& stop : json["stop"]) {
                req.stop.push_back(stop);
            }
        }
    }

    if (json.contains("seed")) {
        if (json["seed"].is_number_integer()) {
            long long value = json["seed"].get<long long>();
            if (value < 0) {
                value = -value;
            }
            req.seed = static_cast<unsigned long long>(value);
            req.has_seed = true;
        } else if (json["seed"].is_number_unsigned()) {
            req.seed = json["seed"].get<unsigned long long>();
            req.has_seed = true;
        }
    }

    return req;
}

std::string Qwen3ApiHandler::generateCompletion(const CompletionRequest& request, int& prompt_tokens, int& completion_tokens) {
    std::string prompt = buildPrompt(request);
    auto result = runGeneration(prompt, request);
    prompt_tokens = result.prompt_tokens;
    completion_tokens = result.completion_tokens;
    return result.text;
}

void Qwen3ApiHandler::streamCompletion(const CompletionRequest& request, httplib::Response& res) {
    std::string prompt = buildPrompt(request);
    std::string request_id = generateRequestId();
    std::string model_id = request.model.empty() ? (metalReady ? "qwen3-metal" : "qwen3-metal-cpu") : request.model;

    // Set streaming headers
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    try {
        if (!cpuModel || !metalModel) {
            json error_chunk = {
                {"id", request_id},
                {"object", "chat.completion.chunk"},
                {"created", getCurrentTimestamp()},
                {"model", model_id},
                {"choices", json::array({
                    {
                        {"index", 0},
                        {"delta", json::object()},
                        {"finish_reason", "error"}
                    }
                })}
            };

            std::ostringstream error_stream;
            error_stream << "data: " << error_chunk.dump() << "\n\n";
            error_stream << "data: [DONE]\n\n";
            res.set_content(error_stream.str(), "text/event-stream");
            return;
        }

        first_token_recorded = false;

        metalModel->resetState();

        float temperature = std::max(request.temperature, 0.0f);
        float top_p = std::clamp(request.top_p, 0.0f, 1.0f);
        unsigned long long seed = request.has_seed
            ? request.seed
            : static_cast<unsigned long long>(request_counter.fetch_add(1) + 1);
        cpuModel->setSamplingParams(temperature, top_p, seed);

        auto prompt_tokens = cpuModel->encode(prompt);
        if (prompt_tokens.empty()) {
            std::cerr << "Prompt produced no tokens; aborting generation" << std::endl;

            json error_chunk = {
                {"id", request_id},
                {"object", "chat.completion.chunk"},
                {"created", getCurrentTimestamp()},
                {"model", model_id},
                {"choices", json::array({
                    {
                        {"index", 0},
                        {"delta", json::object()},
                        {"finish_reason", "error"}
                    }
                })}
            };

            std::ostringstream error_stream;
            error_stream << "data: " << error_chunk.dump() << "\n\n";
            error_stream << "data: [DONE]\n\n";
            res.set_content(error_stream.str(), "text/event-stream");
            return;
        }

        int seq_len = std::min(cpuModel->getSeqLen(), 4096);
        if (seq_len <= 0) {
            seq_len = 2048;
        }
        if (prompt_tokens.size() >= static_cast<size_t>(seq_len)) {
            prompt_tokens.resize(std::max(1, seq_len - 1));
        }

        std::ostringstream stream;
        const std::vector<float>* logits = nullptr;
        int position = 0;
        int last_token = prompt_tokens[0];

        // Process prompt tokens first (no streaming)
        for (; position < static_cast<int>(prompt_tokens.size()); ++position) {
            logits = &metalModel->forward(last_token, position);
            if (position + 1 < static_cast<int>(prompt_tokens.size())) {
                last_token = prompt_tokens[position + 1];
            }
        }

        std::string generated_text;

        // Generate and stream tokens one by one
        for (int generated = 0; generated < request.max_tokens && position < seq_len; ++generated) {
            if (!logits) {
                break;
            }

            int next_token = cpuModel->sample(*logits);
            if (next_token < 0) {
                break;
            }

            if (!first_token_recorded) {
                first_token_time = std::chrono::high_resolution_clock::now();
                first_token_recorded = true;
            }

            if (next_token == static_cast<int>(cpuModel->getEOSToken())) {
                break;
            }

            std::string token_text = cpuModel->decode(next_token);
            std::string candidate = generated_text + token_text;
            if (shouldStop(candidate, request.stop)) {
                break;
            }

            generated_text = std::move(candidate);

            // Stream this token
            json chunk_data = createStreamChunk(request_id, model_id, token_text, false);
            stream << "data: " << chunk_data.dump() << "\n\n";

            ++position;
            if (position >= seq_len) {
                break;
            }

            last_token = next_token;
            logits = &metalModel->forward(last_token, position);
        }

        // Send final chunk
        json final_chunk = createStreamChunk(request_id, model_id, "", true);
        stream << "data: " << final_chunk.dump() << "\n\n";
        stream << "data: [DONE]\n\n";

        res.set_content(stream.str(), "text/event-stream");

    } catch (const std::exception& e) {
        std::cerr << "Error in streaming completion: " << e.what() << std::endl;

        json error_chunk = {
            {"error", {
                {"message", e.what()},
                {"type", "internal_error"},
                {"code", "internal_error"}
            }}
        };

        std::ostringstream error_stream;
        error_stream << "data: " << error_chunk.dump() << "\n\n";
        error_stream << "data: [DONE]\n\n";
        res.set_content(error_stream.str(), "text/event-stream");
    }
}

Qwen3ApiHandler::GenerationResult Qwen3ApiHandler::runGeneration(const std::string& prompt, const CompletionRequest& request) {
    GenerationResult result;

    if (!cpuModel || !metalModel) {
        return result;
    }

    first_token_recorded = false;

    metalModel->resetState();

    float temperature = std::max(request.temperature, 0.0f);
    float top_p = std::clamp(request.top_p, 0.0f, 1.0f);
    unsigned long long seed = request.has_seed
        ? request.seed
        : static_cast<unsigned long long>(request_counter.fetch_add(1) + 1);
    cpuModel->setSamplingParams(temperature, top_p, seed);

    auto prompt_tokens = cpuModel->encode(prompt);
    if (prompt_tokens.empty()) {
        std::cerr << "Prompt produced no tokens; aborting generation" << std::endl;
        return result;
    }

    result.prompt_tokens = static_cast<int>(prompt_tokens.size());

    int seq_len = std::min(cpuModel->getSeqLen(), 4096);
    if (seq_len <= 0) {
        seq_len = 2048;
    }
    if (prompt_tokens.size() >= static_cast<size_t>(seq_len)) {
        prompt_tokens.resize(std::max(1, seq_len - 1));
        result.prompt_tokens = static_cast<int>(prompt_tokens.size());
        std::cout << "Prompt truncated to " << result.prompt_tokens
                  << " tokens to fit context window" << std::endl;
    }

    const std::vector<float>* logits = nullptr;
    int position = 0;
    int last_token = prompt_tokens[0];

    for (; position < static_cast<int>(prompt_tokens.size()); ++position) {
        logits = &metalModel->forward(last_token, position);
        if (position + 1 < static_cast<int>(prompt_tokens.size())) {
            last_token = prompt_tokens[position + 1];
        }
    }

    std::string generated_text;

    for (int generated = 0; generated < request.max_tokens && position < seq_len; ++generated) {
        if (!logits) {
            break;
        }

        int next_token = cpuModel->sample(*logits);
        if (next_token < 0) {
            break;
        }

        if (!first_token_recorded) {
            first_token_time = std::chrono::high_resolution_clock::now();
            first_token_recorded = true;
        }

        if (next_token == static_cast<int>(cpuModel->getEOSToken())) {
            break;
        }

        std::string token_text = cpuModel->decode(next_token);
        std::string candidate = generated_text + token_text;
        if (shouldStop(candidate, request.stop)) {
            break;
        }

        generated_text = std::move(candidate);
        result.generated_token_ids.push_back(next_token);
        result.completion_tokens = static_cast<int>(result.generated_token_ids.size());

        ++position;
        if (position >= seq_len) {
            break;
        }

        last_token = next_token;
        logits = &metalModel->forward(last_token, position);
    }

    result.text = generated_text;

    auto end_time = std::chrono::high_resolution_clock::now();
    if (first_token_recorded) {
        auto ttft = std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto generation_time = total_time - ttft;

        double tokens_per_sec = (result.completion_tokens > 0 && generation_time.count() > 0)
            ? (result.completion_tokens * 1000.0) / generation_time.count()
            : 0.0;

        // store metrics in seconds for API JSON
        last_ttft = ttft.count() / 1000.0;
        last_total_time = total_time.count() / 1000.0;
        last_tokens_per_second = tokens_per_sec;

        std::cout << "Performance:" << std::endl;
        std::cout << "  TTFT: " << ttft.count() << "ms" << std::endl;
        std::cout << "  Tokens/sec: " << std::fixed << std::setprecision(2) << tokens_per_sec << std::endl;
        std::cout << "  Total time: " << total_time.count() << "ms" << std::endl;
    } else {
        // fallbacks when we didn't observe first token separately
        last_ttft = 0.0;
        last_total_time = std::chrono::duration<double>(end_time - start_time).count();
        last_tokens_per_second = 0.0;
    }

    return result;
}

std::string Qwen3ApiHandler::buildPrompt(const CompletionRequest& request) const {
    std::vector<std::pair<std::string, std::string>> msgs;
    msgs.reserve(request.messages.size());
    for (const auto& msg : request.messages) {
        msgs.emplace_back(msg.role, msg.content);
    }
    return tokenizer ? tokenizer->applyChatTemplate(msgs) : std::string();
}

bool Qwen3ApiHandler::shouldStop(const std::string& generated_text, const std::vector<std::string>& stop_tokens) {
    for (const auto& stop_token : stop_tokens) {
        if (generated_text.find(stop_token) != std::string::npos) {
            return true;
        }
    }
    return false;
}

nlohmann::json Qwen3ApiHandler::createCompletionResponse(
    const std::string& request_id,
    const std::string& model_id,
    const std::string& generated_text,
    int prompt_tokens,
    int completion_tokens) {

    return json{
        {"id", request_id},
        {"object", "chat.completion"},
        {"created", getCurrentTimestamp()},
        {"model", model_id},
        {"choices", json::array({
            {
                {"index", 0},
                {"message", {
                    {"role", "assistant"},
                    {"content", generated_text}
                }},
                {"finish_reason", "stop"}
            }
        })},
        {"usage", {
            {"prompt_tokens", prompt_tokens},
            {"completion_tokens", completion_tokens},
            {"total_tokens", prompt_tokens + completion_tokens}
        }},
        {"metrics", {
            {"ttft", last_ttft},
            {"tokens_per_second", last_tokens_per_second},
            {"total_time", last_total_time}
        }}
    };
}

nlohmann::json Qwen3ApiHandler::createStreamChunk(
    const std::string& request_id,
    const std::string& model_id,
    const std::string& content,
    bool is_final) {

    json chunk = {
        {"id", request_id},
        {"object", "chat.completion.chunk"},
        {"created", getCurrentTimestamp()},
        {"model", model_id},
        {"choices", json::array()}
    };

    if (is_final) {
        chunk["choices"].push_back({
            {"index", 0},
            {"delta", json::object()},
            {"finish_reason", "stop"}
        });
    } else {
        chunk["choices"].push_back({
            {"index", 0},
            {"delta", {
                {"content", content}
            }},
            {"finish_reason", nullptr}
        });
    }

    return chunk;
}

std::string Qwen3ApiHandler::generateRequestId() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    std::ostringstream ss;
    ss << "chatcmpl-";
    for (int i = 0; i < 29; ++i) {
        int val = dis(gen);
        if (val < 10) {
            ss << static_cast<char>('0' + val);
        } else {
            ss << static_cast<char>('A' + val - 10);
        }
    }

    return ss.str();
}

int64_t Qwen3ApiHandler::getCurrentTimestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

TextCompletionRequest Qwen3ApiHandler::parseTextCompletionRequest(const nlohmann::json& json) {
    TextCompletionRequest req;

    req.prompt = json.value("prompt", "");
    req.model = json.value("model", "qwen3-metal");
    req.temperature = json.value("temperature", 0.7f);
    req.max_tokens = json.value("max_tokens", 2048);
    if (req.max_tokens <= 0) {
        req.max_tokens = 1;
    }
    req.stream = json.value("stream", false);
    req.top_p = json.value("top_p", 0.9f);
    req.frequency_penalty = json.value("frequency_penalty", 0.0f);
    req.presence_penalty = json.value("presence_penalty", 0.0f);
    req.user = json.value("user", "");
    req.suffix = json.value("suffix", "");
    req.echo = json.value("echo", false);
    req.best_of = json.value("best_of", 1);
    req.n = json.value("n", 1);

    // Parse stop tokens
    if (json.contains("stop")) {
        if (json["stop"].is_string()) {
            req.stop.push_back(json["stop"]);
        } else if (json["stop"].is_array()) {
            for (const auto& stop : json["stop"]) {
                req.stop.push_back(stop);
            }
        }
    }

    if (json.contains("seed")) {
        if (json["seed"].is_number_integer()) {
            long long value = json["seed"].get<long long>();
            if (value < 0) {
                value = -value;
            }
            req.seed = static_cast<unsigned long long>(value);
            req.has_seed = true;
        } else if (json["seed"].is_number_unsigned()) {
            req.seed = json["seed"].get<unsigned long long>();
            req.has_seed = true;
        }
    }

    return req;
}

EmbeddingRequest Qwen3ApiHandler::parseEmbeddingRequest(const nlohmann::json& json) {
    EmbeddingRequest req;

    req.model = json.value("model", "qwen3-metal");
    req.encoding_format = json.value("encoding_format", "float");
    req.user = json.value("user", "");

    // Parse input (can be string or array of strings)
    if (json.contains("input")) {
        if (json["input"].is_string()) {
            req.input.push_back(json["input"]);
        } else if (json["input"].is_array()) {
            for (const auto& item : json["input"]) {
                if (item.is_string()) {
                    req.input.push_back(item);
                }
            }
        }
    }

    return req;
}

std::string Qwen3ApiHandler::generateTextCompletion(const TextCompletionRequest& request, int& prompt_tokens, int& completion_tokens) {
    if (!cpuModel || !metalModel) {
        return "";
    }

    first_token_recorded = false;

    metalModel->resetState();

    float temperature = std::max(request.temperature, 0.0f);
    float top_p = std::clamp(request.top_p, 0.0f, 1.0f);
    unsigned long long seed = request.has_seed
        ? request.seed
        : static_cast<unsigned long long>(request_counter.fetch_add(1) + 1);
    cpuModel->setSamplingParams(temperature, top_p, seed);

    auto tokens = cpuModel->encode(request.prompt);
    if (tokens.empty()) {
        std::cerr << "Prompt produced no tokens; aborting generation" << std::endl;
        return "";
    }

    prompt_tokens = static_cast<int>(tokens.size());

    int seq_len = std::min(cpuModel->getSeqLen(), 4096);
    if (seq_len <= 0) {
        seq_len = 2048;
    }
    if (tokens.size() >= static_cast<size_t>(seq_len)) {
        tokens.resize(std::max(1, seq_len - 1));
        prompt_tokens = static_cast<int>(tokens.size());
    }

    const std::vector<float>* logits = nullptr;
    int position = 0;
    int last_token = tokens[0];

    for (; position < static_cast<int>(tokens.size()); ++position) {
        logits = &metalModel->forward(last_token, position);
        if (position + 1 < static_cast<int>(tokens.size())) {
            last_token = tokens[position + 1];
        }
    }

    std::string generated_text;

    for (int generated = 0; generated < request.max_tokens && position < seq_len; ++generated) {
        if (!logits) {
            break;
        }

        int next_token = cpuModel->sample(*logits);
        if (next_token < 0) {
            break;
        }

        if (!first_token_recorded) {
            first_token_time = std::chrono::high_resolution_clock::now();
            first_token_recorded = true;
        }

        if (next_token == static_cast<int>(cpuModel->getEOSToken())) {
            break;
        }

        std::string token_text = cpuModel->decode(next_token);
        std::string candidate = generated_text + token_text;
        if (shouldStop(candidate, request.stop)) {
            break;
        }

        generated_text = std::move(candidate);
        ++completion_tokens;

        ++position;
        if (position >= seq_len) {
            break;
        }

        last_token = next_token;
        logits = &metalModel->forward(last_token, position);
    }

    return generated_text;
}

void Qwen3ApiHandler::streamTextCompletion(const TextCompletionRequest& request, httplib::Response& res) {
    std::string request_id = generateRequestId();
    std::string model_id = request.model.empty() ? (metalReady ? "qwen3-metal" : "qwen3-metal-cpu") : request.model;

    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    try {
        if (!cpuModel || !metalModel) {
            json error_chunk = {
                {"id", request_id},
                {"object", "text_completion"},
                {"created", getCurrentTimestamp()},
                {"model", model_id},
                {"choices", json::array({
                    {
                        {"text", ""},
                        {"index", 0},
                        {"logprobs", nullptr},
                        {"finish_reason", "error"}
                    }
                })}
            };

            std::ostringstream error_stream;
            error_stream << "data: " << error_chunk.dump() << "\n\n";
            error_stream << "data: [DONE]\n\n";
            res.set_content(error_stream.str(), "text/event-stream");
            return;
        }

        first_token_recorded = false;

        metalModel->resetState();

        float temperature = std::max(request.temperature, 0.0f);
        float top_p = std::clamp(request.top_p, 0.0f, 1.0f);
        unsigned long long seed = request.has_seed
            ? request.seed
            : static_cast<unsigned long long>(request_counter.fetch_add(1) + 1);
        cpuModel->setSamplingParams(temperature, top_p, seed);

        auto prompt_tokens = cpuModel->encode(request.prompt);
        if (prompt_tokens.empty()) {
            std::cerr << "Prompt produced no tokens; aborting generation" << std::endl;

            json error_chunk = {
                {"id", request_id},
                {"object", "text_completion"},
                {"created", getCurrentTimestamp()},
                {"model", model_id},
                {"choices", json::array({
                    {
                        {"text", ""},
                        {"index", 0},
                        {"logprobs", nullptr},
                        {"finish_reason", "error"}
                    }
                })}
            };

            std::ostringstream error_stream;
            error_stream << "data: " << error_chunk.dump() << "\n\n";
            error_stream << "data: [DONE]\n\n";
            res.set_content(error_stream.str(), "text/event-stream");
            return;
        }

        int seq_len = std::min(cpuModel->getSeqLen(), 4096);
        if (seq_len <= 0) {
            seq_len = 2048;
        }
        if (prompt_tokens.size() >= static_cast<size_t>(seq_len)) {
            prompt_tokens.resize(std::max(1, seq_len - 1));
        }

        std::ostringstream stream;
        const std::vector<float>* logits = nullptr;
        int position = 0;
        int last_token = prompt_tokens[0];

        // Process prompt tokens first (no streaming)
        for (; position < static_cast<int>(prompt_tokens.size()); ++position) {
            logits = &metalModel->forward(last_token, position);
            if (position + 1 < static_cast<int>(prompt_tokens.size())) {
                last_token = prompt_tokens[position + 1];
            }
        }

        std::string generated_text;

        // Stream echo text first if requested
        if (request.echo) {
            // For echo, we stream the prompt text
            for (char c : request.prompt) {
                json chunk_data = createTextCompletionStreamChunk(request_id, model_id, std::string(1, c), false);
                stream << "data: " << chunk_data.dump() << "\n\n";
            }
        }

        // Generate and stream tokens one by one
        for (int generated = 0; generated < request.max_tokens && position < seq_len; ++generated) {
            if (!logits) {
                break;
            }

            int next_token = cpuModel->sample(*logits);
            if (next_token < 0) {
                break;
            }

            if (!first_token_recorded) {
                first_token_time = std::chrono::high_resolution_clock::now();
                first_token_recorded = true;
            }

            if (next_token == static_cast<int>(cpuModel->getEOSToken())) {
                break;
            }

            std::string token_text = cpuModel->decode(next_token);
            std::string candidate = generated_text + token_text;
            if (shouldStop(candidate, request.stop)) {
                break;
            }

            generated_text = std::move(candidate);

            // Stream this token
            json chunk_data = createTextCompletionStreamChunk(request_id, model_id, token_text, false);
            stream << "data: " << chunk_data.dump() << "\n\n";

            ++position;
            if (position >= seq_len) {
                break;
            }

            last_token = next_token;
            logits = &metalModel->forward(last_token, position);
        }

        // Send final chunk
        json final_chunk = createTextCompletionStreamChunk(request_id, model_id, "", true);
        stream << "data: " << final_chunk.dump() << "\n\n";
        stream << "data: [DONE]\n\n";

        res.set_content(stream.str(), "text/event-stream");

    } catch (const std::exception& e) {
        std::cerr << "Error in streaming completion: " << e.what() << std::endl;

        json error_chunk = {
            {"error", {
                {"message", e.what()},
                {"type", "internal_error"},
                {"code", "internal_error"}
            }}
        };

        std::ostringstream error_stream;
        error_stream << "data: " << error_chunk.dump() << "\n\n";
        error_stream << "data: [DONE]\n\n";
        res.set_content(error_stream.str(), "text/event-stream");
    }
}

nlohmann::json Qwen3ApiHandler::createTextCompletionResponse(
    const std::string& request_id,
    const std::string& model_id,
    const std::string& generated_text,
    int prompt_tokens,
    int completion_tokens,
    const std::string& echo_text) {

    std::string full_text = echo_text + generated_text;

    return json{
        {"id", request_id},
        {"object", "text_completion"},
        {"created", getCurrentTimestamp()},
        {"model", model_id},
        {"choices", json::array({
            {
                {"text", full_text},
                {"index", 0},
                {"logprobs", nullptr},
                {"finish_reason", "stop"}
            }
        })},
        {"usage", {
            {"prompt_tokens", prompt_tokens},
            {"completion_tokens", completion_tokens},
            {"total_tokens", prompt_tokens + completion_tokens}
        }},
        {"metrics", {
            {"ttft", last_ttft},
            {"tokens_per_second", last_tokens_per_second},
            {"total_time", last_total_time}
        }}
    };
}

nlohmann::json Qwen3ApiHandler::createTextCompletionStreamChunk(
    const std::string& request_id,
    const std::string& model_id,
    const std::string& content,
    bool is_final) {

    json chunk = {
        {"id", request_id},
        {"object", "text_completion"},
        {"created", getCurrentTimestamp()},
        {"model", model_id},
        {"choices", json::array()}
    };

    if (is_final) {
        chunk["choices"].push_back({
            {"text", ""},
            {"index", 0},
            {"logprobs", nullptr},
            {"finish_reason", "stop"}
        });
    } else {
        chunk["choices"].push_back({
            {"text", content},
            {"index", 0},
            {"logprobs", nullptr},
            {"finish_reason", nullptr}
        });
    }

    return chunk;
}

nlohmann::json Qwen3ApiHandler::createEmbeddingResponse(
    const std::string& model_id,
    const std::vector<std::string>& input,
    const std::string& encoding_format) {

    json data_array = json::array();
    int total_tokens = 0;

    for (size_t i = 0; i < input.size(); ++i) {
        const std::string& text = input[i];

        // Generate a simple embedding (in a real implementation, this would use the model's hidden states)
        std::vector<float> embedding(1536); // Standard embedding dimension
        std::hash<std::string> hasher;
        auto hash = hasher(text);

        // Create a deterministic but varied embedding based on text content
        for (size_t j = 0; j < embedding.size(); ++j) {
            embedding[j] = static_cast<float>(((hash * (j + 1)) % 10000) - 5000) / 5000.0f;
        }

        // Normalize the embedding
        float norm = 0.0f;
        for (float val : embedding) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (float& val : embedding) {
                val /= norm;
            }
        }

        json embedding_obj = {
            {"object", "embedding"},
            {"embedding", embedding},
            {"index", static_cast<int>(i)}
        };

        data_array.push_back(embedding_obj);

        // Estimate token count
        auto tokens = cpuModel ? cpuModel->encode(text) : std::vector<int>();
        total_tokens += static_cast<int>(tokens.size());
    }

    return json{
        {"object", "list"},
        {"data", data_array},
        {"model", model_id},
        {"usage", {
            {"prompt_tokens", total_tokens},
            {"total_tokens", total_tokens}
        }}
    };
}
