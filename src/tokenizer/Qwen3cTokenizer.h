#pragma once

#include <string>
#include <vector>

// Forward declarations to match original qwen3.c
struct Qwen3cTokenizer_Internal;

/**
 * C++ wrapper for the original qwen3.c tokenizer
 * This provides a clean C++ interface while using the original implementation
 */
class Qwen3cTokenizer {
public:
    Qwen3cTokenizer();
    ~Qwen3cTokenizer();

    // Initialize tokenizer from checkpoint path (matches original qwen3.c)
    bool initialize(const std::string& checkpoint_path, int vocab_size, bool enable_thinking = false);

    // Tokenizer interface
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    std::string decode(int token) const;

    // Token information
    int getVocabSize() const;
    int getBOSToken() const;
    int getEOSToken() const;

    // Chat template support
    std::string applyChatTemplate(const std::vector<std::pair<std::string, std::string>>& messages) const;

    // Count tokens without full encoding
    size_t countTokens(const std::string& text) const;

private:
    Qwen3cTokenizer_Internal* impl;
    bool initialized;
};