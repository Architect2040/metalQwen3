#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

class Qwen3Tokenizer {
public:
    Qwen3Tokenizer();
    ~Qwen3Tokenizer();

    // Initialize with tokenizer model/config
    bool initialize(const std::string& tokenizer_path = "");

    // Text <-> Token conversion
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    std::string decode(int token) const;

    // Token information
    int getVocabSize() const { return vocab_size; }
    int getBOSToken() const { return bos_token; }
    int getEOSToken() const { return eos_token; }
    int getPADToken() const { return pad_token; }

    // Chat template support
    std::string applyChatTemplate(const std::vector<std::pair<std::string, std::string>>& messages) const;

    // Utility functions
    size_t countTokens(const std::string& text) const;
    bool isSpecialToken(int token) const;

private:
    int vocab_size;
    int bos_token;
    int eos_token;
    int pad_token;

    // Simple vocabulary mapping for demonstration
    std::unordered_map<std::string, int> str_to_token;
    std::unordered_map<int, std::string> token_to_str;

    // Initialize basic vocabulary
    void initializeBasicVocab();

    // Simple tokenization (placeholder for real tokenizer)
    std::vector<std::string> tokenizeText(const std::string& text) const;
};