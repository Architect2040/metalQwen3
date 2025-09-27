#include "Qwen3Tokenizer.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <regex>

Qwen3Tokenizer::Qwen3Tokenizer() : vocab_size(0), bos_token(1), eos_token(2), pad_token(0) {
    // Initialize with basic vocabulary
    initializeBasicVocab();
}

Qwen3Tokenizer::~Qwen3Tokenizer() = default;

bool Qwen3Tokenizer::initialize(const std::string& tokenizer_path) {
    if (tokenizer_path.empty()) {
        std::cout << "Using built-in simple tokenizer for demonstration" << std::endl;
        return true;
    }

    // TODO: Load actual tokenizer model (e.g., sentencepiece, tiktoken)
    std::cout << "Loading tokenizer from: " << tokenizer_path << std::endl;

    // For now, use the basic vocabulary
    return true;
}

void Qwen3Tokenizer::initializeBasicVocab() {
    // Special tokens
    token_to_str[0] = "<pad>";
    token_to_str[1] = "<bos>";
    token_to_str[2] = "<eos>";
    token_to_str[3] = "<unk>";

    str_to_token["<pad>"] = 0;
    str_to_token["<bos>"] = 1;
    str_to_token["<eos>"] = 2;
    str_to_token["<unk>"] = 3;

    // Common words and characters
    int token_id = 4;

    // Single characters
    for (char c = 'a'; c <= 'z'; ++c) {
        std::string s(1, c);
        token_to_str[token_id] = s;
        str_to_token[s] = token_id++;
    }

    for (char c = 'A'; c <= 'Z'; ++c) {
        std::string s(1, c);
        token_to_str[token_id] = s;
        str_to_token[s] = token_id++;
    }

    for (char c = '0'; c <= '9'; ++c) {
        std::string s(1, c);
        token_to_str[token_id] = s;
        str_to_token[s] = token_id++;
    }

    // Common punctuation and spaces
    std::vector<std::string> common_chars = {
        " ", ".", ",", "!", "?", ";", ":", "'", "\"", "-", "_",
        "(", ")", "[", "]", "{", "}", "/", "\\", "@", "#",
        "$", "%", "^", "&", "*", "+", "=", "<", ">", "|",
        "~", "`", "\n", "\t"
    };

    for (const auto& ch : common_chars) {
        if (str_to_token.find(ch) == str_to_token.end()) {
            token_to_str[token_id] = ch;
            str_to_token[ch] = token_id++;
        }
    }

    // Common words (simplified vocabulary)
    std::vector<std::string> common_words = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have",
        "I", "it", "for", "not", "on", "with", "he", "as", "you",
        "do", "at", "this", "but", "his", "by", "from", "they",
        "she", "or", "an", "will", "my", "one", "all", "would",
        "there", "their", "what", "so", "up", "out", "if", "about",
        "who", "get", "which", "go", "me", "when", "make", "can",
        "like", "time", "no", "just", "him", "know", "take", "people",
        "into", "year", "your", "good", "some", "could", "them", "see",
        "other", "than", "then", "now", "look", "only", "come", "its",
        "over", "think", "also", "back", "after", "use", "two", "how",
        "our", "work", "first", "well", "way", "even", "new", "want",
        "because", "any", "these", "give", "day", "most", "us"
    };

    for (const auto& word : common_words) {
        if (str_to_token.find(word) == str_to_token.end()) {
            token_to_str[token_id] = word;
            str_to_token[word] = token_id++;
        }
    }

    vocab_size = token_id;
    std::cout << "Initialized basic tokenizer with " << vocab_size << " tokens" << std::endl;
}

std::vector<std::string> Qwen3Tokenizer::tokenizeText(const std::string& text) const {
    std::vector<std::string> tokens;

    // Simple whitespace tokenization as fallback
    std::istringstream iss(text);
    std::string word;

    while (iss >> word) {
        // Check if the whole word exists in vocabulary
        if (str_to_token.find(word) != str_to_token.end()) {
            tokens.push_back(word);
        } else {
            // Break down into characters
            for (char c : word) {
                std::string char_str(1, c);
                tokens.push_back(char_str);
            }
        }
    }

    return tokens;
}

std::vector<int> Qwen3Tokenizer::encode(const std::string& text) const {
    std::vector<int> token_ids;

    auto tokens = tokenizeText(text);

    for (const auto& token : tokens) {
        auto it = str_to_token.find(token);
        if (it != str_to_token.end()) {
            token_ids.push_back(it->second);
        } else {
            token_ids.push_back(str_to_token.at("<unk>"));
        }
    }

    return token_ids;
}

std::string Qwen3Tokenizer::decode(const std::vector<int>& tokens) const {
    std::ostringstream result;

    for (int token : tokens) {
        auto it = token_to_str.find(token);
        if (it != token_to_str.end()) {
            result << it->second;
        } else {
            result << "<unk>";
        }
    }

    return result.str();
}

std::string Qwen3Tokenizer::decode(int token) const {
    auto it = token_to_str.find(token);
    if (it != token_to_str.end()) {
        return it->second;
    }
    return "<unk>";
}

std::string Qwen3Tokenizer::applyChatTemplate(const std::vector<std::pair<std::string, std::string>>& messages) const {
    std::ostringstream result;

    // Apply Qwen3-style chat template
    for (const auto& [role, content] : messages) {
        if (role == "system") {
            result << "<|im_start|>system\n" << content << "<|im_end|>\n";
        } else if (role == "user") {
            result << "<|im_start|>user\n" << content << "<|im_end|>\n";
        } else if (role == "assistant") {
            result << "<|im_start|>assistant\n" << content << "<|im_end|>\n";
        }
    }

    // Add assistant start for generation
    result << "<|im_start|>assistant\n";

    return result.str();
}

size_t Qwen3Tokenizer::countTokens(const std::string& text) const {
    return encode(text).size();
}

bool Qwen3Tokenizer::isSpecialToken(int token) const {
    return token == bos_token || token == eos_token || token == pad_token || token == 3; // <unk>
}