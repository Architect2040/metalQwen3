#include "Qwen3cTokenizer.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdio>

// Include relevant parts from original qwen3.c
// We'll embed the original tokenizer structures and functions

extern "C" {

// Original qwen3.c tokenizer structures
typedef struct {
    char **vocab;
    float *merge_scores;
    int vocab_size;
    unsigned int max_token_length;
    unsigned int bos_token_id;
    unsigned int eos_token_id;
    char prompt_template[1024];
    char system_prompt_template[1024];
} Qwen3cTokenizer_Original;

// Original qwen3.c functions (simplified versions)
void load_prompt_template(char *checkpoint_path, char *out_template, int with_system_prompt, int enable_thinking) {
    char prompt_path[1024];

    strcpy(prompt_path, checkpoint_path);
    if (with_system_prompt)
        strcat(prompt_path, enable_thinking ? ".template.with-system-and-thinking" : ".template.with-system");
    else
        strcat(prompt_path, enable_thinking ? ".template.with-thinking" : ".template");

    memset(out_template, 0, 1024);
    FILE *file = fopen(prompt_path, "rb");
    if (!file) {
        std::cerr << "Warning: Couldn't load prompt template " << prompt_path
                  << ", using default" << std::endl;
        // Use a default template if file not found
        if (with_system_prompt) {
            strcpy(out_template, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n");
        } else {
            strcpy(out_template, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n");
        }
        return;
    }
    fread(out_template, 1024, 1, file);
    fclose(file);
}

void build_qwen3c_tokenizer(Qwen3cTokenizer_Original *t, char *checkpoint_path, int vocab_size, int enable_thinking) {
    char tokenizer_path[1024];

    strcpy(tokenizer_path, checkpoint_path);
    strcat(tokenizer_path, ".tokenizer");

    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->merge_scores = (float *)malloc(vocab_size * sizeof(float));

    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) {
        std::cerr << "Warning: Couldn't load tokenizer model " << tokenizer_path
                  << ", creating minimal tokenizer" << std::endl;

        // Create a minimal tokenizer for testing
        fread(&t->max_token_length, sizeof(int), 1, file);
        t->max_token_length = 64;
        t->bos_token_id = 1;
        t->eos_token_id = 2;

        // Allocate minimal vocab
        for (int i = 0; i < vocab_size; i++) {
            t->vocab[i] = (char*)malloc(32);
            snprintf(t->vocab[i], 32, "<token_%d>", i);
            t->merge_scores[i] = 0.0f;
        }

        // Set some basic tokens
        if (vocab_size > 0) strcpy(t->vocab[0], "<pad>");
        if (vocab_size > 1) strcpy(t->vocab[1], "<bos>");
        if (vocab_size > 2) strcpy(t->vocab[2], "<eos>");
        if (vocab_size > 3) strcpy(t->vocab[3], "<unk>");

        load_prompt_template(checkpoint_path, t->prompt_template, 0, enable_thinking);
        load_prompt_template(checkpoint_path, t->system_prompt_template, 1, enable_thinking);
        return;
    }

    fread(&t->max_token_length, sizeof(int), 1, file);
    fread(&t->bos_token_id, sizeof(int), 1, file);
    fread(&t->eos_token_id, sizeof(int), 1, file);

    for (int i = 0; i < vocab_size; i++) {
        unsigned char len;
        fread(&len, sizeof(unsigned char), 1, file);
        t->vocab[i] = (char *)malloc(len + 1);
        fread(t->vocab[i], len, 1, file);
        t->vocab[i][len] = '\0'; // add the string terminating token
        fread(&t->merge_scores[i], sizeof(float), 1, file);
    }
    fclose(file);

    // Load prompt templates
    load_prompt_template(checkpoint_path, t->prompt_template, 0, enable_thinking);
    load_prompt_template(checkpoint_path, t->system_prompt_template, 1, enable_thinking);
}

void free_qwen3c_tokenizer(Qwen3cTokenizer_Original *t) {
    if (!t) return;

    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->merge_scores);
}

char *decode_qwen3c(Qwen3cTokenizer_Original *t, int token) {
    if (token < 0 || token >= t->vocab_size) {
        return t->vocab[3]; // return <unk> for invalid tokens
    }
    return t->vocab[token];
}

int str_lookup_qwen3c(char *str, char **vocab, int vocab_size) {
    // find a match for str in vocab, return its index or -1 if not found
    for (int i = 0; i < vocab_size; i++)
        if (!strcmp(str, vocab[i]))
            return i;
    return -1;
}

void encode_qwen3c(Qwen3cTokenizer_Original *t, char *text, int *tokens, int *n_tokens) {
    // Simplified encoding for demo - in reality this would be the full BPE algorithm
    // For now, do basic word-level tokenization

    *n_tokens = 0;
    if (!text || strlen(text) == 0) return;

    // Simple whitespace tokenization as fallback
    char *text_copy = strdup(text);
    char *token = strtok(text_copy, " \t\n\r");

    while (token && *n_tokens < 2048) { // max tokens limit
        int token_id = str_lookup_qwen3c(token, t->vocab, t->vocab_size);
        if (token_id == -1) {
            token_id = 3; // <unk> token
        }
        tokens[(*n_tokens)++] = token_id;
        token = strtok(NULL, " \t\n\r");
    }

    free(text_copy);
}

} // extern "C"

// Internal implementation structure
struct Qwen3cTokenizer_Internal {
    Qwen3cTokenizer_Original tokenizer;
    std::string checkpoint_path;
};

Qwen3cTokenizer::Qwen3cTokenizer() : impl(nullptr), initialized(false) {
    impl = new Qwen3cTokenizer_Internal();
    memset(&impl->tokenizer, 0, sizeof(impl->tokenizer));
}

Qwen3cTokenizer::~Qwen3cTokenizer() {
    if (impl) {
        if (initialized) {
            free_qwen3c_tokenizer(&impl->tokenizer);
        }
        delete impl;
    }
}

bool Qwen3cTokenizer::initialize(const std::string& checkpoint_path, int vocab_size, bool enable_thinking) {
    if (!impl) return false;

    impl->checkpoint_path = checkpoint_path;

    // Use the original qwen3.c tokenizer build function
    char* path_str = const_cast<char*>(checkpoint_path.c_str());
    build_qwen3c_tokenizer(&impl->tokenizer, path_str, vocab_size, enable_thinking ? 1 : 0);

    initialized = true;
    std::cout << "Qwen3c tokenizer initialized with vocab_size=" << vocab_size << std::endl;
    return true;
}

std::vector<int> Qwen3cTokenizer::encode(const std::string& text) const {
    if (!initialized || !impl) return {};

    // Allocate buffer for tokens
    int tokens[2048];
    int n_tokens = 0;

    char* text_str = const_cast<char*>(text.c_str());
    encode_qwen3c(&impl->tokenizer, text_str, tokens, &n_tokens);

    std::vector<int> result(tokens, tokens + n_tokens);
    return result;
}

std::string Qwen3cTokenizer::decode(const std::vector<int>& tokens) const {
    if (!initialized || !impl) return "";

    std::string result;
    for (int token : tokens) {
        char* token_str = decode_qwen3c(&impl->tokenizer, token);
        if (token_str) {
            result += token_str;
        }
    }
    return result;
}

std::string Qwen3cTokenizer::decode(int token) const {
    if (!initialized || !impl) return "";

    char* token_str = decode_qwen3c(&impl->tokenizer, token);
    return token_str ? std::string(token_str) : "";
}

int Qwen3cTokenizer::getVocabSize() const {
    return impl ? impl->tokenizer.vocab_size : 0;
}

int Qwen3cTokenizer::getBOSToken() const {
    return impl ? impl->tokenizer.bos_token_id : 1;
}

int Qwen3cTokenizer::getEOSToken() const {
    return impl ? impl->tokenizer.eos_token_id : 2;
}

std::string Qwen3cTokenizer::applyChatTemplate(const std::vector<std::pair<std::string, std::string>>& messages) const {
    if (!initialized || !impl || messages.empty()) return "";

    std::string result;

    // Use the loaded prompt templates from original qwen3.c
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& [role, content] = messages[i];

        if (role == "system") {
            // System messages use system template
            char formatted[4096];
            snprintf(formatted, sizeof(formatted), impl->tokenizer.system_prompt_template, content.c_str(), "");
            result += formatted;
        } else if (role == "user") {
            char formatted[4096];
            if (i == 0) {
                snprintf(formatted, sizeof(formatted), impl->tokenizer.prompt_template, content.c_str());
            } else {
                snprintf(formatted, sizeof(formatted), "<|im_start|>user\n%s<|im_end|>\n", content.c_str());
            }
            result += formatted;
        } else if (role == "assistant") {
            result += "<|im_start|>assistant\n" + content + "<|im_end|>\n";
        }
    }

    // Add assistant start for generation
    result += "<|im_start|>assistant\n";

    return result;
}

size_t Qwen3cTokenizer::countTokens(const std::string& text) const {
    return encode(text).size();
}