#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <cctype>
#include <algorithm>
#include <stdexcept>
#include <libstemmer.h>

namespace rag_engine::text {

class Tokenizer {
public:
    Tokenizer(const std::string& language = "english") {
        stemmer_ = sb_stemmer_new(language.c_str(), "UTF_8");
        if (!stemmer_) {
            throw std::runtime_error("Failed to initialize stemmer for language: " + language);
        }

        // Add some basic stop words (in a real system, this would be loaded from a file or extensive list)
        if (language == "english") {
            stop_words_ = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "is", "are", "was", "were", "it", "this", "that"};
        }
    }

    ~Tokenizer() {
        if (stemmer_) {
            sb_stemmer_delete(stemmer_);
        }
    }

    // Disable copy to avoid double free of stemmer
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::string current_token;

        for (char c : text) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                current_token += std::tolower(static_cast<unsigned char>(c));
            } else if (!current_token.empty()) {
                process_and_add_token(current_token, tokens);
                current_token.clear();
            }
        }

        if (!current_token.empty()) {
            process_and_add_token(current_token, tokens);
        }

        return tokens;
    }

private:
    void process_and_add_token(std::string& token, std::vector<std::string>& tokens) {
        if (stop_words_.count(token) > 0) {
            return;
        }

        const sb_symbol* stemmed = sb_stemmer_stem(stemmer_, reinterpret_cast<const sb_symbol*>(token.c_str()), token.length());
        if (stemmed) {
             tokens.emplace_back(reinterpret_cast<const char*>(stemmed), sb_stemmer_length(stemmer_));
        } else {
             tokens.push_back(token); // Fallback
        }
    }

    struct sb_stemmer* stemmer_;
    std::unordered_set<std::string> stop_words_;
};

} // namespace rag_engine::text
