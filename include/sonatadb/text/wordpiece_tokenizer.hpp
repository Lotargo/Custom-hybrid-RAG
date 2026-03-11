#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <algorithm>

namespace sonatadb::text {

class WordPieceTokenizer {
public:
    WordPieceTokenizer() = default;

    // Загрузка словаря из tokenizer.json
    void load_from_json(const std::string& filepath) {
        std::ifstream f(filepath);
        if (!f.is_open()) {
            throw std::runtime_error("Failed to open tokenizer.json file: " + filepath);
        }

        nlohmann::json j;
        try {
            f >> j;
        } catch (const nlohmann::json::parse_error& e) {
            throw std::runtime_error("Failed to parse tokenizer.json: " + std::string(e.what()));
        }

        // Загружаем vocabulary
        if (j.contains("model") && j["model"].contains("vocab")) {
            for (auto& [key, value] : j["model"]["vocab"].items()) {
                vocab_[key] = value.get<int>();
            }
        } else {
             throw std::runtime_error("tokenizer.json missing 'model.vocab'");
        }

        // Обработка unk_token
        if (j.contains("model") && j["model"].contains("unk_token")) {
             unk_token_ = j["model"]["unk_token"].get<std::string>();
             if (vocab_.count(unk_token_)) {
                 unk_token_id_ = vocab_[unk_token_];
             }
        }

        // Читаем макс. длину слова из настроек, если есть
        if (j.contains("model") && j["model"].contains("max_input_chars_per_word")) {
            max_input_chars_per_word_ = j["model"]["max_input_chars_per_word"].get<int>();
        }
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        std::vector<std::string> words = pre_tokenize(text);

        for (const std::string& word : words) {
            std::vector<int> word_ids = encode_word(word);
            ids.insert(ids.end(), word_ids.begin(), word_ids.end());
        }

        return ids;
    }

private:
    std::unordered_map<std::string, int> vocab_;
    std::string unk_token_ = "[UNK]";
    int unk_token_id_ = -1;
    size_t max_input_chars_per_word_ = 100;
    std::string continuing_subword_prefix_ = "##";

    // Простой пре-токенизатор: приводим к нижнему регистру (по желанию,
    // но WordPiece обычно работает с lowercase BERT) и разбиваем по пробелам.
    std::vector<std::string> pre_tokenize(const std::string& text) const {
        std::vector<std::string> words;
        std::string current_word;

        for (char c : text) {
            // Для BERT/WordPiece часто делается базовая очистка и приведение к нижнему регистру.
            // Здесь мы сделаем упрощенную версию.
            if (std::isspace(static_cast<unsigned char>(c)) || std::ispunct(static_cast<unsigned char>(c))) {
                if (!current_word.empty()) {
                    words.push_back(current_word);
                    current_word.clear();
                }
                if (std::ispunct(static_cast<unsigned char>(c))) {
                    words.push_back(std::string(1, static_cast<char>(std::tolower(static_cast<unsigned char>(c)))));
                }
            } else {
                current_word += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
        }
        if (!current_word.empty()) {
            words.push_back(current_word);
        }
        return words;
    }

    // Алгоритм "Longest-Match-First" (MaxMatch) для WordPiece
    std::vector<int> encode_word(const std::string& word) const {
        if (word.length() > max_input_chars_per_word_) {
            return {unk_token_id_};
        }

        std::vector<int> ids;
        bool is_bad = false;
        size_t start = 0;

        while (start < word.length()) {
            size_t end = word.length();
            std::string cur_substr;
            int cur_id = -1;

            while (start < end) {
                std::string sub_str = word.substr(start, end - start);
                if (start > 0) {
                    sub_str = continuing_subword_prefix_ + sub_str;
                }

                if (vocab_.count(sub_str)) {
                    cur_substr = sub_str;
                    cur_id = vocab_.at(sub_str);
                    break;
                }
                end--;
            }

            if (cur_id == -1) {
                is_bad = true;
                break;
            }

            ids.push_back(cur_id);
            start = end;
        }

        if (is_bad) {
            return {unk_token_id_};
        }

        return ids;
    }
};

} // namespace sonatadb::text
