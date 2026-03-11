#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace sonatadb::text {

// Pair hash for unordered_map
struct PairHash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        // A simple hash combine
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

class BpeTokenizer {
public:
    BpeTokenizer() = default;

    // Загрузка словаря и мержей из tokenizer.json
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

        // Загружаем merges
        if (j.contains("model") && j["model"].contains("merges")) {
            int merge_rank = 0;
            for (auto& merge : j["model"]["merges"]) {
                std::string merge_str = merge.get<std::string>();
                size_t space_pos = merge_str.find(' ');
                if (space_pos != std::string::npos) {
                    std::string first = merge_str.substr(0, space_pos);
                    std::string second = merge_str.substr(space_pos + 1);
                    merges_[{first, second}] = merge_rank++;
                }
            }
        } else {
             throw std::runtime_error("tokenizer.json missing 'model.merges'");
        }

        // Обработка unk_token (упрощенно)
        if (j.contains("model") && j["model"].contains("unk_token")) {
             unk_token_ = j["model"]["unk_token"].get<std::string>();
             if (vocab_.count(unk_token_)) {
                 unk_token_id_ = vocab_[unk_token_];
             }
        }
    }

    // Токенизация текста и возврат списка ID (простая реализация)
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        // Для BPE необходимо предварительно разбить текст на слова.
        // В рамках данной задачи напишем простой пре-токенайзер, разбивающий по пробелам и пунктуации
        // (в реальных системах используется regex типа GPT-2).
        std::vector<std::string> words = pre_tokenize(text);

        for (const std::string& word : words) {
            std::vector<int> word_ids = encode_word(word);
            ids.insert(ids.end(), word_ids.begin(), word_ids.end());
        }

        return ids;
    }

private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merges_;
    std::string unk_token_ = "<unk>";
    int unk_token_id_ = -1;

    // Простой пре-токенизатор (разбивка по пробелам, в продакшене используется Regex)
    std::vector<std::string> pre_tokenize(const std::string& text) const {
        std::vector<std::string> words;
        std::string current_word;

        // В BPE часто добавляется префикс, например 'Ġ' для пробелов.
        // Для упрощения мы просто разбиваем.
        bool is_first = true;

        for (char c : text) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!current_word.empty()) {
                    words.push_back(current_word);
                    current_word.clear();
                }
                is_first = false;
            } else {
                if (!is_first && current_word.empty()) {
                     // Упрощенный маркер начала слова, если не первое.
                     // current_word += "Ġ"; // Optional prefix for some tokenizers (like RoBERTa)
                }
                current_word += c;
            }
        }
        if (!current_word.empty()) {
            words.push_back(current_word);
        }
        return words;
    }

    // BPE алгоритм для одного слова
    std::vector<int> encode_word(const std::string& word) const {
        if (vocab_.count(word)) {
            return {vocab_.at(word)};
        }

        // Разбиваем слово на символы
        std::vector<std::string> splits;
        for (char c : word) {
            splits.push_back(std::string(1, c));
        }

        while (splits.size() > 1) {
            // Ищем лучшую пару для слияния
            int best_rank = -1;
            int best_pair_idx = -1;

            for (size_t i = 0; i < splits.size() - 1; ++i) {
                auto pair = std::make_pair(splits[i], splits[i+1]);
                if (merges_.count(pair)) {
                    int rank = merges_.at(pair);
                    if (best_rank == -1 || rank < best_rank) {
                        best_rank = rank;
                        best_pair_idx = static_cast<int>(i);
                    }
                }
            }

            // Если пар для слияния больше нет
            if (best_pair_idx == -1) {
                break;
            }

            // Выполняем слияние
            std::vector<std::string> new_splits;
            for (size_t i = 0; i < splits.size(); ++i) {
                if (static_cast<int>(i) == best_pair_idx) {
                    new_splits.push_back(splits[i] + splits[i+1]);
                    i++; // Пропускаем следующий токен, так как он слит
                } else {
                    new_splits.push_back(splits[i]);
                }
            }
            splits = std::move(new_splits);
        }

        // Преобразуем токены в ID
        std::vector<int> ids;
        for (const std::string& token : splits) {
            if (vocab_.count(token)) {
                ids.push_back(vocab_.at(token));
            } else if (unk_token_id_ != -1) {
                ids.push_back(unk_token_id_);
            }
        }

        return ids;
    }
};

} // namespace sonatadb::text
