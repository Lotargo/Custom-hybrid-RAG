#include <benchmark/benchmark.h>
#include "sonatadb/text/bpe_tokenizer.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace sonatadb::text;

// Создадим фейковый tokenizer.json для бенчмарка
void setup_dummy_tokenizer_json(const std::string& path) {
    nlohmann::json j;
    j["model"]["vocab"] = {
        {"t", 0}, {"h", 1}, {"e", 2}, {"q", 3}, {"u", 4}, {"i", 5}, {"c", 6}, {"k", 7},
        {"b", 8}, {"r", 9}, {"o", 10}, {"w", 11}, {"n", 12}, {"f", 13}, {"x", 14},
        {"j", 15}, {"m", 16}, {"p", 17}, {"s", 18}, {"v", 19}, {"l", 20}, {"a", 21},
        {"z", 22}, {"y", 23}, {"d", 24}, {"g", 25}, {"th", 26}, {"the", 27},
        {"qu", 28}, {"qui", 29}, {"quic", 30}, {"quick", 31},
        {"br", 32}, {"bro", 33}, {"brow", 34}, {"brown", 35},
        {"fo", 36}, {"fox", 37},
        {"ju", 38}, {"jum", 39}, {"jump", 40}, {"jumps", 41},
        {"ov", 42}, {"ove", 43}, {"over", 44},
        {"la", 45}, {"laz", 46}, {"lazy", 47},
        {"do", 48}, {"dog", 49}
    };
    j["model"]["merges"] = {
        "t h", "th e", "q u", "qu i", "qui c", "quic k",
        "b r", "br o", "bro w", "brow n", "f o", "fo x",
        "j u", "ju m", "jum p", "jump s", "o v", "ov e", "ove r",
        "l a", "la z", "laz y", "d o", "do g"
    };

    std::ofstream out(path);
    out << j.dump(4);
}

static void BM_BpeTokenizeShortText(benchmark::State& state) {
    std::string test_file = "dummy_tokenizer.json";
    setup_dummy_tokenizer_json(test_file);

    BpeTokenizer tokenizer;
    tokenizer.load_from_json(test_file);

    std::string text = "the quick brown fox jumps over the lazy dog";

    for (auto _ : state) {
        std::vector<int> ids = tokenizer.encode(text);
        benchmark::DoNotOptimize(ids);
    }

    // Cleanup
    std::remove(test_file.c_str());
}

static void BM_BpeTokenizeLongText(benchmark::State& state) {
    std::string test_file = "dummy_tokenizer.json";
    setup_dummy_tokenizer_json(test_file);

    BpeTokenizer tokenizer;
    tokenizer.load_from_json(test_file);

    std::string base_text = "the quick brown fox jumps over the lazy dog ";
    std::string text;
    for (int i = 0; i < 100; ++i) {
        text += base_text;
    }

    for (auto _ : state) {
        std::vector<int> ids = tokenizer.encode(text);
        benchmark::DoNotOptimize(ids);
    }

    // Cleanup
    std::remove(test_file.c_str());
}

BENCHMARK(BM_BpeTokenizeShortText);
BENCHMARK(BM_BpeTokenizeLongText);
