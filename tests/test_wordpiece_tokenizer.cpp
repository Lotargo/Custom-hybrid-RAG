#include <gtest/gtest.h>
#include "sonatadb/text/wordpiece_tokenizer.hpp"
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>

using namespace sonatadb::text;

class WordPieceTokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        nlohmann::json j;
        j["model"]["vocab"] = {
            {"[UNK]", 0},
            {"un", 1},
            {"##aff", 2},
            {"##able", 3},
            {"unaffable", 4}, // Не будет использоваться, если мы ищем "un" + "##aff" (если оно не целиком)
            {"test", 5},
            {"##ing", 6},
            {"##s", 7}
        };
        j["model"]["unk_token"] = "[UNK]";
        j["model"]["max_input_chars_per_word"] = 100;

        std::ofstream out("test_wp_tokenizer.json");
        out << j.dump(4);
    }

    void TearDown() override {
        std::remove("test_wp_tokenizer.json");
    }
};

TEST_F(WordPieceTokenizerTest, SubwordTokenization) {
    WordPieceTokenizer tokenizer;
    ASSERT_NO_THROW(tokenizer.load_from_json("test_wp_tokenizer.json"));

    // Слово "unaffable". У нас есть "unaffable" целиком в словаре.
    // Но давайте протестируем "unaffables" (нет целиком).
    // Алгоритм попытается:
    // "unaffables" -> нет
    // "unaffable" -> да (ID: 4)
    // Остаток: "s". Попытается найти "##s" -> да (ID: 7)
    std::vector<int> expected1 = {4, 7};
    std::vector<int> ids1 = tokenizer.encode("unaffables");
    EXPECT_EQ(ids1, expected1);

    // "testing" -> "test" (5), "##ing" (6)
    std::vector<int> expected2 = {5, 6};
    std::vector<int> ids2 = tokenizer.encode("testing");
    EXPECT_EQ(ids2, expected2);
}

TEST_F(WordPieceTokenizerTest, UnknownToken) {
    WordPieceTokenizer tokenizer;
    ASSERT_NO_THROW(tokenizer.load_from_json("test_wp_tokenizer.json"));

    // "unknown_word". Алгоритм не найдет совпадений и должен вернуть [UNK] (0)
    std::vector<int> expected = {0};
    std::vector<int> ids = tokenizer.encode("unknown");

    EXPECT_EQ(ids, expected);
}
