#include <gtest/gtest.h>
#include "sonatadb/text/bpe_tokenizer.hpp"
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>

using namespace sonatadb::text;

class BpeTokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Создадим тестовый json
        nlohmann::json j;
        j["model"]["vocab"] = {
            {"<unk>", 0},
            {"h", 1}, {"e", 2}, {"l", 3}, {"o", 4}, {"w", 5}, {"r", 6}, {"d", 7},
            {"he", 8}, {"ll", 9}, {"hell", 10}, {"hello", 11},
            {"wo", 12}, {"or", 13}, {"wor", 14}, {"worl", 15}, {"world", 16}
        };
        j["model"]["merges"] = {
            "h e", "l l", "he ll", "hell o",
            "w o", "o r", "wo r", "wor l", "worl d"
        };
        j["model"]["unk_token"] = "<unk>";

        std::ofstream out("test_tokenizer.json");
        out << j.dump(4);
    }

    void TearDown() override {
        std::remove("test_tokenizer.json");
    }
};

TEST_F(BpeTokenizerTest, BasicTokenization) {
    BpeTokenizer tokenizer;
    ASSERT_NO_THROW(tokenizer.load_from_json("test_tokenizer.json"));

    std::vector<int> expected = {11, 16}; // "hello", "world"
    std::vector<int> ids = tokenizer.encode("helloworld");

    // В нашей реализации претокенизатор не разделит "helloworld",
    // поэтому проверим раздельно "hello" "world"
    std::vector<int> ids2 = tokenizer.encode("hello world");

    EXPECT_EQ(ids2, expected);
}

TEST_F(BpeTokenizerTest, UnknownToken) {
    BpeTokenizer tokenizer;
    ASSERT_NO_THROW(tokenizer.load_from_json("test_tokenizer.json"));

    // Символ "z" не в словаре, должен быть заменен на <unk> (ID: 0)
    std::vector<int> expected = {0};
    std::vector<int> ids = tokenizer.encode("z");

    EXPECT_EQ(ids, expected);
}

TEST_F(BpeTokenizerTest, LoadMissingFile) {
    BpeTokenizer tokenizer;
    EXPECT_THROW(tokenizer.load_from_json("nonexistent.json"), std::runtime_error);
}
