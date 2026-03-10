#include <gtest/gtest.h>
#include "rag_engine/text/tokenizer.hpp"
#include <vector>
#include <string>

TEST(TokenizerTest, BasicTokenization) {
    rag_engine::text::Tokenizer tokenizer("english");

    std::string text = "The Quick Brown Fox jumps over the lazy dog! 123";
    std::vector<std::string> expected = {"quick", "brown", "fox", "jump", "over", "lazi", "dog", "123"};

    std::vector<std::string> tokens = tokenizer.tokenize(text);

    EXPECT_EQ(tokens, expected);
}

TEST(TokenizerTest, EmptyString) {
    rag_engine::text::Tokenizer tokenizer("english");
    std::vector<std::string> tokens = tokenizer.tokenize("");
    EXPECT_TRUE(tokens.empty());
}
