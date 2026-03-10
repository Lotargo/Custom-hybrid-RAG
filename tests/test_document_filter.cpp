#include <gtest/gtest.h>
#include "rag_engine/index/document_filter.hpp"

using namespace rag_engine::index;

TEST(DocumentFilterTest, InitializationAndContains) {
    DocumentFilter filter;
    filter.add(10);
    filter.add(20);

    EXPECT_TRUE(filter.contains(10));
    EXPECT_TRUE(filter.contains(20));
    EXPECT_FALSE(filter.contains(30));
    EXPECT_EQ(filter.size(), 2);

    filter.remove(10);
    EXPECT_FALSE(filter.contains(10));
    EXPECT_EQ(filter.size(), 1);
}

TEST(DocumentFilterTest, VectorInitialization) {
    std::vector<uint32_t> ids = {1, 2, 3, 4, 5};
    DocumentFilter filter(ids);

    EXPECT_EQ(filter.size(), 5);
    EXPECT_TRUE(filter.contains(3));
    EXPECT_FALSE(filter.contains(6));
}

TEST(DocumentFilterTest, LogicalAnd) {
    DocumentFilter f1({1, 2, 3, 4, 5});
    DocumentFilter f2({4, 5, 6, 7});

    DocumentFilter result = f1 & f2;

    EXPECT_EQ(result.size(), 2);
    EXPECT_TRUE(result.contains(4));
    EXPECT_TRUE(result.contains(5));
    EXPECT_FALSE(result.contains(3));
    EXPECT_FALSE(result.contains(6));
}

TEST(DocumentFilterTest, LogicalOr) {
    DocumentFilter f1({1, 2});
    DocumentFilter f2({2, 3});

    DocumentFilter result = f1 | f2;

    EXPECT_EQ(result.size(), 3);
    EXPECT_TRUE(result.contains(1));
    EXPECT_TRUE(result.contains(2));
    EXPECT_TRUE(result.contains(3));
    EXPECT_FALSE(result.contains(4));
}

TEST(DocumentFilterTest, LogicalAndNot) {
    DocumentFilter f1({1, 2, 3, 4});
    DocumentFilter f2({3, 4, 5});

    DocumentFilter result = f1 - f2;

    EXPECT_EQ(result.size(), 2);
    EXPECT_TRUE(result.contains(1));
    EXPECT_TRUE(result.contains(2));
    EXPECT_FALSE(result.contains(3));
    EXPECT_FALSE(result.contains(4));
    EXPECT_FALSE(result.contains(5));
}
