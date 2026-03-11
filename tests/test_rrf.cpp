#include <gtest/gtest.h>
#include "sonatadb/index/rrf.hpp"

using namespace sonatadb::index;

TEST(RRFTest, FuseResults) {
    // Two simple lists: Vector and Lexical
    std::vector<SearchResult> hnsw_results = {
        {1, 0.1}, // best
        {2, 0.3},
        {3, 0.5}
    };

    std::vector<SearchResult> bm25_results = {
        {3, 10.5}, // best
        {1, 5.0},
        {4, 2.0}
    };

    std::vector<std::vector<SearchResult>> lists = {hnsw_results, bm25_results};
    auto fused = reciprocal_rank_fusion(lists, 60, 100);

    // Results logic with k=60:
    // doc 1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639 + 0.01612 = 0.0325
    // doc 3: 1/(60+3) + 1/(60+1) = 1/63 + 1/61 = 0.01587 + 0.01639 = 0.0322
    // doc 2: 1/(60+2) = 1/62 = 0.01612
    // doc 4: 1/(60+3) = 1/63 = 0.01587
    // Ordered expected: doc 1, doc 3, doc 2, doc 4

    ASSERT_EQ(fused.size(), 4);
    EXPECT_EQ(fused[0].id, 1);
    EXPECT_EQ(fused[1].id, 3);
    EXPECT_EQ(fused[2].id, 2);
    EXPECT_EQ(fused[3].id, 4);

    EXPECT_NEAR(fused[0].distance, 1.0f/61.0f + 1.0f/62.0f, 1e-5);
    EXPECT_NEAR(fused[1].distance, 1.0f/63.0f + 1.0f/61.0f, 1e-5);
}

TEST(RRFTest, EmptyLists) {
    std::vector<std::vector<SearchResult>> lists = {};
    auto fused = reciprocal_rank_fusion(lists, 60, 100);
    EXPECT_TRUE(fused.empty());

    lists = {{}, {}};
    fused = reciprocal_rank_fusion(lists, 60, 100);
    EXPECT_TRUE(fused.empty());
}

TEST(RRFTest, TopNLimit) {
    std::vector<SearchResult> list1;
    for (uint32_t i = 0; i < 200; ++i) {
        list1.push_back({i, static_cast<float>(i)});
    }

    std::vector<std::vector<SearchResult>> lists = {list1};
    auto fused = reciprocal_rank_fusion(lists, 60, 50);

    EXPECT_EQ(fused.size(), 50);
}
