#include <gtest/gtest.h>
#include "rag_engine/index/bloom_filter.hpp"

using namespace rag_engine::index;

TEST(BloomFilterTest, Initialization) {
    BloomFilter bf(1000, 0.01);
    EXPECT_GT(bf.bit_size(), 0);
}

TEST(BloomFilterTest, AddAndContainsInt) {
    BloomFilter bf(100, 0.01);

    bf.add(10);
    bf.add(42);
    bf.add(100);

    EXPECT_TRUE(bf.contains(10));
    EXPECT_TRUE(bf.contains(42));
    EXPECT_TRUE(bf.contains(100));

    EXPECT_FALSE(bf.contains(11));
    EXPECT_FALSE(bf.contains(43));
    EXPECT_FALSE(bf.contains(99));
}

TEST(BloomFilterTest, AddAndContainsString) {
    BloomFilter bf(100, 0.01);

    bf.add("hello");
    bf.add("world");

    EXPECT_TRUE(bf.contains("hello"));
    EXPECT_TRUE(bf.contains("world"));

    EXPECT_FALSE(bf.contains("hell"));
    EXPECT_FALSE(bf.contains("worlds"));
}

TEST(BloomFilterTest, FalsePositiveRate) {
    size_t expected = 10000;
    double fp_rate = 0.01;
    BloomFilter bf(expected, fp_rate);

    for (size_t i = 0; i < expected; i++) {
        bf.add(static_cast<uint32_t>(i));
    }

    size_t false_positives = 0;
    size_t checks = 10000;
    for (size_t i = expected; i < expected + checks; i++) {
        if (bf.contains(static_cast<uint32_t>(i))) {
            false_positives++;
        }
    }

    double actual_fp_rate = static_cast<double>(false_positives) / checks;
    // The actual FP rate should be close to the target FP rate (0.01)
    EXPECT_LT(actual_fp_rate, fp_rate * 2.0); // Allow some margin
}
