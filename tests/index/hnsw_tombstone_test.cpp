#include <gtest/gtest.h>
#include "sonatadb/index/hnsw.hpp"
#include <filesystem>
#include <vector>

using namespace sonatadb::index;

class HnswTombstoneTest : public ::testing::Test {
protected:
    void SetUp() override {
        filepath = "test_hnsw_tombstone.bin";
        compact_filepath = "test_hnsw_compact.bin";
        if (std::filesystem::exists(filepath)) {
            std::filesystem::remove(filepath);
        }
        if (std::filesystem::exists(filepath + ".tombstones")) {
            std::filesystem::remove(filepath + ".tombstones");
        }
        if (std::filesystem::exists(compact_filepath)) {
            std::filesystem::remove(compact_filepath);
        }
    }

    void TearDown() override {
        if (std::filesystem::exists(filepath)) {
            std::filesystem::remove(filepath);
        }
        if (std::filesystem::exists(filepath + ".tombstones")) {
            std::filesystem::remove(filepath + ".tombstones");
        }
        if (std::filesystem::exists(compact_filepath)) {
            std::filesystem::remove(compact_filepath);
        }
        if (std::filesystem::exists(compact_filepath + ".tombstones")) {
            std::filesystem::remove(compact_filepath + ".tombstones");
        }
    }

    std::string filepath;
    std::string compact_filepath;
};

TEST_F(HnswTombstoneTest, MarkDeletedAndSearch) {
    HnswConfig config;
    config.dim = 2;
    config.max_elements = 100;
    config.M = 4;
    config.M0 = 8;

    HnswIndex index(filepath, config);

    std::vector<float> v0 = {0.0f, 0.0f};
    std::vector<float> v1 = {1.0f, 1.0f};
    std::vector<float> v2 = {0.1f, 0.1f}; // Very close to v0

    index.add(0, v0);
    index.add(1, v1);
    index.add(2, v2);

    auto results1 = index.search(v0, 3, 10);
    ASSERT_EQ(results1.size(), 3);
    EXPECT_EQ(results1[0].id, 0);
    EXPECT_EQ(results1[1].id, 2);
    EXPECT_EQ(results1[2].id, 1);

    // Mark node 0 as deleted
    index.markDeleted(0);
    EXPECT_TRUE(index.isDeleted(0));

    // Search again
    auto results2 = index.search(v0, 3, 10);
    ASSERT_EQ(results2.size(), 2);
    // Node 2 should now be the closest
    EXPECT_EQ(results2[0].id, 2);
    EXPECT_EQ(results2[1].id, 1);
}

TEST_F(HnswTombstoneTest, PersistTombstones) {
    HnswConfig config;
    config.dim = 2;
    config.max_elements = 100;

    {
        HnswIndex index(filepath, config);
        std::vector<float> v0 = {0.0f, 0.0f};
        index.add(0, v0);
        index.markDeleted(0);
        index.sync();
    }

    EXPECT_TRUE(std::filesystem::exists(filepath + ".tombstones"));

    {
        HnswIndex index2(filepath, config, true);
        EXPECT_TRUE(index2.isDeleted(0));
    }
}

TEST_F(HnswTombstoneTest, Compaction) {
    HnswConfig config;
    config.dim = 2;
    config.max_elements = 100;

    {
        HnswIndex index(filepath, config);
        std::vector<float> v0 = {0.0f, 0.0f};
        std::vector<float> v1 = {1.0f, 1.0f};
        std::vector<float> v2 = {2.0f, 2.0f};

        index.add(0, v0);
        index.add(1, v1); // We will delete this one
        index.add(2, v2);

        index.markDeleted(1);

        auto mapping = index.compact(compact_filepath);
        EXPECT_EQ(mapping.size(), 3);
        EXPECT_EQ(mapping[0], 0);
        EXPECT_EQ(mapping[1], UINT32_MAX);
        EXPECT_EQ(mapping[2], 1);
    }

    {
        HnswIndex compacted(compact_filepath, config, true);
        EXPECT_EQ(compacted.size(), 2);

        // Since we re-index sequentially, original id 0 -> new id 0, original id 2 -> new id 1
        std::vector<float> query = {2.0f, 2.0f};
        auto results = compacted.search(query, 1, 10);
        ASSERT_EQ(results.size(), 1);
        EXPECT_EQ(results[0].id, 1); // The node that used to be id 2 is now id 1
    }
}
