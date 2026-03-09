#include <gtest/gtest.h>
#include "rag_engine/storage/metadata_store.hpp"
#include <filesystem>
#include <string>

namespace {
    const std::string TEST_DIR = "test_metadata_store_dir";
}

class MetadataStoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::filesystem::remove_all(TEST_DIR);
    }

    void TearDown() override {
        std::filesystem::remove_all(TEST_DIR);
    }
};

TEST_F(MetadataStoreTest, BasicPutGetRemove) {
    using namespace rag_engine::storage;

    MetadataStore store(TEST_DIR);

    // Test Put and Get
    store.put("doc_1", "meta_1");
    store.put("doc_2", "meta_2");

    auto val1 = store.get("doc_1");
    ASSERT_TRUE(val1.has_value());
    EXPECT_EQ(val1.value(), "meta_1");

    auto val2 = store.get("doc_2");
    ASSERT_TRUE(val2.has_value());
    EXPECT_EQ(val2.value(), "meta_2");

    // Test non-existent key
    auto val3 = store.get("doc_3");
    EXPECT_FALSE(val3.has_value());

    // Test Remove
    store.remove("doc_1");
    auto val1_after_remove = store.get("doc_1");
    EXPECT_FALSE(val1_after_remove.has_value());
}

TEST_F(MetadataStoreTest, SyncAndReopen) {
    using namespace rag_engine::storage;

    {
        MetadataStore store(TEST_DIR);
        store.put("sync_key", "sync_value");
        store.sync();
    } // Store is closed here

    // Reopen
    MetadataStore store2(TEST_DIR);
    auto val = store2.get("sync_key");
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(val.value(), "sync_value");
}
