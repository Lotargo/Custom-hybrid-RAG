#include <gtest/gtest.h>
#include "sonatadb/c_api/sonatadb.h"
#include <vector>
#include <string>

class CApiTest : public ::testing::Test {
protected:
    void SetUp() override {
        filepath_ = "test_c_api_hnsw.bin";
        std::remove(filepath_.c_str());
    }

    void TearDown() override {
        std::remove(filepath_.c_str());
    }

    std::string filepath_;
};

TEST_F(CApiTest, LifecycleAndOperations) {
    sonatadb_hnsw_t* index = nullptr;

    // Create
    int res = sonatadb_hnsw_create(filepath_.c_str(), 128, 100, 16, 200, false, &index);
    ASSERT_EQ(res, 0);
    ASSERT_NE(index, nullptr);

    // Insert
    std::vector<float> vec1(128, 0.1f);
    res = sonatadb_hnsw_insert(index, vec1.data(), 1);
    ASSERT_EQ(res, 0);

    std::vector<float> vec2(128, 0.2f);
    res = sonatadb_hnsw_insert(index, vec2.data(), 2);
    ASSERT_EQ(res, 0);

    // Search
    std::vector<sonatadb_search_result_t> results(10);
    size_t num_results = 0;

    // Exact match for vec1
    res = sonatadb_hnsw_search(index, vec1.data(), 10, 50, results.data(), &num_results);
    ASSERT_EQ(res, 0);
    ASSERT_GT(num_results, 0);
    EXPECT_EQ(results[0].id, 1); // Closest should be vec1

    // Free
    sonatadb_hnsw_free(index);
}

TEST_F(CApiTest, ErrorHandling) {
    sonatadb_hnsw_t* index = nullptr;

    // Invalid argument test
    int res = sonatadb_hnsw_create(nullptr, 128, 100, 16, 200, false, &index);
    ASSERT_NE(res, 0);

    char error_msg[256];
    size_t err_len = sonatadb_get_last_error_message(error_msg, sizeof(error_msg));
    ASSERT_GT(err_len, 0);
    EXPECT_STREQ(error_msg, "Invalid arguments: filepath or index_out is null");
}
