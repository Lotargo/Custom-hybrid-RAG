#include <gtest/gtest.h>
#include "rag_engine/index/hnsw.hpp"
#include <vector>
#include <random>
#include <filesystem>
#include <thread>

using namespace rag_engine::index;

class HnswIndexTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_file = "test_hnsw.bin";
        if (std::filesystem::exists(test_file)) {
            std::filesystem::remove(test_file);
        }
    }

    void TearDown() override {
        if (std::filesystem::exists(test_file)) {
            std::filesystem::remove(test_file);
        }
    }

    std::string test_file;
    HnswConfig config{128, 1000, 16, 32, 100, 42};
};

TEST_F(HnswIndexTest, BasicInsertAndSearch) {
    HnswIndex index(test_file, config);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    // Insert 100 elements
    std::vector<std::vector<float>> data(100, std::vector<float>(config.dim));
    for (size_t i = 0; i < 100; ++i) {
        for (size_t d = 0; d < config.dim; ++d) {
            data[i][d] = dist(gen);
        }
        index.add(i, data[i]);
    }

    EXPECT_EQ(index.size(), 100);

    // Search for the exact 0-th element
    auto results = index.search(data[0], 5, 50);
    EXPECT_FALSE(results.empty());
    EXPECT_EQ(results[0].id, 0); // Self should be closest
    EXPECT_NEAR(results[0].distance, 0.0f, 1e-5f);
}

TEST_F(HnswIndexTest, Persistence) {
    std::vector<std::vector<float>> data(50, std::vector<float>(config.dim));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    {
        HnswIndex index(test_file, config);
        for (size_t i = 0; i < 50; ++i) {
            for (size_t d = 0; d < config.dim; ++d) {
                data[i][d] = dist(gen);
            }
            index.add(i, data[i]);
        }
        index.sync();
    } // Index goes out of scope and is synced

    // Re-open readonly
    {
        HnswIndex index2(test_file, config, true);
        EXPECT_EQ(index2.size(), 50);

        auto results = index2.search(data[42], 5, 50);
        EXPECT_EQ(results[0].id, 42);
    }
}

TEST_F(HnswIndexTest, MultithreadedInsert) {
    HnswIndex index(test_file, config);

    std::vector<std::thread> threads;
    size_t num_threads = 4;
    size_t elements_per_thread = 200;

    std::vector<std::vector<float>> data(num_threads * elements_per_thread, std::vector<float>(config.dim));
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (auto& vec : data) {
        for (auto& val : vec) {
            val = dist(gen);
        }
    }

    auto worker = [&](size_t thread_idx) {
        for (size_t i = 0; i < elements_per_thread; ++i) {
            size_t id = thread_idx * elements_per_thread + i;
            index.add(id, data[id]);
        }
    };

    for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(index.size(), num_threads * elements_per_thread);

    auto results = index.search(data[10], 1, 50);
    EXPECT_EQ(results[0].id, 10);
}
