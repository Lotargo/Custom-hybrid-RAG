#include <gtest/gtest.h>
#include "rag_engine/index/hnsw.hpp"
#include "../sift_parser.hpp"
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <iostream>

using namespace rag_engine::index;
using namespace rag_engine::test_utils;

class HnswRecallTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Path to the downloaded siftsmall dataset
        dataset_dir = "../assets/datasets/siftsmall";
        index_file = "sift_test_index.bin";

        // Remove old index file if it exists
        if (std::filesystem::exists(index_file)) {
            std::filesystem::remove(index_file);
        }
        if (std::filesystem::exists(index_file + ".tombstones")) {
            std::filesystem::remove(index_file + ".tombstones");
        }
    }

    void TearDown() override {
        // Cleanup index files
        if (std::filesystem::exists(index_file)) {
            std::filesystem::remove(index_file);
        }
        if (std::filesystem::exists(index_file + ".tombstones")) {
            std::filesystem::remove(index_file + ".tombstones");
        }
    }

    std::string dataset_dir;
    std::string index_file;
};

// Calculate Recall@K
double calculate_recall(const std::vector<std::vector<SearchResult>>& results,
                        const std::vector<std::vector<int>>& ground_truth,
                        size_t k) {
    size_t total_queries = results.size();
    size_t hits = 0;

    for (size_t i = 0; i < total_queries; ++i) {
        const auto& result_cands = results[i];
        const auto& truth_cands = ground_truth[i];

        // We check if any of the top-K ground truth IDs are in the top-K results
        // SIFT 1M / SIFTSmall uses Recall@1 (is the actual nearest neighbor in the top-K results)
        // Let's compute Recall@K (Intersection size / K) or Recall@1 for standard ANN benchmarks.

        // For standard ANN benchmarks (like ann-benchmarks), Recall is defined as:
        // Number of returned results that are in the true top-K / K
        size_t local_hits = 0;
        size_t current_k = std::min(k, result_cands.size());

        for (size_t j = 0; j < current_k; ++j) {
            uint32_t res_id = result_cands[j].id;
            // Check if res_id is in the top-K of ground truth
            for (size_t t = 0; t < k; ++t) {
                if (res_id == static_cast<uint32_t>(truth_cands[t])) {
                    local_hits++;
                    break;
                }
            }
        }
        hits += local_hits;
    }

    return static_cast<double>(hits) / (total_queries * k);
}

// Calculate Recall@1@K: Is the *true nearest neighbor* (truth[0]) present in the top-K returned results?
// This is another very common metric for ANN (e.g. SIFT1M paper)
double calculate_recall_1_at_k(const std::vector<std::vector<SearchResult>>& results,
                               const std::vector<std::vector<int>>& ground_truth,
                               size_t k) {
    size_t total_queries = results.size();
    size_t hits = 0;

    for (size_t i = 0; i < total_queries; ++i) {
        const auto& result_cands = results[i];
        int true_nn = ground_truth[i][0]; // The actual closest neighbor

        size_t current_k = std::min(k, result_cands.size());
        for (size_t j = 0; j < current_k; ++j) {
            if (result_cands[j].id == static_cast<uint32_t>(true_nn)) {
                hits++;
                break;
            }
        }
    }

    return static_cast<double>(hits) / total_queries;
}

TEST_F(HnswRecallTest, SiftSmallRecallGreaterThan95) {
    // Check if dataset exists
    std::string base_file = dataset_dir + "/siftsmall_base.fvecs";
    std::string query_file = dataset_dir + "/siftsmall_query.fvecs";
    std::string truth_file = dataset_dir + "/siftsmall_groundtruth.ivecs";

    if (!std::filesystem::exists(base_file)) {
        GTEST_SKIP() << "SIFTSmall dataset not found at " << dataset_dir << ". Skipping test.";
    }

    // Load data
    std::cout << "Loading dataset..." << std::endl;
    auto base_vectors = SiftParser::read_fvecs(base_file);
    auto query_vectors = SiftParser::read_fvecs(query_file);
    auto ground_truth = SiftParser::read_ivecs(truth_file);

    ASSERT_EQ(base_vectors.size(), 10000);
    ASSERT_EQ(query_vectors.size(), 100);
    ASSERT_EQ(ground_truth.size(), 100);

    size_t dim = base_vectors[0].size();
    ASSERT_EQ(dim, 128);

    // Initialize HNSW
    HnswConfig config;
    config.dim = dim;
    config.max_elements = base_vectors.size();
    config.M = 16;
    config.M0 = 32;
    config.ef_construction = 100;

    std::cout << "Building HNSW index..." << std::endl;
    HnswIndex index(index_file, config, false);

    for (size_t i = 0; i < base_vectors.size(); ++i) {
        index.add(i, base_vectors[i]);
    }
    index.sync();

    // Query
    std::cout << "Querying..." << std::endl;
    size_t k = 10;
    size_t ef_search = 50;
    std::vector<std::vector<SearchResult>> all_results;
    all_results.reserve(query_vectors.size());

    for (const auto& q : query_vectors) {
        all_results.push_back(index.search(q, k, ef_search));
    }

    // Evaluate
    double recall_at_10 = calculate_recall(all_results, ground_truth, 10);
    double recall_1_at_10 = calculate_recall_1_at_k(all_results, ground_truth, 10);

    std::cout << "Recall@10: " << recall_at_10 << std::endl;
    std::cout << "Recall@1@10: " << recall_1_at_10 << std::endl;

    // HNSW should easily hit > 0.95 on SIFTSmall with ef=50
    EXPECT_GE(recall_1_at_10, 0.95);
    EXPECT_GE(recall_at_10, 0.90); // Exact top-10 overlap is usually slightly lower but should be very high
}
