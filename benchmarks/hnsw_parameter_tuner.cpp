#include "sonatadb/index/hnsw.hpp"
#include "../tests/sift_parser.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>

using namespace sonatadb::index;
using namespace sonatadb::test_utils;

// Helper to calculate recall
double calc_recall(const std::vector<std::vector<SearchResult>>& results,
                   const std::vector<std::vector<int>>& truth,
                   size_t k) {
    size_t hits = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        int true_nn = truth[i][0];
        size_t limit = std::min(k, results[i].size());
        for (size_t j = 0; j < limit; ++j) {
            if (results[i][j].id == static_cast<uint32_t>(true_nn)) {
                hits++;
                break;
            }
        }
    }
    return static_cast<double>(hits) / results.size();
}

void print_help(const char* prog) {
    std::cerr << "Usage: " << prog << " <base.fvecs> <query.fvecs> <groundtruth.ivecs> [max_base_elements]\n";
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_help(argv[0]);
        return 1;
    }

    std::string base_file = argv[1];
    std::string query_file = argv[2];
    std::string truth_file = argv[3];
    size_t max_base = (argc >= 5) ? std::stoull(argv[4]) : 0;

    std::cout << "Loading dataset...\n";
    auto base_vectors = SiftParser::read_fvecs(base_file, max_base);
    auto query_vectors = SiftParser::read_fvecs(query_file);
    auto ground_truth = SiftParser::read_ivecs(truth_file);

    std::cout << "Loaded " << base_vectors.size() << " base vectors, "
              << query_vectors.size() << " queries.\n\n";

    if (base_vectors.empty() || query_vectors.empty() || ground_truth.empty()) {
        std::cerr << "Empty dataset loaded.\n";
        return 1;
    }

    size_t dim = base_vectors[0].size();

    // Parameters to tune
    std::vector<size_t> M_values = {16, 32};
    std::vector<size_t> efConstruct_values = {100, 200};
    std::vector<size_t> efSearch_values = {50, 100};

    std::cout << "====================================================================================\n";
    std::cout << std::left << std::setw(6) << "M"
              << std::setw(15) << "efConstruction"
              << std::setw(12) << "efSearch"
              << std::setw(15) << "BuildTime(ms)"
              << std::setw(15) << "QueryTime(ms)"
              << std::setw(12) << "Recall@10"
              << "\n";
    std::cout << "====================================================================================\n";

    for (size_t m : M_values) {
        for (size_t efC : efConstruct_values) {

            std::string index_path = "tuner_index.bin";
            if (std::filesystem::exists(index_path)) std::filesystem::remove(index_path);
            if (std::filesystem::exists(index_path + ".tombstones")) std::filesystem::remove(index_path + ".tombstones");

            HnswConfig config;
            config.dim = dim;
            config.max_elements = base_vectors.size();
            config.M = m;
            config.M0 = m * 2;
            config.ef_construction = efC;

            auto t_build_start = std::chrono::high_resolution_clock::now();
            {
                HnswIndex index(index_path, config, false);
                for (size_t i = 0; i < base_vectors.size(); ++i) {
                    index.add(i, base_vectors[i]);
                }
                index.sync();
            }
            auto t_build_end = std::chrono::high_resolution_clock::now();
            double build_ms = std::chrono::duration<double, std::milli>(t_build_end - t_build_start).count();

            // Load read-only for queries
            HnswIndex index_ro(index_path, config, true);

            for (size_t efS : efSearch_values) {

                auto t_query_start = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<SearchResult>> all_results;
                all_results.reserve(query_vectors.size());

                for (const auto& q : query_vectors) {
                    all_results.push_back(index_ro.search(q, 10, efS));
                }

                auto t_query_end = std::chrono::high_resolution_clock::now();
                double query_ms = std::chrono::duration<double, std::milli>(t_query_end - t_query_start).count();

                double recall = calc_recall(all_results, ground_truth, 10);

                std::cout << std::left << std::setw(6) << m
                          << std::setw(15) << efC
                          << std::setw(12) << efS
                          << std::setw(15) << std::fixed << std::setprecision(2) << build_ms
                          << std::setw(15) << std::fixed << std::setprecision(2) << query_ms
                          << std::setw(12) << std::fixed << std::setprecision(4) << recall
                          << "\n";
            }

            // Clean up index for next iteration
            if (std::filesystem::exists(index_path)) std::filesystem::remove(index_path);
            if (std::filesystem::exists(index_path + ".tombstones")) std::filesystem::remove(index_path + ".tombstones");
        }
    }
    std::cout << "====================================================================================\n";

    return 0;
}
