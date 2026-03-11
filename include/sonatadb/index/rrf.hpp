#pragma once

#include "sonatadb/index/types.hpp"
#include <vector>

namespace sonatadb::index {

/**
 * Fuse results from multiple search engines (e.g., Vector Search + Keyword Search)
 * using the Reciprocal Rank Fusion (RRF) algorithm.
 *
 * RRF score = sum(1 / (k + rank)) for each rank in the provided result lists.
 * Note: Lower distance means better rank in HNSW.
 * Higher TF/IDF score means better rank in Keyword search.
 * This function expects that the lists are ALREADY SORTED in the order of relevance
 * (best match at index 0).
 *
 * @param results_list A list of search results (each is a sorted vector of SearchResult).
 * @param k The RRF constant (usually 60).
 * @param top_n Number of results to return.
 * @return A fused and sorted list of top_n results. The `distance` field will contain the RRF score (higher is better).
 */
std::vector<SearchResult> reciprocal_rank_fusion(
    const std::vector<std::vector<SearchResult>>& results_list,
    size_t k = 60,
    size_t top_n = 100
);

} // namespace sonatadb::index
