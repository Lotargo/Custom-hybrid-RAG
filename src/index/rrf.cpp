#include "rag_engine/index/rrf.hpp"
#include <unordered_map>
#include <algorithm>

namespace rag_engine::index {

std::vector<SearchResult> reciprocal_rank_fusion(
    const std::vector<std::vector<SearchResult>>& results_list,
    size_t k,
    size_t top_n
) {
    std::unordered_map<uint32_t, float> aggregated_scores;

    // RRF assigns a score = 1 / (k + rank), where rank is 1-based (rank=1 for best match).
    // The lists should already be sorted by their original engines from best to worst.
    for (const auto& list : results_list) {
        size_t rank = 1;
        for (const auto& item : list) {
            float rrf_score = 1.0f / static_cast<float>(k + rank);
            aggregated_scores[item.id] += rrf_score;
            rank++;
        }
    }

    std::vector<SearchResult> fused_results;
    fused_results.reserve(aggregated_scores.size());
    for (const auto& [id, score] : aggregated_scores) {
        // Distance field is abused here to store RRF score. Note: for RRF, HIGHER is better.
        fused_results.push_back({id, score});
    }

    // Sort by RRF score descending
    std::sort(fused_results.begin(), fused_results.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.distance > b.distance;
    });

    if (fused_results.size() > top_n) {
        fused_results.resize(top_n);
    }

    return fused_results;
}

} // namespace rag_engine::index
