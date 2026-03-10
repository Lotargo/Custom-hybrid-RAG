#pragma once

#include <vector>
#include <cstdint>

namespace rag_engine::index {

// Result of a query, mapping a document ID to its distance/score
struct SearchResult {
    uint32_t id;
    float distance; // For HNSW (L2/Cosine), smaller is better. For BM25, larger is better.
                    // This is important to consider depending on the context.

    bool operator<(const SearchResult& other) const {
        return distance < other.distance;
    }
    bool operator>(const SearchResult& other) const {
        return distance > other.distance;
    }
};

} // namespace rag_engine::index
