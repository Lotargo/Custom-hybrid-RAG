#pragma once

#include <atomic>
#include <cstdint>
#include <vector>
#include <memory>
#include <shared_mutex>
#include <span>
#include <string>
#include <optional>
#include "rag_engine/storage/mmap_file.hpp"
#include "rag_engine/math/simd_dispatch.hpp"
#include <roaring/roaring.hh>
#include "rag_engine/index/types.hpp"

namespace rag_engine::index {

// Spinlock for fine-grained node locking during neighbor updates
class Spinlock {
public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // spin
        }
    }
    void unlock() {
        flag_.clear(std::memory_order_release);
    }
private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

// Represents a node in the HNSW graph, stored flat in mmap memory
struct HnswNode {
    // We store the vector inline or separately depending on size, but for flat mmap
    // it's easier to store them as arrays. We will use a separate block for vectors
    // and a separate block for neighbors to ensure cache locality.

    // We will just define offsets here.
    // Instead of actual pointers, we will use uint32_t indices.
    uint32_t id;
    uint32_t max_level;
};

// Configuration for the HNSW index
struct HnswConfig {
    size_t dim;            // Vector dimensionality
    size_t max_elements;   // Maximum number of elements to pre-allocate
    size_t M = 16;         // Max number of connections per layer (except level 0)
    size_t M0 = 32;        // Max number of connections for layer 0
    size_t ef_construction = 100; // Size of the dynamic candidate list during construction
    uint32_t seed = 42;    // Random seed for level generation
};

class HnswIndex {
public:
    HnswIndex(const std::string& filepath, HnswConfig config, bool read_only = false);
    ~HnswIndex();

    // Prevent copy/move for simplicity, as it manages raw mmap pointers
    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;

    // Add a vector to the index. Thread-safe.
    void add(uint32_t id, std::span<const float> vector);

    // Search for the k nearest neighbors. Thread-safe.
    std::vector<SearchResult> search(std::span<const float> query, size_t k, size_t ef_search) const;

    // Persist changes to disk (msync)
    void sync();

    // Get current number of elements
    size_t size() const {
        return num_elements_.load(std::memory_order_acquire);
    }

    // Mark a node as deleted (Tombstoning)
    void markDeleted(uint32_t id);

    // Check if a node is deleted
    bool isDeleted(uint32_t id) const;

    // Compact the index by removing deleted nodes and writing to a new file
    // Returns a mapping of old_id to new_id (where missing/deleted nodes are e.g. -1 or ignored in mapping)
    // We return a vector of size `size()` where result[old_id] = new_id, and UINT32_MAX if deleted.
    std::vector<uint32_t> compact(const std::string& new_filepath) const;

private:
    // Memory layout header
    struct Header {
        uint32_t magic;
        uint32_t version;
        size_t dim;
        size_t max_elements;
        size_t M;
        size_t M0;
        uint32_t entry_point;
        int32_t max_level;
        size_t num_elements;
        // Padding to ensure the next structure (nodes array) is 64-byte aligned
        char padding[64 - (sizeof(uint32_t)*3 + sizeof(size_t)*5 + sizeof(int32_t)) % 64];
    };

    HnswConfig config_;
    bool read_only_mode_ = false;
    std::unique_ptr<storage::MmapFile> mmap_file_;
    Header* header_;

    // In-memory locks for each node (not persisted to mmap)
    std::vector<Spinlock> node_locks_;

    // Tombstone bitmap for deleted nodes
    roaring::Roaring deleted_nodes_;
    mutable std::shared_mutex deleted_nodes_lock_;

    std::string filepath_;

    // Pointers to mapped regions
    HnswNode* nodes_;
    float* vectors_;       // flat array: max_elements * dim
    uint32_t* levels_;     // sizes of layers for each node
    uint32_t* links0_;     // layer 0 links: max_elements * M0
    uint32_t* links_high_; // high layers links: max_elements * max_levels * M

    std::atomic<size_t> num_elements_{0};
    std::atomic<int32_t> max_level_{-1};
    std::atomic<uint32_t> entry_point_{0};

    // Global lock for changing the entry point (rare)
    std::shared_mutex entry_point_lock_;

    int getRandomLevel();
    float distance(std::span<const float> a, std::span<const float> b) const;

    // HNSW internals
    std::vector<SearchResult> searchLayer(
        std::span<const float> query,
        uint32_t entry_point,
        size_t ef,
        int layer
    ) const;

    void mutuallyConnect(uint32_t id1, uint32_t id2, int layer);

    std::span<uint32_t> getLinks(uint32_t id, int layer);
    std::span<const uint32_t> getLinks(uint32_t id, int layer) const;
    void setLinks(uint32_t id, int layer, const std::vector<uint32_t>& new_links);

    std::span<const float> getVector(uint32_t id) const;

    std::vector<uint32_t> selectNeighbors(
        std::span<const float> query,
        const std::vector<SearchResult>& candidates,
        size_t M,
        int layer,
        bool extend_candidates,
        bool keep_pruned
    ) const;
};

} // namespace rag_engine::index
