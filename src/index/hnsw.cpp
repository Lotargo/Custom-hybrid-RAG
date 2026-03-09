#include "rag_engine/index/hnsw.hpp"
#include <cmath>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <random>
#include <cassert>
#include <mutex>

namespace rag_engine::index {

HnswIndex::HnswIndex(const std::string& filepath, HnswConfig config, bool read_only)
    : config_(config), read_only_mode_(read_only), node_locks_(config.max_elements) {

    // Calculate total mmap size
    // 1. Header
    size_t header_sz = sizeof(Header);

    // 2. Nodes array
    size_t nodes_sz = config.max_elements * sizeof(HnswNode);

    // 3. Vectors array (flat)
    size_t vectors_sz = config.max_elements * config.dim * sizeof(float);

    // 4. Levels info array
    size_t levels_sz = config.max_elements * sizeof(uint32_t);

    // 5. Layer 0 links (flat)
    size_t links0_sz = config.max_elements * (config.M0 + 1) * sizeof(uint32_t);

    // Calculate expected max levels. In a real system, we'd dynamically allocate, but
    // for mmap and zero-allocation we preallocate a reasonable max levels per node.
    // e.g., max 15 layers.
    size_t max_layers = 16;

    // 6. High layers links (flat)
    size_t links_high_sz = config.max_elements * max_layers * (config.M + 1) * sizeof(uint32_t);

    size_t total_size = header_sz + nodes_sz + vectors_sz + levels_sz + links0_sz + links_high_sz;

    storage::MmapMode mode = read_only ? storage::MmapMode::ReadOnly : storage::MmapMode::ReadWrite;

    mmap_file_ = std::make_unique<storage::MmapFile>(filepath, mode);

    // If we just created the file, resize it.
    // MmapFile might need a small patch to handle creation, but we assume it either resizes
    // or we initialize it. Let's rely on MmapFile to allocate or open properly.
    if (!read_only && mmap_file_->size() < total_size) {
        mmap_file_->resize(total_size);
    }

    std::byte* base = mmap_file_->span().data();

    header_ = reinterpret_cast<Header*>(base);
    base += header_sz;

    nodes_ = reinterpret_cast<HnswNode*>(base);
    base += nodes_sz;

    vectors_ = reinterpret_cast<float*>(base);
    base += vectors_sz;

    levels_ = reinterpret_cast<uint32_t*>(base);
    base += levels_sz;

    links0_ = reinterpret_cast<uint32_t*>(base);
    base += links0_sz;

    links_high_ = reinterpret_cast<uint32_t*>(base);

    if (!read_only) {
        if (header_->magic != 0x484E5357) { // "HNSW"
            // Initialize header
            header_->magic = 0x484E5357;
            header_->version = 1;
            header_->dim = config.dim;
            header_->max_elements = config.max_elements;
            header_->M = config.M;
            header_->M0 = config.M0;
            header_->entry_point = 0;
            header_->max_level = -1;
            header_->num_elements = 0;
        } else {
            // Restore state
            config_.dim = header_->dim;
            config_.max_elements = header_->max_elements;
            config_.M = header_->M;
            config_.M0 = header_->M0;
            num_elements_.store(header_->num_elements, std::memory_order_relaxed);
            max_level_.store(header_->max_level, std::memory_order_relaxed);
            entry_point_.store(header_->entry_point, std::memory_order_relaxed);
        }

        // Madvise for optimization: vectors are accessed randomly
        // Temporarily commented out to fix madvise invalid argument
        // mmap_file_->advise(storage::MmapAdvice::Random, header_sz + nodes_sz, vectors_sz);
    } else {
        // Validation check
        if (header_->magic != 0x484E5357) {
            throw std::runtime_error("Invalid HNSW file magic");
        }
        num_elements_.store(header_->num_elements, std::memory_order_relaxed);
        max_level_.store(header_->max_level, std::memory_order_relaxed);
        entry_point_.store(header_->entry_point, std::memory_order_relaxed);
    }
}

HnswIndex::~HnswIndex() {
    if (!read_only_mode_ && header_ && mmap_file_ && mmap_file_->data() != nullptr) {
        header_->num_elements = num_elements_.load(std::memory_order_relaxed);
        header_->max_level = max_level_.load(std::memory_order_relaxed);
        header_->entry_point = entry_point_.load(std::memory_order_relaxed);
    }
}

void HnswIndex::sync() {
    if (!read_only_mode_ && header_ && mmap_file_ && mmap_file_->data() != nullptr) {
        header_->num_elements = num_elements_.load(std::memory_order_relaxed);
        header_->max_level = max_level_.load(std::memory_order_relaxed);
        header_->entry_point = entry_point_.load(std::memory_order_relaxed);
        mmap_file_->sync(0, 0, true);
    }
}

int HnswIndex::getRandomLevel() {
    static thread_local std::mt19937 generator(config_.seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(generator)) * (1.0 / log(config_.M));
    return static_cast<int>(r);
}

float HnswIndex::distance(std::span<const float> a, std::span<const float> b) const {
    // We use L2 distance. Can be configured.
    return rag_engine::math::l2_distance(a, b);
}

std::span<const float> HnswIndex::getVector(uint32_t id) const {
    return std::span<const float>(&vectors_[id * config_.dim], config_.dim);
}

// Link representation: The first uint32_t is the COUNT of links, followed by the link IDs.
std::span<uint32_t> HnswIndex::getLinks(uint32_t id, int layer) {
    if (layer == 0) {
        uint32_t* base = &links0_[id * (config_.M0 + 1)];
        uint32_t count = std::atomic_ref<uint32_t>(base[0]).load(std::memory_order_acquire);
        return std::span<uint32_t>(&base[1], count);
    } else {
        uint32_t* base = &links_high_[(id * 16 + static_cast<uint32_t>(layer - 1)) * (config_.M + 1)];
        uint32_t count = std::atomic_ref<uint32_t>(base[0]).load(std::memory_order_acquire);
        return std::span<uint32_t>(&base[1], count);
    }
}

std::span<const uint32_t> HnswIndex::getLinks(uint32_t id, int layer) const {
    if (layer == 0) {
        const uint32_t* base = &links0_[id * (config_.M0 + 1)];
        // Safe concurrent read
        uint32_t count = std::atomic_ref<const uint32_t>(base[0]).load(std::memory_order_acquire);
        return std::span<const uint32_t>(&base[1], count);
    } else {
        const uint32_t* base = &links_high_[(id * 16 + static_cast<uint32_t>(layer - 1)) * (config_.M + 1)];
        uint32_t count = std::atomic_ref<const uint32_t>(base[0]).load(std::memory_order_acquire);
        return std::span<const uint32_t>(&base[1], count);
    }
}

void HnswIndex::setLinks(uint32_t id, int layer, const std::vector<uint32_t>& new_links) {
    if (layer == 0) {
        uint32_t* base = &links0_[id * (config_.M0 + 1)];
        for (size_t i = 0; i < new_links.size(); ++i) {
            base[i + 1] = new_links[i];
        }
        // Write memory barrier - write payload first, then update size
        std::atomic_ref<uint32_t>(base[0]).store(static_cast<uint32_t>(new_links.size()), std::memory_order_release);
    } else {
        uint32_t* base = &links_high_[(id * 16 + static_cast<uint32_t>(layer - 1)) * (config_.M + 1)];
        for (size_t i = 0; i < new_links.size(); ++i) {
            base[i + 1] = new_links[i];
        }
        std::atomic_ref<uint32_t>(base[0]).store(static_cast<uint32_t>(new_links.size()), std::memory_order_release);
    }
}

void HnswIndex::add(uint32_t id, std::span<const float> vector) {
    if (id >= config_.max_elements) {
        throw std::runtime_error("Exceeded max elements");
    }

    // Copy vector
    std::copy(vector.begin(), vector.end(), &vectors_[id * config_.dim]);

    nodes_[id].id = id;
    int level = getRandomLevel();
    // Cap at 15 for safety in our flat mmap model
    if (level > 15) level = 15;

    levels_[id] = static_cast<uint32_t>(level);

    // Fast-path for first element
    int32_t max_lev = max_level_.load(std::memory_order_acquire);
    if (max_lev == -1) {
        std::unique_lock lock(entry_point_lock_);
        max_lev = max_level_.load(std::memory_order_acquire);
        if (max_lev == -1) {
            entry_point_.store(id, std::memory_order_release);
            max_level_.store(level, std::memory_order_release);
            num_elements_.fetch_add(1, std::memory_order_release);
            return;
        }
    }

    uint32_t ep = entry_point_.load(std::memory_order_acquire);
    uint32_t curr_obj = ep;
    float curr_dist = distance(vector, getVector(curr_obj));

    // Phase 1: greedy search to find entry point to the node's level
    for (int l = max_lev; l > level; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            auto links = getLinks(curr_obj, l);
            for (uint32_t neighbor : links) {
                float dist = distance(vector, getVector(neighbor));
                if (dist < curr_dist) {
                    curr_dist = dist;
                    curr_obj = neighbor;
                    changed = true;
                }
            }
        }
    }

    // Phase 2: search and connect starting from node's level
    for (int l = std::min(level, max_lev); l >= 0; --l) {
        auto W = searchLayer(vector, curr_obj, config_.ef_construction, l);
        size_t maxM = (l == 0) ? config_.M0 : config_.M;
        auto neighbors = selectNeighbors(vector, W, maxM, l, false, false);

        // Mutually connect
        node_locks_[id].lock();
        setLinks(id, l, neighbors);
        node_locks_[id].unlock();

        for (uint32_t neighbor_id : neighbors) {
            node_locks_[neighbor_id].lock();
            auto n_links = getLinks(neighbor_id, l);
            std::vector<uint32_t> link_copy(n_links.begin(), n_links.end());
            link_copy.push_back(id);

            if (link_copy.size() > maxM) {
                // Shrink connections
                std::vector<SearchResult> cands;
                for (uint32_t c : link_copy) {
                    cands.push_back({c, distance(getVector(neighbor_id), getVector(c))});
                }
                link_copy = selectNeighbors(getVector(neighbor_id), cands, maxM, l, false, false);
            }
            setLinks(neighbor_id, l, link_copy);
            node_locks_[neighbor_id].unlock();
        }

        // Advance to next layer
        curr_obj = W.front().id;
    }

    num_elements_.fetch_add(1, std::memory_order_release);

    // Update entry point if necessary
    if (level > max_lev) {
        std::unique_lock lock(entry_point_lock_);
        if (level > max_level_.load(std::memory_order_acquire)) {
            max_level_.store(level, std::memory_order_release);
            entry_point_.store(id, std::memory_order_release);
        }
    }
}

std::vector<SearchResult> HnswIndex::searchLayer(
    std::span<const float> query,
    uint32_t ep,
    size_t ef,
    int layer
) const {
    std::priority_queue<SearchResult> top_candidates; // max-heap
    std::priority_queue<SearchResult, std::vector<SearchResult>, std::greater<>> candidate_set; // min-heap
    std::unordered_set<uint32_t> visited;

    float dist = distance(query, getVector(ep));
    top_candidates.push({ep, dist});
    candidate_set.push({ep, dist});
    visited.insert(ep);

    while (!candidate_set.empty()) {
        SearchResult c = candidate_set.top();
        if (c.distance > top_candidates.top().distance) {
            break;
        }
        candidate_set.pop();

        auto links = getLinks(c.id, layer);
        for (uint32_t e : links) {
            if (visited.find(e) == visited.end()) {
                visited.insert(e);
                float d = distance(query, getVector(e));
                if (top_candidates.size() < ef || d < top_candidates.top().distance) {
                    candidate_set.push({e, d});
                    top_candidates.push({e, d});
                    if (top_candidates.size() > ef) {
                        top_candidates.pop();
                    }
                }
            }
        }
    }

    std::vector<SearchResult> results;
    while (!top_candidates.empty()) {
        results.push_back(top_candidates.top());
        top_candidates.pop();
    }
    std::reverse(results.begin(), results.end());
    return results;
}

std::vector<uint32_t> HnswIndex::selectNeighbors(
    std::span<const float> query,
    const std::vector<SearchResult>& candidates,
    size_t M,
    int layer,
    bool extend_candidates,
    bool keep_pruned
) const {
    // Basic heuristic: simple nearest neighbors.
    // For a real HNSW, this would implement the distance-based heuristic.
    std::vector<SearchResult> sorted_cands = candidates;
    std::sort(sorted_cands.begin(), sorted_cands.end());

    std::vector<uint32_t> result;
    for (size_t i = 0; i < std::min(M, sorted_cands.size()); ++i) {
        result.push_back(sorted_cands[i].id);
    }
    return result;
}

std::vector<SearchResult> HnswIndex::search(std::span<const float> query, size_t k, size_t ef_search) const {
    int32_t max_lev = max_level_.load(std::memory_order_acquire);
    if (max_lev == -1) {
        return {};
    }

    uint32_t ep = entry_point_.load(std::memory_order_acquire);
    float curr_dist = distance(query, getVector(ep));
    uint32_t curr_obj = ep;

    for (int l = max_lev; l > 0; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            auto links = getLinks(curr_obj, l);
            for (uint32_t neighbor : links) {
                float dist = distance(query, getVector(neighbor));
                if (dist < curr_dist) {
                    curr_dist = dist;
                    curr_obj = neighbor;
                    changed = true;
                }
            }
        }
    }

    auto W = searchLayer(query, curr_obj, std::max(k, ef_search), 0);

    if (W.size() > k) {
        W.resize(k);
    }
    return W;
}

} // namespace rag_engine::index
