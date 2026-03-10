#pragma once

#include <roaring/roaring.hh>
#include <vector>
#include <cstdint>

namespace rag_engine::index {

/**
 * A wrapper around roaring::Roaring bitmaps for filtering document IDs.
 * Allows logical operations (AND, OR, ANDNOT) to combine filters for metadata filtering
 * before or during HNSW/BM25 retrieval.
 */
class DocumentFilter {
public:
    DocumentFilter() = default;

    /**
     * Create a filter from a list of document IDs.
     */
    explicit DocumentFilter(const std::vector<uint32_t>& ids);

    /**
     * Create a filter from an initializer list of document IDs.
     */
    DocumentFilter(std::initializer_list<uint32_t> ids);

    /**
     * Create a filter directly from a Roaring bitmap.
     */
    explicit DocumentFilter(roaring::Roaring bitmap);

    /**
     * Add a single document ID to the filter.
     */
    void add(uint32_t id);

    /**
     * Remove a single document ID from the filter.
     */
    void remove(uint32_t id);

    /**
     * Check if a document ID is allowed by this filter.
     */
    bool contains(uint32_t id) const;

    /**
     * Get the number of documents in the filter.
     */
    uint64_t size() const;

    /**
     * Logical AND with another filter (intersection).
     * Modifies this filter in-place.
     */
    DocumentFilter& operator&=(const DocumentFilter& other);

    /**
     * Logical OR with another filter (union).
     * Modifies this filter in-place.
     */
    DocumentFilter& operator|=(const DocumentFilter& other);

    /**
     * Logical AND NOT with another filter (difference).
     * Modifies this filter in-place.
     */
    DocumentFilter& operator-=(const DocumentFilter& other);

    /**
     * Logical AND. Returns a new filter.
     */
    DocumentFilter operator&(const DocumentFilter& other) const;

    /**
     * Logical OR. Returns a new filter.
     */
    DocumentFilter operator|(const DocumentFilter& other) const;

    /**
     * Logical AND NOT. Returns a new filter.
     */
    DocumentFilter operator-(const DocumentFilter& other) const;

    /**
     * Get underlying Roaring bitmap.
     */
    const roaring::Roaring& get_bitmap() const { return bitmap_; }

private:
    roaring::Roaring bitmap_;
};

} // namespace rag_engine::index
