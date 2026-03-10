#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>

namespace rag_engine::index {

/**
 * A fast, 0-allocation (after initialization) Bloom Filter implementation.
 * Designed to filter out false positives when querying metadata (e.g., whether
 * a document has a certain attribute or ID).
 */
class BloomFilter {
public:
    /**
     * Initialize the Bloom Filter.
     * @param expected_elements Estimated number of elements to be inserted.
     * @param false_positive_rate Desired false positive probability (e.g., 0.01 for 1%).
     */
    BloomFilter(size_t expected_elements, double false_positive_rate);

    /**
     * Add a document ID to the filter.
     */
    void add(uint32_t element);

    /**
     * Add a string key (e.g., an attribute or keyword) to the filter.
     */
    void add(const std::string& element);

    /**
     * Add a raw byte buffer.
     */
    void add(const uint8_t* data, size_t len);

    /**
     * Check if a document ID might be in the set.
     * @return True if possibly in set, False if definitely not in set.
     */
    bool contains(uint32_t element) const;

    /**
     * Check if a string key might be in the set.
     */
    bool contains(const std::string& element) const;

    /**
     * Check if raw bytes might be in the set.
     */
    bool contains(const uint8_t* data, size_t len) const;

    /**
     * Clear all elements from the filter.
     */
    void clear();

    /**
     * Returns the size of the internal bitset in bits.
     */
    size_t bit_size() const { return bits_; }

private:
    std::vector<uint64_t> bitset_;
    size_t bits_;
    size_t num_hashes_;

    // Compute two 64-bit hashes using a custom hash function (similar to MurmurHash3)
    void hash(const uint8_t* data, size_t len, uint64_t out[2]) const;
};

} // namespace rag_engine::index
