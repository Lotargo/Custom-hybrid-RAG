#include "sonatadb/index/bloom_filter.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace sonatadb::index {

// MurmurHash3 core logic for 128-bit hash (x64) simplified
static inline uint64_t rotl64(uint64_t x, int8_t r) {
    return (x << r) | (x >> (64 - r));
}

#define MURMUR_GETBLOCK(p, i) (p[i])

static void MurmurHash3_x64_128(const void* key, const int len, const uint32_t seed, void* out) {
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = len / 16;

    uint64_t h1 = seed;
    uint64_t h2 = seed;

    const uint64_t c1 = 0x87c37b91114253d5LLU;
    const uint64_t c2 = 0x4cf5ad432745937fLLU;

    const uint64_t* blocks = (const uint64_t*)(data);

    for (int i = 0; i < nblocks; i++) {
        uint64_t k1 = MURMUR_GETBLOCK(blocks, i * 2 + 0);
        uint64_t k2 = MURMUR_GETBLOCK(blocks, i * 2 + 1);

        k1 *= c1; k1 = rotl64(k1, 31); k1 *= c2; h1 ^= k1;
        h1 = rotl64(h1, 27); h1 += h2; h1 = h1 * 5 + 0x52dce729;

        k2 *= c2; k2 = rotl64(k2, 33); k2 *= c1; h2 ^= k2;
        h2 = rotl64(h2, 31); h2 += h1; h2 = h2 * 5 + 0x38495ab5;
    }

    const uint8_t* tail = (const uint8_t*)(data + nblocks * 16);
    uint64_t k1 = 0;
    uint64_t k2 = 0;

    switch (len & 15) {
        case 15: k2 ^= ((uint64_t)tail[14]) << 48;
        case 14: k2 ^= ((uint64_t)tail[13]) << 40;
        case 13: k2 ^= ((uint64_t)tail[12]) << 32;
        case 12: k2 ^= ((uint64_t)tail[11]) << 24;
        case 11: k2 ^= ((uint64_t)tail[10]) << 16;
        case 10: k2 ^= ((uint64_t)tail[9]) << 8;
        case  9: k2 ^= ((uint64_t)tail[8]) << 0;
                 k2 *= c2; k2 = rotl64(k2, 33); k2 *= c1; h2 ^= k2;
        case  8: k1 ^= ((uint64_t)tail[7]) << 56;
        case  7: k1 ^= ((uint64_t)tail[6]) << 48;
        case  6: k1 ^= ((uint64_t)tail[5]) << 40;
        case  5: k1 ^= ((uint64_t)tail[4]) << 32;
        case  4: k1 ^= ((uint64_t)tail[3]) << 24;
        case  3: k1 ^= ((uint64_t)tail[2]) << 16;
        case  2: k1 ^= ((uint64_t)tail[1]) << 8;
        case  1: k1 ^= ((uint64_t)tail[0]) << 0;
                 k1 *= c1; k1 = rotl64(k1, 31); k1 *= c2; h1 ^= k1;
    };

    h1 ^= len; h2 ^= len;
    h1 += h2; h2 += h1;

    auto fmix64 = [](uint64_t k) {
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdLLU;
        k ^= k >> 33;
        k *= 0xc4ceb9fe1a85ec53LLU;
        k ^= k >> 33;
        return k;
    };

    h1 = fmix64(h1);
    h2 = fmix64(h2);

    h1 += h2; h2 += h1;

    ((uint64_t*)out)[0] = h1;
    ((uint64_t*)out)[1] = h2;
}

BloomFilter::BloomFilter(size_t expected_elements, double false_positive_rate) {
    if (expected_elements == 0) {
        expected_elements = 1;
    }
    // Calculate optimal sizes
    // m = ceil((n * log(p)) / log(1 / pow(2, log(2))));
    bits_ = std::ceil((expected_elements * std::log(false_positive_rate)) / std::log(1.0 / std::pow(2.0, std::log(2.0))));

    // k = round((m / n) * log(2));
    num_hashes_ = std::round((static_cast<double>(bits_) / expected_elements) * std::log(2.0));

    if (num_hashes_ == 0) num_hashes_ = 1;

    size_t vector_size = (bits_ + 63) / 64;
    bitset_.assign(vector_size, 0);
}

void BloomFilter::hash(const uint8_t* data, size_t len, uint64_t out[2]) const {
    MurmurHash3_x64_128(data, len, 0, out);
}

void BloomFilter::add(const uint8_t* data, size_t len) {
    uint64_t h[2];
    hash(data, len, h);

    for (size_t i = 0; i < num_hashes_; i++) {
        // Double hashing technique: hash_i = (h1 + i * h2) % bits
        uint64_t index = (h[0] + i * h[1]) % bits_;
        bitset_[index / 64] |= (1ULL << (index % 64));
    }
}

void BloomFilter::add(uint32_t element) {
    add(reinterpret_cast<const uint8_t*>(&element), sizeof(element));
}

void BloomFilter::add(const std::string& element) {
    add(reinterpret_cast<const uint8_t*>(element.data()), element.size());
}

bool BloomFilter::contains(const uint8_t* data, size_t len) const {
    if (bits_ == 0) return false;

    uint64_t h[2];
    hash(data, len, h);

    for (size_t i = 0; i < num_hashes_; i++) {
        uint64_t index = (h[0] + i * h[1]) % bits_;
        if ((bitset_[index / 64] & (1ULL << (index % 64))) == 0) {
            return false;
        }
    }
    return true;
}

bool BloomFilter::contains(uint32_t element) const {
    return contains(reinterpret_cast<const uint8_t*>(&element), sizeof(element));
}

bool BloomFilter::contains(const std::string& element) const {
    return contains(reinterpret_cast<const uint8_t*>(element.data()), element.size());
}

void BloomFilter::clear() {
    std::fill(bitset_.begin(), bitset_.end(), 0);
}

} // namespace sonatadb::index
