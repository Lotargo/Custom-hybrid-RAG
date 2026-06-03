#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>
#include "sonatadb/storage/mmap_file.hpp"
#include <memory>
#include "sonatadb/compression/elias_fano.hpp"
#include "sonatadb/compression/varint.hpp"

namespace sonatadb::index {

struct DiskPosting {
    uint64_t doc_id;
    uint64_t tf;
};

class DiskInvertedIndex {
public:
    DiskInvertedIndex(const std::string& filepath) {
        mmap_file_ = std::make_unique<sonatadb::storage::MmapFile>(filepath, sonatadb::storage::MmapMode::ReadOnly);
        data_ = static_cast<const uint8_t*>(mmap_file_->data());
        size_ = mmap_file_->size();

        parse_header();
        parse_dictionary();
    }

    ~DiskInvertedIndex() = default;

    // Disable copy
    DiskInvertedIndex(const DiskInvertedIndex&) = delete;
    DiskInvertedIndex& operator=(const DiskInvertedIndex&) = delete;

    std::vector<DiskPosting> get_postings(const std::string& term) const {
        auto it = dictionary_.find(term);
        if (it == dictionary_.end()) {
            return {}; // Term not found
        }

        size_t offset = postings_start_offset_ + it->second;

        // Read EF Metadata
        auto read_uint64 = [&]() {
            uint64_t val = 0;
            for (int i=0; i<8; i++) val |= (static_cast<uint64_t>(data_[offset++]) << (i*8));
            return val;
        };

        uint64_t num_elements = read_uint64();
        uint64_t lower_bits_size = read_uint64();

        // Read lower bits
        uint64_t lower_bits_vec_size = read_uint64();
        std::vector<uint64_t> lower_bits(lower_bits_vec_size);
        for (uint64_t i = 0; i < lower_bits_vec_size; i++) {
            lower_bits[i] = read_uint64();
        }

        // Read upper bits
        uint64_t upper_bits_vec_size = read_uint64();
        std::vector<uint64_t> upper_bits(upper_bits_vec_size);
        for (uint64_t i = 0; i < upper_bits_vec_size; i++) {
            upper_bits[i] = read_uint64();
        }

        // Read TF Varint size
        uint64_t tf_buffer_size = read_uint64();

        // Create EliasFanoDecoder
        sonatadb::compression::EliasFanoDecoder ef_decoder(lower_bits, upper_bits, num_elements, lower_bits_size);

        // Decode TFs
        size_t tf_offset = offset;
        std::vector<uint64_t> tfs;
        tfs.reserve(num_elements);
        while (tf_offset < offset + tf_buffer_size) {
            tfs.push_back(sonatadb::compression::Varint::decode(data_, size_, tf_offset));
        }

        // Combine into Postings
        std::vector<DiskPosting> postings;
        postings.reserve(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            postings.push_back({ef_decoder.next(), tfs[i]});
        }

        return postings;
    }

private:
    void parse_header() {
        if (size_ < 4 || data_[0] != 'R' || data_[1] != 'A' || data_[2] != 'G' || data_[3] != 'I') {
            throw std::runtime_error("Invalid index file format: missing magic bytes");
        }

        size_t offset = 4;
        uint32_t version = *reinterpret_cast<const uint32_t*>(data_ + offset); offset += 4;
        if (version != 1) {
            throw std::runtime_error("Unsupported index file version");
        }

        num_docs_ = *reinterpret_cast<const uint64_t*>(data_ + offset); offset += 8;
        max_doc_id_ = *reinterpret_cast<const uint64_t*>(data_ + offset); offset += 8;
        num_terms_ = *reinterpret_cast<const uint64_t*>(data_ + offset); offset += 8;

        dictionary_start_offset_ = offset;
    }

    void parse_dictionary() {
        size_t offset = dictionary_start_offset_;
        uint64_t dict_size = *reinterpret_cast<const uint64_t*>(data_ + offset); offset += 8;
        size_t dict_end = offset + dict_size;

        while (offset < dict_end) {
            uint32_t term_len = 0;
            for (int i=0; i<4; i++) term_len |= (data_[offset++] << (i*8));

            std::string term(reinterpret_cast<const char*>(data_ + offset), term_len);
            offset += term_len;

            uint64_t posting_offset = 0;
            for (int i=0; i<8; i++) posting_offset |= (static_cast<uint64_t>(data_[offset++]) << (i*8));

            dictionary_[term] = posting_offset;
        }

        postings_start_offset_ = offset + 8; // skip postings_size (8 bytes)
    }

    std::unique_ptr<sonatadb::storage::MmapFile> mmap_file_;
    const uint8_t* data_;
    size_t size_;

    uint64_t num_docs_;
    uint64_t max_doc_id_;
    uint64_t num_terms_;

    size_t dictionary_start_offset_;
    size_t postings_start_offset_;

    std::unordered_map<std::string, uint64_t> dictionary_;
};

} // namespace sonatadb::index
