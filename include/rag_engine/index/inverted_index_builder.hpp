#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <iostream>

#include "rag_engine/text/tokenizer.hpp"
#include "rag_engine/compression/elias_fano.hpp"
#include "rag_engine/compression/varint.hpp"

namespace rag_engine::index {

struct Posting {
    uint64_t doc_id;
    uint64_t tf; // term frequency
};

class InvertedIndexBuilder {
public:
    InvertedIndexBuilder(const std::string& language = "english") : tokenizer_(language), max_doc_id_(0), num_docs_(0) {}

    void add_document(uint64_t doc_id, const std::string& text) {
        if (doc_id > max_doc_id_) {
            max_doc_id_ = doc_id;
        }
        num_docs_++;

        std::vector<std::string> tokens = tokenizer_.tokenize(text);

        // Count term frequencies for this document
        std::unordered_map<std::string, uint64_t> term_counts;
        for (const auto& token : tokens) {
            term_counts[token]++;
        }

        // Add to posting lists
        for (const auto& [term, count] : term_counts) {
            auto& postings = index_[term];
            // Ensure doc_ids are sorted. In a real system, documents are usually added in order of doc_id.
            if (!postings.empty() && postings.back().doc_id >= doc_id) {
                // If not, we'd need to insert sorted, but let's assume monotonic doc_ids for now, or just sort at the end.
                 throw std::runtime_error("Documents must be added in increasing order of doc_id");
            }
            postings.push_back({doc_id, count});
        }
    }

    // Serializes the in-memory index to a custom binary file format.
    void serialize(const std::string& filepath) const {
        std::ofstream out(filepath, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open file for writing: " + filepath);
        }

        // 1. Write Header: Magic bytes, version, num_docs, max_doc_id, num_terms
        const char magic[4] = {'R', 'A', 'G', 'I'}; // RAG Index
        out.write(magic, 4);
        uint32_t version = 1;
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&num_docs_), sizeof(num_docs_));
        out.write(reinterpret_cast<const char*>(&max_doc_id_), sizeof(max_doc_id_));

        uint64_t num_terms = index_.size();
        out.write(reinterpret_cast<const char*>(&num_terms), sizeof(num_terms));

        // 2. Prepare Dictionary and Posting Lists
        // We need to write the dictionary (term -> offset in posting list data).
        // Since we don't know the offsets until we write the posting lists, we'll write them to a buffer first.

        std::vector<uint8_t> dictionary_buffer;
        std::vector<uint8_t> postings_buffer;

        for (const auto& [term, postings] : index_) {
            // Encode term
            uint32_t term_len = term.length();
            // Using direct copy for dict buffer instead of varint for simplicity of reading
            for (int i=0; i<sizeof(term_len); i++) dictionary_buffer.push_back((term_len >> (i*8)) & 0xFF);
            for (char c : term) dictionary_buffer.push_back(c);

            // Record offset
            uint64_t offset = postings_buffer.size();
            for (int i=0; i<sizeof(offset); i++) dictionary_buffer.push_back((offset >> (i*8)) & 0xFF);

            // Compress doc_ids using Elias-Fano
            compression::EliasFanoEncoder ef_encoder(postings.size(), max_doc_id_);
            for (const auto& p : postings) {
                ef_encoder.add(p.doc_id);
            }

            // Write EF metadata
            uint64_t num_elements = ef_encoder.get_num_elements();
            uint64_t lower_bits_size = ef_encoder.get_lower_bits_size();

            auto write_uint64 = [&postings_buffer](uint64_t val) {
                for (int i=0; i<8; i++) postings_buffer.push_back((val >> (i*8)) & 0xFF);
            };

            write_uint64(num_elements);
            write_uint64(lower_bits_size);

            // Write lower bits
            const auto& lower_bits = ef_encoder.get_lower_bits();
            uint64_t lower_bits_vec_size = lower_bits.size();
            write_uint64(lower_bits_vec_size);
            for (uint64_t val : lower_bits) write_uint64(val);

            // Write upper bits
            const auto& upper_bits = ef_encoder.get_upper_bits();
            uint64_t upper_bits_vec_size = upper_bits.size();
            write_uint64(upper_bits_vec_size);
            for (uint64_t val : upper_bits) write_uint64(val);

            // Compress TFs using Varint
            std::vector<uint8_t> tf_buffer;
            for (const auto& p : postings) {
                compression::Varint::encode(p.tf, tf_buffer);
            }

            uint64_t tf_buffer_size = tf_buffer.size();
            write_uint64(tf_buffer_size);
            postings_buffer.insert(postings_buffer.end(), tf_buffer.begin(), tf_buffer.end());
        }

        // 3. Write Dictionary Size and Dictionary
        uint64_t dict_size = dictionary_buffer.size();
        out.write(reinterpret_cast<const char*>(&dict_size), sizeof(dict_size));
        out.write(reinterpret_cast<const char*>(dictionary_buffer.data()), dictionary_buffer.size());

        // 4. Write Posting Lists Data
        uint64_t postings_size = postings_buffer.size();
        out.write(reinterpret_cast<const char*>(&postings_size), sizeof(postings_size));
        out.write(reinterpret_cast<const char*>(postings_buffer.data()), postings_buffer.size());

        out.close();
    }

private:
    rag_engine::text::Tokenizer tokenizer_;
    std::unordered_map<std::string, std::vector<Posting>> index_;
    uint64_t max_doc_id_;
    uint64_t num_docs_;
};

} // namespace rag_engine::index
