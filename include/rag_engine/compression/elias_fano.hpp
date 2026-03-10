#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <bit>

namespace rag_engine::compression {

class EliasFanoEncoder {
public:
    EliasFanoEncoder(size_t max_elements, uint64_t max_value) {
        num_elements_ = 0;
        max_value_ = max_value;
        if (max_elements > 0) {
            lower_bits_ = std::bit_width(max_value / max_elements);
        } else {
            lower_bits_ = 0;
        }

        upper_bits_ = (max_value >> lower_bits_) + 1;

        lower_bits_array_.reserve((max_elements * lower_bits_ + 63) / 64);
        upper_bits_array_.reserve((upper_bits_ + max_elements + 63) / 64);

        last_value_ = 0;
        last_upper_ = 0;
        upper_bits_offset_ = 0;
    }

    void add(uint64_t value) {
        if (value < last_value_ && num_elements_ > 0) {
            throw std::runtime_error("EliasFanoEncoder: Values must be non-decreasing");
        }
        last_value_ = value;

        uint64_t lower = value & ((1ULL << lower_bits_) - 1);
        uint64_t upper = value >> lower_bits_;

        // Append lower bits
        if (lower_bits_ > 0) {
            size_t bit_pos = num_elements_ * lower_bits_;
            size_t word_idx = bit_pos / 64;
            size_t bit_offset = bit_pos % 64;

            if (word_idx >= lower_bits_array_.size()) {
                lower_bits_array_.push_back(0);
            }

            lower_bits_array_[word_idx] |= (lower << bit_offset);
            if (bit_offset + lower_bits_ > 64) {
                if (word_idx + 1 >= lower_bits_array_.size()) {
                    lower_bits_array_.push_back(0);
                }
                lower_bits_array_[word_idx + 1] |= (lower >> (64 - bit_offset));
            }
        }

        // Append upper bits
        uint64_t upper_diff = upper - last_upper_;
        for (uint64_t i = 0; i < upper_diff; ++i) {
            append_upper_bit(0);
        }
        append_upper_bit(1);
        last_upper_ = upper;
        num_elements_++;
    }

    const std::vector<uint64_t>& get_lower_bits() const { return lower_bits_array_; }
    const std::vector<uint64_t>& get_upper_bits() const { return upper_bits_array_; }
    size_t get_num_elements() const { return num_elements_; }
    uint64_t get_max_value() const { return max_value_; }
    size_t get_lower_bits_size() const { return lower_bits_; }

private:
    void append_upper_bit(uint64_t bit) {
        size_t word_idx = upper_bits_offset_ / 64;
        size_t bit_offset = upper_bits_offset_ % 64;

        if (word_idx >= upper_bits_array_.size()) {
            upper_bits_array_.push_back(0);
        }

        upper_bits_array_[word_idx] |= (bit << bit_offset);
        upper_bits_offset_++;
    }

    size_t num_elements_;
    uint64_t max_value_;
    size_t lower_bits_;
    size_t upper_bits_;

    std::vector<uint64_t> lower_bits_array_;
    std::vector<uint64_t> upper_bits_array_;

    uint64_t last_value_;
    uint64_t last_upper_;
    size_t upper_bits_offset_;
};

class EliasFanoDecoder {
public:
    EliasFanoDecoder(const std::vector<uint64_t>& lower_bits,
                     const std::vector<uint64_t>& upper_bits,
                     size_t num_elements,
                     size_t lower_bits_size)
        : lower_bits_(lower_bits), upper_bits_(upper_bits),
          num_elements_(num_elements), lower_bits_size_(lower_bits_size),
          current_index_(0), current_upper_value_(0), current_upper_pos_(0) {

        if (num_elements_ > 0) {
             advance_upper();
        }
    }

    bool has_next() const {
        return current_index_ < num_elements_;
    }

    uint64_t next() {
        if (!has_next()) {
            throw std::runtime_error("EliasFanoDecoder: No more elements");
        }

        uint64_t lower = 0;
        if (lower_bits_size_ > 0) {
            size_t bit_pos = current_index_ * lower_bits_size_;
            size_t word_idx = bit_pos / 64;
            size_t bit_offset = bit_pos % 64;

            lower = (lower_bits_[word_idx] >> bit_offset) & ((1ULL << lower_bits_size_) - 1);
            if (bit_offset + lower_bits_size_ > 64) {
                lower |= (lower_bits_[word_idx + 1] << (64 - bit_offset)) & ((1ULL << lower_bits_size_) - 1);
            }
        }

        uint64_t upper = current_upper_value_;
        uint64_t value = (upper << lower_bits_size_) | lower;

        current_index_++;
        if (has_next()) {
            advance_upper();
        }

        return value;
    }

private:
    void advance_upper() {
        while (true) {
            size_t word_idx = current_upper_pos_ / 64;
            size_t bit_offset = current_upper_pos_ % 64;

            if (word_idx >= upper_bits_.size()) {
                 break; // Should not happen if data is valid
            }

            uint64_t bit = (upper_bits_[word_idx] >> bit_offset) & 1;
            current_upper_pos_++;

            if (bit == 1) {
                break;
            } else {
                current_upper_value_++;
            }
        }
    }

    const std::vector<uint64_t>& lower_bits_;
    const std::vector<uint64_t>& upper_bits_;
    size_t num_elements_;
    size_t lower_bits_size_;

    size_t current_index_;
    uint64_t current_upper_value_;
    size_t current_upper_pos_;
};

} // namespace rag_engine::compression
