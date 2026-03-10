#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

namespace rag_engine::compression {

class Varint {
public:
    static void encode(uint64_t value, std::vector<uint8_t>& buffer) {
        while (value >= 0x80) {
            buffer.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
            value >>= 7;
        }
        buffer.push_back(static_cast<uint8_t>(value));
    }

    // Decodes a single value from the buffer starting at the given offset.
    // Updates the offset to the byte immediately following the decoded Varint.
    static uint64_t decode(const uint8_t* buffer, size_t size, size_t& offset) {
        uint64_t value = 0;
        int shift = 0;
        while (offset < size) {
            uint8_t byte = buffer[offset++];
            value |= static_cast<uint64_t>(byte & 0x7F) << shift;
            if ((byte & 0x80) == 0) {
                return value;
            }
            shift += 7;
        }
        throw std::runtime_error("Invalid Varint sequence");
    }
};

} // namespace rag_engine::compression
