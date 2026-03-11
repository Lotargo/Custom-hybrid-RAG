#include "network/protocol.h"

#include <stdexcept>
#include <cstring>
#include <bit>

namespace rag_engine::network {

// Реализация CRC32 (полином 0xEDB88320)
uint32_t Protocol::calculate_crc32(std::span<const uint8_t> data) {
    uint32_t crc = 0xFFFFFFFF;
    for (uint8_t byte : data) {
        crc ^= byte;
        for (int i = 0; i < 8; i++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    return ~crc;
}

std::vector<uint8_t> Protocol::serialize(CommandID command, std::span<const uint8_t> payload) {
    std::vector<uint8_t> packet;
    packet.reserve(HEADER_SIZE + payload.size());

    // 1. Magic byte
    packet.push_back(MAGIC_BYTE);

    // 2. Length (4 bytes, Little Endian)
    uint32_t length = static_cast<uint32_t>(payload.size());
    // Если мы на системе с другим порядком байт, нужно было бы конвертировать в LE.
    // Пока предполагаем x86_64/ARM (Little Endian).
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("Big endian systems are not supported yet");
    }
    uint8_t len_bytes[4];
    std::memcpy(len_bytes, &length, 4);
    packet.insert(packet.end(), len_bytes, len_bytes + 4);

    // 3. Command ID (1 byte)
    packet.push_back(static_cast<uint8_t>(command));

    // 4. CRC32 (4 bytes, Little Endian)
    uint32_t crc = calculate_crc32(payload);
    uint8_t crc_bytes[4];
    std::memcpy(crc_bytes, &crc, 4);
    packet.insert(packet.end(), crc_bytes, crc_bytes + 4);

    // 5. Payload
    packet.insert(packet.end(), payload.begin(), payload.end());

    return packet;
}

Protocol::ParseResult Protocol::parse_header(std::span<const uint8_t> data) {
    if (data.size() < HEADER_SIZE) {
        return {ParseStatus::NeedMoreData, std::nullopt};
    }

    if (data[0] != MAGIC_BYTE) {
        return {ParseStatus::InvalidMagic, std::nullopt};
    }

    PacketHeader header;
    header.magic = data[0];

    std::memcpy(&header.length, data.data() + 1, 4);

    header.command = static_cast<CommandID>(data[5]);

    std::memcpy(&header.crc32, data.data() + 6, 4);

    return {ParseStatus::Ok, header};
}

bool Protocol::validate_payload(const PacketHeader& header, std::span<const uint8_t> payload) {
    if (payload.size() != header.length) {
        return false;
    }
    return calculate_crc32(payload) == header.crc32;
}

} // namespace rag_engine::network
