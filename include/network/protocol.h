#pragma once

#include <cstdint>
#include <vector>
#include <optional>
#include <span>

namespace rag_engine::network {

// Формат пакета:
// 1 байт: Magic Byte (0x52 'R')
// 4 байта: Length (длина Payload, Little Endian)
// 1 байт: Command ID
// 4 байта: CRC32 (от Payload, Little Endian)
// Length байт: Payload

constexpr uint8_t MAGIC_BYTE = 0x52; // 'R' for RAG
constexpr size_t HEADER_SIZE = 1 + 4 + 1 + 4; // Magic + Length + Command + CRC32 = 10 байт

enum class CommandID : uint8_t {
    Ping = 0x01,
    InsertDocument = 0x02,
    Search = 0x03,
    Error = 0xFF
};

struct PacketHeader {
    uint8_t magic;
    uint32_t length;
    CommandID command;
    uint32_t crc32;
};

struct Packet {
    PacketHeader header;
    std::vector<uint8_t> payload;
};

class Protocol {
public:
    // Сериализация пакета в буфер
    static std::vector<uint8_t> serialize(CommandID command, std::span<const uint8_t> payload);

    // Результат парсинга
    enum class ParseStatus {
        Ok,
        NeedMoreData,
        InvalidMagic
    };

    struct ParseResult {
        ParseStatus status;
        std::optional<PacketHeader> header;
    };

    // Попытка распарсить заголовок из буфера (минимум 10 байт)
    static ParseResult parse_header(std::span<const uint8_t> data);

    // Валидация payload (проверка CRC32)
    static bool validate_payload(const PacketHeader& header, std::span<const uint8_t> payload);

    // Вычисление CRC32 (реализация IEEE 802.3)
    static uint32_t calculate_crc32(std::span<const uint8_t> data);
};

} // namespace rag_engine::network
