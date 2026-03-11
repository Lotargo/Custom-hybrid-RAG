#include <gtest/gtest.h>
#include "network/protocol.h"

using namespace rag_engine::network;

TEST(ProtocolTests, SerializeAndParse) {
    std::vector<uint8_t> payload = {0x01, 0x02, 0x03, 0x04};
    auto packet = Protocol::serialize(CommandID::Ping, payload);

    ASSERT_EQ(packet.size(), HEADER_SIZE + payload.size());
    EXPECT_EQ(packet[0], MAGIC_BYTE);

    auto parse_result = Protocol::parse_header(packet);
    ASSERT_EQ(parse_result.status, Protocol::ParseStatus::Ok);

    auto header = parse_result.header.value();
    EXPECT_EQ(header.magic, MAGIC_BYTE);
    EXPECT_EQ(header.length, payload.size());
    EXPECT_EQ(header.command, CommandID::Ping);

    std::span<const uint8_t> parsed_payload(packet.data() + HEADER_SIZE, header.length);
    EXPECT_TRUE(Protocol::validate_payload(header, parsed_payload));
}

TEST(ProtocolTests, ParseInvalidMagic) {
    std::vector<uint8_t> data(HEADER_SIZE, 0x00); // Invalid magic byte
    auto parse_result = Protocol::parse_header(data);
    EXPECT_EQ(parse_result.status, Protocol::ParseStatus::InvalidMagic);
    EXPECT_FALSE(parse_result.header.has_value());
}

TEST(ProtocolTests, ValidateInvalidPayload) {
    std::vector<uint8_t> payload = {0x01, 0x02, 0x03, 0x04};
    auto packet = Protocol::serialize(CommandID::Ping, payload);

    auto header = Protocol::parse_header(packet).header.value();

    // Corrupt payload
    packet[HEADER_SIZE] = 0xFF;
    std::span<const uint8_t> corrupted_payload(packet.data() + HEADER_SIZE, header.length);

    EXPECT_FALSE(Protocol::validate_payload(header, corrupted_payload));
}
