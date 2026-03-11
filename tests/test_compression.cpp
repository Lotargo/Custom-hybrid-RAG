#include <gtest/gtest.h>
#include "sonatadb/compression/varint.hpp"
#include "sonatadb/compression/elias_fano.hpp"
#include <vector>

TEST(CompressionTest, VarintEncodeDecode) {
    using namespace sonatadb::compression;

    std::vector<uint64_t> values = {0, 1, 127, 128, 255, 256, 16383, 16384, 1ULL << 32, 1ULL << 63};
    std::vector<uint8_t> buffer;

    for (uint64_t v : values) {
        Varint::encode(v, buffer);
    }

    size_t offset = 0;
    for (uint64_t expected : values) {
        uint64_t decoded = Varint::decode(buffer.data(), buffer.size(), offset);
        EXPECT_EQ(decoded, expected);
    }
    EXPECT_EQ(offset, buffer.size());
}

TEST(CompressionTest, EliasFanoBasic) {
    using namespace sonatadb::compression;

    std::vector<uint64_t> values = {2, 5, 13, 27, 42, 100, 256, 1024, 4096};
    uint64_t max_val = 4096;

    EliasFanoEncoder encoder(values.size(), max_val);
    for (uint64_t v : values) {
        encoder.add(v);
    }

    EliasFanoDecoder decoder(encoder.get_lower_bits(), encoder.get_upper_bits(), encoder.get_num_elements(), encoder.get_lower_bits_size());

    std::vector<uint64_t> decoded;
    while (decoder.has_next()) {
        decoded.push_back(decoder.next());
    }

    EXPECT_EQ(decoded, values);
}
