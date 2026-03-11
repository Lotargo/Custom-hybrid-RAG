#include <gtest/gtest.h>
#include "sonatadb/storage/wal.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace {
    const std::string TEST_FILE = "test_wal.bin";
}

class WalTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::filesystem::remove(TEST_FILE);
    }

    void TearDown() override {
        std::filesystem::remove(TEST_FILE);
    }
};

TEST_F(WalTest, BasicAppendAndRecover) {
    using namespace sonatadb::storage;

    std::vector<WalEntry> recovered_entries;
    auto cb = [&](const WalEntry& e) {
        recovered_entries.push_back(e);
    };

    {
        WriteAheadLog wal(TEST_FILE);

        // Append records
        wal.append(WalOpcode::INSERT, "key1", "payload1");
        wal.append(WalOpcode::UPDATE, "key2", "payload2");
        wal.append(WalOpcode::DELETE, "key3", "payload3");

        wal.sync();
    } // WAL closed

    // Recover
    WriteAheadLog wal2(TEST_FILE);
    wal2.recover(cb);

    ASSERT_EQ(recovered_entries.size(), 3);

    EXPECT_EQ(recovered_entries[0].opcode, WalOpcode::INSERT);
    EXPECT_EQ(recovered_entries[0].key, "key1");
    EXPECT_EQ(recovered_entries[0].payload, "payload1");

    EXPECT_EQ(recovered_entries[1].opcode, WalOpcode::UPDATE);
    EXPECT_EQ(recovered_entries[1].key, "key2");
    EXPECT_EQ(recovered_entries[1].payload, "payload2");

    EXPECT_EQ(recovered_entries[2].opcode, WalOpcode::DELETE);
    EXPECT_EQ(recovered_entries[2].key, "key3");
    EXPECT_EQ(recovered_entries[2].payload, "payload3");
}

TEST_F(WalTest, EmptyRecover) {
    using namespace sonatadb::storage;

    std::vector<WalEntry> recovered_entries;
    auto cb = [&](const WalEntry& e) {
        recovered_entries.push_back(e);
    };

    {
        WriteAheadLog wal(TEST_FILE);
    } // Empty WAL created and closed

    WriteAheadLog wal2(TEST_FILE);
    wal2.recover(cb);

    EXPECT_EQ(recovered_entries.size(), 0);
}
