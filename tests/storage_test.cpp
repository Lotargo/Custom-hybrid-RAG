#include <gtest/gtest.h>
#include "rag_engine/storage/mmap_file.hpp"

#include <fstream>
#include <filesystem>
#include <string>

namespace {
    const std::string TEST_FILE = "test_mmap.bin";
}

class MmapFileTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a dummy file with some data
        std::ofstream out(TEST_FILE, std::ios::binary);
        const char data[] = "Hello, Mmap!";
        out.write(data, sizeof(data) - 1);
        out.close();
    }

    void TearDown() override {
        if (std::filesystem::exists(TEST_FILE)) {
            std::filesystem::remove(TEST_FILE);
        }
    }
};

TEST_F(MmapFileTest, ReadOnlyMapping) {
    using namespace rag_engine::storage;

    MmapFile file(TEST_FILE, MmapMode::ReadOnly);

    EXPECT_EQ(file.size(), 12);
    ASSERT_NE(file.data(), nullptr);

    auto span = file.span();
    EXPECT_EQ(span.size(), 12);

    std::string content(reinterpret_cast<const char*>(span.data()), span.size());
    EXPECT_EQ(content, "Hello, Mmap!");
}

TEST_F(MmapFileTest, ReadWriteMapping) {
    using namespace rag_engine::storage;

    {
        MmapFile file(TEST_FILE, MmapMode::ReadWrite);
        EXPECT_EQ(file.size(), 12);

        auto span = file.span();
        // Modify the mapped memory
        std::byte* ptr = const_cast<std::byte*>(span.data());
        ptr[0] = static_cast<std::byte>('J'); // "Jello, Mmap!"

        file.sync(); // msync
    }

    // Re-open and verify
    MmapFile file2(TEST_FILE, MmapMode::ReadOnly);
    auto span2 = file2.span();
    std::string content(reinterpret_cast<const char*>(span2.data()), span2.size());
    EXPECT_EQ(content, "Jello, Mmap!");
}

TEST_F(MmapFileTest, ResizeMapping) {
    using namespace rag_engine::storage;

    MmapFile file(TEST_FILE, MmapMode::ReadWrite);
    file.resize(100);

    EXPECT_EQ(file.size(), 100);

    auto span = file.span();
    EXPECT_EQ(span.size(), 100);

    // Write to the newly allocated space
    std::byte* ptr = const_cast<std::byte*>(span.data());
    ptr[99] = static_cast<std::byte>('X');

    file.sync();

    EXPECT_EQ(std::filesystem::file_size(TEST_FILE), 100);
}

TEST_F(MmapFileTest, Advise) {
    using namespace rag_engine::storage;

    MmapFile file(TEST_FILE, MmapMode::ReadOnly);
    EXPECT_NO_THROW(file.advise(MmapAdvice::Sequential));
    EXPECT_NO_THROW(file.advise(MmapAdvice::Random));
    EXPECT_NO_THROW(file.advise(MmapAdvice::WillNeed));
    EXPECT_NO_THROW(file.advise(MmapAdvice::DontNeed));
}

TEST_F(MmapFileTest, EmptyFileMapping) {
    using namespace rag_engine::storage;

    const std::string EMPTY_FILE = "empty_test.bin";
    std::ofstream out(EMPTY_FILE, std::ios::binary);
    out.close();

    {
        MmapFile file(EMPTY_FILE, MmapMode::ReadOnly);
        EXPECT_EQ(file.size(), 0);
        EXPECT_EQ(file.data(), nullptr);
    }

    {
        MmapFile file(EMPTY_FILE, MmapMode::ReadWrite);
        EXPECT_EQ(file.size(), 0);
        file.resize(10);
        EXPECT_EQ(file.size(), 10);
        EXPECT_NE(file.data(), nullptr);
    }

    std::filesystem::remove(EMPTY_FILE);
}
