#pragma once

#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <span>
#include <cstdint>

namespace rag_engine::storage {

class WalError : public std::runtime_error {
public:
    explicit WalError(const std::string& msg) : std::runtime_error(msg) {}
};

enum class WalOpcode : uint8_t {
    INSERT = 1,
    UPDATE = 2,
    DELETE = 3
};

struct WalEntry {
    WalOpcode opcode;
    std::string key;
    std::string payload; // Binary or JSON data depending on context
};

class WriteAheadLog {
public:
    explicit WriteAheadLog(const std::string& path);
    ~WriteAheadLog();

    // Prevent copying
    WriteAheadLog(const WriteAheadLog&) = delete;
    WriteAheadLog& operator=(const WriteAheadLog&) = delete;

    // Allow moving
    WriteAheadLog(WriteAheadLog&& other) noexcept;
    WriteAheadLog& operator=(WriteAheadLog&& other) noexcept;

    // Append a record to the log
    // Returns the byte offset of the start of the record
    size_t append(WalOpcode opcode, const std::string& key, const std::string& payload);

    // Forces the OS to flush written data to disk
    void sync();

    // Recovers data from the WAL.
    // Takes a callback that will be called for each valid entry.
    void recover(std::function<void(const WalEntry&)> callback);

    // Get the current file size
    size_t size() const { return size_; }

private:
    void cleanup();

    int fd_ = -1;
    size_t size_ = 0;
    std::string path_;
};

} // namespace rag_engine::storage