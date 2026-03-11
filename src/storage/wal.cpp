#include "sonatadb/storage/wal.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>

namespace sonatadb::storage {

// The header structure for each WAL record
#pragma pack(push, 1)
struct WalRecordHeader {
    uint8_t opcode;
    uint32_t key_len;
    uint32_t payload_len;
    uint32_t checksum; // Simple XOR checksum for data integrity checks
};
#pragma pack(pop)

// Utility for simple checksums
static uint32_t calculate_checksum(const WalRecordHeader& header, const std::string& key, const std::string& payload) {
    uint32_t checksum = 0;
    checksum ^= header.opcode;
    checksum ^= header.key_len;
    checksum ^= header.payload_len;

    for (char c : key) checksum ^= static_cast<uint8_t>(c);
    for (char c : payload) checksum ^= static_cast<uint8_t>(c);

    return checksum;
}

WriteAheadLog::WriteAheadLog(const std::string& path) : path_(path) {
    // Open in append-only mode. O_DIRECT can be used in advanced setups
    int flags = O_WRONLY | O_APPEND | O_CREAT;
    mode_t fd_mode = S_IRUSR | S_IWUSR; // 0600

    fd_ = ::open(path.c_str(), flags, fd_mode);
    if (fd_ == -1) {
        throw WalError("Failed to open WAL file " + path + ": " + std::strerror(errno));
    }

    struct stat st;
    if (::fstat(fd_, &st) == -1) {
        cleanup();
        throw WalError("Failed to stat WAL file " + path + ": " + std::strerror(errno));
    }

    size_ = st.st_size;
}

WriteAheadLog::~WriteAheadLog() {
    cleanup();
}

WriteAheadLog::WriteAheadLog(WriteAheadLog&& other) noexcept
    : fd_(other.fd_), size_(other.size_), path_(std::move(other.path_)) {
    other.fd_ = -1;
    other.size_ = 0;
}

WriteAheadLog& WriteAheadLog::operator=(WriteAheadLog&& other) noexcept {
    if (this != &other) {
        cleanup();
        fd_ = other.fd_;
        size_ = other.size_;
        path_ = std::move(other.path_);
        other.fd_ = -1;
        other.size_ = 0;
    }
    return *this;
}

void WriteAheadLog::cleanup() {
    if (fd_ != -1) {
        // Best effort sync before closing
        ::fsync(fd_);
        ::close(fd_);
        fd_ = -1;
    }
}

size_t WriteAheadLog::append(WalOpcode opcode, const std::string& key, const std::string& payload) {
    if (fd_ == -1) throw WalError("WAL file is not open");

    WalRecordHeader header;
    header.opcode = static_cast<uint8_t>(opcode);
    header.key_len = static_cast<uint32_t>(key.size());
    header.payload_len = static_cast<uint32_t>(payload.size());
    header.checksum = calculate_checksum(header, key, payload);

    // Assemble the buffer to minimize syscalls
    std::vector<char> buffer(sizeof(WalRecordHeader) + key.size() + payload.size());

    char* ptr = buffer.data();
    std::memcpy(ptr, &header, sizeof(WalRecordHeader));
    ptr += sizeof(WalRecordHeader);

    std::memcpy(ptr, key.data(), key.size());
    ptr += key.size();

    std::memcpy(ptr, payload.data(), payload.size());

    // Write to file
    ssize_t bytes_written = ::write(fd_, buffer.data(), buffer.size());
    if (bytes_written == -1 || static_cast<size_t>(bytes_written) != buffer.size()) {
        throw WalError("Failed to write to WAL file: " + std::string(std::strerror(errno)));
    }

    size_t current_offset = size_;
    size_ += buffer.size();

    return current_offset;
}

void WriteAheadLog::sync() {
    if (fd_ != -1) {
        if (::fsync(fd_) == -1) {
            throw WalError("Failed to fsync WAL file: " + std::string(std::strerror(errno)));
        }
    }
}

void WriteAheadLog::recover(std::function<void(const WalEntry&)> callback) {
    // Open a separate read-only descriptor for recovery
    int read_fd = ::open(path_.c_str(), O_RDONLY);
    if (read_fd == -1) {
        throw WalError("Failed to open WAL file for recovery " + path_ + ": " + std::strerror(errno));
    }

    struct stat st;
    if (::fstat(read_fd, &st) == -1) {
        ::close(read_fd);
        throw WalError("Failed to stat WAL file for recovery " + path_ + ": " + std::strerror(errno));
    }

    size_t file_size = st.st_size;
    size_t offset = 0;

    while (offset < file_size) {
        WalRecordHeader header;
        ssize_t header_read = ::pread(read_fd, &header, sizeof(WalRecordHeader), offset);

        if (header_read != sizeof(WalRecordHeader)) {
            // Unexpected EOF or error, stop recovery (corrupt tail is normal after crashes)
            break;
        }

        size_t total_record_size = sizeof(WalRecordHeader) + header.key_len + header.payload_len;
        if (offset + total_record_size > file_size) {
            // Incomplete record
            break;
        }

        std::string key(header.key_len, '\0');
        ssize_t key_read = ::pread(read_fd, key.data(), header.key_len, offset + sizeof(WalRecordHeader));
        if (key_read != static_cast<ssize_t>(header.key_len)) {
            break;
        }

        std::string payload(header.payload_len, '\0');
        ssize_t payload_read = ::pread(read_fd, payload.data(), header.payload_len, offset + sizeof(WalRecordHeader) + header.key_len);
        if (payload_read != static_cast<ssize_t>(header.payload_len)) {
            break;
        }

        // Verify checksum
        uint32_t calculated_checksum = calculate_checksum(header, key, payload);
        if (calculated_checksum != header.checksum) {
            // Corrupt record, stop recovery
            break;
        }

        WalEntry entry;
        entry.opcode = static_cast<WalOpcode>(header.opcode);
        entry.key = key;
        entry.payload = payload;

        // Callback
        callback(entry);

        offset += total_record_size;
    }

    ::close(read_fd);
}

} // namespace sonatadb::storage