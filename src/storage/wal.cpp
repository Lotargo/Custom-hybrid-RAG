#include "sonatadb/storage/wal.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include <cstring>
#include <cerrno>
#include <vector>
#include <fstream>
#include <string>

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

#ifdef _WIN32
WriteAheadLog::WriteAheadLog(const std::string& path) : path_(path) {
    std::wstring wpath(path.begin(), path.end());
    HANDLE hFile = CreateFileW(
        wpath.c_str(),
        GENERIC_WRITE,
        FILE_SHARE_READ,
        nullptr,
        OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        throw WalError("Failed to open WAL file " + path + ": error code " + std::to_string(GetLastError()));
    }

    fd_ = hFile;

    // Move to the end of the file (append mode) and get the size
    LARGE_INTEGER liDistance = { 0 };
    LARGE_INTEGER liNewPointer;
    if (!SetFilePointerEx(hFile, liDistance, &liNewPointer, FILE_END)) {
        cleanup();
        throw WalError("Failed to seek WAL file " + path + ": error code " + std::to_string(GetLastError()));
    }

    size_ = static_cast<size_t>(liNewPointer.QuadPart);
}
#else
WriteAheadLog::WriteAheadLog(const std::string& path) : path_(path) {
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
#endif

WriteAheadLog::~WriteAheadLog() {
    cleanup();
}

WriteAheadLog::WriteAheadLog(WriteAheadLog&& other) noexcept
    : fd_(other.fd_), size_(other.size_), path_(std::move(other.path_)) {
#ifdef _WIN32
    other.fd_ = (void*)-1;
#else
    other.fd_ = -1;
#endif
    other.size_ = 0;
}

WriteAheadLog& WriteAheadLog::operator=(WriteAheadLog&& other) noexcept {
    if (this != &other) {
        cleanup();
        fd_ = other.fd_;
        size_ = other.size_;
        path_ = std::move(other.path_);
#ifdef _WIN32
        other.fd_ = (void*)-1;
#else
        other.fd_ = -1;
#endif
        other.size_ = 0;
    }
    return *this;
}

void WriteAheadLog::cleanup() {
#ifdef _WIN32
    if (fd_ != (void*)-1) {
        FlushFileBuffers(fd_);
        CloseHandle(fd_);
        fd_ = (void*)-1;
    }
#else
    if (fd_ != -1) {
        ::fsync(fd_);
        ::close(fd_);
        fd_ = -1;
    }
#endif
}

size_t WriteAheadLog::append(WalOpcode opcode, const std::string& key, const std::string& payload) {
#ifdef _WIN32
    if (fd_ == (void*)-1) throw WalError("WAL file is not open");
#else
    if (fd_ == -1) throw WalError("WAL file is not open");
#endif

    WalRecordHeader header;
    header.opcode = static_cast<uint8_t>(opcode);
    header.key_len = static_cast<uint32_t>(key.size());
    header.payload_len = static_cast<uint32_t>(payload.size());
    header.checksum = calculate_checksum(header, key, payload);

    std::vector<char> buffer(sizeof(WalRecordHeader) + key.size() + payload.size());

    char* ptr = buffer.data();
    std::memcpy(ptr, &header, sizeof(WalRecordHeader));
    ptr += sizeof(WalRecordHeader);

    std::memcpy(ptr, key.data(), key.size());
    ptr += key.size();

    std::memcpy(ptr, payload.data(), payload.size());

#ifdef _WIN32
    DWORD bytes_written = 0;
    if (!WriteFile(fd_, buffer.data(), static_cast<DWORD>(buffer.size()), &bytes_written, nullptr) || 
        bytes_written != buffer.size()) {
        throw WalError("Failed to write to WAL file: error code " + std::to_string(GetLastError()));
    }
#else
    ssize_t bytes_written = ::write(fd_, buffer.data(), buffer.size());
    if (bytes_written == -1 || static_cast<size_t>(bytes_written) != buffer.size()) {
        throw WalError("Failed to write to WAL file: " + std::string(std::strerror(errno)));
    }
#endif

    size_t current_offset = size_;
    size_ += buffer.size();

    return current_offset;
}

void WriteAheadLog::sync() {
#ifdef _WIN32
    if (fd_ != (void*)-1) {
        if (!FlushFileBuffers(fd_)) {
            throw WalError("Failed to FlushFileBuffers WAL file: error code " + std::to_string(GetLastError()));
        }
    }
#else
    if (fd_ != -1) {
        if (::fsync(fd_) == -1) {
            throw WalError("Failed to fsync WAL file: " + std::string(std::strerror(errno)));
        }
    }
#endif
}

void WriteAheadLog::recover(std::function<void(const WalEntry&)> callback) {
    // Open a separate read-only stream for recovery using portable std::ifstream
#ifdef _WIN32
    // Convert path to wide string for Unicode on Windows std::ifstream
    std::wstring wpath(path_.begin(), path_.end());
    std::ifstream infile(wpath, std::ios::binary);
#else
    std::ifstream infile(path_, std::ios::binary);
#endif

    if (!infile) {
        throw WalError("Failed to open WAL file for recovery " + path_);
    }

    while (infile) {
        WalRecordHeader header;
        infile.read(reinterpret_cast<char*>(&header), sizeof(WalRecordHeader));
        
        if (infile.gcount() != sizeof(WalRecordHeader)) {
            // Expected EOF or incomplete record at the end of the file
            break;
        }

        std::string key(header.key_len, '\0');
        infile.read(key.data(), header.key_len);
        if (infile.gcount() != static_cast<std::streamsize>(header.key_len)) {
            break;
        }

        std::string payload(header.payload_len, '\0');
        infile.read(payload.data(), header.payload_len);
        if (infile.gcount() != static_cast<std::streamsize>(header.payload_len)) {
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

        callback(entry);
    }
}

} // namespace sonatadb::storage