#pragma once

#include <string>
#include <string_view>
#include <stdexcept>
#include <span>
#include <cstdint>
#include <cstddef>

namespace sonatadb::storage {

enum class MmapMode {
    ReadOnly,
    ReadWrite
};

enum class MmapAdvice {
    Normal,
    Random,
    Sequential,
    WillNeed,
    DontNeed
};

class MmapError : public std::runtime_error {
public:
    explicit MmapError(const std::string& msg) : std::runtime_error(msg) {}
};

class MmapFile {
public:
    // Opens a file and mmaps it
    MmapFile(const std::string& path, MmapMode mode);

    // Unmaps and closes the file
    ~MmapFile();

    // Prevent copying
    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;

    // Allow moving
    MmapFile(MmapFile&& other) noexcept;
    MmapFile& operator=(MmapFile&& other) noexcept;

    // Get a pointer to the mapped memory
    void* data() { return data_; }
    const void* data() const { return data_; }

    // Get as a span of bytes
    std::span<std::byte> span() {
        return std::span<std::byte>(static_cast<std::byte*>(data_), size_);
    }
    std::span<const std::byte> span() const {
        return std::span<const std::byte>(static_cast<const std::byte*>(data_), size_);
    }

    // Get the size of the mapped file
    size_t size() const { return size_; }

    // Synchronize a region with disk (msync)
    void sync(size_t offset = 0, size_t length = 0, bool async = false);

    // Provide advice to the OS (madvise)
    void advise(MmapAdvice advice, size_t offset = 0, size_t length = 0);

    // Resize the file (only allowed if ReadWrite).
    // This will unmap and remap! Pointers will be invalidated!
    void resize(size_t new_size);

private:
    void cleanup();

#ifdef _WIN32
    void* fd_ = (void*)-1;             // Windows HANDLE (INVALID_HANDLE_VALUE)
    void* mapping_handle_ = nullptr;   // Windows file mapping HANDLE
#else
    int fd_ = -1;
#endif
    void* data_ = nullptr;
    size_t size_ = 0;
    MmapMode mode_ = MmapMode::ReadOnly;
    std::string path_;
};

} // namespace sonatadb::storage
