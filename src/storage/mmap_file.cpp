#include "rag_engine/storage/mmap_file.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <utility>

namespace rag_engine::storage {

MmapFile::MmapFile(const std::string& path, MmapMode mode)
    : mode_(mode), path_(path) {

    int flags = (mode == MmapMode::ReadOnly) ? O_RDONLY : O_RDWR | O_CREAT;
    mode_t fd_mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH; // 0644

    fd_ = ::open(path.c_str(), flags, fd_mode);
    if (fd_ == -1) {
        throw MmapError("Failed to open file " + path + ": " + std::strerror(errno));
    }

    struct stat st;
    if (::fstat(fd_, &st) == -1) {
        cleanup();
        throw MmapError("Failed to stat file " + path + ": " + std::strerror(errno));
    }

    size_ = st.st_size;

    // Handle empty file case
    if (size_ == 0) {
        // If it's read-only and empty, there's nothing to map.
        // We'll leave data_ as nullptr.
        // If it's ReadWrite and empty, the user might call resize() later to allocate space.
        return;
    }

    int prot = PROT_READ;
    if (mode == MmapMode::ReadWrite) {
        prot |= PROT_WRITE;
    }

    // MAP_SHARED is typical for memory-mapped files
    data_ = ::mmap(nullptr, size_, prot, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        cleanup();
        throw MmapError("Failed to mmap file " + path + ": " + std::strerror(errno));
    }
}

MmapFile::~MmapFile() {
    cleanup();
}

MmapFile::MmapFile(MmapFile&& other) noexcept
    : fd_(other.fd_),
      data_(other.data_),
      size_(other.size_),
      mode_(other.mode_),
      path_(std::move(other.path_)) {
    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
}

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
    if (this != &other) {
        cleanup();

        fd_ = other.fd_;
        data_ = other.data_;
        size_ = other.size_;
        mode_ = other.mode_;
        path_ = std::move(other.path_);

        other.fd_ = -1;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void MmapFile::cleanup() {
    if (data_ && data_ != MAP_FAILED) {
        ::munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
    size_ = 0;
}

void MmapFile::sync(size_t offset, size_t length, bool async) {
    if (!data_ || mode_ != MmapMode::ReadWrite) return;

    if (length == 0) {
        length = size_ - offset;
    }

    if (offset + length > size_) {
        throw MmapError("msync bounds out of range");
    }

    int flags = async ? MS_ASYNC : MS_SYNC;

    // Pointer math for syncing specific offsets
    void* addr = static_cast<char*>(data_) + offset;

    if (::msync(addr, length, flags) == -1) {
        throw MmapError("msync failed on " + path_ + ": " + std::strerror(errno));
    }
}

void MmapFile::advise(MmapAdvice advice, size_t offset, size_t length) {
    if (!data_) return;

    if (length == 0) {
        length = size_ - offset;
    }

    if (offset + length > size_) {
        throw MmapError("madvise bounds out of range");
    }

    int madv = MADV_NORMAL;
    switch (advice) {
        case MmapAdvice::Normal:     madv = MADV_NORMAL; break;
        case MmapAdvice::Random:     madv = MADV_RANDOM; break;
        case MmapAdvice::Sequential: madv = MADV_SEQUENTIAL; break;
        case MmapAdvice::WillNeed:   madv = MADV_WILLNEED; break;
        case MmapAdvice::DontNeed:   madv = MADV_DONTNEED; break;
    }

    void* addr = static_cast<char*>(data_) + offset;

    if (::madvise(addr, length, madv) == -1) {
        throw MmapError("madvise failed on " + path_ + ": " + std::strerror(errno));
    }
}

void MmapFile::resize(size_t new_size) {
    if (mode_ != MmapMode::ReadWrite) {
        throw MmapError("Cannot resize a read-only mmap file: " + path_);
    }
    if (fd_ == -1) {
        throw MmapError("Invalid file descriptor for resize: " + path_);
    }

    // Unmap current
    if (data_ && data_ != MAP_FAILED) {
        ::munmap(data_, size_);
        data_ = nullptr;
    }

    // Truncate file
    if (::ftruncate(fd_, new_size) == -1) {
        cleanup();
        throw MmapError("ftruncate failed on " + path_ + ": " + std::strerror(errno));
    }

    size_ = new_size;

    if (size_ > 0) {
        int prot = PROT_READ | PROT_WRITE;
        data_ = ::mmap(nullptr, size_, prot, MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            cleanup();
            throw MmapError("Failed to remap file after resize " + path_ + ": " + std::strerror(errno));
        }
    }
}

} // namespace rag_engine::storage
