#include "sonatadb/storage/mmap_file.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include <cstring>
#include <cerrno>
#include <utility>
#include <string>
#include <stdexcept>

namespace sonatadb::storage {

#ifdef _WIN32
MmapFile::MmapFile(const std::string& path, MmapMode mode)
    : mode_(mode), path_(path) {

    DWORD dwDesiredAccess = GENERIC_READ;
    DWORD dwShareMode = FILE_SHARE_READ;
    DWORD dwCreationDisposition = OPEN_EXISTING;
    
    if (mode == MmapMode::ReadWrite) {
        dwDesiredAccess |= GENERIC_WRITE;
        dwShareMode |= FILE_SHARE_WRITE;
        dwCreationDisposition = OPEN_ALWAYS;
    }

    std::wstring wpath(path.begin(), path.end());
    HANDLE hFile = CreateFileW(
        wpath.c_str(),
        dwDesiredAccess,
        dwShareMode,
        nullptr,
        dwCreationDisposition,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        throw MmapError("Failed to open file " + path + ": error code " + std::to_string(GetLastError()));
    }

    fd_ = hFile;

    LARGE_INTEGER liSize;
    if (!GetFileSizeEx(hFile, &liSize)) {
        cleanup();
        throw MmapError("Failed to get file size " + path + ": error code " + std::to_string(GetLastError()));
    }

    size_ = static_cast<size_t>(liSize.QuadPart);

    if (size_ == 0) {
        return;
    }

    DWORD flProtect = (mode == MmapMode::ReadOnly) ? PAGE_READONLY : PAGE_READWRITE;
    HANDLE hMapping = CreateFileMappingW(
        hFile,
        nullptr,
        flProtect,
        0,
        0,
        nullptr
    );

    if (!hMapping) {
        cleanup();
        throw MmapError("Failed to create file mapping for " + path + ": error code " + std::to_string(GetLastError()));
    }

    mapping_handle_ = hMapping;

    DWORD dwMapAccess = (mode == MmapMode::ReadOnly) ? FILE_MAP_READ : FILE_MAP_ALL_ACCESS;
    data_ = MapViewOfFile(
        hMapping,
        dwMapAccess,
        0,
        0,
        0
    );

    if (!data_) {
        cleanup();
        throw MmapError("Failed to map view of file " + path + ": error code " + std::to_string(GetLastError()));
    }
}
#else
MmapFile::MmapFile(const std::string& path, MmapMode mode)
    : mode_(mode), path_(path) {

    int flags = (mode == MmapMode::ReadOnly) ? O_RDONLY : O_RDWR | O_CREAT;
    mode_t fd_mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

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

    if (size_ == 0) {
        return;
    }

    int prot = PROT_READ;
    if (mode == MmapMode::ReadWrite) {
        prot |= PROT_WRITE;
    }

    data_ = ::mmap(nullptr, size_, prot, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        cleanup();
        throw MmapError("Failed to mmap file " + path + ": " + std::strerror(errno));
    }
}
#endif

MmapFile::~MmapFile() {
    cleanup();
}

MmapFile::MmapFile(MmapFile&& other) noexcept
    : fd_(other.fd_),
#ifdef _WIN32
      mapping_handle_(other.mapping_handle_),
#endif
      data_(other.data_),
      size_(other.size_),
      mode_(other.mode_),
      path_(std::move(other.path_)) {
#ifdef _WIN32
    other.fd_ = (void*)-1;
    other.mapping_handle_ = nullptr;
#else
    other.fd_ = -1;
#endif
    other.data_ = nullptr;
    other.size_ = 0;
}

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
    if (this != &other) {
        cleanup();

        fd_ = other.fd_;
#ifdef _WIN32
        mapping_handle_ = other.mapping_handle_;
#endif
        data_ = other.data_;
        size_ = other.size_;
        mode_ = other.mode_;
        path_ = std::move(other.path_);

#ifdef _WIN32
        other.fd_ = (void*)-1;
        other.mapping_handle_ = nullptr;
#else
        other.fd_ = -1;
#endif
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void MmapFile::cleanup() {
    if (data_) {
#ifdef _WIN32
        UnmapViewOfFile(data_);
#else
        ::munmap(data_, size_);
#endif
        data_ = nullptr;
    }
#ifdef _WIN32
    if (mapping_handle_) {
        CloseHandle(mapping_handle_);
        mapping_handle_ = nullptr;
    }
    if (fd_ != (void*)-1) {
        CloseHandle(fd_);
        fd_ = (void*)-1;
    }
#else
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
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

    void* addr = static_cast<char*>(data_) + offset;

#ifdef _WIN32
    if (!FlushViewOfFile(addr, length)) {
        throw MmapError("FlushViewOfFile failed on " + path_ + ": error code " + std::to_string(GetLastError()));
    }
    if (!async) {
        if (!FlushFileBuffers(fd_)) {
            throw MmapError("FlushFileBuffers failed on " + path_ + ": error code " + std::to_string(GetLastError()));
        }
    }
#else
    int flags = async ? MS_ASYNC : MS_SYNC;
    if (::msync(addr, length, flags) == -1) {
        throw MmapError("msync failed on " + path_ + ": " + std::strerror(errno));
    }
#endif
}

void MmapFile::advise(MmapAdvice advice, size_t offset, size_t length) {
#ifdef _WIN32
    // No-op on Windows, advising is configured at file open time.
    (void)advice;
    (void)offset;
    (void)length;
#else
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
#endif
}

void MmapFile::resize(size_t new_size) {
    if (mode_ != MmapMode::ReadWrite) {
        throw MmapError("Cannot resize a read-only mmap file: " + path_);
    }

#ifdef _WIN32
    if (fd_ == (void*)-1) {
        throw MmapError("Invalid file descriptor for resize: " + path_);
    }

    if (data_) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (mapping_handle_) {
        CloseHandle(mapping_handle_);
        mapping_handle_ = nullptr;
    }

    LARGE_INTEGER liSize;
    liSize.QuadPart = static_cast<LONGLONG>(new_size);
    if (!SetFilePointerEx(fd_, liSize, nullptr, FILE_BEGIN)) {
        cleanup();
        throw MmapError("SetFilePointerEx failed on " + path_ + ": error code " + std::to_string(GetLastError()));
    }

    if (!SetEndOfFile(fd_)) {
        cleanup();
        throw MmapError("SetEndOfFile failed on " + path_ + ": error code " + std::to_string(GetLastError()));
    }

    size_ = new_size;

    if (size_ > 0) {
        HANDLE hMapping = CreateFileMappingW(
            fd_,
            nullptr,
            PAGE_READWRITE,
            0,
            0,
            nullptr
        );

        if (!hMapping) {
            cleanup();
            throw MmapError("Failed to recreate file mapping after resize " + path_ + ": error code " + std::to_string(GetLastError()));
        }

        mapping_handle_ = hMapping;

        data_ = MapViewOfFile(
            hMapping,
            FILE_MAP_ALL_ACCESS,
            0,
            0,
            0
        );

        if (!data_) {
            cleanup();
            throw MmapError("Failed to remap view after resize " + path_ + ": error code " + std::to_string(GetLastError()));
        }
    }
#else
    if (fd_ == -1) {
        throw MmapError("Invalid file descriptor for resize: " + path_);
    }

    if (data_ && data_ != MAP_FAILED) {
        ::munmap(data_, size_);
        data_ = nullptr;
    }

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
#endif
}

} // namespace sonatadb::storage
