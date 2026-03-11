#include "sonatadb/network/ring_buffer.h"

#include <cstring>
#include <algorithm>

namespace sonatadb::network {

RingBuffer::RingBuffer(size_t capacity)
    : buffer_(capacity, 0), capacity_(capacity) {}

bool RingBuffer::write(std::span<const uint8_t> data) {
    if (data.empty()) return true;
    if (free_space() < data.size()) return false;

    size_t till_end = capacity_ - write_pos_;
    if (data.size() <= till_end) {
        std::memcpy(&buffer_[write_pos_], data.data(), data.size());
        write_pos_ += data.size();
        if (write_pos_ == capacity_) write_pos_ = 0;
    } else {
        std::memcpy(&buffer_[write_pos_], data.data(), till_end);
        std::memcpy(&buffer_[0], data.data() + till_end, data.size() - till_end);
        write_pos_ = data.size() - till_end;
    }

    if (write_pos_ == read_pos_) {
        is_full_ = true;
    }

    return true;
}

size_t RingBuffer::peek(std::span<uint8_t> out_data) const {
    size_t to_read = std::min(out_data.size(), available());
    if (to_read == 0) return 0;

    size_t first_part;
    if (write_pos_ > read_pos_) {
        first_part = write_pos_ - read_pos_;
    } else {
        first_part = capacity_ - read_pos_;
    }

    if (to_read <= first_part) {
        std::memcpy(out_data.data(), &buffer_[read_pos_], to_read);
    } else {
        std::memcpy(out_data.data(), &buffer_[read_pos_], first_part);
        std::memcpy(out_data.data() + first_part, &buffer_[0], to_read - first_part);
    }

    return to_read;
}

std::span<const uint8_t> RingBuffer::read() const {
    if (available() == 0) return {};

    size_t read_len;
    if (write_pos_ > read_pos_) {
        read_len = write_pos_ - read_pos_;
    } else {
        read_len = capacity_ - read_pos_;
    }

    return {&buffer_[read_pos_], read_len};
}

void RingBuffer::advance(size_t bytes) {
    if (bytes == 0) return;
    if (bytes > available()) {
        throw std::runtime_error("Advancing beyond available data in RingBuffer");
    }

    read_pos_ = (read_pos_ + bytes) % capacity_;
    is_full_ = false;
}

size_t RingBuffer::available() const {
    if (is_full_) return capacity_;
    if (write_pos_ >= read_pos_) {
        return write_pos_ - read_pos_;
    }
    return capacity_ - read_pos_ + write_pos_;
}

size_t RingBuffer::free_space() const {
    return capacity_ - available();
}

void RingBuffer::clear() {
    read_pos_ = 0;
    write_pos_ = 0;
    is_full_ = false;
}

std::span<uint8_t> RingBuffer::write_buffer() {
    if (is_full_) return {};

    size_t write_len;
    if (write_pos_ >= read_pos_) {
        write_len = capacity_ - write_pos_;
    } else {
        write_len = read_pos_ - write_pos_;
    }

    return {&buffer_[write_pos_], write_len};
}

void RingBuffer::commit_write(size_t bytes) {
    if (bytes > free_space()) {
        throw std::runtime_error("Committing more bytes than free space");
    }

    write_pos_ = (write_pos_ + bytes) % capacity_;
    if (write_pos_ == read_pos_ && bytes > 0) {
        is_full_ = true;
    }
}

} // namespace sonatadb::network
