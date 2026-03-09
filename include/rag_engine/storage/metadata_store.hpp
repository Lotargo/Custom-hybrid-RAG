#pragma once

#include <string>
#include <memory>
#include <optional>
#include <stdexcept>

// Forward declaration of RocksDB types to avoid polluting headers
namespace rocksdb {
    class DB;
}

namespace rag_engine::storage {

class MetadataError : public std::runtime_error {
public:
    explicit MetadataError(const std::string& msg) : std::runtime_error(msg) {}
};

class MetadataStore {
public:
    explicit MetadataStore(const std::string& path);
    ~MetadataStore();

    // Prevent copying
    MetadataStore(const MetadataStore&) = delete;
    MetadataStore& operator=(const MetadataStore&) = delete;

    // Allow moving
    MetadataStore(MetadataStore&& other) noexcept;
    MetadataStore& operator=(MetadataStore&& other) noexcept;

    // K/V Operations
    void put(const std::string& key, const std::string& value);
    std::optional<std::string> get(const std::string& key) const;
    void remove(const std::string& key);

    // Explicit sync to disk
    void sync();

private:
    void cleanup();

    std::unique_ptr<rocksdb::DB> db_;
    std::string path_;
};

} // namespace rag_engine::storage