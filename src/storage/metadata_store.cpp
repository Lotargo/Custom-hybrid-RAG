#include "sonatadb/storage/metadata_store.hpp"
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/status.h>

namespace sonatadb::storage {

MetadataStore::MetadataStore(const std::string& path) : path_(path) {
    rocksdb::Options options;
    options.create_if_missing = true;

    // Optimize for low latency, as we have WAL for main crash-safety
    options.OptimizeLevelStyleCompaction();

    rocksdb::DB* db_ptr = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(options, path, &db_ptr);

    if (!status.ok()) {
        throw MetadataError("Failed to open MetadataStore at " + path + ": " + status.ToString());
    }

    db_.reset(db_ptr);
}

MetadataStore::~MetadataStore() {
    cleanup();
}

MetadataStore::MetadataStore(MetadataStore&& other) noexcept
    : db_(std::move(other.db_)), path_(std::move(other.path_)) {
}

MetadataStore& MetadataStore::operator=(MetadataStore&& other) noexcept {
    if (this != &other) {
        cleanup();
        db_ = std::move(other.db_);
        path_ = std::move(other.path_);
    }
    return *this;
}

void MetadataStore::cleanup() {
    if (db_) {
        // Force flush before destruction to ensure data is safe
        rocksdb::FlushOptions fopts;
        fopts.wait = true;
        db_->Flush(fopts);

        // Let the unique_ptr delete the DB
        db_.reset();
    }
}

void MetadataStore::put(const std::string& key, const std::string& value) {
    if (!db_) throw MetadataError("Database is closed");

    rocksdb::WriteOptions wopts;
    // We disable synchronous writing per-key if WAL is managed externally.
    wopts.sync = false;

    rocksdb::Status status = db_->Put(wopts, key, value);
    if (!status.ok()) {
        throw MetadataError("Failed to put key " + key + ": " + status.ToString());
    }
}

std::optional<std::string> MetadataStore::get(const std::string& key) const {
    if (!db_) throw MetadataError("Database is closed");

    std::string value;
    rocksdb::ReadOptions ropts;

    rocksdb::Status status = db_->Get(ropts, key, &value);

    if (status.IsNotFound()) {
        return std::nullopt;
    }

    if (!status.ok()) {
        throw MetadataError("Failed to get key " + key + ": " + status.ToString());
    }

    return value;
}

void MetadataStore::remove(const std::string& key) {
    if (!db_) throw MetadataError("Database is closed");

    rocksdb::WriteOptions wopts;
    wopts.sync = false;

    rocksdb::Status status = db_->Delete(wopts, key);
    if (!status.ok()) {
        throw MetadataError("Failed to remove key " + key + ": " + status.ToString());
    }
}

void MetadataStore::sync() {
    if (!db_) throw MetadataError("Database is closed");

    rocksdb::FlushOptions fopts;
    fopts.wait = true;
    rocksdb::Status status = db_->Flush(fopts);

    if (!status.ok()) {
        throw MetadataError("Failed to sync database: " + status.ToString());
    }
}

} // namespace sonatadb::storage