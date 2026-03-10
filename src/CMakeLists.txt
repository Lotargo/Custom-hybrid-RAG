target_sources(rag_engine_core PRIVATE
    dummy.cpp
    storage/mmap_file.cpp
    storage/metadata_store.cpp
    storage/wal.cpp
    index/hnsw.cpp
)

find_package(RocksDB CONFIG REQUIRED)
find_package(roaring CONFIG REQUIRED)
target_link_libraries(rag_engine_core PRIVATE RocksDB::rocksdb roaring::roaring)
