target_sources(rag_engine_core PRIVATE
    dummy.cpp
    storage/mmap_file.cpp
    storage/metadata_store.cpp
    storage/wal.cpp
    index/hnsw.cpp
)

find_package(RocksDB CONFIG REQUIRED)
find_package(roaring CONFIG REQUIRED)
find_path(LIBSTEMMER_INCLUDE_DIR libstemmer.h)
find_library(LIBSTEMMER_LIBRARY stemmer)
target_link_libraries(rag_engine_core PRIVATE RocksDB::rocksdb roaring::roaring ${LIBSTEMMER_LIBRARY})
