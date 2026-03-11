target_sources(rag_engine_core PRIVATE
    dummy.cpp
    storage/mmap_file.cpp
    storage/metadata_store.cpp
    storage/wal.cpp
    index/hnsw.cpp
    index/rrf.cpp
    index/bloom_filter.cpp
    index/document_filter.cpp
    inference/thread_pool.cpp
    network/protocol.cpp
    network/ring_buffer.cpp
    network/iouring_server.cpp
)

find_package(RocksDB CONFIG REQUIRED)
find_package(roaring CONFIG REQUIRED)
find_path(LIBSTEMMER_INCLUDE_DIR libstemmer.h)
find_library(LIBSTEMMER_LIBRARY stemmer)

# Подключаем liburing
find_path(LIBURING_INCLUDE_DIRS liburing.h)
find_library(LIBURING_LIBRARIES uring)

target_link_libraries(rag_engine_core PRIVATE RocksDB::rocksdb roaring::roaring ${LIBSTEMMER_LIBRARY} ${LIBURING_LIBRARIES})
