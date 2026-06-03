target_sources(sonatadb_core PRIVATE
    dummy.cpp
    storage/mmap_file.cpp
    storage/metadata_store.cpp
    storage/wal.cpp
    index/hnsw.cpp
    index/rrf.cpp
    index/bloom_filter.cpp
    index/document_filter.cpp
    inference/thread_pool.cpp
    inference/onnx_runner.cpp
    inference/inference_pool.cpp
    network/protocol.cpp
    network/ring_buffer.cpp
    network/tcp_server.cpp
)

find_package(RocksDB CONFIG REQUIRED)
find_package(roaring CONFIG REQUIRED)
find_path(LIBSTEMMER_INCLUDE_DIR libstemmer.h)
find_library(LIBSTEMMER_LIBRARY stemmer)
find_package(onnxruntime CONFIG REQUIRED)

target_link_libraries(sonatadb_core PRIVATE RocksDB::rocksdb roaring::roaring ${LIBSTEMMER_LIBRARY} onnxruntime::onnxruntime)

# Подключаем liburing только для Linux при наличии
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_path(LIBURING_INCLUDE_DIRS liburing.h)
    find_library(LIBURING_LIBRARIES uring)
    
    target_link_libraries(sonatadb_core PRIVATE ${LIBURING_LIBRARIES})
    target_compile_definitions(sonatadb_core PRIVATE SONATADB_USE_IO_URING)
endif()

if(WIN32)
    target_link_libraries(sonatadb_core PRIVATE ws2_32)
endif()

# C-API Shared Library
add_library(sonatadb_c SHARED c_api/sonatadb_c.cpp)
target_link_libraries(sonatadb_c PRIVATE sonatadb_core roaring::roaring)
if(WIN32)
    target_link_libraries(sonatadb_c PRIVATE ws2_32)
endif()
target_include_directories(sonatadb_c PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)

