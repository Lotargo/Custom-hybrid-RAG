find_package(GTest CONFIG REQUIRED)

add_executable(sonatadb_tests
    test_main.cpp
    math/metrics_test.cpp
    math/metrics_avx2_test.cpp
    math/metrics_avx512_test.cpp
    math/simd_dispatch_test.cpp
    storage_test.cpp
    metadata_store_test.cpp
    wal_test.cpp
    index/test_hnsw.cpp
    index/hnsw_tombstone_test.cpp
    index/test_hnsw_sift_recall.cpp
    test_compression.cpp
    test_tokenizer.cpp
    test_bpe_tokenizer.cpp
    test_wordpiece_tokenizer.cpp
    test_inverted_index.cpp
    test_rrf.cpp
    test_bloom_filter.cpp
    test_document_filter.cpp
    test_thread_pool.cpp
    network/protocol_tests.cpp
    network/ring_buffer_tests.cpp
    network/iouring_server_tests.cpp
)

target_link_libraries(sonatadb_tests PRIVATE
    GTest::gtest_main
    sonatadb_core
    nlohmann_json::nlohmann_json
)

include(GoogleTest)
gtest_discover_tests(sonatadb_tests)
