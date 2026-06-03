find_package(GTest CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

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
    c_api_test.cpp
    ../src/c_api/sonatadb_c.cpp
    test_onnx_runner.cpp
    network/tcp_server_tests.cpp
)

target_link_libraries(sonatadb_tests PRIVATE
    GTest::gtest_main
    sonatadb_core
    nlohmann_json::nlohmann_json
)

if(WIN32)
    target_link_libraries(sonatadb_tests PRIVATE ws2_32)
endif()


# Повторно используем прекомпилированные заголовки для ускорения сборки тестов
target_precompile_headers(sonatadb_tests REUSE_FROM sonatadb_core)

include(GoogleTest)
gtest_discover_tests(sonatadb_tests)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Generate dummy ONNX model for tests
add_custom_command(
    TARGET sonatadb_tests PRE_BUILD
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/generate_dummy_model.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating dummy ONNX model for tests"
)
