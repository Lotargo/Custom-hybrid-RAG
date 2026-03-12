find_package(benchmark CONFIG REQUIRED)
find_package(roaring CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(sonatadb_benchmarks
    bench_main.cpp
    math/metrics_bench.cpp
    storage_benchmark.cpp
    thread_pool_benchmark.cpp
    bpe_benchmark.cpp
    onnx_benchmark.cpp
)

target_link_libraries(sonatadb_benchmarks PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    sonatadb_core
    nlohmann_json::nlohmann_json
)

add_executable(hnsw_parameter_tuner hnsw_parameter_tuner.cpp)
target_link_libraries(hnsw_parameter_tuner PRIVATE sonatadb_core roaring::roaring-headers-cpp)

# Generate dummy ONNX model for benchmarks if it doesn't exist
add_custom_command(
    TARGET sonatadb_benchmarks PRE_BUILD
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../tests/generate_dummy_model.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating dummy ONNX model for benchmarks"
)
