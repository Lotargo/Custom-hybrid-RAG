find_package(benchmark CONFIG REQUIRED)
find_package(roaring CONFIG REQUIRED)

add_executable(sonatadb_benchmarks
    bench_main.cpp
    math/metrics_bench.cpp
    storage_benchmark.cpp
    thread_pool_benchmark.cpp
    bpe_benchmark.cpp
)

target_link_libraries(sonatadb_benchmarks PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    sonatadb_core
    nlohmann_json::nlohmann_json
)

add_executable(hnsw_parameter_tuner hnsw_parameter_tuner.cpp)
target_link_libraries(hnsw_parameter_tuner PRIVATE sonatadb_core roaring::roaring-headers-cpp)
