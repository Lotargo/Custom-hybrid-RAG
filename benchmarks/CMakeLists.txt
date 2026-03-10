find_package(benchmark CONFIG REQUIRED)
find_package(roaring CONFIG REQUIRED)

add_executable(rag_engine_benchmarks
    bench_main.cpp
    math/metrics_bench.cpp
    storage_benchmark.cpp
)

target_link_libraries(rag_engine_benchmarks PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    rag_engine_core
)

add_executable(hnsw_parameter_tuner hnsw_parameter_tuner.cpp)
target_link_libraries(hnsw_parameter_tuner PRIVATE rag_engine_core roaring::roaring-headers-cpp)
