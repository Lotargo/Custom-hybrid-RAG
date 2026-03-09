find_package(benchmark CONFIG REQUIRED)

add_executable(rag_engine_benchmarks
    bench_main.cpp
    math/metrics_bench.cpp
)

target_link_libraries(rag_engine_benchmarks PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    rag_engine_core
)
