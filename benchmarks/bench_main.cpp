#include <benchmark/benchmark.h>

static void BM_DummyBenchmark(benchmark::State& state) {
    for (auto _ : state) {
        int result = 1 + 1;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_DummyBenchmark);

// benchmark_main provides main() function
