#include <benchmark/benchmark.h>
#include "rag_engine/inference/thread_pool.hpp"
#include <vector>
#include <numeric>

#if defined(__linux__)
#include <sched.h>
#endif

using namespace rag_engine::inference;

// Helper to get available cores for benchmarking
std::vector<int> get_available_cores() {
    std::vector<int> cores;
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) == 0) {
        for (int i = 0; i < CPU_SETSIZE; ++i) {
            if (CPU_ISSET(i, &cpuset)) {
                cores.push_back(i);
                if (cores.size() >= 4) break; // Limit to 4 for consistency in benchmark
            }
        }
    }
#endif
    if (cores.empty()) {
        cores = {0, 0, 0, 0}; // Fallback
    }
    return cores;
}

static void BM_ThreadPool_EmptyTask(benchmark::State& state) {
    auto cores = get_available_cores();
    AffinityThreadPool pool(cores);

    for (auto _ : state) {
        auto future = pool.submit([]() { benchmark::DoNotOptimize(1); });
        future.wait();
    }
}
BENCHMARK(BM_ThreadPool_EmptyTask)->UseRealTime();

static void BM_ThreadPool_SubmitLatency(benchmark::State& state) {
    auto cores = get_available_cores();
    AffinityThreadPool pool(cores);

    for (auto _ : state) {
        // Just measure submit without waiting to see lock overhead
        pool.submit([]() { benchmark::DoNotOptimize(1); });
    }
}
BENCHMARK(BM_ThreadPool_SubmitLatency)->UseRealTime();

static void BM_ThreadPool_BatchTasks(benchmark::State& state) {
    auto cores = get_available_cores();
    AffinityThreadPool pool(cores);
    const int num_tasks = state.range(0);

    for (auto _ : state) {
        std::vector<std::future<int>> futures;
        futures.reserve(num_tasks);

        for (int i = 0; i < num_tasks; ++i) {
            futures.push_back(pool.submit([i]() { return i * 2; }));
        }

        for (auto& f : futures) {
            benchmark::DoNotOptimize(f.get());
        }
    }
}
BENCHMARK(BM_ThreadPool_BatchTasks)->Range(10, 10000)->UseRealTime();
