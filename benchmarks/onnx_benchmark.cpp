#include <benchmark/benchmark.h>
#include "sonatadb/inference/onnx_runner.hpp"
#include "sonatadb/inference/inference_pool.hpp"
#include <vector>
#include <fstream>
#include <stdexcept>

// Since we generate a dummy model during test build, we can assume it exists in the working directory
// or we generate one for the benchmark if needed. For simplicity, let's assume it exists.

using namespace sonatadb::inference;

static std::vector<uint8_t> load_dummy_model() {
    // Attempt to load from current directory, which in cmake build is usually the build/benchmarks or similar
    // Fallback path check might be needed.
    std::ifstream file("dummy_model.onnx", std::ios::binary);
    if (!file.is_open()) {
        file.open("../tests/dummy_model.onnx", std::ios::binary); // fallback if running from build
    }
    if (!file.is_open()) {
        file.open("../../tests/dummy_model.onnx", std::ios::binary);
    }
    if (!file.is_open()) {
        throw std::runtime_error("Could not find dummy_model.onnx for benchmark");
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static void BM_OnnxRunner_Sync(benchmark::State& state) {
    try {
        auto model_data = load_dummy_model();
        OnnxModelRunner runner(model_data);

        std::vector<int64_t> input_ids(state.range(0), 42);

        for (auto _ : state) {
            auto output = runner.run(input_ids);
            benchmark::DoNotOptimize(output);
        }
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}
BENCHMARK(BM_OnnxRunner_Sync)->Arg(16)->Arg(128)->Arg(512);

static void BM_OnnxRunner_ThreadPool(benchmark::State& state) {
    try {
        auto model_data = load_dummy_model();
        std::vector<int> cores = {0}; // bind to core 0
        InferencePool pool(cores, model_data);

        std::vector<int64_t> input_ids(state.range(0), 42);

        for (auto _ : state) {
            auto future = pool.submit_inference(input_ids);
            auto output = future.get();
            benchmark::DoNotOptimize(output);
        }
    } catch (const std::exception& e) {
        state.SkipWithError(e.what());
    }
}
BENCHMARK(BM_OnnxRunner_ThreadPool)->Arg(16)->Arg(128)->Arg(512);

BENCHMARK_MAIN();
