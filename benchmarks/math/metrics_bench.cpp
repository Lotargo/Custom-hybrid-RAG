#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include "rag_engine/math/metrics.hpp"
#include "rag_engine/math/metrics_avx2.hpp"

using namespace rag_engine::math;

static std::vector<float> generate_random_vector(size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::vector<float> vec(size);
    for(size_t i = 0; i < size; ++i) {
        vec[i] = dis(gen);
    }
    return vec;
}

static void BM_DotProductScalar(benchmark::State& state) {
    auto a = generate_random_vector(static_cast<size_t>(state.range(0)));
    auto b = generate_random_vector(static_cast<size_t>(state.range(0)));

    for (auto _ : state) {
        float result = dot_product_scalar<float>(a, b);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_DotProductScalar)->Range(8, 8192);

static void BM_DotProductAVX2(benchmark::State& state) {
    auto a = generate_random_vector(static_cast<size_t>(state.range(0)));
    auto b = generate_random_vector(static_cast<size_t>(state.range(0)));

    for (auto _ : state) {
        float result = dot_product_avx2(a, b);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_DotProductAVX2)->Range(8, 8192);

static void BM_L2DistanceScalar(benchmark::State& state) {
    auto a = generate_random_vector(static_cast<size_t>(state.range(0)));
    auto b = generate_random_vector(static_cast<size_t>(state.range(0)));

    for (auto _ : state) {
        float result = l2_distance_scalar<float>(a, b);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_L2DistanceScalar)->Range(8, 8192);

static void BM_L2DistanceAVX2(benchmark::State& state) {
    auto a = generate_random_vector(static_cast<size_t>(state.range(0)));
    auto b = generate_random_vector(static_cast<size_t>(state.range(0)));

    for (auto _ : state) {
        float result = l2_distance_avx2(a, b);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_L2DistanceAVX2)->Range(8, 8192);
