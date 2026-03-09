#include <benchmark/benchmark.h>
#include "rag_engine/storage/mmap_file.hpp"

#include <fstream>
#include <filesystem>
#include <vector>
#include <string>

namespace {
    const std::string BENCH_FILE = "bench_mmap.bin";
    constexpr size_t FILE_SIZE = 100 * 1024 * 1024; // 100 MB
}

class MmapFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) {
        if (!std::filesystem::exists(BENCH_FILE)) {
            std::ofstream out(BENCH_FILE, std::ios::binary);
            std::vector<char> buffer(1024 * 1024, 'X');
            for (size_t i = 0; i < FILE_SIZE / buffer.size(); ++i) {
                out.write(buffer.data(), buffer.size());
            }
        }
    }

    void TearDown(const ::benchmark::State& state) {
        // Keep the file to reuse across runs to save setup time.
        // Uncomment to remove after all runs:
        // if (std::filesystem::exists(BENCH_FILE)) {
        //     std::filesystem::remove(BENCH_FILE);
        // }
    }
};

BENCHMARK_F(MmapFixture, StdIfstreamRead)(benchmark::State& state) {
    for (auto _ : state) {
        std::ifstream in(BENCH_FILE, std::ios::binary);
        if (!in) {
            state.SkipWithError("Failed to open bench file");
            return;
        }

        std::vector<char> buffer(FILE_SIZE);
        in.read(buffer.data(), FILE_SIZE);
        benchmark::DoNotOptimize(buffer.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * FILE_SIZE);
}

BENCHMARK_F(MmapFixture, MmapRead)(benchmark::State& state) {
    for (auto _ : state) {
        rag_engine::storage::MmapFile file(BENCH_FILE, rag_engine::storage::MmapMode::ReadOnly);
        // Advise the OS that we're going to read sequentially
        file.advise(rag_engine::storage::MmapAdvice::Sequential);

        auto span = file.span();

        // Emulate reading by touching all pages
        uint64_t sum = 0;
        const auto* data = span.data();
        for (size_t i = 0; i < span.size(); i += 4096) {
            sum += static_cast<uint64_t>(data[i]);
        }
        benchmark::DoNotOptimize(sum);
        benchmark::DoNotOptimize(data);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * FILE_SIZE);
}
