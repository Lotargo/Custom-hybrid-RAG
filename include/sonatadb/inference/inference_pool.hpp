#pragma once

#include "sonatadb/inference/thread_pool.hpp"
#include "sonatadb/inference/onnx_runner.hpp"
#include <vector>
#include <memory>
#include <future>
#include <string>
#include <span>

namespace sonatadb::inference {

/**
 * @brief Combines AffinityThreadPool with OnnxModelRunner instances to run
 * parallel inference tasks seamlessly.
 *
 * Creates one instance of OnnxModelRunner per thread to avoid locking contention
 * on a single ONNX session, as Session::Run is thread-safe but may have internal contention.
 */
class InferencePool {
public:
    /**
     * @brief Creates a pool of threads tied to specific cores, each with its own ONNX model instance.
     *
     * @param core_ids List of CPU cores to bind threads to.
     * @param model_path Path to the .onnx model.
     * @param config ONNX configuration.
     */
    InferencePool(const std::vector<int>& core_ids, const std::string& model_path, const OnnxRunnerConfig& config = {});

    /**
     * @brief Creates a pool using model bytes instead of a file.
     */
    InferencePool(const std::vector<int>& core_ids, std::span<const uint8_t> model_data, const OnnxRunnerConfig& config = {});

    ~InferencePool();

    /**
     * @brief Submits an inference task to the pool.
     */
    std::future<std::vector<float>> submit_inference(std::vector<int64_t> input_ids,
                                                     std::vector<int64_t> attention_mask = {},
                                                     std::vector<int64_t> token_type_ids = {});

private:
    std::unique_ptr<AffinityThreadPool> thread_pool_;

    // We use thread_local to hold the model runner for each thread, but since we use
    // standard threads in a pool, we can either use thread_local or pass it explicitly.
    // However, assigning thread_local index requires running on those specific threads.

    // To ensure thread indices are assigned safely, we wrap tasks with the runner.
    // Actually, `Ort::Session::Run` is thread-safe. To save memory (models can be large),
    // it's usually better to share a single ONNX session and let ORT handle threads.
    // BUT we have a custom AffinityThreadPool.
    // So we'll share a single OnnxModelRunner across all our threads.
    std::unique_ptr<OnnxModelRunner> runner_;
};

} // namespace sonatadb::inference
