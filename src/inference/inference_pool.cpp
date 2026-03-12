#include "sonatadb/inference/inference_pool.hpp"

namespace sonatadb::inference {

InferencePool::InferencePool(const std::vector<int>& core_ids, const std::string& model_path, const OnnxRunnerConfig& config) {
    runner_ = std::make_unique<OnnxModelRunner>(model_path, config);
    thread_pool_ = std::make_unique<AffinityThreadPool>(core_ids);
}

InferencePool::InferencePool(const std::vector<int>& core_ids, std::span<const uint8_t> model_data, const OnnxRunnerConfig& config) {
    runner_ = std::make_unique<OnnxModelRunner>(model_data, config);
    thread_pool_ = std::make_unique<AffinityThreadPool>(core_ids);
}

InferencePool::~InferencePool() = default;

std::future<std::vector<float>> InferencePool::submit_inference(std::vector<int64_t> input_ids,
                                                                std::vector<int64_t> attention_mask,
                                                                std::vector<int64_t> token_type_ids) {
    // We move the vectors into the lambda to avoid copying, but the runner takes span.
    // The lambda needs to be mutable to move from the captured vectors.
    // We share the runner across threads. Ort::Session::Run is thread-safe.
    return thread_pool_->submit([this,
                                 input_ids = std::move(input_ids),
                                 attention_mask = std::move(attention_mask),
                                 token_type_ids = std::move(token_type_ids)]() mutable -> std::vector<float> {
        return runner_->run(input_ids, attention_mask, token_type_ids);
    });
}

} // namespace sonatadb::inference
