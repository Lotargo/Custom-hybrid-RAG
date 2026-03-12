#pragma once

#include <string>
#include <vector>
#include <memory>
#include <span>
#include <stdexcept>

// Forward declarations for ONNX Runtime C++ API to avoid including heavy headers in our header
namespace Ort {
    class Env;
    class Session;
    class MemoryInfo;
}

namespace sonatadb::inference {

/**
 * @brief Configuration for the ONNX Runtime session.
 */
struct OnnxRunnerConfig {
    int intra_op_num_threads = 1;
    int inter_op_num_threads = 1;
    bool use_cpu_execution_provider = true;
    // Add other execution providers (e.g., CUDA, TensorRT, CoreML) here if needed.
};

/**
 * @brief A wrapper around ONNX Runtime for running embedding models.
 *
 * Optimized for C++20 with std::span for zero-copy where possible.
 */
class OnnxModelRunner {
public:
    /**
     * @brief Constructs an OnnxModelRunner and loads the model from disk.
     *
     * @param model_path Path to the .onnx model file.
     * @param config Configuration for the ONNX Runtime session.
     */
    OnnxModelRunner(const std::string& model_path, const OnnxRunnerConfig& config = {});

    /**
     * @brief Constructs an OnnxModelRunner and loads the model from memory.
     *
     * @param model_data The raw bytes of the .onnx model.
     * @param config Configuration for the ONNX Runtime session.
     */
    OnnxModelRunner(std::span<const uint8_t> model_data, const OnnxRunnerConfig& config = {});

    ~OnnxModelRunner();

    // Prevent copying because of internal state (pointers to ORT objects)
    OnnxModelRunner(const OnnxModelRunner&) = delete;
    OnnxModelRunner& operator=(const OnnxModelRunner&) = delete;

    OnnxModelRunner(OnnxModelRunner&&) noexcept;
    OnnxModelRunner& operator=(OnnxModelRunner&&) noexcept;

    /**
     * @brief Runs inference on a single sequence of tokens.
     *
     * @param input_ids The token IDs (e.g., from BPE/WordPiece).
     * @param attention_mask The attention mask (1 for real tokens, 0 for padding). If empty, assumes all 1s.
     * @param token_type_ids The token type IDs (usually 0). If empty, assumes all 0s.
     * @return A vector containing the embedding (e.g., float32 array).
     */
    std::vector<float> run(std::span<const int64_t> input_ids,
                           std::span<const int64_t> attention_mask = {},
                           std::span<const int64_t> token_type_ids = {});

    /**
     * @brief Get the expected dimension of the output embedding.
     */
    size_t get_output_dim() const;

private:
    void initialize_session(const std::string* model_path, std::span<const uint8_t> model_data, const OnnxRunnerConfig& config);

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_c_str_;
    std::vector<const char*> output_names_c_str_;

    size_t output_dim_ = 0;
};

} // namespace sonatadb::inference
