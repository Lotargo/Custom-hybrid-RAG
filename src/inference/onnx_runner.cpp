#include "sonatadb/inference/onnx_runner.hpp"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace sonatadb::inference {

OnnxModelRunner::OnnxModelRunner(const std::string& model_path, const OnnxRunnerConfig& config) {
    initialize_session(&model_path, {}, config);
}

OnnxModelRunner::OnnxModelRunner(std::span<const uint8_t> model_data, const OnnxRunnerConfig& config) {
    initialize_session(nullptr, model_data, config);
}

OnnxModelRunner::~OnnxModelRunner() = default;

OnnxModelRunner::OnnxModelRunner(OnnxModelRunner&&) noexcept = default;
OnnxModelRunner& OnnxModelRunner::operator=(OnnxModelRunner&&) noexcept = default;

void OnnxModelRunner::initialize_session(const std::string* model_path, std::span<const uint8_t> model_data, const OnnxRunnerConfig& config) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "sonatadb_inference");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(config.intra_op_num_threads);
    session_options.SetInterOpNumThreads(config.inter_op_num_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    if (model_path) {
#ifdef _WIN32
        std::wstring w_model_path(model_path->begin(), model_path->end());
        session_ = std::make_unique<Ort::Session>(*env_, w_model_path.c_str(), session_options);
#else
        session_ = std::make_unique<Ort::Session>(*env_, model_path->c_str(), session_options);
#endif
    } else {
        session_ = std::make_unique<Ort::Session>(*env_, model_data.data(), model_data.size(), session_options);
    }

    memory_info_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_input_nodes = session_->GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
        Ort::AllocatedStringPtr name = session_->GetInputNameAllocated(i, allocator);
        input_names_.push_back(name.get());
    }

    size_t num_output_nodes = session_->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
        Ort::AllocatedStringPtr name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.push_back(name.get());
    }

    for (const auto& str : input_names_) {
        input_names_c_str_.push_back(str.c_str());
    }
    for (const auto& str : output_names_) {
        output_names_c_str_.push_back(str.c_str());
    }

    if (!output_names_.empty()) {
        Ort::TypeInfo type_info = session_->GetOutputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_shape = tensor_info.GetShape();
        if (!output_shape.empty()) {
            output_dim_ = output_shape.back(); // Usually [batch, seq_len, hidden_dim] or [batch, hidden_dim]
        }
    }
}

size_t OnnxModelRunner::get_output_dim() const {
    return output_dim_;
}

std::vector<float> OnnxModelRunner::run(std::span<const int64_t> input_ids,
                                        std::span<const int64_t> attention_mask,
                                        std::span<const int64_t> token_type_ids) {
    if (input_ids.empty()) {
        throw std::invalid_argument("input_ids cannot be empty");
    }

    size_t seq_len = input_ids.size();

    std::vector<int64_t> mask;
    if (attention_mask.empty()) {
        mask.resize(seq_len, 1);
        attention_mask = {mask.data(), mask.size()};
    } else if (attention_mask.size() != seq_len) {
        throw std::invalid_argument("attention_mask size must match input_ids size");
    }

    std::vector<int64_t> type_ids;
    if (token_type_ids.empty() && std::find(input_names_.begin(), input_names_.end(), "token_type_ids") != input_names_.end()) {
        type_ids.resize(seq_len, 0);
        token_type_ids = {type_ids.data(), type_ids.size()};
    } else if (!token_type_ids.empty() && token_type_ids.size() != seq_len) {
        throw std::invalid_argument("token_type_ids size must match input_ids size");
    }

    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(seq_len)};
    std::vector<Ort::Value> input_tensors;

    // We expect standard transformer inputs: "input_ids", "attention_mask", "token_type_ids"
    // However, some models only take input_ids and attention_mask.

    // Convert to non-const for ORT API. We promise ORT won't modify the data by using memory info.
    int64_t* input_ids_ptr = const_cast<int64_t*>(input_ids.data());

    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        *memory_info_, input_ids_ptr, input_ids.size(), input_shape.data(), input_shape.size()));

    if (attention_mask.size() == seq_len) {
        int64_t* attention_mask_ptr = const_cast<int64_t*>(attention_mask.data());
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            *memory_info_, attention_mask_ptr, attention_mask.size(), input_shape.data(), input_shape.size()));
    }

    if (token_type_ids.size() == seq_len) {
        int64_t* token_type_ids_ptr = const_cast<int64_t*>(token_type_ids.data());
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            *memory_info_, token_type_ids_ptr, token_type_ids.size(), input_shape.data(), input_shape.size()));
    }

    // Match input names with provided tensors. We assume the order matches input_names_.
    // For a generic pipeline, this might need more robust matching if the ONNX model is arbitrary.

    std::vector<const char*> run_input_names;
    std::vector<Ort::Value> run_input_tensors;

    for (const auto& name : input_names_) {
        if (name == "input_ids") {
            run_input_names.push_back(name.c_str());
            run_input_tensors.push_back(std::move(input_tensors[0]));
        } else if (name == "attention_mask") {
            run_input_names.push_back(name.c_str());
            run_input_tensors.push_back(std::move(input_tensors[1]));
        } else if (name == "token_type_ids") {
            run_input_names.push_back(name.c_str());
            run_input_tensors.push_back(std::move(input_tensors[2]));
        } else if (name == "input") { // Some models just use "input" for simple dense layers
             run_input_names.push_back(name.c_str());
             run_input_tensors.push_back(std::move(input_tensors[0])); // just feed input_ids
        } else {
             // throw or ignore for simpler testing?
             // std::cerr << "Warning: Unknown input name " << name << std::endl;
        }
    }

    Ort::RunOptions run_options;
    auto output_tensors = session_->Run(run_options, run_input_names.data(), run_input_tensors.data(),
                                        run_input_names.size(), output_names_c_str_.data(), output_names_c_str_.size());

    // Expect output to be float. We take the first output.
    if (output_tensors.empty()) {
        throw std::runtime_error("No output tensors returned");
    }

    Ort::Value& output_tensor = output_tensors.front();
    auto type_info = output_tensor.GetTensorTypeAndShapeInfo();

    if (type_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("Expected float output tensor");
    }

    const float* float_array = output_tensor.GetTensorMutableData<float>();
    size_t count = type_info.GetElementCount();

    // Basic pooling (mean pooling) for sequence outputs, or just copy if it's already pooled.
    std::vector<int64_t> output_shape = type_info.GetShape();

    std::vector<float> result;
    if (output_shape.size() == 3) { // [batch, seq_len, hidden_dim]
        int64_t batch = output_shape[0]; // 1
        int64_t seq = output_shape[1];
        int64_t hidden = output_shape[2];

        if (batch != 1) {
            throw std::runtime_error("Batch size > 1 not supported yet");
        }

        // Example: mean pooling using attention mask (if available)
        result.resize(hidden, 0.0f);
        float sum_mask = 0.0f;

        for (int64_t i = 0; i < seq; ++i) {
            float mask_val = attention_mask.empty() ? 1.0f : static_cast<float>(attention_mask[i]);
            sum_mask += mask_val;

            for (int64_t j = 0; j < hidden; ++j) {
                result[j] += float_array[i * hidden + j] * mask_val;
            }
        }

        if (sum_mask > 0.0f) {
            for (int64_t j = 0; j < hidden; ++j) {
                result[j] /= sum_mask;
            }
        }
    } else if (output_shape.size() == 2) { // [batch, hidden_dim]
        result.assign(float_array, float_array + count);
    } else {
        // Flat or other shape
        result.assign(float_array, float_array + count);
    }

    return result;
}

} // namespace sonatadb::inference
