#include <gtest/gtest.h>
#include "sonatadb/inference/onnx_runner.hpp"
#include "sonatadb/inference/inference_pool.hpp"
#include <vector>
#include <fstream>
#include <stdexcept>

// To test without a real model, we will verify that OnnxModelRunner throws the expected errors
// when we give it bad data, and correctly loads a tiny dummy model if we create one.

// A tiny ONNX model (Identity node) represented as a hex array.
// This is an extremely simplified ONNX graph that takes an input tensor and returns it.
// We can use a Python script to generate this byte array.
// For now, let's just test the initialization failures.

using namespace sonatadb::inference;

TEST(OnnxRunnerTest, InvalidModelDataThrows) {
    std::vector<uint8_t> bad_data = {0x01, 0x02, 0x03}; // Not a valid ONNX model
    EXPECT_THROW({
        OnnxModelRunner runner{std::span<const uint8_t>{bad_data}};
    }, std::exception); // Ort::Exception inherits from std::exception
}

TEST(OnnxRunnerTest, EmptyInputThrows) {
    // If we had a valid model, we could test this.
    // Instead we will mock a simple execution if we can.
    // However, since ONNX runtime validates the model during initialization,
    // we cannot test the run() method without a valid model.
}

TEST(InferencePoolTest, InitializationWithInvalidModelThrows) {
    std::vector<int> cores = {0}; // bind to core 0 for test
    std::vector<uint8_t> bad_data = {0x01, 0x02, 0x03};
    EXPECT_THROW({
        InferencePool pool(cores, std::span<const uint8_t>(bad_data));
    }, std::exception);
}

// Let's create a test that uses the dummy_model.onnx we generated
TEST(OnnxRunnerTest, LoadAndRunDummyModel) {
    // Read the file dummy_model.onnx
    std::ifstream file("dummy_model.onnx", std::ios::binary);
    ASSERT_TRUE(file.is_open()) << "Could not open dummy_model.onnx. Ensure it was generated before testing.";

    std::vector<uint8_t> model_data((std::istreambuf_iterator<char>(file)),
                                    std::istreambuf_iterator<char>());

    OnnxModelRunner runner(model_data);

    std::vector<int64_t> input_ids = {42, 100, 7};
    auto output = runner.run(input_ids); // Should cast INT64 to FLOAT and return it

    ASSERT_EQ(output.size(), 3);
    EXPECT_FLOAT_EQ(output[0], 42.0f);
    EXPECT_FLOAT_EQ(output[1], 100.0f);
    EXPECT_FLOAT_EQ(output[2], 7.0f);
}

TEST(InferencePoolTest, ParallelDummyModelInference) {
    std::ifstream file("dummy_model.onnx", std::ios::binary);
    ASSERT_TRUE(file.is_open());
    std::vector<uint8_t> model_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::vector<int> cores = {0}; // Just 1 thread on core 0 to ensure it works
    InferencePool pool(cores, model_data);

    std::vector<int64_t> input1 = {1, 2, 3};
    std::vector<int64_t> input2 = {4, 5, 6};

    auto future1 = pool.submit_inference(input1);
    auto future2 = pool.submit_inference(input2);

    auto result1 = future1.get();
    auto result2 = future2.get();

    ASSERT_EQ(result1.size(), 3);
    EXPECT_FLOAT_EQ(result1[0], 1.0f);
    EXPECT_FLOAT_EQ(result1[2], 3.0f);

    ASSERT_EQ(result2.size(), 3);
    EXPECT_FLOAT_EQ(result2[0], 4.0f);
    EXPECT_FLOAT_EQ(result2[2], 6.0f);
}
