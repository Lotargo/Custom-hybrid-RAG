#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace rag_engine::test_utils {

class SiftParser {
public:
    static std::vector<std::vector<float>> read_fvecs(const std::string& filepath, size_t max_vectors = 0) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open fvecs file: " + filepath);
        }

        std::vector<std::vector<float>> data;
        while (file.peek() != EOF) {
            int dim;
            if (!file.read(reinterpret_cast<char*>(&dim), sizeof(int))) {
                break;
            }

            if (dim <= 0) {
                throw std::runtime_error("Invalid dimension in fvecs file.");
            }

            std::vector<float> vec(dim);
            if (!file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float))) {
                break;
            }

            data.push_back(std::move(vec));

            if (max_vectors > 0 && data.size() >= max_vectors) {
                break;
            }
        }
        return data;
    }

    static std::vector<std::vector<int>> read_ivecs(const std::string& filepath, size_t max_vectors = 0) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open ivecs file: " + filepath);
        }

        std::vector<std::vector<int>> data;
        while (file.peek() != EOF) {
            int k;
            if (!file.read(reinterpret_cast<char*>(&k), sizeof(int))) {
                break;
            }

            if (k <= 0) {
                throw std::runtime_error("Invalid length in ivecs file.");
            }

            std::vector<int> vec(k);
            if (!file.read(reinterpret_cast<char*>(vec.data()), k * sizeof(int))) {
                break;
            }

            data.push_back(std::move(vec));

            if (max_vectors > 0 && data.size() >= max_vectors) {
                break;
            }
        }
        return data;
    }
};

} // namespace rag_engine::test_utils
