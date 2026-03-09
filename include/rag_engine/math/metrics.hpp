#pragma once

#include <cmath>
#include <concepts>
#include <numeric>
#include <span>
#include <stdexcept>

namespace rag_engine::math {

// Концепт для проверки, что тип является арифметическим (float, double, int и т.д.)
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

/**
 * @brief Вычисляет скалярное произведение (Dot Product) двух векторов.
 * @param a Первый вектор (через std::span)
 * @param b Второй вектор (через std::span)
 * @return Скалярное произведение
 */
template <Arithmetic T>
inline T dot_product_scalar(std::span<const T> a, std::span<const T> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    T result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

/**
 * @brief Вычисляет L2 дистанцию (Евклидово расстояние в квадрате) между двумя векторами.
 * @param a Первый вектор (через std::span)
 * @param b Второй вектор (через std::span)
 * @return Квадрат L2 дистанции
 */
template <Arithmetic T>
inline T l2_distance_scalar(std::span<const T> a, std::span<const T> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    T result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        T diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

/**
 * @brief Вычисляет косинусное сходство (Cosine Similarity) между двумя векторами.
 * @param a Первый вектор (через std::span)
 * @param b Второй вектор (через std::span)
 * @return Косинусное сходство
 */
template <Arithmetic T>
inline T cosine_similarity_scalar(std::span<const T> a, std::span<const T> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    T dot = 0;
    T norm_a = 0;
    T norm_b = 0;

    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a == 0 || norm_b == 0) {
        return 0; // Или бросать исключение, если векторы нулевые
    }

    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

} // namespace rag_engine::math
