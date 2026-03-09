#pragma once

#include <span>
#include <stdexcept>
#include <immintrin.h>

namespace rag_engine::math {

/**
 * @brief Вычисляет скалярное произведение (Dot Product) двух векторов используя AVX2.
 *        Работает с float32. Длина вектора не обязана быть кратной 8, "хвост" обрабатывается скалярно.
 */
inline float dot_product_avx2(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    size_t i = 0;
    size_t size = a.size();

    // Аккумуляторы из 8 float
    __m256 sum256 = _mm256_setzero_ps();

    // Обрабатываем блоки по 8 элементов
    for (; i + 7 < size; i += 8) {
        // Загружаем невыровненные данные
        __m256 vecA = _mm256_loadu_ps(&a[i]);
        __m256 vecB = _mm256_loadu_ps(&b[i]);

        // Fused Multiply-Add: sum256 += vecA * vecB
        sum256 = _mm256_fmadd_ps(vecA, vecB, sum256);
    }

    // Горизонтальное сложение внутри 256-битного регистра
    // sum256 = [a, b, c, d, e, f, g, h]
    // 1. Извлекаем верхнюю и нижнюю 128-битные части и складываем
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
    // sum128 = [a+e, b+f, c+g, d+h] = [A, B, C, D]

    // 2. Складываем половины sum128
    __m128 sum128_high = _mm_movehl_ps(sum128, sum128); // [C, D, C, D]
    sum128 = _mm_add_ps(sum128, sum128_high); // [A+C, B+D, C, D]

    // 3. Складываем оставшиеся 2 элемента
    __m128 sum128_dup = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1, 1, 1, 1)); // [B+D, B+D, ...]
    sum128 = _mm_add_ss(sum128, sum128_dup);

    float result = _mm_cvtss_f32(sum128);

    // Обрабатываем хвост (если размер не кратен 8)
    for (; i < size; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

/**
 * @brief Вычисляет L2 дистанцию (Евклидово расстояние в квадрате) используя AVX2.
 */
inline float l2_distance_avx2(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    size_t i = 0;
    size_t size = a.size();

    __m256 sum256 = _mm256_setzero_ps();

    for (; i + 7 < size; i += 8) {
        __m256 vecA = _mm256_loadu_ps(&a[i]);
        __m256 vecB = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(vecA, vecB);
        sum256 = _mm256_fmadd_ps(diff, diff, sum256);
    }

    // Горизонтальное сложение
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
    __m128 sum128_high = _mm_movehl_ps(sum128, sum128);
    sum128 = _mm_add_ps(sum128, sum128_high);
    __m128 sum128_dup = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1, 1, 1, 1));
    sum128 = _mm_add_ss(sum128, sum128_dup);

    float result = _mm_cvtss_f32(sum128);

    for (; i < size; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }

    return result;
}

/**
 * @brief Вычисляет Косинусное сходство используя AVX2.
 */
inline float cosine_similarity_avx2(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    size_t i = 0;
    size_t size = a.size();

    __m256 dot256 = _mm256_setzero_ps();
    __m256 normA256 = _mm256_setzero_ps();
    __m256 normB256 = _mm256_setzero_ps();

    for (; i + 7 < size; i += 8) {
        __m256 vecA = _mm256_loadu_ps(&a[i]);
        __m256 vecB = _mm256_loadu_ps(&b[i]);

        dot256 = _mm256_fmadd_ps(vecA, vecB, dot256);
        normA256 = _mm256_fmadd_ps(vecA, vecA, normA256);
        normB256 = _mm256_fmadd_ps(vecB, vecB, normB256);
    }

    // Горизонтальное сложение для dot
    __m128 sum128_dot = _mm_add_ps(_mm256_castps256_ps128(dot256), _mm256_extractf128_ps(dot256, 1));
    __m128 high_dot = _mm_movehl_ps(sum128_dot, sum128_dot);
    sum128_dot = _mm_add_ps(sum128_dot, high_dot);
    __m128 dup_dot = _mm_shuffle_ps(sum128_dot, sum128_dot, _MM_SHUFFLE(1, 1, 1, 1));
    sum128_dot = _mm_add_ss(sum128_dot, dup_dot);
    float dot = _mm_cvtss_f32(sum128_dot);

    // Горизонтальное сложение для normA
    __m128 sum128_a = _mm_add_ps(_mm256_castps256_ps128(normA256), _mm256_extractf128_ps(normA256, 1));
    __m128 high_a = _mm_movehl_ps(sum128_a, sum128_a);
    sum128_a = _mm_add_ps(sum128_a, high_a);
    __m128 dup_a = _mm_shuffle_ps(sum128_a, sum128_a, _MM_SHUFFLE(1, 1, 1, 1));
    sum128_a = _mm_add_ss(sum128_a, dup_a);
    float norm_a = _mm_cvtss_f32(sum128_a);

    // Горизонтальное сложение для normB
    __m128 sum128_b = _mm_add_ps(_mm256_castps256_ps128(normB256), _mm256_extractf128_ps(normB256, 1));
    __m128 high_b = _mm_movehl_ps(sum128_b, sum128_b);
    sum128_b = _mm_add_ps(sum128_b, high_b);
    __m128 dup_b = _mm_shuffle_ps(sum128_b, sum128_b, _MM_SHUFFLE(1, 1, 1, 1));
    sum128_b = _mm_add_ss(sum128_b, dup_b);
    float norm_b = _mm_cvtss_f32(sum128_b);

    for (; i < size; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 0.0f;
    }

    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

} // namespace rag_engine::math
