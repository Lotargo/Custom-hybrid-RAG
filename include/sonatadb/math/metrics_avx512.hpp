#pragma once

#include <span>
#include <stdexcept>
#include <cmath>
#include "sonatadb/math/metrics.hpp"

// Защита от компиляции на не-x86 архитектурах (например, ARM)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <immintrin.h>

namespace sonatadb::math {

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f")))
#endif
inline float dot_product_avx512(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    size_t i = 0;
    size_t size = a.size();

#if defined(__AVX512F__) || defined(__GNUC__) || defined(__clang__)
    __m512 sum512 = _mm512_setzero_ps();

    for (; i + 15 < size; i += 16) {
        __m512 vecA = _mm512_loadu_ps(&a[i]);
        __m512 vecB = _mm512_loadu_ps(&b[i]);
        sum512 = _mm512_fmadd_ps(vecA, vecB, sum512);
    }

    float result = _mm512_reduce_add_ps(sum512);
#else
    float result = 0.0f;
#endif

    // Обрабатываем хвост (если размер не кратен 16)
    for (; i < size; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f")))
#endif
inline float l2_distance_avx512(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    size_t i = 0;
    size_t size = a.size();

#if defined(__AVX512F__) || defined(__GNUC__) || defined(__clang__)
    __m512 sum512 = _mm512_setzero_ps();

    for (; i + 15 < size; i += 16) {
        __m512 vecA = _mm512_loadu_ps(&a[i]);
        __m512 vecB = _mm512_loadu_ps(&b[i]);
        __m512 diff = _mm512_sub_ps(vecA, vecB);
        sum512 = _mm512_fmadd_ps(diff, diff, sum512);
    }

    float result = _mm512_reduce_add_ps(sum512);
#else
    float result = 0.0f;
#endif

    for (; i < size; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }

    return result;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f")))
#endif
inline float cosine_similarity_avx512(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    size_t i = 0;
    size_t size = a.size();

#if defined(__AVX512F__) || defined(__GNUC__) || defined(__clang__)
    __m512 dot512 = _mm512_setzero_ps();
    __m512 normA512 = _mm512_setzero_ps();
    __m512 normB512 = _mm512_setzero_ps();

    for (; i + 15 < size; i += 16) {
        __m512 vecA = _mm512_loadu_ps(&a[i]);
        __m512 vecB = _mm512_loadu_ps(&b[i]);

        dot512 = _mm512_fmadd_ps(vecA, vecB, dot512);
        normA512 = _mm512_fmadd_ps(vecA, vecA, normA512);
        normB512 = _mm512_fmadd_ps(vecB, vecB, normB512);
    }

    float dot = _mm512_reduce_add_ps(dot512);
    float norm_a = _mm512_reduce_add_ps(normA512);
    float norm_b = _mm512_reduce_add_ps(normB512);
#else
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
#endif

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

} // namespace sonatadb::math

#else // Если это не x86 архитектура (например, ARM)

namespace sonatadb::math {

inline float dot_product_avx512(std::span<const float> a, std::span<const float> b) {
    // Этот код никогда не должен вызываться на не-x86 архитектурах благодаря simd_dispatch
    return dot_product_scalar(a, b);
}

inline float l2_distance_avx512(std::span<const float> a, std::span<const float> b) {
    return l2_distance_scalar(a, b);
}

inline float cosine_similarity_avx512(std::span<const float> a, std::span<const float> b) {
    return cosine_similarity_scalar(a, b);
}

} // namespace sonatadb::math

#endif
