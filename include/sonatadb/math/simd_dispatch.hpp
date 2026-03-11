#pragma once

#include <span>
#include <stdexcept>
#include "sonatadb/math/metrics.hpp"
#include "sonatadb/math/metrics_avx2.hpp"
#include "sonatadb/math/metrics_avx512.hpp"

namespace sonatadb::math {

enum class SimdType {
    Scalar,
    AVX2,
    AVX512,
    NEON,
    WASM_SIMD128,
    Auto // Выбирает наилучший доступный на текущем железе
};

// Функция определения лучшего поддерживаемого типа SIMD в runtime
inline SimdType get_best_simd_type() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_cpu_supports("avx512f")) {
        return SimdType::AVX512;
    }
    if (__builtin_cpu_supports("avx2")) {
        return SimdType::AVX2;
    }
#else
    // Fallback or specific implementation for MSVC could be added here.
    // For now, assume auto-dispatch via CPUID is pending for MSVC.
#endif
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
    // В будущем тут будет проверка NEON (на ARMv8 он почти всегда есть)
    return SimdType::NEON;
#elif defined(__wasm__) && defined(__wasm_simd128__)
    return SimdType::WASM_SIMD128;
#endif
    return SimdType::Scalar;
}

// Глобальный кэш для лучшего доступного SIMD
inline const SimdType BEST_AVAILABLE_SIMD = get_best_simd_type();

// Единый фасад для Dot Product
inline float dot_product(std::span<const float> a, std::span<const float> b, SimdType simd = SimdType::Auto) {
    if (simd == SimdType::Auto) {
        simd = BEST_AVAILABLE_SIMD;
    }

    switch (simd) {
        case SimdType::AVX512:
            return dot_product_avx512(a, b);

        case SimdType::AVX2:
            return dot_product_avx2(a, b);
        case SimdType::NEON:
        case SimdType::WASM_SIMD128:
        case SimdType::Scalar:
        default:
            return dot_product_scalar<float>(a, b);
    }
}

// Единый фасад для L2 Distance
inline float l2_distance(std::span<const float> a, std::span<const float> b, SimdType simd = SimdType::Auto) {
    if (simd == SimdType::Auto) {
        simd = BEST_AVAILABLE_SIMD;
    }

    switch (simd) {
        case SimdType::AVX512:
            return l2_distance_avx512(a, b);
        case SimdType::AVX2:
            return l2_distance_avx2(a, b);
        case SimdType::NEON:
        case SimdType::WASM_SIMD128:
        case SimdType::Scalar:
        default:
            return l2_distance_scalar<float>(a, b);
    }
}

// Единый фасад для Cosine Similarity
inline float cosine_similarity(std::span<const float> a, std::span<const float> b, SimdType simd = SimdType::Auto) {
    if (simd == SimdType::Auto) {
        simd = BEST_AVAILABLE_SIMD;
    }

    switch (simd) {
        case SimdType::AVX512:
            return cosine_similarity_avx512(a, b);
        case SimdType::AVX2:
            return cosine_similarity_avx2(a, b);
        case SimdType::NEON:
        case SimdType::WASM_SIMD128:
        case SimdType::Scalar:
        default:
            return cosine_similarity_scalar<float>(a, b);
    }
}

} // namespace sonatadb::math
