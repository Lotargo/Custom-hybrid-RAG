#include <gtest/gtest.h>
#include <vector>
#include "rag_engine/math/metrics.hpp"
#include "rag_engine/math/metrics_avx512.hpp"
#include "rag_engine/math/simd_dispatch.hpp"

using namespace rag_engine::math;

TEST(MetricsAVX512Test, DotProduct) {
    if (get_best_simd_type() != SimdType::AVX512) {
        GTEST_SKIP() << "Skipping AVX-512 tests: CPU does not support AVX-512 instructions.";
    }

    std::vector<float> a(30, 1.0f); // 30 elements to test non-multiple of 16
    std::vector<float> b(30, 2.0f);

    float scalar_result = dot_product_scalar<float>(a, b);
    float avx512_result = dot_product_avx512(a, b);

    EXPECT_FLOAT_EQ(scalar_result, avx512_result);
    EXPECT_FLOAT_EQ(avx512_result, 60.0f);
}

TEST(MetricsAVX512Test, L2Distance) {
    if (get_best_simd_type() != SimdType::AVX512) {
        GTEST_SKIP() << "Skipping AVX-512 tests: CPU does not support AVX-512 instructions.";
    }

    std::vector<float> a(35);
    std::vector<float> b(35);
    for (size_t i = 0; i < 35; ++i) {
        a[i] = static_cast<float>(i) * 1.0f;
        b[i] = static_cast<float>(i) * 0.5f;
    }

    float scalar_result = l2_distance_scalar<float>(a, b);
    float avx512_result = l2_distance_avx512(a, b);

    EXPECT_FLOAT_EQ(scalar_result, avx512_result);
}

TEST(MetricsAVX512Test, CosineSimilarity) {
    if (get_best_simd_type() != SimdType::AVX512) {
        GTEST_SKIP() << "Skipping AVX-512 tests: CPU does not support AVX-512 instructions.";
    }

    std::vector<float> a(40, 2.0f);
    std::vector<float> b(40, 4.0f);

    float scalar_result = cosine_similarity_scalar<float>(a, b);
    float avx512_result = cosine_similarity_avx512(a, b);

    EXPECT_FLOAT_EQ(scalar_result, avx512_result);
}
