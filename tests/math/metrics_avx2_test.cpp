#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "rag_engine/math/metrics.hpp"
#include "rag_engine/math/metrics_avx2.hpp"

using namespace rag_engine::math;

TEST(MetricsAVX2Test, DotProduct) {
    std::vector<float> a(15, 1.0f); // 15 elements to test non-multiple of 8
    std::vector<float> b(15, 2.0f);

    float scalar_result = dot_product_scalar<float>(a, b);
    float avx2_result = dot_product_avx2(a, b);

    EXPECT_FLOAT_EQ(scalar_result, avx2_result);
    EXPECT_FLOAT_EQ(avx2_result, 30.0f);
}

TEST(MetricsAVX2Test, L2Distance) {
    std::vector<float> a(20);
    std::vector<float> b(20);
    for (size_t i = 0; i < 20; ++i) {
        a[i] = static_cast<float>(i) * 1.0f;
        b[i] = static_cast<float>(i) * 0.5f;
    }

    float scalar_result = l2_distance_scalar<float>(a, b);
    float avx2_result = l2_distance_avx2(a, b);

    EXPECT_FLOAT_EQ(scalar_result, avx2_result);
}

TEST(MetricsAVX2Test, CosineSimilarity) {
    std::vector<float> a(10, 2.0f);
    std::vector<float> b(10, 4.0f);

    float scalar_result = cosine_similarity_scalar<float>(a, b);
    float avx2_result = cosine_similarity_avx2(a, b);

    EXPECT_FLOAT_EQ(scalar_result, avx2_result);
}
