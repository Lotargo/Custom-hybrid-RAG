#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "sonatadb/math/metrics.hpp"

using namespace sonatadb::math;

TEST(MetricsTest, DotProductScalar) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};

    float result = dot_product_scalar<float>(a, b);
    EXPECT_FLOAT_EQ(result, 32.0f); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}

TEST(MetricsTest, L2DistanceScalar) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};

    float result = l2_distance_scalar<float>(a, b);
    EXPECT_FLOAT_EQ(result, 27.0f); // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
}

TEST(MetricsTest, CosineSimilarityScalar) {
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f, 0.0f};

    float result = cosine_similarity_scalar<float>(a, b);
    EXPECT_FLOAT_EQ(result, 0.0f); // Ортогональные векторы

    std::vector<float> c = {2.0f, 0.0f, 0.0f};
    float result2 = cosine_similarity_scalar<float>(a, c);
    EXPECT_FLOAT_EQ(result2, 1.0f); // Коллинеарные векторы
}

TEST(MetricsTest, InvalidArgument) {
    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f};

    EXPECT_THROW(dot_product_scalar<float>(a, b), std::invalid_argument);
    EXPECT_THROW(l2_distance_scalar<float>(a, b), std::invalid_argument);
    EXPECT_THROW(cosine_similarity_scalar<float>(a, b), std::invalid_argument);
}
