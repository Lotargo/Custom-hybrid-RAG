#include <gtest/gtest.h>
#include <vector>
#include "sonatadb/math/simd_dispatch.hpp"

using namespace sonatadb::math;

TEST(SimdDispatchTest, AutoDispatchWorks) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {4.0f, 3.0f, 2.0f, 1.0f};

    // Просто убеждаемся, что фасады не падают
    EXPECT_NO_THROW(dot_product(a, b));
    EXPECT_NO_THROW(l2_distance(a, b));
    EXPECT_NO_THROW(cosine_similarity(a, b));
}

TEST(SimdDispatchTest, Correctness) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {4.0f, 3.0f, 2.0f, 1.0f};

    float dot_scalar = dot_product_scalar<float>(a, b);
    float dot_auto = dot_product(a, b);

    EXPECT_FLOAT_EQ(dot_scalar, dot_auto);

    float l2_scalar = l2_distance_scalar<float>(a, b);
    float l2_auto = l2_distance(a, b);

    EXPECT_FLOAT_EQ(l2_scalar, l2_auto);

    float cos_scalar = cosine_similarity_scalar<float>(a, b);
    float cos_auto = cosine_similarity(a, b);

    EXPECT_FLOAT_EQ(cos_scalar, cos_auto);
}
