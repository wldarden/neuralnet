#include <neuralnet/activation.h>

#include <gtest/gtest.h>

#include <cmath>

namespace nn = neuralnet;

TEST(ActivationTest, ReluPositive) {
    EXPECT_FLOAT_EQ(nn::activate(nn::Activation::ReLU, 2.5f), 2.5f);
}

TEST(ActivationTest, ReluNegative) {
    EXPECT_FLOAT_EQ(nn::activate(nn::Activation::ReLU, -1.0f), 0.0f);
}

TEST(ActivationTest, ReluZero) {
    EXPECT_FLOAT_EQ(nn::activate(nn::Activation::ReLU, 0.0f), 0.0f);
}

TEST(ActivationTest, SigmoidZero) {
    EXPECT_FLOAT_EQ(nn::activate(nn::Activation::Sigmoid, 0.0f), 0.5f);
}

TEST(ActivationTest, SigmoidLargePositive) {
    EXPECT_NEAR(nn::activate(nn::Activation::Sigmoid, 10.0f), 1.0f, 1e-4f);
}

TEST(ActivationTest, SigmoidLargeNegative) {
    EXPECT_NEAR(nn::activate(nn::Activation::Sigmoid, -10.0f), 0.0f, 1e-4f);
}

TEST(ActivationTest, TanhZero) {
    EXPECT_FLOAT_EQ(nn::activate(nn::Activation::Tanh, 0.0f), 0.0f);
}

TEST(ActivationTest, TanhPositive) {
    EXPECT_NEAR(nn::activate(nn::Activation::Tanh, 1.0f), std::tanh(1.0f), 1e-6f);
}

TEST(ActivationTest, TanhNegative) {
    EXPECT_NEAR(nn::activate(nn::Activation::Tanh, -1.0f), std::tanh(-1.0f), 1e-6f);
}

TEST(ActivationTest, ActivateVectorRelu) {
    std::vector<float> vals = {-1.0f, 0.0f, 2.0f, -0.5f};
    nn::activate_inplace(nn::Activation::ReLU, vals);
    EXPECT_FLOAT_EQ(vals[0], 0.0f);
    EXPECT_FLOAT_EQ(vals[1], 0.0f);
    EXPECT_FLOAT_EQ(vals[2], 2.0f);
    EXPECT_FLOAT_EQ(vals[3], 0.0f);
}

// --- Linear ---

TEST(ActivationTest, LinearPassthrough) {
    EXPECT_FLOAT_EQ(nn::activate(nn::Activation::Linear, 3.14f), 3.14f);
    EXPECT_FLOAT_EQ(nn::activate(nn::Activation::Linear, -2.5f), -2.5f);
    EXPECT_FLOAT_EQ(nn::activate(nn::Activation::Linear, 0.0f), 0.0f);
}

// --- Gaussian ---

TEST(ActivationTest, GaussianZero) {
    EXPECT_FLOAT_EQ(nn::activate(nn::Activation::Gaussian, 0.0f), 1.0f);
}

TEST(ActivationTest, GaussianDecays) {
    float at_1 = nn::activate(nn::Activation::Gaussian, 1.0f);
    float at_2 = nn::activate(nn::Activation::Gaussian, 2.0f);
    EXPECT_GT(at_1, 0.0f);
    EXPECT_LT(at_1, 1.0f);
    EXPECT_GT(at_2, 0.0f);
    EXPECT_LT(at_2, at_1);  // further from 0 = smaller
}

TEST(ActivationTest, GaussianSymmetric) {
    EXPECT_FLOAT_EQ(nn::activate(nn::Activation::Gaussian, 1.5f),
                    nn::activate(nn::Activation::Gaussian, -1.5f));
}

// --- Sine ---

TEST(ActivationTest, SineZero) {
    EXPECT_NEAR(nn::activate(nn::Activation::Sine, 0.0f), 0.0f, 1e-6f);
}

TEST(ActivationTest, SinePiHalf) {
    EXPECT_NEAR(nn::activate(nn::Activation::Sine, 3.14159265f / 2.0f), 1.0f, 1e-5f);
}

TEST(ActivationTest, SineNegative) {
    float pos = nn::activate(nn::Activation::Sine, 1.0f);
    float neg = nn::activate(nn::Activation::Sine, -1.0f);
    EXPECT_NEAR(pos, -neg, 1e-6f);  // odd function
}

// --- Abs ---

TEST(ActivationTest, AbsPositive) {
    EXPECT_FLOAT_EQ(nn::activate(nn::Activation::Abs, 3.5f), 3.5f);
}

TEST(ActivationTest, AbsNegative) {
    EXPECT_FLOAT_EQ(nn::activate(nn::Activation::Abs, -3.5f), 3.5f);
}

TEST(ActivationTest, AbsZero) {
    EXPECT_FLOAT_EQ(nn::activate(nn::Activation::Abs, 0.0f), 0.0f);
}
