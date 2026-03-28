#include <neuralnet/layer.h>

#include <gtest/gtest.h>

#include <vector>

namespace nn = neuralnet;

TEST(LayerTest, ConstructionSetsCorrectDimensions) {
    // 3 inputs, 2 outputs
    std::vector<float> weights = {
        1.0f, 0.0f, 0.0f,  // neuron 0 weights
        0.0f, 1.0f, 0.0f   // neuron 1 weights
    };
    std::vector<float> biases = {0.0f, 0.0f};

    nn::Layer layer(3, 2, weights, biases, nn::Activation::ReLU);

    EXPECT_EQ(layer.input_size(), 3);
    EXPECT_EQ(layer.output_size(), 2);
}

TEST(LayerTest, ForwardPassIdentityWeights) {
    // 2 inputs, 2 outputs, identity-like weights, no bias, ReLU
    std::vector<float> weights = {
        1.0f, 0.0f,  // neuron 0: pass through input 0
        0.0f, 1.0f   // neuron 1: pass through input 1
    };
    std::vector<float> biases = {0.0f, 0.0f};
    nn::Layer layer(2, 2, weights, biases, nn::Activation::ReLU);

    std::vector<float> input = {3.0f, 5.0f};
    auto output = layer.forward(input);

    ASSERT_EQ(output.size(), 2);
    EXPECT_FLOAT_EQ(output[0], 3.0f);
    EXPECT_FLOAT_EQ(output[1], 5.0f);
}

TEST(LayerTest, ForwardPassWithBias) {
    std::vector<float> weights = {1.0f, 1.0f};  // 2 inputs, 1 output
    std::vector<float> biases = {-3.0f};
    nn::Layer layer(2, 1, weights, biases, nn::Activation::ReLU);

    std::vector<float> input = {1.0f, 1.0f};
    auto output = layer.forward(input);

    // (1*1 + 1*1) + (-3) = -1, ReLU(-1) = 0
    ASSERT_EQ(output.size(), 1);
    EXPECT_FLOAT_EQ(output[0], 0.0f);
}

TEST(LayerTest, ForwardPassSigmoid) {
    std::vector<float> weights = {0.0f};  // 1 input, 1 output
    std::vector<float> biases = {0.0f};
    nn::Layer layer(1, 1, weights, biases, nn::Activation::Sigmoid);

    std::vector<float> input = {0.0f};
    auto output = layer.forward(input);

    // sigmoid(0) = 0.5
    EXPECT_FLOAT_EQ(output[0], 0.5f);
}

TEST(LayerTest, GetLastActivations) {
    std::vector<float> weights = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> biases = {0.0f, 0.0f};
    nn::Layer layer(2, 2, weights, biases, nn::Activation::ReLU);

    std::vector<float> input = {3.0f, -2.0f};
    (void)layer.forward(input);

    auto activations = layer.get_last_activations();
    ASSERT_EQ(activations.size(), 2);
    EXPECT_FLOAT_EQ(activations[0], 3.0f);
    EXPECT_FLOAT_EQ(activations[1], 0.0f);  // ReLU(-2) = 0
}

TEST(LayerTest, PerNodeActivations) {
    // 2 inputs → 3 outputs, identity weights, zero bias
    // Node 0: ReLU, Node 1: Tanh, Node 2: Abs
    std::vector<float> weights = {
        1.0f, 0.0f,  // node 0 reads input 0
        0.0f, 1.0f,  // node 1 reads input 1
        1.0f, 1.0f,  // node 2 reads both
    };
    std::vector<float> biases = {0.0f, 0.0f, 0.0f};
    std::vector<nn::Activation> acts = {
        nn::Activation::ReLU,
        nn::Activation::Tanh,
        nn::Activation::Abs,
    };
    nn::Layer layer(2, 3, weights, biases, acts);

    EXPECT_TRUE(layer.has_per_node_activations());
    ASSERT_EQ(layer.activations().size(), 3);
    EXPECT_EQ(layer.activations()[0], nn::Activation::ReLU);
    EXPECT_EQ(layer.activations()[1], nn::Activation::Tanh);
    EXPECT_EQ(layer.activations()[2], nn::Activation::Abs);

    // input = {-2.0, 0.5}
    std::vector<float> input = {-2.0f, 0.5f};
    auto output = layer.forward(input);

    // Node 0: ReLU(-2.0) = 0.0
    EXPECT_FLOAT_EQ(output[0], 0.0f);
    // Node 1: Tanh(0.5) ≈ 0.4621
    EXPECT_NEAR(output[1], std::tanh(0.5f), 1e-5f);
    // Node 2: Abs(-2.0 + 0.5) = Abs(-1.5) = 1.5
    EXPECT_FLOAT_EQ(output[2], 1.5f);
}

TEST(LayerTest, PerLayerStillWorks) {
    // Verify the old single-activation constructor still works
    std::vector<float> weights = {1.0f};
    std::vector<float> biases = {0.0f};
    nn::Layer layer(1, 1, weights, biases, nn::Activation::Gaussian);

    EXPECT_FALSE(layer.has_per_node_activations());
    EXPECT_EQ(layer.activation(), nn::Activation::Gaussian);

    auto output = layer.forward(std::vector<float>{0.0f});
    // Gaussian(0) = exp(0) = 1.0
    EXPECT_FLOAT_EQ(output[0], 1.0f);
}
