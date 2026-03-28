#include <neuralnet/network.h>

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

namespace nn = neuralnet;

TEST(NetworkTest, ConstructFromTopologyAndWeights) {
    // 2 inputs -> 3 hidden (ReLU) -> 1 output (Sigmoid)
    nn::NetworkTopology topo;
    topo.input_size = 2;
    topo.layers = {
        {.output_size = 3, .activation = nn::Activation::ReLU},
        {.output_size = 1, .activation = nn::Activation::Sigmoid},
    };

    // Layer 0: 2*3 weights + 3 biases = 9
    // Layer 1: 3*1 weights + 1 bias = 4
    // Total: 13 floats
    std::vector<float> weights(13, 0.0f);

    auto net = nn::Network(topo, weights);
    EXPECT_EQ(net.input_size(), 2);
    EXPECT_EQ(net.output_size(), 1);
    EXPECT_EQ(net.total_weights(), 13);
}

TEST(NetworkTest, ForwardPassProducesCorrectOutputSize) {
    nn::NetworkTopology topo;
    topo.input_size = 4;
    topo.layers = {
        {.output_size = 8, .activation = nn::Activation::ReLU},
        {.output_size = 3, .activation = nn::Activation::Tanh},
    };

    std::vector<float> weights(4 * 8 + 8 + 8 * 3 + 3, 0.1f);
    auto net = nn::Network(topo, weights);

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    auto output = net.forward(input);

    EXPECT_EQ(output.size(), 3);
}

TEST(NetworkTest, ForwardPassKnownValues) {
    // Single layer: 2 inputs -> 1 output (ReLU), weights=[1, 1], bias=[0]
    nn::NetworkTopology topo;
    topo.input_size = 2;
    topo.layers = {{.output_size = 1, .activation = nn::Activation::ReLU}};

    std::vector<float> weights = {1.0f, 1.0f, 0.0f};  // 2 weights + 1 bias
    auto net = nn::Network(topo, weights);

    auto output = net.forward({3.0f, 4.0f});

    ASSERT_EQ(output.size(), 1);
    EXPECT_FLOAT_EQ(output[0], 7.0f);  // (3+4) ReLU = 7
}

TEST(NetworkTest, GetAllWeightsRoundTrip) {
    nn::NetworkTopology topo;
    topo.input_size = 2;
    topo.layers = {
        {.output_size = 3, .activation = nn::Activation::ReLU},
        {.output_size = 1, .activation = nn::Activation::Sigmoid},
    };

    std::vector<float> original_weights(13);
    std::iota(original_weights.begin(), original_weights.end(), 1.0f);

    auto net = nn::Network(topo, original_weights);
    auto extracted = net.get_all_weights();

    EXPECT_EQ(extracted, original_weights);
}

TEST(NetworkTest, GetActivationsPerLayer) {
    nn::NetworkTopology topo;
    topo.input_size = 2;
    topo.layers = {
        {.output_size = 2, .activation = nn::Activation::ReLU},
        {.output_size = 1, .activation = nn::Activation::Sigmoid},
    };

    // Layer 0: identity-ish weights
    // Layer 1: sum both inputs
    std::vector<float> weights = {
        1.0f, 0.0f,   // L0 neuron 0
        0.0f, 1.0f,   // L0 neuron 1
        0.0f, 0.0f,   // L0 biases
        1.0f, 1.0f,   // L1 neuron 0
        0.0f,          // L1 bias
    };

    auto net = nn::Network(topo, weights);
    (void)net.forward({5.0f, -3.0f});

    auto all_activations = net.get_all_activations();
    ASSERT_EQ(all_activations.size(), 2);  // 2 layers

    // Layer 0: ReLU(5) = 5, ReLU(-3) = 0
    EXPECT_FLOAT_EQ(all_activations[0][0], 5.0f);
    EXPECT_FLOAT_EQ(all_activations[0][1], 0.0f);

    // Layer 1: sigmoid(5*1 + 0*1 + 0) = sigmoid(5) ≈ 0.9933
    EXPECT_NEAR(all_activations[1][0], 0.9933f, 0.001f);
}
