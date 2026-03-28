#include <neuralnet/network.h>
#include <neuralnet/serialization.h>

#include <gtest/gtest.h>

#include <numeric>
#include <sstream>

namespace nn = neuralnet;

TEST(SerializationTest, RoundTripBinary) {
    nn::NetworkTopology topo;
    topo.input_size = 3;
    topo.layers = {
        {.output_size = 4, .activation = nn::Activation::ReLU},
        {.output_size = 2, .activation = nn::Activation::Tanh},
    };

    std::vector<float> weights(3 * 4 + 4 + 4 * 2 + 2);
    std::iota(weights.begin(), weights.end(), 0.1f);

    auto original = nn::Network(topo, weights);

    std::stringstream ss;
    nn::serialize(original, ss);
    auto restored = nn::deserialize(ss);

    EXPECT_EQ(restored.input_size(), original.input_size());
    EXPECT_EQ(restored.output_size(), original.output_size());
    EXPECT_EQ(restored.get_all_weights(), original.get_all_weights());

    // Verify forward pass produces same result
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    EXPECT_EQ(restored.forward(input), original.forward(input));
}

TEST(SerializationTest, EmptyStreamThrows) {
    std::stringstream ss;
    EXPECT_THROW((void)nn::deserialize(ss), std::runtime_error);
}
