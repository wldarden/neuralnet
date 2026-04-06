#include <neuralnet/adapt.h>
#include <gtest/gtest.h>

#include <random>
#include <vector>

namespace nn = neuralnet;

// -- adapt_input tests --------------------------------------------------------

TEST(AdaptInputTest, ExactMatch) {
    auto result = nn::adapt_input({"a", "b", "c"}, {1.0f, 2.0f, 3.0f}, {"a", "b", "c"});
    EXPECT_EQ(result, (std::vector<float>{1.0f, 2.0f, 3.0f}));
}

TEST(AdaptInputTest, Reorder) {
    auto result = nn::adapt_input({"a", "b", "c"}, {1.0f, 2.0f, 3.0f}, {"c", "a", "b"});
    EXPECT_EQ(result, (std::vector<float>{3.0f, 1.0f, 2.0f}));
}

TEST(AdaptInputTest, MissingInputs) {
    auto result = nn::adapt_input({"a"}, {1.0f}, {"a", "b"}, 0.5f);
    EXPECT_EQ(result, (std::vector<float>{1.0f, 0.5f}));
}

TEST(AdaptInputTest, ExtraInputs) {
    auto result = nn::adapt_input({"a", "b", "c"}, {1.0f, 2.0f, 3.0f}, {"b"});
    EXPECT_EQ(result, (std::vector<float>{2.0f}));
}

TEST(AdaptInputTest, Mixed) {
    auto result = nn::adapt_input({"a", "b", "c"}, {1.0f, 2.0f, 3.0f}, {"d", "c", "a"}, -1.0f);
    EXPECT_EQ(result, (std::vector<float>{-1.0f, 3.0f, 1.0f}));
}

// -- adapt_topology_inputs tests ----------------------------------------------

TEST(AdaptTopologyTest, AddColumns) {
    nn::NetworkTopology topo;
    topo.input_size = 2;
    topo.input_ids = {"a", "b"};
    topo.layers = {{.output_size = 2, .activation = nn::Activation::ReLU}};

    // Weights: [o0_a, o0_b, o1_a, o1_b] + biases [b0, b1]
    std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f};

    std::mt19937 rng(42);
    auto result = nn::adapt_topology_inputs(topo, weights, {"a", "b", "c"}, rng);

    EXPECT_EQ(result.adapted_topology.input_size, 3u);
    EXPECT_EQ(result.adapted_topology.input_ids.size(), 3u);
    EXPECT_EQ(result.added_ids, std::vector<std::string>{"c"});
    EXPECT_TRUE(result.removed_ids.empty());

    // Preserved columns
    EXPECT_FLOAT_EQ(result.adapted_weights[0], 1.0f);  // o0_a
    EXPECT_FLOAT_EQ(result.adapted_weights[1], 2.0f);  // o0_b
    // [2] = random (o0_c)
    EXPECT_FLOAT_EQ(result.adapted_weights[3], 3.0f);  // o1_a
    EXPECT_FLOAT_EQ(result.adapted_weights[4], 4.0f);  // o1_b
    // [5] = random (o1_c)
    // Biases preserved
    EXPECT_FLOAT_EQ(result.adapted_weights[6], 0.1f);
    EXPECT_FLOAT_EQ(result.adapted_weights[7], 0.2f);
}

TEST(AdaptTopologyTest, RemoveColumns) {
    nn::NetworkTopology topo;
    topo.input_size = 3;
    topo.input_ids = {"a", "b", "c"};
    topo.layers = {{.output_size = 1, .activation = nn::Activation::ReLU}};

    std::vector<float> weights = {1.0f, 2.0f, 3.0f, 0.5f};

    std::mt19937 rng(42);
    auto result = nn::adapt_topology_inputs(topo, weights, {"a", "c"}, rng);

    EXPECT_EQ(result.adapted_topology.input_size, 2u);
    EXPECT_EQ(result.removed_ids, std::vector<std::string>{"b"});
    EXPECT_TRUE(result.added_ids.empty());

    EXPECT_FLOAT_EQ(result.adapted_weights[0], 1.0f);  // a kept
    EXPECT_FLOAT_EQ(result.adapted_weights[1], 3.0f);  // c kept
    EXPECT_FLOAT_EQ(result.adapted_weights[2], 0.5f);  // bias
}

TEST(AdaptTopologyTest, ReportsAddedAndRemoved) {
    nn::NetworkTopology topo;
    topo.input_size = 3;
    topo.input_ids = {"a", "b", "c"};
    topo.layers = {{.output_size = 1, .activation = nn::Activation::ReLU}};

    std::vector<float> weights = {1.0f, 2.0f, 3.0f, 0.0f};

    std::mt19937 rng(42);
    auto result = nn::adapt_topology_inputs(topo, weights, {"b", "d", "e"}, rng);

    EXPECT_EQ(result.added_ids.size(), 2u);
    EXPECT_EQ(result.removed_ids.size(), 2u);
}

TEST(AdaptTopologyTest, PreservesHigherLayers) {
    nn::NetworkTopology topo;
    topo.input_size = 2;
    topo.input_ids = {"a", "b"};
    topo.layers = {
        {.output_size = 2, .activation = nn::Activation::ReLU},
        {.output_size = 1, .activation = nn::Activation::Sigmoid},
    };

    // L0: 2*2 weights + 2 biases = 6, L1: 2*1 weights + 1 bias = 3
    std::vector<float> weights = {1, 2, 3, 4, 0.1f, 0.2f, 10, 20, 0.5f};

    std::mt19937 rng(42);
    auto result = nn::adapt_topology_inputs(topo, weights, {"a", "b", "c"}, rng);

    // New L0: 2*3 weights + 2 biases = 8, L1 starts at offset 8
    EXPECT_FLOAT_EQ(result.adapted_weights[8], 10.0f);
    EXPECT_FLOAT_EQ(result.adapted_weights[9], 20.0f);
    EXPECT_FLOAT_EQ(result.adapted_weights[10], 0.5f);
}

TEST(AdaptTopologyTest, PreservesOutputIds) {
    nn::NetworkTopology topo;
    topo.input_size = 2;
    topo.input_ids = {"a", "b"};
    topo.layers = {{.output_size = 1, .activation = nn::Activation::ReLU}};
    topo.output_ids = {"out"};

    std::vector<float> weights = {1.0f, 2.0f, 0.0f};

    std::mt19937 rng(42);
    auto result = nn::adapt_topology_inputs(topo, weights, {"a"}, rng);

    EXPECT_EQ(result.adapted_topology.output_ids, std::vector<std::string>{"out"});
}

TEST(AdaptTopologyTest, ThrowsIfSourceHasNoIds) {
    nn::NetworkTopology topo;
    topo.input_size = 2;
    topo.layers = {{.output_size = 1, .activation = nn::Activation::ReLU}};

    std::vector<float> weights = {1.0f, 2.0f, 0.0f};

    std::mt19937 rng(42);
    EXPECT_THROW(
        (void)nn::adapt_topology_inputs(topo, weights, {"a", "b"}, rng),
        std::invalid_argument);
}
