#include <neuralnet/graph_network.h>
#include <neuralnet/neural_node_props.h>
#include <neuralnet/neural_neat_policy.h>
#include <evolve/neat_operators.h>
#include <evolve/node_role.h>

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>

namespace nn = neuralnet;
namespace ev = evolve;

nn::NeuralGenome make_simple_genome() {
    std::mt19937 rng(42);
    auto policy = nn::make_neural_neat_policy(nn::NeuralMutationConfig{}, nn::NodeType::Stateless, nn::Activation::Tanh);
    return ev::create_minimal_genome<nn::NeuralNodeProps>(2, 1, policy, rng);
}

TEST(GraphNetworkTest, ConstructFromMinimalGenome) {
    auto genome = make_simple_genome();
    nn::GraphNetwork net(genome);
    EXPECT_EQ(net.input_size(), 2);
    EXPECT_EQ(net.output_size(), 1);
    EXPECT_EQ(net.num_nodes(), 3);
    EXPECT_EQ(net.num_connections(), 2);
}

TEST(GraphNetworkTest, ValidationRejectsZeroInputs) {
    nn::NeuralGenome genome;
    genome.nodes.push_back({.id = 0, .role = ev::NodeRole::Output,
        .props = {.activation = nn::Activation::Tanh,
        .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}});
    EXPECT_THROW(nn::GraphNetwork{genome}, std::invalid_argument);
}

TEST(GraphNetworkTest, ValidationRejectsZeroOutputs) {
    nn::NeuralGenome genome;
    genome.nodes.push_back({.id = 0, .role = ev::NodeRole::Input,
        .props = {.activation = nn::Activation::Tanh,
        .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}});
    EXPECT_THROW(nn::GraphNetwork{genome}, std::invalid_argument);
}

TEST(GraphNetworkTest, ValidationRejectsDuplicateNodeIDs) {
    nn::NeuralGenome genome;
    genome.nodes.push_back({.id = 0, .role = ev::NodeRole::Input,
        .props = {.activation = nn::Activation::Tanh,
        .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}});
    genome.nodes.push_back({.id = 0, .role = ev::NodeRole::Output,
        .props = {.activation = nn::Activation::Tanh,
        .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}});
    EXPECT_THROW(nn::GraphNetwork{genome}, std::invalid_argument);
}

TEST(GraphNetworkTest, ValidationRejectsDanglingConnection) {
    nn::NeuralGenome genome;
    genome.nodes.push_back({.id = 0, .role = ev::NodeRole::Input,
        .props = {.activation = nn::Activation::Tanh,
        .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}});
    genome.nodes.push_back({.id = 1, .role = ev::NodeRole::Output,
        .props = {.activation = nn::Activation::Tanh,
        .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}});
    genome.connections.push_back({.from_node = 0, .to_node = 99,
        .weight = 1.0f, .enabled = true, .innovation = 0});
    EXPECT_THROW(nn::GraphNetwork{genome}, std::invalid_argument);
}

TEST(GraphNetworkTest, ValidationRejectsConnectionToInput) {
    nn::NeuralGenome genome;
    genome.nodes.push_back({.id = 0, .role = ev::NodeRole::Input,
        .props = {.activation = nn::Activation::Tanh,
        .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}});
    genome.nodes.push_back({.id = 1, .role = ev::NodeRole::Input,
        .props = {.activation = nn::Activation::Tanh,
        .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}});
    genome.nodes.push_back({.id = 2, .role = ev::NodeRole::Output,
        .props = {.activation = nn::Activation::Tanh,
        .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}});
    genome.connections.push_back({.from_node = 2, .to_node = 0,
        .weight = 1.0f, .enabled = true, .innovation = 0});
    EXPECT_THROW(nn::GraphNetwork{genome}, std::invalid_argument);
}

// Task 4 — stateless forward pass tests

TEST(GraphNetworkTest, ForwardStateless_SimpleSum) {
    nn::NeuralGenome genome;
    genome.nodes = {
        {.id = 0, .role = ev::NodeRole::Input, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 1, .role = ev::NodeRole::Input, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 2, .role = ev::NodeRole::Output, .props = {.activation = nn::Activation::Tanh,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 2, .weight = 1.0f, .enabled = true, .innovation = 0},
        {.from_node = 1, .to_node = 2, .weight = 1.0f, .enabled = true, .innovation = 1},
    };
    nn::GraphNetwork net(genome);
    auto output = net.forward(std::vector<float>{0.5f, 0.5f});
    ASSERT_EQ(output.size(), 1);
    EXPECT_NEAR(output[0], std::tanh(1.0f), 1e-5f);
}

TEST(GraphNetworkTest, ForwardStateless_WithHiddenNode) {
    nn::NeuralGenome genome;
    genome.nodes = {
        {.id = 0, .role = ev::NodeRole::Input, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 1, .role = ev::NodeRole::Output, .props = {.activation = nn::Activation::Tanh,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 2, .role = ev::NodeRole::Hidden, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 2, .weight = 2.0f, .enabled = true, .innovation = 0},
        {.from_node = 2, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 1},
    };
    nn::GraphNetwork net(genome);
    auto output = net.forward(std::vector<float>{3.0f});
    ASSERT_EQ(output.size(), 1);
    EXPECT_NEAR(output[0], std::tanh(6.0f), 1e-5f);
}

TEST(GraphNetworkTest, ForwardStateless_DisabledConnectionIgnored) {
    nn::NeuralGenome genome;
    genome.nodes = {
        {.id = 0, .role = ev::NodeRole::Input, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 1, .role = ev::NodeRole::Output, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 5.0f, .enabled = false, .innovation = 0},
    };
    nn::GraphNetwork net(genome);
    auto output = net.forward(std::vector<float>{10.0f});
    ASSERT_EQ(output.size(), 1);
    EXPECT_FLOAT_EQ(output[0], 0.0f);
}

TEST(GraphNetworkTest, ForwardStateless_BiasWorks) {
    nn::NeuralGenome genome;
    genome.nodes = {
        {.id = 0, .role = ev::NodeRole::Input, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 1, .role = ev::NodeRole::Output, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 3.0f, .tau = 1.0f}},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 0},
    };
    nn::GraphNetwork net(genome);
    auto output = net.forward(std::vector<float>{2.0f});
    ASSERT_EQ(output.size(), 1);
    EXPECT_FLOAT_EQ(output[0], 5.0f);
}

TEST(GraphNetworkTest, ForwardStateless_InputSizeMismatchThrows) {
    auto genome = make_simple_genome();
    nn::GraphNetwork net(genome);
    EXPECT_THROW((void)net.forward(std::vector<float>{1.0f}), std::invalid_argument);
    EXPECT_THROW((void)net.forward(std::vector<float>{1.0f, 2.0f, 3.0f}), std::invalid_argument);
}

TEST(GraphNetworkTest, ForwardStateless_MultipleOutputs) {
    nn::NeuralGenome genome;
    genome.nodes = {
        {.id = 0, .role = ev::NodeRole::Input, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 1, .role = ev::NodeRole::Output, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 2, .role = ev::NodeRole::Output, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 0},
        {.from_node = 0, .to_node = 2, .weight = 2.0f, .enabled = true, .innovation = 1},
    };
    nn::GraphNetwork net(genome);
    auto output = net.forward(std::vector<float>{3.0f});
    ASSERT_EQ(output.size(), 2);
    EXPECT_FLOAT_EQ(output[0], 3.0f);
    EXPECT_FLOAT_EQ(output[1], 6.0f);
}

// Task 5 — CTRNN + recurrence tests

TEST(GraphNetworkTest, CTRNN_SlowlyApproachesTarget) {
    nn::NeuralGenome genome;
    genome.nodes = {
        {.id = 0, .role = ev::NodeRole::Input, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 1, .role = ev::NodeRole::Output, .props = {.activation = nn::Activation::Tanh,
         .type = nn::NodeType::CTRNN, .bias = 0.0f, .tau = 10.0f}},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 0},
    };
    nn::GraphNetwork net(genome);
    float target = std::tanh(1.0f);
    auto out1 = net.forward(std::vector<float>{1.0f});
    EXPECT_GT(out1[0], 0.0f);
    EXPECT_LT(out1[0], target);
    for (int i = 0; i < 100; ++i) (void)net.forward(std::vector<float>{1.0f});
    auto out_final = net.forward(std::vector<float>{1.0f});
    EXPECT_NEAR(out_final[0], target, 0.01f);
}

TEST(GraphNetworkTest, CTRNN_FastTauActsLikeStateless) {
    nn::NeuralGenome genome;
    genome.nodes = {
        {.id = 0, .role = ev::NodeRole::Input, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 1, .role = ev::NodeRole::Output, .props = {.activation = nn::Activation::Tanh,
         .type = nn::NodeType::CTRNN, .bias = 0.0f, .tau = 1.0f}},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 0},
    };
    nn::GraphNetwork net(genome);
    for (int i = 0; i < 5; ++i) (void)net.forward(std::vector<float>{1.0f});
    auto out = net.forward(std::vector<float>{1.0f});
    EXPECT_NEAR(out[0], std::tanh(1.0f), 0.02f);
}

TEST(GraphNetworkTest, ResetState_ClearsCTRNN) {
    nn::NeuralGenome genome;
    genome.nodes = {
        {.id = 0, .role = ev::NodeRole::Input, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 1, .role = ev::NodeRole::Output, .props = {.activation = nn::Activation::Tanh,
         .type = nn::NodeType::CTRNN, .bias = 0.0f, .tau = 10.0f}},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 0},
    };
    nn::GraphNetwork net(genome);
    (void)net.forward(std::vector<float>{1.0f});
    (void)net.forward(std::vector<float>{1.0f});
    net.reset_state();
    auto after_reset = net.forward(std::vector<float>{1.0f});
    nn::GraphNetwork fresh(genome);
    auto first_tick = fresh.forward(std::vector<float>{1.0f});
    EXPECT_FLOAT_EQ(after_reset[0], first_tick[0]);
}

TEST(GraphNetworkTest, RecurrentConnection_UsesLastTick) {
    nn::NeuralGenome genome;
    genome.nodes = {
        {.id = 0, .role = ev::NodeRole::Input, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 1, .role = ev::NodeRole::Output, .props = {.activation = nn::Activation::Tanh,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 0},
        {.from_node = 1, .to_node = 1, .weight = 0.5f, .enabled = true, .innovation = 1},
    };
    nn::GraphNetwork net(genome);
    auto out1 = net.forward(std::vector<float>{1.0f});
    float expected1 = std::tanh(1.0f);
    EXPECT_NEAR(out1[0], expected1, 1e-5f);
    auto out2 = net.forward(std::vector<float>{1.0f});
    float expected2 = std::tanh(1.0f + 0.5f * expected1);
    EXPECT_NEAR(out2[0], expected2, 1e-5f);
}

TEST(GraphNetworkTest, RecurrentCycle_HiddenToHidden) {
    nn::NeuralGenome genome;
    genome.nodes = {
        {.id = 0, .role = ev::NodeRole::Input, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 1, .role = ev::NodeRole::Output, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 2, .role = ev::NodeRole::Hidden, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 3, .role = ev::NodeRole::Hidden, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 2, .weight = 1.0f, .enabled = true, .innovation = 0},
        {.from_node = 2, .to_node = 3, .weight = 1.0f, .enabled = true, .innovation = 1},
        {.from_node = 3, .to_node = 2, .weight = 0.5f, .enabled = true, .innovation = 2},
        {.from_node = 2, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 3},
    };
    nn::GraphNetwork net(genome);
    auto out1 = net.forward(std::vector<float>{1.0f});
    ASSERT_EQ(out1.size(), 1);
    auto out2 = net.forward(std::vector<float>{1.0f});
    EXPECT_GE(out2[0], out1[0]);
}

TEST(GraphNetworkTest, DeadNode_ExcludedFromEvaluation) {
    nn::NeuralGenome genome;
    genome.nodes = {
        {.id = 0, .role = ev::NodeRole::Input, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 1, .role = ev::NodeRole::Output, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 2, .role = ev::NodeRole::Hidden, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 100.0f, .tau = 1.0f}},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 0},
    };
    nn::GraphNetwork net(genome);
    auto output = net.forward(std::vector<float>{5.0f});
    ASSERT_EQ(output.size(), 1);
    EXPECT_FLOAT_EQ(output[0], 5.0f);
}

TEST(GraphNetworkTest, DtOverride) {
    nn::NeuralGenome genome;
    genome.nodes = {
        {.id = 0, .role = ev::NodeRole::Input, .props = {.activation = nn::Activation::ReLU,
         .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 1, .role = ev::NodeRole::Output, .props = {.activation = nn::Activation::Tanh,
         .type = nn::NodeType::CTRNN, .bias = 0.0f, .tau = 10.0f}},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 0},
    };
    nn::GraphNetwork net1(genome);
    nn::GraphNetwork net2(genome);
    auto out1 = net1.forward(std::vector<float>{1.0f});
    auto out2 = net2.forward(std::vector<float>{1.0f}, 0.1f);
    EXPECT_GT(out1[0], out2[0]);
}
