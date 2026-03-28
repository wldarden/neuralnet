#include <neuralnet/neural_node_props.h>
#include <neuralnet/neural_neat_policy.h>
#include <neuralnet/node_types.h>
#include <evolve/neat_operators.h>
#include <evolve/node_role.h>
#include <gtest/gtest.h>
#include <random>

namespace nn = neuralnet;
namespace ev = evolve;

TEST(NodeTypesTest, EnumsHaveExpectedValues) {
    EXPECT_EQ(static_cast<uint8_t>(nn::NodeType::Stateless), 0);
    EXPECT_EQ(static_cast<uint8_t>(nn::NodeType::CTRNN), 1);
    EXPECT_EQ(static_cast<uint8_t>(ev::NodeRole::Input), 0);
    EXPECT_EQ(static_cast<uint8_t>(ev::NodeRole::Hidden), 1);
    EXPECT_EQ(static_cast<uint8_t>(ev::NodeRole::Output), 2);
}

TEST(GraphGenomeTest, EmptyGenome) {
    nn::NeuralGenome genome;
    EXPECT_TRUE(genome.nodes.empty());
    EXPECT_TRUE(genome.connections.empty());
}

TEST(GraphGenomeTest, NodeGeneFields) {
    nn::NeuralNodeGene node{
        .id = 0,
        .role = ev::NodeRole::Input,
        .props = {
            .activation = nn::Activation::ReLU,
            .type = nn::NodeType::Stateless,
            .bias = 0.0f,
            .tau = 1.0f,
        },
    };
    EXPECT_EQ(node.id, 0);
    EXPECT_EQ(node.role, ev::NodeRole::Input);
}

TEST(ConnectionGeneTest, Fields) {
    ev::ConnectionGene conn{
        .from_node = 0,
        .to_node = 2,
        .weight = 0.5f,
        .enabled = true,
        .innovation = 0,
    };
    EXPECT_EQ(conn.from_node, 0);
    EXPECT_EQ(conn.to_node, 2);
    EXPECT_FLOAT_EQ(conn.weight, 0.5f);
    EXPECT_TRUE(conn.enabled);
}

TEST(GraphGenomeTest, CreateMinimal_2Inputs_1Output) {
    std::mt19937 rng(42);
    auto policy = nn::make_neural_neat_policy(nn::NeuralMutationConfig{}, nn::NodeType::Stateless, nn::Activation::Tanh);
    auto genome = ev::create_minimal_genome<nn::NeuralNodeProps>(2, 1, policy, rng);
    EXPECT_EQ(genome.nodes.size(), 3);
    EXPECT_EQ(genome.connections.size(), 2);
    EXPECT_EQ(genome.nodes[0].id, 0);
    EXPECT_EQ(genome.nodes[0].role, ev::NodeRole::Input);
    EXPECT_EQ(genome.nodes[1].id, 1);
    EXPECT_EQ(genome.nodes[1].role, ev::NodeRole::Input);
    EXPECT_EQ(genome.nodes[2].id, 2);
    EXPECT_EQ(genome.nodes[2].role, ev::NodeRole::Output);
    EXPECT_EQ(genome.nodes[2].props.type, nn::NodeType::Stateless);
    EXPECT_EQ(genome.nodes[2].props.activation, nn::Activation::Tanh);
    EXPECT_EQ(genome.connections[0].from_node, 0);
    EXPECT_EQ(genome.connections[0].to_node, 2);
    EXPECT_TRUE(genome.connections[0].enabled);
    EXPECT_EQ(genome.connections[1].from_node, 1);
    EXPECT_EQ(genome.connections[1].to_node, 2);
    EXPECT_TRUE(genome.connections[1].enabled);
}

TEST(GraphGenomeTest, CreateMinimal_InnovationNumbersSequential) {
    std::mt19937 rng(42);
    auto policy = nn::make_neural_neat_policy(nn::NeuralMutationConfig{}, nn::NodeType::CTRNN, nn::Activation::Tanh);
    auto genome = ev::create_minimal_genome<nn::NeuralNodeProps>(3, 2, policy, rng);
    EXPECT_EQ(genome.connections.size(), 6);
    for (uint32_t i = 0; i < 6; ++i) {
        EXPECT_EQ(genome.connections[i].innovation, i);
    }
}

TEST(GraphGenomeTest, CreateMinimal_CTRNNOutputs) {
    std::mt19937 rng(42);
    auto policy = nn::make_neural_neat_policy(nn::NeuralMutationConfig{}, nn::NodeType::CTRNN, nn::Activation::Tanh);
    auto genome = ev::create_minimal_genome<nn::NeuralNodeProps>(2, 2, policy, rng);
    for (std::size_t i = 2; i < genome.nodes.size(); ++i) {
        EXPECT_EQ(genome.nodes[i].props.type, nn::NodeType::CTRNN);
        EXPECT_GT(genome.nodes[i].props.tau, 0.0f);
    }
}

TEST(GraphGenomeTest, CreateMinimal_WeightsAreRandom) {
    std::mt19937 rng1(42);
    auto policy1 = nn::make_neural_neat_policy(nn::NeuralMutationConfig{}, nn::NodeType::Stateless, nn::Activation::Tanh);
    auto g1 = ev::create_minimal_genome<nn::NeuralNodeProps>(2, 1, policy1, rng1);
    std::mt19937 rng2(99);
    auto policy2 = nn::make_neural_neat_policy(nn::NeuralMutationConfig{}, nn::NodeType::Stateless, nn::Activation::Tanh);
    auto g2 = ev::create_minimal_genome<nn::NeuralNodeProps>(2, 1, policy2, rng2);
    bool any_different = false;
    for (std::size_t i = 0; i < g1.connections.size(); ++i) {
        if (g1.connections[i].weight != g2.connections[i].weight) any_different = true;
    }
    EXPECT_TRUE(any_different);
}
