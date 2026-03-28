#include <neuralnet/serialization.h>
#include <neuralnet/graph_network.h>
#include <neuralnet/neural_node_props.h>
#include <neuralnet/neural_neat_policy.h>
#include <evolve/neat_operators.h>

#include <gtest/gtest.h>

#include <random>
#include <sstream>
#include <variant>

namespace nn = neuralnet;
namespace ev = evolve;

TEST(SerializationV2Test, GraphNetworkRoundTrip) {
    std::mt19937 rng(42);
    auto policy = nn::make_neural_neat_policy(nn::NeuralMutationConfig{}, nn::NodeType::CTRNN, nn::Activation::Tanh);
    auto genome = ev::create_minimal_genome<nn::NeuralNodeProps>(2, 1, policy, rng);
    nn::GraphNetwork net(genome);
    (void)net.forward(std::vector<float>{1.0f, 0.5f});
    (void)net.forward(std::vector<float>{0.3f, 0.7f});

    std::stringstream ss;
    nn::save(net, ss);
    ss.seekg(0);
    auto loaded = nn::load(ss);

    ASSERT_TRUE(std::holds_alternative<nn::GraphNetwork>(loaded));
    auto& loaded_net = std::get<nn::GraphNetwork>(loaded);
    EXPECT_EQ(loaded_net.input_size(), 2);
    EXPECT_EQ(loaded_net.output_size(), 1);
    EXPECT_EQ(loaded_net.num_nodes(), net.num_nodes());
    EXPECT_EQ(loaded_net.num_connections(), net.num_connections());

    auto out_original = net.forward(std::vector<float>{1.0f, 1.0f});
    auto out_loaded = loaded_net.forward(std::vector<float>{1.0f, 1.0f});
    ASSERT_EQ(out_original.size(), out_loaded.size());
    for (std::size_t i = 0; i < out_original.size(); ++i) {
        EXPECT_NEAR(out_original[i], out_loaded[i], 1e-5f);
    }
}

TEST(SerializationV2Test, MLPRoundTrip_VersionedFormat) {
    nn::NetworkTopology topo;
    topo.input_size = 2;
    topo.layers = {{.output_size = 3, .activation = nn::Activation::ReLU},
                   {.output_size = 1, .activation = nn::Activation::Sigmoid}};
    std::vector<float> weights(13, 0.5f);
    nn::Network net(topo, weights);

    std::stringstream ss;
    nn::save(net, ss);
    ss.seekg(0);
    auto loaded = nn::load(ss);

    ASSERT_TRUE(std::holds_alternative<nn::Network>(loaded));
    auto& loaded_net = std::get<nn::Network>(loaded);
    EXPECT_EQ(loaded_net.input_size(), 2);
    EXPECT_EQ(loaded_net.output_size(), 1);
    EXPECT_EQ(loaded_net.get_all_weights(), weights);
}

TEST(SerializationV2Test, LegacyFormatStillLoads) {
    nn::NetworkTopology topo;
    topo.input_size = 2;
    topo.layers = {{.output_size = 1, .activation = nn::Activation::ReLU}};
    std::vector<float> weights = {1.0f, 1.0f, 0.0f};
    nn::Network net(topo, weights);

    std::stringstream ss;
    nn::serialize(net, ss);  // Legacy format
    ss.seekg(0);
    auto loaded = nn::load(ss);

    ASSERT_TRUE(std::holds_alternative<nn::Network>(loaded));
    auto& loaded_net = std::get<nn::Network>(loaded);
    EXPECT_EQ(loaded_net.get_all_weights(), weights);
}
