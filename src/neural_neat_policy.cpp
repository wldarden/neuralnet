#include <neuralnet/neural_neat_policy.h>
#include <neuralnet/activation.h>
#include <evolve/node_role.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>

namespace neuralnet {

void mutate_biases(NeuralGenome& genome, const NeuralMutationConfig& config,
                   std::mt19937& rng) {
    std::uniform_real_distribution<float> chance(0.0f, 1.0f);
    std::normal_distribution<float> perturb(0.0f, config.bias_perturb_strength);

    for (auto& node : genome.nodes) {
        if (node.role == evolve::NodeRole::Input) continue;
        if (chance(rng) < config.bias_mutate_rate) {
            node.props.bias += perturb(rng);
        }
    }
}

void mutate_tau(NeuralGenome& genome, const NeuralMutationConfig& config,
                std::mt19937& rng) {
    std::uniform_real_distribution<float> chance(0.0f, 1.0f);
    std::normal_distribution<float> perturb(0.0f, config.tau_perturb_strength);

    for (auto& node : genome.nodes) {
        if (node.props.type != NodeType::CTRNN) continue;
        if (chance(rng) < config.tau_mutate_rate) {
            node.props.tau += perturb(rng);
            node.props.tau = std::clamp(node.props.tau, config.tau_min, config.tau_max);
        }
    }
}

void mutate_node_types(NeuralGenome& genome, const NeuralMutationConfig& config,
                       std::mt19937& rng) {
    std::uniform_real_distribution<float> chance(0.0f, 1.0f);

    for (auto& node : genome.nodes) {
        if (node.role == evolve::NodeRole::Input) continue;
        if (chance(rng) < config.node_type_mutate_rate) {
            if (node.props.type == NodeType::Stateless) {
                node.props.type = NodeType::CTRNN;
                node.props.tau = 1.0f;
            } else {
                node.props.type = NodeType::Stateless;
            }
        }
    }
}

void mutate_activations(NeuralGenome& genome, const NeuralMutationConfig& config,
                        std::mt19937& rng) {
    constexpr std::array<Activation, ACTIVATION_COUNT> activations = {
        Activation::ReLU, Activation::Sigmoid, Activation::Tanh,
        Activation::Linear, Activation::Gaussian, Activation::Sine, Activation::Abs};

    std::uniform_real_distribution<float> chance(0.0f, 1.0f);
    std::uniform_int_distribution<int> act_dist(0, ACTIVATION_COUNT - 1);

    for (auto& node : genome.nodes) {
        if (node.role == evolve::NodeRole::Input) continue;
        if (chance(rng) < config.activation_mutate_rate) {
            node.props.activation = activations[act_dist(rng)];
        }
    }
}

[[nodiscard]] evolve::NeatPolicy<NeuralNodeProps> make_neural_neat_policy(
    const NeuralMutationConfig& config,
    NodeType default_output_type,
    Activation default_output_activation) {

    return evolve::NeatPolicy<NeuralNodeProps>{
        .init_node_props = [](NeuralNodeProps& props, std::mt19937& rng) {
            constexpr std::array<Activation, ACTIVATION_COUNT> activations = {
                Activation::ReLU, Activation::Sigmoid, Activation::Tanh,
                Activation::Linear, Activation::Gaussian, Activation::Sine, Activation::Abs};
            std::uniform_int_distribution<int> act_dist(0, ACTIVATION_COUNT - 1);

            props.activation = activations[act_dist(rng)];
            props.type = NodeType::Stateless;
            props.bias = 0.0f;
            props.tau = 1.0f;
        },

        .merge_node_props = [](NeuralNodeProps& child,
                                const NeuralNodeProps& parent_a,
                                const NeuralNodeProps& parent_b,
                                std::mt19937& rng) {
            std::uniform_int_distribution<int> coin(0, 1);
            child = coin(rng) ? parent_a : parent_b;
        },

        .mutate_properties = [config](NeuralGenome& genome, std::mt19937& rng) {
            mutate_biases(genome, config, rng);
            mutate_tau(genome, config, rng);
            mutate_node_types(genome, config, rng);
            mutate_activations(genome, config, rng);
        },

        .init_output_node_props = [default_output_type, default_output_activation](
                                       NeuralNodeProps& props, std::mt19937& rng) {
            props.activation = default_output_activation;
            props.type = default_output_type;
            props.bias = 0.0f;
            if (default_output_type == NodeType::CTRNN) {
                std::uniform_real_distribution<float> tau_dist(0.5f, 5.0f);
                props.tau = tau_dist(rng);
            } else {
                props.tau = 1.0f;
            }
        },
    };
}

} // namespace neuralnet
