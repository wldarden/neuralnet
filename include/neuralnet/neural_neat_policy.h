#pragma once
#include <neuralnet/neural_node_props.h>
#include <evolve/neat_policy.h>
#include <random>

namespace neuralnet {

struct NeuralMutationConfig {
    float bias_mutate_rate        = 0.40f;
    float bias_perturb_strength   = 0.2f;
    float tau_mutate_rate         = 0.20f;
    float tau_perturb_strength    = 0.1f;
    float tau_min                 = 0.1f;
    float tau_max                 = 100.0f;
    float node_type_mutate_rate   = 0.05f;
    float activation_mutate_rate  = 0.05f;
};

[[nodiscard]] evolve::NeatPolicy<NeuralNodeProps> make_neural_neat_policy(
    const NeuralMutationConfig& config,
    NodeType default_output_type = NodeType::Stateless,
    Activation default_output_activation = Activation::Tanh);

void mutate_biases(NeuralGenome& genome, const NeuralMutationConfig& config, std::mt19937& rng);
void mutate_tau(NeuralGenome& genome, const NeuralMutationConfig& config, std::mt19937& rng);
void mutate_node_types(NeuralGenome& genome, const NeuralMutationConfig& config, std::mt19937& rng);
void mutate_activations(NeuralGenome& genome, const NeuralMutationConfig& config, std::mt19937& rng);

} // namespace neuralnet
