#pragma once

#include <neuralnet/activation.h>
#include <neuralnet/layer.h>

#include <cstddef>
#include <span>
#include <vector>

namespace neuralnet {

/// Describes one layer in the network topology.
struct LayerDef {
    std::size_t output_size;
    Activation activation;                      // default activation for all nodes
    std::vector<Activation> node_activations;   // per-node overrides (empty = use default)
};

/// Describes the full network shape (used to construct a Network from flat weights).
struct NetworkTopology {
    std::size_t input_size;
    std::vector<LayerDef> layers;
};

/// A feedforward neural network: ordered sequence of dense layers.
/// Immutable after construction — to mutate, extract weights, modify, reconstruct.
class Network {
public:
    /// Construct from a topology and a flat weight vector.
    /// The weight vector is sliced into per-layer weights and biases
    /// in order: [layer0_weights, layer0_biases, layer1_weights, layer1_biases, ...].
    Network(const NetworkTopology& topology, std::span<const float> flat_weights);

    /// Forward pass: input -> output.
    [[nodiscard]] std::vector<float> forward(std::span<const float> input) const;

    /// Convenience overload accepting a braced-init-list of floats.
    [[nodiscard]] std::vector<float> forward(std::initializer_list<float> input) const {
        return forward(std::span<const float>{input.begin(), input.size()});
    }

    /// Extract all weights as a flat vector (same format as constructor input).
    [[nodiscard]] std::vector<float> get_all_weights() const;

    /// Get activation values per layer from the last forward pass.
    [[nodiscard]] std::vector<std::vector<float>> get_all_activations() const;

    [[nodiscard]] std::size_t input_size() const noexcept;
    [[nodiscard]] std::size_t output_size() const noexcept;
    [[nodiscard]] std::size_t total_weights() const noexcept;
    [[nodiscard]] const NetworkTopology& topology() const noexcept;

private:
    NetworkTopology topology_;
    std::vector<Layer> layers_;
    std::size_t total_weights_;
};

} // namespace neuralnet
