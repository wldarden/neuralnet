#pragma once

#include <neuralnet/activation.h>

#include <cstddef>
#include <span>
#include <vector>

namespace neuralnet {

/// A single dense layer: output = activate(weights * input + bias).
/// Weights stored in row-major order: weights[output_idx * input_size + input_idx].
/// Supports per-layer activation (all nodes same) or per-node activation (each node different).
class Layer {
public:
    /// Per-layer activation: all nodes use the same function.
    Layer(std::size_t input_size,
          std::size_t output_size,
          std::span<const float> weights,
          std::span<const float> biases,
          Activation activation);

    /// Per-node activation: each node can use a different function.
    Layer(std::size_t input_size,
          std::size_t output_size,
          std::span<const float> weights,
          std::span<const float> biases,
          std::span<const Activation> activations);

    /// Run the layer on an input vector. Returns the activated output.
    [[nodiscard]] const std::vector<float>& forward(std::span<const float> input) const;

    [[nodiscard]] std::size_t input_size() const noexcept { return input_size_; }
    [[nodiscard]] std::size_t output_size() const noexcept { return output_size_; }

    /// Returns the default activation (first node's activation if per-node).
    [[nodiscard]] Activation activation() const noexcept { return activations_[0]; }

    /// Returns true if nodes have mixed activations.
    [[nodiscard]] bool has_per_node_activations() const noexcept { return per_node_; }

    /// Per-node activation array (size = output_size).
    [[nodiscard]] std::span<const Activation> activations() const noexcept { return activations_; }

    /// Returns activation values from the last forward pass.
    [[nodiscard]] std::span<const float> get_last_activations() const noexcept;

    /// Read-only access to weights and biases.
    [[nodiscard]] std::span<const float> weights() const noexcept;
    [[nodiscard]] std::span<const float> biases() const noexcept;

private:
    std::size_t input_size_;
    std::size_t output_size_;
    bool per_node_ = false;
    std::vector<Activation> activations_;  // [output_size_], uniform or mixed
    std::vector<float> weights_;  // row-major: [output_size_ x input_size_]
    std::vector<float> biases_;   // [output_size_]
    mutable std::vector<float> last_activations_;  // cached for inspector
};

} // namespace neuralnet
