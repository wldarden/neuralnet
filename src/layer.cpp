#include <neuralnet/layer.h>

#include <stdexcept>
#include <string>

namespace neuralnet {

Layer::Layer(std::size_t input_size,
             std::size_t output_size,
             std::span<const float> weights,
             std::span<const float> biases,
             Activation activation)
    : input_size_(input_size),
      output_size_(output_size),
      per_node_(false),
      activations_(output_size, activation),
      weights_(weights.begin(), weights.end()),
      biases_(biases.begin(), biases.end()),
      last_activations_(output_size, 0.0f) {
    if (weights.size() != input_size * output_size) {
        throw std::invalid_argument("Weight size mismatch: expected "
            + std::to_string(input_size * output_size) + " got " + std::to_string(weights.size()));
    }
    if (biases.size() != output_size) {
        throw std::invalid_argument("Bias size mismatch: expected "
            + std::to_string(output_size) + " got " + std::to_string(biases.size()));
    }
}

Layer::Layer(std::size_t input_size,
             std::size_t output_size,
             std::span<const float> weights,
             std::span<const float> biases,
             std::span<const Activation> activations)
    : input_size_(input_size),
      output_size_(output_size),
      per_node_(true),
      activations_(activations.begin(), activations.end()),
      weights_(weights.begin(), weights.end()),
      biases_(biases.begin(), biases.end()),
      last_activations_(output_size, 0.0f) {
    if (weights.size() != input_size * output_size) {
        throw std::invalid_argument("Weight size mismatch: expected "
            + std::to_string(input_size * output_size) + " got " + std::to_string(weights.size()));
    }
    if (biases.size() != output_size) {
        throw std::invalid_argument("Bias size mismatch: expected "
            + std::to_string(output_size) + " got " + std::to_string(biases.size()));
    }
    if (activations.size() != output_size) {
        throw std::invalid_argument("Activations size mismatch: expected "
            + std::to_string(output_size) + " got " + std::to_string(activations.size()));
    }
}

const std::vector<float>& Layer::forward(std::span<const float> input) const {
    if (input.size() != input_size_) {
        throw std::invalid_argument("Layer input size mismatch: expected "
            + std::to_string(input_size_) + " got " + std::to_string(input.size()));
    }

    for (std::size_t o = 0; o < output_size_; ++o) {
        float sum = biases_[o];
        for (std::size_t i = 0; i < input_size_; ++i) {
            sum += weights_[o * input_size_ + i] * input[i];
        }
        last_activations_[o] = activate(activations_[o], sum);
    }

    return last_activations_;
}

std::span<const float> Layer::get_last_activations() const noexcept {
    return last_activations_;
}

std::span<const float> Layer::weights() const noexcept {
    return weights_;
}

std::span<const float> Layer::biases() const noexcept {
    return biases_;
}

} // namespace neuralnet
