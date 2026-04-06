#include <neuralnet/network.h>

#include <stdexcept>
#include <string>

namespace neuralnet {

Network::Network(const NetworkTopology& topology, std::span<const float> flat_weights)
    : topology_(topology), total_weights_(flat_weights.size()) {

    std::size_t offset = 0;
    std::size_t prev_size = topology.input_size;

    for (const auto& layer_def : topology.layers) {
        const auto weight_count = prev_size * layer_def.output_size;
        const auto bias_count = layer_def.output_size;

        if (offset + weight_count + bias_count > flat_weights.size()) {
            throw std::invalid_argument("Insufficient weights: need at least "
                + std::to_string(offset + weight_count + bias_count)
                + " but got " + std::to_string(flat_weights.size()));
        }

        auto weights = flat_weights.subspan(offset, weight_count);
        offset += weight_count;

        auto biases = flat_weights.subspan(offset, bias_count);
        offset += bias_count;

        if (!layer_def.node_activations.empty()) {
            layers_.emplace_back(prev_size, layer_def.output_size, weights, biases,
                                 std::span<const Activation>(layer_def.node_activations));
        } else {
            layers_.emplace_back(prev_size, layer_def.output_size, weights, biases,
                                 layer_def.activation);
        }

        prev_size = layer_def.output_size;
    }

    if (offset != flat_weights.size()) {
        throw std::invalid_argument("Excess weights: used "
            + std::to_string(offset) + " of " + std::to_string(flat_weights.size()));
    }
}

std::vector<float> Network::forward(std::span<const float> input) const {
    if (input.size() != topology_.input_size) {
        throw std::invalid_argument("Input size mismatch: expected "
            + std::to_string(topology_.input_size) + " got " + std::to_string(input.size()));
    }

    if (layers_.empty()) {
        return {input.begin(), input.end()};
    }

    // Chain forward: each layer writes to its own cache, next layer reads from it.
    // No intermediate allocations — only the final return copies.
    const std::vector<float>* last_output = &layers_[0].forward(input);
    for (std::size_t i = 1; i < layers_.size(); ++i) {
        last_output = &layers_[i].forward(*last_output);
    }
    return {last_output->begin(), last_output->end()};
}

std::vector<float> Network::get_all_weights() const {
    std::vector<float> result;
    result.reserve(total_weights_);
    for (const auto& layer : layers_) {
        auto w = layer.weights();
        result.insert(result.end(), w.begin(), w.end());
        auto b = layer.biases();
        result.insert(result.end(), b.begin(), b.end());
    }
    return result;
}

std::vector<std::vector<float>> Network::get_all_activations() const {
    std::vector<std::vector<float>> result;
    result.reserve(layers_.size());
    for (const auto& layer : layers_) {
        auto acts = layer.get_last_activations();
        result.emplace_back(acts.begin(), acts.end());
    }
    return result;
}

std::size_t Network::input_size() const noexcept {
    return topology_.input_size;
}

std::size_t Network::output_size() const noexcept {
    return layers_.empty() ? topology_.input_size : layers_.back().output_size();
}

std::size_t Network::total_weights() const noexcept {
    return total_weights_;
}

const NetworkTopology& Network::topology() const noexcept {
    return topology_;
}

const std::vector<std::string>& Network::input_ids() const noexcept {
    return topology_.input_ids;
}

const std::vector<std::string>& Network::output_ids() const noexcept {
    return topology_.output_ids;
}

} // namespace neuralnet
