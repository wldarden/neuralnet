#pragma once

#include <neuralnet/neural_node_props.h>

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace neuralnet {

class GraphNetwork {
public:
    explicit GraphNetwork(const NeuralGenome& genome, float dt = 1.0f);

    [[nodiscard]] std::vector<float> forward(std::span<const float> input);
    [[nodiscard]] std::vector<float> forward(std::span<const float> input, float dt_override);

    void reset_state();

    [[nodiscard]] std::span<const float> get_node_states() const noexcept;
    [[nodiscard]] std::span<const float> get_node_outputs() const noexcept;
    void set_node_states(std::span<const float> states);
    [[nodiscard]] const NeuralGenome& genome() const noexcept;

    [[nodiscard]] std::size_t input_size() const noexcept;
    [[nodiscard]] std::size_t output_size() const noexcept;
    [[nodiscard]] std::size_t num_nodes() const noexcept;
    [[nodiscard]] std::size_t num_connections() const noexcept;

private:
    void build_topology();
    void validate() const;

    NeuralGenome genome_;
    float dt_;
    std::size_t num_inputs_ = 0;
    std::size_t num_outputs_ = 0;

    std::vector<float> node_states_;
    std::vector<float> node_outputs_;
    std::vector<uint32_t> eval_order_;

    std::vector<std::vector<std::pair<uint32_t, float>>> feedforward_by_target_;
    std::vector<std::vector<std::pair<uint32_t, float>>> recurrent_by_target_;

    std::unordered_map<uint32_t, uint32_t> id_to_index_;
    std::vector<uint32_t> input_indices_;
    std::vector<uint32_t> output_indices_;

    std::vector<NodeType> node_types_;
    std::vector<Activation> node_activations_;
    std::vector<float> node_biases_;
    std::vector<float> node_taus_;
};

} // namespace neuralnet
