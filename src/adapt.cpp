#include <neuralnet/adapt.h>

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace neuralnet {

std::vector<float> adapt_input(
    const std::vector<std::string>& source_ids,
    const std::vector<float>& values,
    const std::vector<std::string>& target_ids,
    float default_value) {

    std::unordered_map<std::string, std::size_t> source_map;
    source_map.reserve(source_ids.size());
    for (std::size_t i = 0; i < source_ids.size(); ++i) {
        source_map[source_ids[i]] = i;
    }

    std::vector<float> result(target_ids.size());
    for (std::size_t j = 0; j < target_ids.size(); ++j) {
        const auto it = source_map.find(target_ids[j]);
        if (it != source_map.end()) {
            result[j] = values[it->second];
        } else {
            result[j] = default_value;
        }
    }
    return result;
}

AdaptResult adapt_topology_inputs(
    const NetworkTopology& source_topology,
    const std::vector<float>& source_weights,
    const std::vector<std::string>& target_input_ids,
    std::mt19937& rng) {

    if (source_topology.input_ids.empty()) {
        throw std::invalid_argument(
            "adapt_topology_inputs requires source topology to have input_ids");
    }
    if (source_topology.layers.empty()) {
        throw std::invalid_argument(
            "adapt_topology_inputs requires source topology to have at least one layer");
    }

    // Build source id -> column index map.
    std::unordered_map<std::string, std::size_t> source_map;
    source_map.reserve(source_topology.input_ids.size());
    for (std::size_t i = 0; i < source_topology.input_ids.size(); ++i) {
        source_map[source_topology.input_ids[i]] = i;
    }

    const auto new_input_size = target_input_ids.size();
    const auto old_input_size = source_topology.input_size;
    const auto output_size = source_topology.layers[0].output_size;
    const auto old_l0_weights = old_input_size * output_size;
    const auto old_l0_biases = output_size;

    // Build new layer 0 weight matrix (row-major: weights[row * cols + col]).
    std::vector<float> new_l0(output_size * new_input_size);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    for (std::size_t o = 0; o < output_size; ++o) {
        for (std::size_t j = 0; j < new_input_size; ++j) {
            const auto it = source_map.find(target_input_ids[j]);
            if (it != source_map.end()) {
                const auto c = it->second;
                new_l0[o * new_input_size + j] =
                    source_weights[o * old_input_size + c];
            } else {
                new_l0[o * new_input_size + j] = dist(rng);
            }
        }
    }

    // Assemble adapted weights: new L0 weights + old L0 biases + remaining layers.
    std::vector<float> adapted_weights;
    adapted_weights.reserve(
        new_l0.size() + old_l0_biases +
        (source_weights.size() - old_l0_weights - old_l0_biases));

    // New layer 0 weights.
    adapted_weights.insert(adapted_weights.end(), new_l0.begin(), new_l0.end());

    // Layer 0 biases (unchanged).
    const auto bias_start = source_weights.begin()
        + static_cast<std::ptrdiff_t>(old_l0_weights);
    const auto bias_end = bias_start
        + static_cast<std::ptrdiff_t>(old_l0_biases);
    adapted_weights.insert(adapted_weights.end(), bias_start, bias_end);

    // All remaining layers (unchanged).
    const auto rest_start = bias_end;
    adapted_weights.insert(adapted_weights.end(), rest_start, source_weights.end());

    // Build adapted topology.
    NetworkTopology adapted_topology = source_topology;
    adapted_topology.input_size = new_input_size;
    adapted_topology.input_ids = target_input_ids;

    // Compute added IDs (in target but not source).
    std::vector<std::string> added_ids;
    for (const auto& id : target_input_ids) {
        if (source_map.find(id) == source_map.end()) {
            added_ids.push_back(id);
        }
    }

    // Compute removed IDs (in source but not target).
    std::unordered_set<std::string> target_set(
        target_input_ids.begin(), target_input_ids.end());
    std::vector<std::string> removed_ids;
    for (const auto& id : source_topology.input_ids) {
        if (target_set.find(id) == target_set.end()) {
            removed_ids.push_back(id);
        }
    }

    return AdaptResult{
        .adapted_weights = std::move(adapted_weights),
        .adapted_topology = std::move(adapted_topology),
        .added_ids = std::move(added_ids),
        .removed_ids = std::move(removed_ids),
    };
}

} // namespace neuralnet
