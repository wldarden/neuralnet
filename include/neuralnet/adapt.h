#pragma once

#include <neuralnet/network.h>

#include <random>
#include <string>
#include <vector>

namespace neuralnet {

/// Reorder/pad/trim a float vector from source IDs to target IDs.
/// - Present in both: copied to correct position in output.
/// - In target but not source (missing): filled with default_value.
/// - In source but not target (extra): ignored.
/// Returns vector sized to target_ids.size().
[[nodiscard]] std::vector<float> adapt_input(
    const std::vector<std::string>& source_ids,
    const std::vector<float>& values,
    const std::vector<std::string>& target_ids,
    float default_value = 0.0f);

/// Result of adapting a topology's input layout.
struct AdaptResult {
    std::vector<float> adapted_weights;
    NetworkTopology adapted_topology;
    std::vector<std::string> added_ids;
    std::vector<std::string> removed_ids;
};

/// Adapt a topology's first-layer weights to match a new input ID set.
/// Requires source_topology.input_ids to be non-empty (throws if empty).
/// - Columns for matching IDs: preserved in new order.
/// - Columns for missing IDs: filled with small random weights.
/// - Columns for extra IDs: dropped.
/// Higher layers unchanged. Output IDs preserved.
[[nodiscard]] AdaptResult adapt_topology_inputs(
    const NetworkTopology& source_topology,
    const std::vector<float>& source_weights,
    const std::vector<std::string>& target_input_ids,
    std::mt19937& rng);

} // namespace neuralnet
