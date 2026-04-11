#include <neuralnet/graph_network.h>
#include <neuralnet/activation.h>
#include <evolve/node_role.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <queue>

namespace neuralnet {

GraphNetwork::GraphNetwork(const NeuralGenome& genome, float dt)
    : genome_(genome), dt_(dt) {
    validate();
    build_topology();
}

void GraphNetwork::validate() const {
    std::size_t input_count = 0;
    std::size_t output_count = 0;
    std::unordered_set<uint32_t> node_ids;
    std::unordered_set<uint32_t> input_ids;

    for (const auto& node : genome_.nodes) {
        if (!node_ids.insert(node.id).second) {
            throw std::invalid_argument("Duplicate node ID: " + std::to_string(node.id));
        }
        if (node.role == evolve::NodeRole::Input) {
            ++input_count;
            input_ids.insert(node.id);
        }
        if (node.role == evolve::NodeRole::Output) ++output_count;
    }

    if (input_count == 0) throw std::invalid_argument("GraphGenome has zero input nodes");
    if (output_count == 0) throw std::invalid_argument("GraphGenome has zero output nodes");

    for (const auto& conn : genome_.connections) {
        if (!node_ids.contains(conn.from_node))
            throw std::invalid_argument("Connection references non-existent from_node: " + std::to_string(conn.from_node));
        if (!node_ids.contains(conn.to_node))
            throw std::invalid_argument("Connection references non-existent to_node: " + std::to_string(conn.to_node));
        if (input_ids.contains(conn.to_node))
            throw std::invalid_argument("Connection targets input node: " + std::to_string(conn.to_node));
    }
}

void GraphNetwork::build_topology() {
    const auto node_count = genome_.nodes.size();

    for (std::size_t idx = 0; idx < node_count; ++idx) {
        id_to_index_[genome_.nodes[idx].id] = static_cast<uint32_t>(idx);
    }

    node_states_.resize(node_count, 0.0f);
    node_outputs_.resize(node_count, 0.0f);
    node_types_.resize(node_count);
    node_activations_.resize(node_count);
    node_biases_.resize(node_count);
    node_taus_.resize(node_count);

    for (std::size_t idx = 0; idx < node_count; ++idx) {
        const auto& node = genome_.nodes[idx];
        node_types_[idx] = node.props.type;
        node_activations_[idx] = node.props.activation;
        node_biases_[idx] = node.props.bias;
        node_taus_[idx] = node.props.tau;

        if (node.role == evolve::NodeRole::Input) {
            input_indices_.push_back(static_cast<uint32_t>(idx));
        } else if (node.role == evolve::NodeRole::Output) {
            output_indices_.push_back(static_cast<uint32_t>(idx));
        }
    }
    num_inputs_ = input_indices_.size();
    num_outputs_ = output_indices_.size();

    std::sort(output_indices_.begin(), output_indices_.end(),
        [this](uint32_t a, uint32_t b) {
            return genome_.nodes[a].id < genome_.nodes[b].id;
        });

    // BFS for reachability from inputs
    std::vector<std::vector<uint32_t>> adj(node_count);
    for (const auto& conn : genome_.connections) {
        if (!conn.enabled) continue;
        adj[id_to_index_[conn.from_node]].push_back(id_to_index_[conn.to_node]);
    }

    std::vector<bool> reachable(node_count, false);
    {
        std::queue<uint32_t> q;
        for (auto idx : input_indices_) {
            reachable[idx] = true;
            q.push(idx);
        }
        while (!q.empty()) {
            auto u = q.front(); q.pop();
            for (auto v : adj[u]) {
                if (!reachable[v]) { reachable[v] = true; q.push(v); }
            }
        }
    }

    // Kahn's algorithm for topological sort
    std::vector<int> topo_position(node_count, -1);
    int pos = 0;
    for (auto idx : input_indices_) {
        topo_position[idx] = pos++;
    }

    std::vector<uint32_t> kahn_in(node_count, 0);
    std::vector<std::vector<uint32_t>> fwd_adj(node_count);
    for (const auto& conn : genome_.connections) {
        if (!conn.enabled) continue;
        auto from_idx = id_to_index_[conn.from_node];
        auto to_idx = id_to_index_[conn.to_node];
        if (!reachable[from_idx] || !reachable[to_idx]) continue;
        if (from_idx == to_idx) continue;
        fwd_adj[from_idx].push_back(to_idx);
        kahn_in[to_idx]++;
    }

    {
        std::queue<uint32_t> q;
        for (auto idx : input_indices_) q.push(idx);
        while (!q.empty()) {
            auto u = q.front(); q.pop();
            for (auto v : fwd_adj[u]) {
                kahn_in[v]--;
                if (kahn_in[v] == 0) {
                    topo_position[v] = pos++;
                    eval_order_.push_back(v);
                    q.push(v);
                }
            }
        }
    }

    // Nodes in cycles: reachable but not yet positioned, and not inputs
    for (std::size_t idx = 0; idx < node_count; ++idx) {
        if (reachable[idx] && topo_position[idx] == -1
            && genome_.nodes[idx].role != evolve::NodeRole::Input) {
            topo_position[idx] = pos++;
            eval_order_.push_back(static_cast<uint32_t>(idx));
        }
    }

    // Classify connections as feedforward or recurrent
    feedforward_by_target_.resize(node_count);
    recurrent_by_target_.resize(node_count);

    for (const auto& conn : genome_.connections) {
        if (!conn.enabled) continue;
        auto from_idx = id_to_index_[conn.from_node];
        auto to_idx = id_to_index_[conn.to_node];
        if (!reachable[from_idx] || !reachable[to_idx]) continue;

        if (from_idx == to_idx) {
            recurrent_by_target_[to_idx].emplace_back(from_idx, conn.weight);
        } else if (topo_position[from_idx] < topo_position[to_idx]) {
            feedforward_by_target_[to_idx].emplace_back(from_idx, conn.weight);
        } else {
            recurrent_by_target_[to_idx].emplace_back(from_idx, conn.weight);
        }
    }
}

const NeuralGenome& GraphNetwork::genome() const noexcept { return genome_; }
std::size_t GraphNetwork::input_size() const noexcept { return num_inputs_; }
std::size_t GraphNetwork::output_size() const noexcept { return num_outputs_; }
std::size_t GraphNetwork::num_nodes() const noexcept { return genome_.nodes.size(); }
std::size_t GraphNetwork::num_connections() const noexcept { return genome_.connections.size(); }
std::span<const float> GraphNetwork::get_node_states() const noexcept { return node_states_; }
std::span<const float> GraphNetwork::get_node_outputs() const noexcept { return node_outputs_; }

void GraphNetwork::reset_state() {
    std::fill(node_states_.begin(), node_states_.end(), 0.0f);
    std::fill(node_outputs_.begin(), node_outputs_.end(), 0.0f);
}

void GraphNetwork::set_node_states(std::span<const float> states) {
    if (states.size() != node_states_.size()) {
        throw std::invalid_argument("State size mismatch");
    }
    std::copy(states.begin(), states.end(), node_states_.begin());
    std::copy(states.begin(), states.end(), node_outputs_.begin());
}

std::vector<float> GraphNetwork::forward(std::span<const float> input) {
    return forward(input, dt_);
}

std::vector<float> GraphNetwork::forward(std::span<const float> input, float dt_override) {
    if (input.size() != num_inputs_) {
        throw std::invalid_argument("Input size mismatch: expected "
            + std::to_string(num_inputs_) + " got " + std::to_string(input.size()));
    }

    for (std::size_t i = 0; i < num_inputs_; ++i) {
        node_outputs_[input_indices_[i]] = input[i];
    }

    for (auto idx : eval_order_) {
        float weighted_sum = node_biases_[idx];
        for (const auto& [src_idx, weight] : feedforward_by_target_[idx]) {
            weighted_sum += weight * node_outputs_[src_idx];
        }
        for (const auto& [src_idx, weight] : recurrent_by_target_[idx]) {
            weighted_sum += weight * node_outputs_[src_idx];
        }

        float activated = activate(node_activations_[idx], weighted_sum);

        if (node_types_[idx] == NodeType::CTRNN) {
            float tau = node_taus_[idx];
            node_states_[idx] += (dt_override / tau) * (-node_states_[idx] + activated);
            node_outputs_[idx] = node_states_[idx];
        } else {
            node_outputs_[idx] = activated;
        }
    }

    std::vector<float> result;
    result.reserve(num_outputs_);
    for (auto idx : output_indices_) {
        result.push_back(node_outputs_[idx]);
    }
    return result;
}

} // namespace neuralnet
