# GraphNetwork + NEAT Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add NEAT-style graph networks with pluggable node types (Stateless, CTRNN) to `libs/neuralnet`, and NEAT evolution operators to `libs/evolve`.

**Architecture:** New `GraphNetwork` type coexists alongside existing MLP `Network`. Data types (enums, structs, genome) live in `libs/neuralnet`. Evolution logic (mutation, crossover, speciation, population) lives in `libs/evolve` with a dependency on `libs/neuralnet`. Versioned serialization wraps both network types.

**Tech Stack:** C++20, CMake, GoogleTest, `-Wall -Wextra -Wpedantic -Werror`

**Spec:** `libs/neuralnet/docs/superpowers/specs/2026-03-21-graph-network-neat-design.md`

**Key conventions (match existing code):**
- Namespace `neuralnet` for all neuralnet types, `evolve` for evolution types
- Test aliases: `namespace nn = neuralnet;`, `namespace ev = evolve;`
- Test fixture naming: `TEST(ClassName, BehaviorDescription)`
- `[[nodiscard]]`, `noexcept` where applicable
- `std::span<const float>` for input arrays
- Row-major weight storage
- Binary serialization with magic numbers
- `#pragma once` for header guards

---

### Task 1: Node Types and GraphGenome Data Structures

**Files:**
- Create: `libs/neuralnet/include/neuralnet/node_types.h`
- Create: `libs/neuralnet/include/neuralnet/graph_genome.h`
- Create: `libs/neuralnet/src/graph_genome.cpp` (empty initially)
- Create: `libs/neuralnet/tests/graph_genome_test.cpp`
- Modify: `libs/neuralnet/CMakeLists.txt`
- Modify: `libs/neuralnet/tests/CMakeLists.txt`

- [ ] **Step 1: Create node_types.h**

```cpp
// libs/neuralnet/include/neuralnet/node_types.h
#pragma once

#include <neuralnet/activation.h>

#include <cstdint>

namespace neuralnet {

enum class NodeType : uint8_t {
    Stateless = 0,
    CTRNN     = 1,
};

enum class NodeRole : uint8_t {
    Input  = 0,
    Hidden = 1,
    Output = 2,
};

struct NodeGene {
    uint32_t id;
    NodeRole role;
    NodeType type;
    Activation activation;
    float bias;
    float tau;
};

} // namespace neuralnet
```

- [ ] **Step 2: Create graph_genome.h**

```cpp
// libs/neuralnet/include/neuralnet/graph_genome.h
#pragma once

#include <neuralnet/node_types.h>

#include <cstdint>
#include <vector>

namespace neuralnet {

struct ConnectionGene {
    uint32_t from_node;
    uint32_t to_node;
    float weight;
    bool enabled;
    uint32_t innovation;
};

struct GraphGenome {
    std::vector<NodeGene> nodes;
    std::vector<ConnectionGene> connections;
};

} // namespace neuralnet
```

- [ ] **Step 3: Create empty graph_genome.cpp**

```cpp
// libs/neuralnet/src/graph_genome.cpp
#include <neuralnet/graph_genome.h>

namespace neuralnet {

// Utilities will be added in Task 2.

} // namespace neuralnet
```

- [ ] **Step 4: Write test that structs compile and have expected defaults**

```cpp
// libs/neuralnet/tests/graph_genome_test.cpp
#include <neuralnet/graph_genome.h>
#include <neuralnet/node_types.h>

#include <gtest/gtest.h>

namespace nn = neuralnet;

TEST(NodeTypesTest, EnumsHaveExpectedValues) {
    EXPECT_EQ(static_cast<uint8_t>(nn::NodeType::Stateless), 0);
    EXPECT_EQ(static_cast<uint8_t>(nn::NodeType::CTRNN), 1);
    EXPECT_EQ(static_cast<uint8_t>(nn::NodeRole::Input), 0);
    EXPECT_EQ(static_cast<uint8_t>(nn::NodeRole::Hidden), 1);
    EXPECT_EQ(static_cast<uint8_t>(nn::NodeRole::Output), 2);
}

TEST(GraphGenomeTest, EmptyGenome) {
    nn::GraphGenome genome;
    EXPECT_TRUE(genome.nodes.empty());
    EXPECT_TRUE(genome.connections.empty());
}

TEST(GraphGenomeTest, NodeGeneFields) {
    nn::NodeGene node{
        .id = 0,
        .role = nn::NodeRole::Input,
        .type = nn::NodeType::Stateless,
        .activation = nn::Activation::ReLU,
        .bias = 0.0f,
        .tau = 1.0f,
    };
    EXPECT_EQ(node.id, 0);
    EXPECT_EQ(node.role, nn::NodeRole::Input);
}

TEST(ConnectionGeneTest, Fields) {
    nn::ConnectionGene conn{
        .from_node = 0,
        .to_node = 2,
        .weight = 0.5f,
        .enabled = true,
        .innovation = 0,
    };
    EXPECT_EQ(conn.from_node, 0);
    EXPECT_EQ(conn.to_node, 2);
    EXPECT_FLOAT_EQ(conn.weight, 0.5f);
    EXPECT_TRUE(conn.enabled);
}
```

- [ ] **Step 5: Update CMakeLists.txt to include new files**

Add `src/graph_genome.cpp` to `libs/neuralnet/CMakeLists.txt`:

```cmake
add_library(neuralnet STATIC
    src/activation.cpp
    src/layer.cpp
    src/network.cpp
    src/serialization.cpp
    src/graph_genome.cpp
)
```

Add test file to `libs/neuralnet/tests/CMakeLists.txt`:

```cmake
target_sources(neuralnet_tests PRIVATE graph_genome_test.cpp)
```

- [ ] **Step 6: Build and run tests**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target neuralnet_tests && ./build/libs/neuralnet/tests/neuralnet_tests --gtest_filter='NodeTypes*:GraphGenome*:ConnectionGene*'`
Expected: All 4 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add libs/neuralnet/include/neuralnet/node_types.h libs/neuralnet/include/neuralnet/graph_genome.h libs/neuralnet/src/graph_genome.cpp libs/neuralnet/tests/graph_genome_test.cpp libs/neuralnet/CMakeLists.txt libs/neuralnet/tests/CMakeLists.txt
git commit -m "feat(neuralnet): add NodeType, NodeRole, NodeGene, ConnectionGene, GraphGenome structs"
```

---

### Task 2: create_minimal_genome

**Files:**
- Modify: `libs/neuralnet/include/neuralnet/graph_genome.h`
- Modify: `libs/neuralnet/src/graph_genome.cpp`
- Modify: `libs/neuralnet/tests/graph_genome_test.cpp`

- [ ] **Step 1: Write failing tests for create_minimal_genome**

Append to `libs/neuralnet/tests/graph_genome_test.cpp`:

```cpp
TEST(GraphGenomeTest, CreateMinimal_2Inputs_1Output) {
    // create_minimal_genome needs an InnovationCounter, but we haven't built
    // the evolve side yet. Use a simple local counter for now.
    // We'll add the real InnovationCounter overload in Task 6.
    std::mt19937 rng(42);
    auto genome = nn::create_minimal_genome(
        2, 1,
        nn::NodeType::Stateless,
        nn::Activation::Tanh,
        rng
    );

    // 2 input nodes + 1 output node = 3 nodes
    EXPECT_EQ(genome.nodes.size(), 3);
    // 2 inputs * 1 output = 2 connections (fully connected)
    EXPECT_EQ(genome.connections.size(), 2);

    // Input nodes: IDs 0, 1
    EXPECT_EQ(genome.nodes[0].id, 0);
    EXPECT_EQ(genome.nodes[0].role, nn::NodeRole::Input);
    EXPECT_EQ(genome.nodes[1].id, 1);
    EXPECT_EQ(genome.nodes[1].role, nn::NodeRole::Input);

    // Output node: ID 2
    EXPECT_EQ(genome.nodes[2].id, 2);
    EXPECT_EQ(genome.nodes[2].role, nn::NodeRole::Output);
    EXPECT_EQ(genome.nodes[2].type, nn::NodeType::Stateless);
    EXPECT_EQ(genome.nodes[2].activation, nn::Activation::Tanh);

    // Connections: input 0 -> output 2, input 1 -> output 2
    EXPECT_EQ(genome.connections[0].from_node, 0);
    EXPECT_EQ(genome.connections[0].to_node, 2);
    EXPECT_TRUE(genome.connections[0].enabled);
    EXPECT_EQ(genome.connections[1].from_node, 1);
    EXPECT_EQ(genome.connections[1].to_node, 2);
    EXPECT_TRUE(genome.connections[1].enabled);
}

TEST(GraphGenomeTest, CreateMinimal_InnovationNumbersSequential) {
    std::mt19937 rng(42);
    auto genome = nn::create_minimal_genome(3, 2, nn::NodeType::CTRNN, nn::Activation::Tanh, rng);

    // 3 inputs * 2 outputs = 6 connections
    EXPECT_EQ(genome.connections.size(), 6);

    // Innovation numbers should be sequential starting from 0
    for (uint32_t i = 0; i < 6; ++i) {
        EXPECT_EQ(genome.connections[i].innovation, i);
    }
}

TEST(GraphGenomeTest, CreateMinimal_CTRNNOutputs) {
    std::mt19937 rng(42);
    auto genome = nn::create_minimal_genome(2, 2, nn::NodeType::CTRNN, nn::Activation::Tanh, rng);

    // Output nodes should be CTRNN with tau > 0
    for (std::size_t i = 2; i < genome.nodes.size(); ++i) {
        EXPECT_EQ(genome.nodes[i].type, nn::NodeType::CTRNN);
        EXPECT_GT(genome.nodes[i].tau, 0.0f);
    }
}

TEST(GraphGenomeTest, CreateMinimal_WeightsAreRandom) {
    std::mt19937 rng1(42);
    auto g1 = nn::create_minimal_genome(2, 1, nn::NodeType::Stateless, nn::Activation::Tanh, rng1);

    std::mt19937 rng2(99);
    auto g2 = nn::create_minimal_genome(2, 1, nn::NodeType::Stateless, nn::Activation::Tanh, rng2);

    // Different seeds should produce different weights
    bool any_different = false;
    for (std::size_t i = 0; i < g1.connections.size(); ++i) {
        if (g1.connections[i].weight != g2.connections[i].weight) {
            any_different = true;
        }
    }
    EXPECT_TRUE(any_different);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target neuralnet_tests 2>&1 | head -20`
Expected: Compilation fails — `create_minimal_genome` not declared.

- [ ] **Step 3: Add declaration to graph_genome.h**

Add to `libs/neuralnet/include/neuralnet/graph_genome.h` before the closing `}`:

```cpp
#include <random>

/// Create a minimal NEAT starting genome: inputs directly connected to outputs.
/// Innovation numbers are assigned sequentially starting from 0.
/// Weights are initialized from uniform distribution [-1, 1].
/// For a version that uses an InnovationCounter, see the overload in evolve/neat_operators.h.
[[nodiscard]] GraphGenome create_minimal_genome(
    std::size_t num_inputs,
    std::size_t num_outputs,
    NodeType default_output_type,
    Activation default_output_activation,
    std::mt19937& rng
);
```

- [ ] **Step 4: Implement create_minimal_genome in graph_genome.cpp**

```cpp
// libs/neuralnet/src/graph_genome.cpp
#include <neuralnet/graph_genome.h>

#include <random>

namespace neuralnet {

GraphGenome create_minimal_genome(
    std::size_t num_inputs,
    std::size_t num_outputs,
    NodeType default_output_type,
    Activation default_output_activation,
    std::mt19937& rng
) {
    GraphGenome genome;
    std::uniform_real_distribution<float> weight_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> tau_dist(0.5f, 5.0f);

    // Input nodes: IDs 0 .. num_inputs-1
    for (std::size_t i = 0; i < num_inputs; ++i) {
        genome.nodes.push_back(NodeGene{
            .id = static_cast<uint32_t>(i),
            .role = NodeRole::Input,
            .type = NodeType::Stateless,
            .activation = Activation::ReLU,  // Ignored for input nodes
            .bias = 0.0f,
            .tau = 1.0f,
        });
    }

    // Output nodes: IDs num_inputs .. num_inputs+num_outputs-1
    for (std::size_t i = 0; i < num_outputs; ++i) {
        float tau = (default_output_type == NodeType::CTRNN)
            ? tau_dist(rng) : 1.0f;
        genome.nodes.push_back(NodeGene{
            .id = static_cast<uint32_t>(num_inputs + i),
            .role = NodeRole::Output,
            .type = default_output_type,
            .activation = default_output_activation,
            .bias = 0.0f,
            .tau = tau,
        });
    }

    // Fully connect inputs to outputs
    uint32_t innovation = 0;
    for (std::size_t in = 0; in < num_inputs; ++in) {
        for (std::size_t out = 0; out < num_outputs; ++out) {
            genome.connections.push_back(ConnectionGene{
                .from_node = static_cast<uint32_t>(in),
                .to_node = static_cast<uint32_t>(num_inputs + out),
                .weight = weight_dist(rng),
                .enabled = true,
                .innovation = innovation++,
            });
        }
    }

    return genome;
}

} // namespace neuralnet
```

- [ ] **Step 5: Build and run tests**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target neuralnet_tests && ./build/libs/neuralnet/tests/neuralnet_tests --gtest_filter='GraphGenome*'`
Expected: All GraphGenome tests PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/neuralnet/include/neuralnet/graph_genome.h libs/neuralnet/src/graph_genome.cpp libs/neuralnet/tests/graph_genome_test.cpp
git commit -m "feat(neuralnet): implement create_minimal_genome for NEAT starting topologies"
```

---

### Task 3: GraphNetwork Construction and Validation

**Files:**
- Create: `libs/neuralnet/include/neuralnet/graph_network.h`
- Create: `libs/neuralnet/src/graph_network.cpp`
- Create: `libs/neuralnet/tests/graph_network_test.cpp`
- Modify: `libs/neuralnet/CMakeLists.txt`
- Modify: `libs/neuralnet/tests/CMakeLists.txt`

- [ ] **Step 1: Write failing tests for construction and validation**

```cpp
// libs/neuralnet/tests/graph_network_test.cpp
#include <neuralnet/graph_network.h>
#include <neuralnet/graph_genome.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace nn = neuralnet;

// Helper to create a simple 2-input, 1-output genome
nn::GraphGenome make_simple_genome() {
    std::mt19937 rng(42);
    return nn::create_minimal_genome(2, 1, nn::NodeType::Stateless, nn::Activation::Tanh, rng);
}

TEST(GraphNetworkTest, ConstructFromMinimalGenome) {
    auto genome = make_simple_genome();
    nn::GraphNetwork net(genome);

    EXPECT_EQ(net.input_size(), 2);
    EXPECT_EQ(net.output_size(), 1);
    EXPECT_EQ(net.num_nodes(), 3);
    EXPECT_EQ(net.num_connections(), 2);
}

TEST(GraphNetworkTest, ValidationRejectsZeroInputs) {
    nn::GraphGenome genome;
    genome.nodes.push_back({.id = 0, .role = nn::NodeRole::Output,
        .type = nn::NodeType::Stateless, .activation = nn::Activation::Tanh,
        .bias = 0.0f, .tau = 1.0f});
    EXPECT_THROW(nn::GraphNetwork(genome), std::invalid_argument);
}

TEST(GraphNetworkTest, ValidationRejectsZeroOutputs) {
    nn::GraphGenome genome;
    genome.nodes.push_back({.id = 0, .role = nn::NodeRole::Input,
        .type = nn::NodeType::Stateless, .activation = nn::Activation::Tanh,
        .bias = 0.0f, .tau = 1.0f});
    EXPECT_THROW(nn::GraphNetwork(genome), std::invalid_argument);
}

TEST(GraphNetworkTest, ValidationRejectsDuplicateNodeIDs) {
    nn::GraphGenome genome;
    genome.nodes.push_back({.id = 0, .role = nn::NodeRole::Input,
        .type = nn::NodeType::Stateless, .activation = nn::Activation::Tanh,
        .bias = 0.0f, .tau = 1.0f});
    genome.nodes.push_back({.id = 0, .role = nn::NodeRole::Output,
        .type = nn::NodeType::Stateless, .activation = nn::Activation::Tanh,
        .bias = 0.0f, .tau = 1.0f});
    EXPECT_THROW(nn::GraphNetwork(genome), std::invalid_argument);
}

TEST(GraphNetworkTest, ValidationRejectsDanglingConnection) {
    nn::GraphGenome genome;
    genome.nodes.push_back({.id = 0, .role = nn::NodeRole::Input,
        .type = nn::NodeType::Stateless, .activation = nn::Activation::Tanh,
        .bias = 0.0f, .tau = 1.0f});
    genome.nodes.push_back({.id = 1, .role = nn::NodeRole::Output,
        .type = nn::NodeType::Stateless, .activation = nn::Activation::Tanh,
        .bias = 0.0f, .tau = 1.0f});
    // Connection targets non-existent node 99
    genome.connections.push_back({.from_node = 0, .to_node = 99,
        .weight = 1.0f, .enabled = true, .innovation = 0});
    EXPECT_THROW(nn::GraphNetwork(genome), std::invalid_argument);
}

TEST(GraphNetworkTest, ValidationRejectsConnectionToInput) {
    nn::GraphGenome genome;
    genome.nodes.push_back({.id = 0, .role = nn::NodeRole::Input,
        .type = nn::NodeType::Stateless, .activation = nn::Activation::Tanh,
        .bias = 0.0f, .tau = 1.0f});
    genome.nodes.push_back({.id = 1, .role = nn::NodeRole::Input,
        .type = nn::NodeType::Stateless, .activation = nn::Activation::Tanh,
        .bias = 0.0f, .tau = 1.0f});
    genome.nodes.push_back({.id = 2, .role = nn::NodeRole::Output,
        .type = nn::NodeType::Stateless, .activation = nn::Activation::Tanh,
        .bias = 0.0f, .tau = 1.0f});
    // Connection targets input node — invalid
    genome.connections.push_back({.from_node = 2, .to_node = 0,
        .weight = 1.0f, .enabled = true, .innovation = 0});
    EXPECT_THROW(nn::GraphNetwork(genome), std::invalid_argument);
}
```

- [ ] **Step 2: Run tests to verify compilation fails**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target neuralnet_tests 2>&1 | head -10`
Expected: Fails — `graph_network.h` not found.

- [ ] **Step 3: Create graph_network.h**

```cpp
// libs/neuralnet/include/neuralnet/graph_network.h
#pragma once

#include <neuralnet/graph_genome.h>

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace neuralnet {

class GraphNetwork {
public:
    explicit GraphNetwork(const GraphGenome& genome, float dt = 1.0f);

    [[nodiscard]] std::vector<float> forward(std::span<const float> input);
    [[nodiscard]] std::vector<float> forward(std::span<const float> input, float dt_override);

    void reset_state();

    [[nodiscard]] std::span<const float> get_node_states() const noexcept;
    void set_node_states(std::span<const float> states);  // For deserialization
    [[nodiscard]] const GraphGenome& genome() const noexcept;

    [[nodiscard]] std::size_t input_size() const noexcept;
    [[nodiscard]] std::size_t output_size() const noexcept;
    [[nodiscard]] std::size_t num_nodes() const noexcept;
    [[nodiscard]] std::size_t num_connections() const noexcept;

private:
    void build_topology();
    void validate() const;

    GraphGenome genome_;
    float dt_;
    std::size_t num_inputs_ = 0;
    std::size_t num_outputs_ = 0;

    std::vector<float> node_states_;
    std::vector<float> node_outputs_;
    std::vector<uint32_t> eval_order_;  // Compact indices in topological order

    // Feedforward connections grouped by target node index
    // feedforward_by_target_[target_idx] = vector of (source_idx, weight)
    std::vector<std::vector<std::pair<uint32_t, float>>> feedforward_by_target_;
    // Recurrent connections grouped by target node index
    std::vector<std::vector<std::pair<uint32_t, float>>> recurrent_by_target_;

    std::unordered_map<uint32_t, uint32_t> id_to_index_;
    std::vector<uint32_t> input_indices_;
    std::vector<uint32_t> output_indices_;  // In node-ID order

    // Per-node properties (compact indexed)
    std::vector<NodeType> node_types_;
    std::vector<Activation> node_activations_;
    std::vector<float> node_biases_;
    std::vector<float> node_taus_;
};

} // namespace neuralnet
```

- [ ] **Step 4: Implement constructor with validation and topology analysis**

```cpp
// libs/neuralnet/src/graph_network.cpp
#include <neuralnet/graph_network.h>
#include <neuralnet/activation.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <queue>

namespace neuralnet {

GraphNetwork::GraphNetwork(const GraphGenome& genome, float dt)
    : genome_(genome), dt_(dt) {
    validate();
    build_topology();
}

void GraphNetwork::validate() const {
    // Check for zero inputs/outputs
    std::size_t input_count = 0;
    std::size_t output_count = 0;
    std::unordered_set<uint32_t> node_ids;

    for (const auto& node : genome_.nodes) {
        if (!node_ids.insert(node.id).second) {
            throw std::invalid_argument("Duplicate node ID: " + std::to_string(node.id));
        }
        if (node.role == NodeRole::Input) ++input_count;
        if (node.role == NodeRole::Output) ++output_count;
    }

    if (input_count == 0) {
        throw std::invalid_argument("GraphGenome has zero input nodes");
    }
    if (output_count == 0) {
        throw std::invalid_argument("GraphGenome has zero output nodes");
    }

    // Build set of input node IDs for O(1) lookup
    std::unordered_set<uint32_t> input_ids;
    for (const auto& node : genome_.nodes) {
        if (node.role == NodeRole::Input) input_ids.insert(node.id);
    }

    // Check connections reference valid nodes and don't target inputs
    for (const auto& conn : genome_.connections) {
        if (node_ids.find(conn.from_node) == node_ids.end()) {
            throw std::invalid_argument(
                "Connection references non-existent from_node: " + std::to_string(conn.from_node));
        }
        if (node_ids.find(conn.to_node) == node_ids.end()) {
            throw std::invalid_argument(
                "Connection references non-existent to_node: " + std::to_string(conn.to_node));
        }
        if (input_ids.count(conn.to_node) > 0) {
            throw std::invalid_argument(
                "Connection targets input node: " + std::to_string(conn.to_node));
        }
    }
}

void GraphNetwork::build_topology() {
    // Build ID-to-index mapping (compact)
    for (uint32_t idx = 0; idx < genome_.nodes.size(); ++idx) {
        id_to_index_[genome_.nodes[idx].id] = idx;
    }

    const auto node_count = genome_.nodes.size();

    // Allocate per-node arrays
    node_states_.resize(num_nodes, 0.0f);
    node_outputs_.resize(num_nodes, 0.0f);
    node_types_.resize(num_nodes);
    node_activations_.resize(num_nodes);
    node_biases_.resize(num_nodes);
    node_taus_.resize(num_nodes);

    // Fill per-node properties and collect input/output indices
    for (uint32_t idx = 0; idx < num_nodes; ++idx) {
        const auto& node = genome_.nodes[idx];
        node_types_[idx] = node.type;
        node_activations_[idx] = node.activation;
        node_biases_[idx] = node.bias;
        node_taus_[idx] = node.tau;

        if (node.role == NodeRole::Input) {
            input_indices_.push_back(idx);
        } else if (node.role == NodeRole::Output) {
            output_indices_.push_back(idx);
        }
    }
    num_inputs_ = input_indices_.size();
    num_outputs_ = output_indices_.size();

    // Sort output_indices_ by node ID so output order is deterministic
    std::sort(output_indices_.begin(), output_indices_.end(),
        [this](uint32_t a, uint32_t b) {
            return genome_.nodes[a].id < genome_.nodes[b].id;
        });

    // Build adjacency list from enabled connections (for reachability + topo sort)
    std::vector<std::vector<uint32_t>> adj(node_count);
    for (const auto& conn : genome_.connections) {
        if (!conn.enabled) continue;
        auto from_idx = id_to_index_[conn.from_node];
        auto to_idx = id_to_index_[conn.to_node];
        adj[from_idx].push_back(to_idx);
    }

    // BFS from input nodes to find reachable nodes
    std::vector<bool> reachable(node_count, false);
    {
        std::queue<uint32_t> bfs_queue;
        for (auto idx : input_indices_) {
            reachable[idx] = true;
            bfs_queue.push(idx);
        }
        while (!bfs_queue.empty()) {
            auto u = bfs_queue.front();
            bfs_queue.pop();
            for (auto v : adj[u]) {
                if (!reachable[v]) {
                    reachable[v] = true;
                    bfs_queue.push(v);
                }
            }
        }
    }

    // Kahn's algorithm for topological sort
    // Assign topological positions: input nodes first, then Kahn's on the rest
    std::vector<int> topo_position(node_count, -1);
    int pos = 0;

    for (auto idx : input_indices_) {
        topo_position[idx] = pos++;
    }

    // Build forward adjacency + in-degree for Kahn's (skip self-loops, unreachable)
    std::vector<uint32_t> kahn_in(node_count, 0);
    std::vector<std::vector<uint32_t>> fwd_adj(node_count);
    for (const auto& conn : genome_.connections) {
        if (!conn.enabled) continue;
        auto from_idx = id_to_index_[conn.from_node];
        auto to_idx = id_to_index_[conn.to_node];
        if (!reachable[from_idx] || !reachable[to_idx]) continue;
        if (from_idx == to_idx) continue; // self-connections are always recurrent
        fwd_adj[from_idx].push_back(to_idx);
        kahn_in[to_idx]++;
    }

    // Seed Kahn's with input nodes
    {
        std::queue<uint32_t> kahn_queue;
        for (auto idx : input_indices_) {
            kahn_queue.push(idx);
        }
        while (!kahn_queue.empty()) {
            auto u = kahn_queue.front();
            kahn_queue.pop();
            for (auto v : fwd_adj[u]) {
                kahn_in[v]--;
                if (kahn_in[v] == 0) {
                    topo_position[v] = pos++;
                    eval_order_.push_back(v);
                    kahn_queue.push(v);
                }
            }
        }
    }

    // Nodes still with kahn_in > 0 are in cycles — add them in arbitrary order
    // (their back-edges will be classified as recurrent)
    for (uint32_t idx = 0; idx < node_count; ++idx) {
        if (reachable[idx] && topo_position[idx] == -1
            && genome_.nodes[idx].role != NodeRole::Input) {
            topo_position[idx] = pos++;
            eval_order_.push_back(idx);
        }
    }

    // Classify connections as feedforward or recurrent based on topological position
    feedforward_by_target_.resize(num_nodes);
    recurrent_by_target_.resize(num_nodes);

    for (const auto& conn : genome_.connections) {
        if (!conn.enabled) continue;
        auto from_idx = id_to_index_[conn.from_node];
        auto to_idx = id_to_index_[conn.to_node];
        if (!reachable[from_idx] || !reachable[to_idx]) continue;

        if (from_idx == to_idx) {
            // Self-connection: always recurrent
            recurrent_by_target_[to_idx].emplace_back(from_idx, conn.weight);
        } else if (topo_position[from_idx] < topo_position[to_idx]) {
            // Feedforward: from appears before to in topological order
            feedforward_by_target_[to_idx].emplace_back(from_idx, conn.weight);
        } else {
            // Recurrent: from appears at same or later position
            recurrent_by_target_[to_idx].emplace_back(from_idx, conn.weight);
        }
    }
}

const GraphGenome& GraphNetwork::genome() const noexcept {
    return genome_;
}

std::size_t GraphNetwork::input_size() const noexcept {
    return num_inputs_;
}

std::size_t GraphNetwork::output_size() const noexcept {
    return num_outputs_;
}

std::size_t GraphNetwork::num_nodes() const noexcept {
    return genome_.nodes.size();
}

std::size_t GraphNetwork::num_connections() const noexcept {
    return genome_.connections.size();
}

std::span<const float> GraphNetwork::get_node_states() const noexcept {
    return node_states_;
}

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
        throw std::invalid_argument(
            "Input size mismatch: expected " + std::to_string(num_inputs_)
            + " got " + std::to_string(input.size()));
    }

    // Step 1: Set input node outputs
    for (std::size_t i = 0; i < num_inputs_; ++i) {
        node_outputs_[input_indices_[i]] = input[i];
    }

    // Step 2: Evaluate nodes in topological order
    for (auto idx : eval_order_) {
        float weighted_sum = node_biases_[idx];

        // Sum feedforward connections (current tick values)
        for (auto [src_idx, weight] : feedforward_by_target_[idx]) {
            weighted_sum += weight * node_outputs_[src_idx];
        }
        // Sum recurrent connections (previous tick values — still in node_outputs_)
        for (auto [src_idx, weight] : recurrent_by_target_[idx]) {
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

    // Step 3: Collect output values in node-ID order
    std::vector<float> result;
    result.reserve(num_outputs_);
    for (auto idx : output_indices_) {
        result.push_back(node_outputs_[idx]);
    }
    return result;
}

} // namespace neuralnet
```

- [ ] **Step 5: Update CMakeLists**

Add `src/graph_network.cpp` to `libs/neuralnet/CMakeLists.txt` source list.
Add `graph_network_test.cpp` to `libs/neuralnet/tests/CMakeLists.txt`.

- [ ] **Step 6: Build and run tests**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target neuralnet_tests && ./build/libs/neuralnet/tests/neuralnet_tests --gtest_filter='GraphNetwork*'`
Expected: All 6 GraphNetwork tests PASS.

- [ ] **Step 7: Commit**

```bash
git add libs/neuralnet/include/neuralnet/graph_network.h libs/neuralnet/src/graph_network.cpp libs/neuralnet/tests/graph_network_test.cpp libs/neuralnet/CMakeLists.txt libs/neuralnet/tests/CMakeLists.txt
git commit -m "feat(neuralnet): add GraphNetwork with construction, validation, and topological sort"
```

---

### Task 4: GraphNetwork Forward Pass — Stateless Nodes

**Files:**
- Modify: `libs/neuralnet/tests/graph_network_test.cpp`

- [ ] **Step 1: Write tests for stateless forward pass with known values**

Append to `libs/neuralnet/tests/graph_network_test.cpp`:

```cpp
TEST(GraphNetworkTest, ForwardStateless_SimpleSum) {
    // 2 inputs -> 1 output (Tanh), weights both 1.0, bias 0
    nn::GraphGenome genome;
    genome.nodes = {
        {.id = 0, .role = nn::NodeRole::Input, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 1, .role = nn::NodeRole::Input, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 2, .role = nn::NodeRole::Output, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::Tanh, .bias = 0.0f, .tau = 1.0f},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 2, .weight = 1.0f, .enabled = true, .innovation = 0},
        {.from_node = 1, .to_node = 2, .weight = 1.0f, .enabled = true, .innovation = 1},
    };

    nn::GraphNetwork net(genome);
    auto output = net.forward(std::vector<float>{0.5f, 0.5f});

    // tanh(0.5 + 0.5) = tanh(1.0) ≈ 0.7616
    ASSERT_EQ(output.size(), 1);
    EXPECT_NEAR(output[0], std::tanh(1.0f), 1e-5f);
}

TEST(GraphNetworkTest, ForwardStateless_WithHiddenNode) {
    // 1 input -> 1 hidden (ReLU) -> 1 output (Tanh)
    // input(0) --w=2.0--> hidden(2) --w=1.0--> output(1)
    nn::GraphGenome genome;
    genome.nodes = {
        {.id = 0, .role = nn::NodeRole::Input, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 1, .role = nn::NodeRole::Output, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::Tanh, .bias = 0.0f, .tau = 1.0f},
        {.id = 2, .role = nn::NodeRole::Hidden, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 2, .weight = 2.0f, .enabled = true, .innovation = 0},
        {.from_node = 2, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 1},
    };

    nn::GraphNetwork net(genome);
    auto output = net.forward(std::vector<float>{3.0f});

    // hidden = ReLU(3.0 * 2.0 + 0) = 6.0
    // output = tanh(6.0 * 1.0 + 0) = tanh(6.0) ≈ 0.99999
    ASSERT_EQ(output.size(), 1);
    EXPECT_NEAR(output[0], std::tanh(6.0f), 1e-5f);
}

TEST(GraphNetworkTest, ForwardStateless_DisabledConnectionIgnored) {
    nn::GraphGenome genome;
    genome.nodes = {
        {.id = 0, .role = nn::NodeRole::Input, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 1, .role = nn::NodeRole::Output, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 5.0f, .enabled = false, .innovation = 0},
    };

    nn::GraphNetwork net(genome);
    auto output = net.forward(std::vector<float>{10.0f});

    // Connection disabled, so output = ReLU(0 + 0) = 0
    ASSERT_EQ(output.size(), 1);
    EXPECT_FLOAT_EQ(output[0], 0.0f);
}

TEST(GraphNetworkTest, ForwardStateless_BiasWorks) {
    nn::GraphGenome genome;
    genome.nodes = {
        {.id = 0, .role = nn::NodeRole::Input, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 1, .role = nn::NodeRole::Output, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 3.0f, .tau = 1.0f},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 0},
    };

    nn::GraphNetwork net(genome);
    auto output = net.forward(std::vector<float>{2.0f});

    // ReLU(2.0 * 1.0 + 3.0) = 5.0
    ASSERT_EQ(output.size(), 1);
    EXPECT_FLOAT_EQ(output[0], 5.0f);
}

TEST(GraphNetworkTest, ForwardStateless_InputSizeMismatchThrows) {
    auto genome = make_simple_genome(); // 2 inputs
    nn::GraphNetwork net(genome);
    EXPECT_THROW(net.forward(std::vector<float>{1.0f}), std::invalid_argument);
    EXPECT_THROW(net.forward(std::vector<float>{1.0f, 2.0f, 3.0f}), std::invalid_argument);
}

TEST(GraphNetworkTest, ForwardStateless_MultipleOutputs) {
    // 1 input -> 2 outputs
    nn::GraphGenome genome;
    genome.nodes = {
        {.id = 0, .role = nn::NodeRole::Input, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 1, .role = nn::NodeRole::Output, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 2, .role = nn::NodeRole::Output, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 0},
        {.from_node = 0, .to_node = 2, .weight = 2.0f, .enabled = true, .innovation = 1},
    };

    nn::GraphNetwork net(genome);
    auto output = net.forward(std::vector<float>{3.0f});

    // Output 1 (id=1): ReLU(3*1) = 3
    // Output 2 (id=2): ReLU(3*2) = 6
    // Outputs in node-ID order
    ASSERT_EQ(output.size(), 2);
    EXPECT_FLOAT_EQ(output[0], 3.0f);
    EXPECT_FLOAT_EQ(output[1], 6.0f);
}
```

- [ ] **Step 2: Build and run tests**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target neuralnet_tests && ./build/libs/neuralnet/tests/neuralnet_tests --gtest_filter='GraphNetwork*'`
Expected: All GraphNetwork tests PASS (construction from Task 3 + forward pass tests).

- [ ] **Step 3: Commit**

```bash
git add libs/neuralnet/tests/graph_network_test.cpp
git commit -m "test(neuralnet): add stateless forward pass tests for GraphNetwork"
```

---

### Task 5: GraphNetwork CTRNN Dynamics and Recurrence

**Files:**
- Modify: `libs/neuralnet/tests/graph_network_test.cpp`

- [ ] **Step 1: Write tests for CTRNN and recurrent behavior**

Append to `libs/neuralnet/tests/graph_network_test.cpp`:

```cpp
TEST(GraphNetworkTest, CTRNN_SlowlyApproachesTarget) {
    // 1 input -> 1 CTRNN output with large tau
    nn::GraphGenome genome;
    genome.nodes = {
        {.id = 0, .role = nn::NodeRole::Input, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 1, .role = nn::NodeRole::Output, .type = nn::NodeType::CTRNN,
         .activation = nn::Activation::Tanh, .bias = 0.0f, .tau = 10.0f},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 0},
    };

    nn::GraphNetwork net(genome);

    // With tau=10, dt=1: state += (1/10) * (-state + tanh(input))
    // Feed constant input of 1.0 — should slowly approach tanh(1.0)
    float target = std::tanh(1.0f);

    auto out1 = net.forward(std::vector<float>{1.0f});
    EXPECT_GT(out1[0], 0.0f);
    EXPECT_LT(out1[0], target);  // Not there yet

    // After many ticks, should be close to target
    for (int i = 0; i < 100; ++i) {
        net.forward(std::vector<float>{1.0f});
    }
    auto out_final = net.forward(std::vector<float>{1.0f});
    EXPECT_NEAR(out_final[0], target, 0.01f);
}

TEST(GraphNetworkTest, CTRNN_FastTauActsLikeStateless) {
    // tau=1.0 with dt=1.0 means state changes ~63% per tick
    nn::GraphGenome genome;
    genome.nodes = {
        {.id = 0, .role = nn::NodeRole::Input, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 1, .role = nn::NodeRole::Output, .type = nn::NodeType::CTRNN,
         .activation = nn::Activation::Tanh, .bias = 0.0f, .tau = 1.0f},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 0},
    };

    nn::GraphNetwork net(genome);

    // After ~5 ticks with tau=1, should be very close to target
    for (int i = 0; i < 5; ++i) {
        net.forward(std::vector<float>{1.0f});
    }
    auto out = net.forward(std::vector<float>{1.0f});
    EXPECT_NEAR(out[0], std::tanh(1.0f), 0.02f);
}

TEST(GraphNetworkTest, ResetState_ClearsCTRNN) {
    nn::GraphGenome genome;
    genome.nodes = {
        {.id = 0, .role = nn::NodeRole::Input, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 1, .role = nn::NodeRole::Output, .type = nn::NodeType::CTRNN,
         .activation = nn::Activation::Tanh, .bias = 0.0f, .tau = 10.0f},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 0},
    };

    nn::GraphNetwork net(genome);
    net.forward(std::vector<float>{1.0f});
    net.forward(std::vector<float>{1.0f});
    auto before_reset = net.forward(std::vector<float>{1.0f});
    EXPECT_GT(before_reset[0], 0.0f);

    net.reset_state();
    auto after_reset = net.forward(std::vector<float>{1.0f});
    // Should be same as first tick from fresh state
    nn::GraphNetwork fresh(genome);
    auto first_tick = fresh.forward(std::vector<float>{1.0f});
    EXPECT_FLOAT_EQ(after_reset[0], first_tick[0]);
}

TEST(GraphNetworkTest, RecurrentConnection_UsesLastTick) {
    // output feeds back to itself via recurrent connection
    // input(0) -> output(1), output(1) -> output(1) [recurrent self-connection]
    nn::GraphGenome genome;
    genome.nodes = {
        {.id = 0, .role = nn::NodeRole::Input, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 1, .role = nn::NodeRole::Output, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::Tanh, .bias = 0.0f, .tau = 1.0f},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 0},
        {.from_node = 1, .to_node = 1, .weight = 0.5f, .enabled = true, .innovation = 1},
    };

    nn::GraphNetwork net(genome);

    // Tick 1: output = tanh(1.0*input + 0.5*0) = tanh(1.0) [prev output was 0]
    auto out1 = net.forward(std::vector<float>{1.0f});
    float expected1 = std::tanh(1.0f);
    EXPECT_NEAR(out1[0], expected1, 1e-5f);

    // Tick 2: output = tanh(1.0*input + 0.5*prev_output)
    auto out2 = net.forward(std::vector<float>{1.0f});
    float expected2 = std::tanh(1.0f + 0.5f * expected1);
    EXPECT_NEAR(out2[0], expected2, 1e-5f);
}

TEST(GraphNetworkTest, RecurrentCycle_HiddenToHidden) {
    // input(0) -> hidden(2) -> output(1), hidden(2) -> hidden(3) -> hidden(2) [cycle]
    nn::GraphGenome genome;
    genome.nodes = {
        {.id = 0, .role = nn::NodeRole::Input, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 1, .role = nn::NodeRole::Output, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 2, .role = nn::NodeRole::Hidden, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 3, .role = nn::NodeRole::Hidden, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 2, .weight = 1.0f, .enabled = true, .innovation = 0},
        {.from_node = 2, .to_node = 3, .weight = 1.0f, .enabled = true, .innovation = 1},
        {.from_node = 3, .to_node = 2, .weight = 0.5f, .enabled = true, .innovation = 2},
        {.from_node = 2, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 3},
    };

    nn::GraphNetwork net(genome);

    // Should not crash, should produce output
    auto out1 = net.forward(std::vector<float>{1.0f});
    ASSERT_EQ(out1.size(), 1);

    // Second tick should differ (recurrent cycle feeds back)
    auto out2 = net.forward(std::vector<float>{1.0f});
    // The cycle means hidden nodes accumulate — output should increase
    EXPECT_GE(out2[0], out1[0]);
}

TEST(GraphNetworkTest, DeadNode_ExcludedFromEvaluation) {
    // input(0) -> output(1), plus an unreachable hidden(2)
    nn::GraphGenome genome;
    genome.nodes = {
        {.id = 0, .role = nn::NodeRole::Input, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 1, .role = nn::NodeRole::Output, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 2, .role = nn::NodeRole::Hidden, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 100.0f, .tau = 1.0f},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 0},
        // hidden(2) has no incoming connections — it's dead/unreachable
    };

    nn::GraphNetwork net(genome);
    auto output = net.forward(std::vector<float>{5.0f});

    // Dead hidden node (with bias=100) should NOT affect output
    // Output = ReLU(5.0 * 1.0 + 0) = 5.0
    ASSERT_EQ(output.size(), 1);
    EXPECT_FLOAT_EQ(output[0], 5.0f);
}

TEST(GraphNetworkTest, DtOverride) {
    nn::GraphGenome genome;
    genome.nodes = {
        {.id = 0, .role = nn::NodeRole::Input, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 1, .role = nn::NodeRole::Output, .type = nn::NodeType::CTRNN,
         .activation = nn::Activation::Tanh, .bias = 0.0f, .tau = 10.0f},
    };
    genome.connections = {
        {.from_node = 0, .to_node = 1, .weight = 1.0f, .enabled = true, .innovation = 0},
    };

    nn::GraphNetwork net1(genome);
    nn::GraphNetwork net2(genome);

    // net1 uses default dt=1.0, net2 uses dt=0.1
    auto out1 = net1.forward(std::vector<float>{1.0f});
    auto out2 = net2.forward(std::vector<float>{1.0f}, 0.1f);

    // Smaller dt should produce smaller state change
    EXPECT_GT(out1[0], out2[0]);
}
```

- [ ] **Step 2: Build and run tests**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target neuralnet_tests && ./build/libs/neuralnet/tests/neuralnet_tests --gtest_filter='GraphNetwork*'`
Expected: All GraphNetwork tests PASS.

- [ ] **Step 3: Commit**

```bash
git add libs/neuralnet/tests/graph_network_test.cpp
git commit -m "test(neuralnet): add CTRNN dynamics, recurrent connection, and dt override tests"
```

---

### Task 6: InnovationCounter

**Files:**
- Create: `libs/evolve/include/evolve/innovation.h`
- Create: `libs/evolve/src/innovation.cpp`
- Create: `libs/evolve/tests/innovation_test.cpp`
- Modify: `libs/evolve/CMakeLists.txt`
- Modify: `libs/evolve/tests/CMakeLists.txt`

- [ ] **Step 1: Write failing tests**

```cpp
// libs/evolve/tests/innovation_test.cpp
#include <evolve/innovation.h>

#include <gtest/gtest.h>

namespace ev = evolve;

TEST(InnovationCounterTest, AssignsSequentialNumbers) {
    ev::InnovationCounter counter;
    EXPECT_EQ(counter.get_or_create(0, 2), 0);
    EXPECT_EQ(counter.get_or_create(1, 2), 1);
    EXPECT_EQ(counter.get_or_create(0, 3), 2);
}

TEST(InnovationCounterTest, SamePairSameGeneration_ReturnsSameNumber) {
    ev::InnovationCounter counter;
    auto first = counter.get_or_create(0, 2);
    auto second = counter.get_or_create(0, 2);
    EXPECT_EQ(first, second);
}

TEST(InnovationCounterTest, NewGeneration_ResetsTracking) {
    ev::InnovationCounter counter;
    auto gen1 = counter.get_or_create(0, 2);
    EXPECT_EQ(gen1, 0);

    counter.new_generation();

    // Same pair in new generation gets a NEW number
    auto gen2 = counter.get_or_create(0, 2);
    EXPECT_EQ(gen2, 1);
}

TEST(InnovationCounterTest, CounterPersistsAcrossGenerations) {
    ev::InnovationCounter counter;
    counter.get_or_create(0, 1);  // 0
    counter.get_or_create(0, 2);  // 1
    counter.new_generation();
    auto next = counter.get_or_create(5, 6);  // 2
    EXPECT_EQ(next, 2);
}
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target evolve_tests 2>&1 | head -10`
Expected: Compilation fails — `evolve/innovation.h` not found.

- [ ] **Step 3: Implement InnovationCounter**

```cpp
// libs/evolve/include/evolve/innovation.h
#pragma once

#include <cstdint>
#include <map>
#include <utility>

namespace evolve {

class InnovationCounter {
public:
    /// Get or assign an innovation number for a connection between two nodes.
    /// If this (from, to) pair was already created this generation, returns the same number.
    uint32_t get_or_create(uint32_t from_node, uint32_t to_node);

    /// Call at the start of each generation to clear the within-generation dedup map.
    void new_generation();

    /// Current next innovation number (useful for testing/debugging).
    [[nodiscard]] uint32_t next_innovation() const noexcept { return next_innovation_; }

private:
    uint32_t next_innovation_ = 0;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> current_generation_;
};

} // namespace evolve
```

```cpp
// libs/evolve/src/innovation.cpp
#include <evolve/innovation.h>

namespace evolve {

uint32_t InnovationCounter::get_or_create(uint32_t from_node, uint32_t to_node) {
    auto key = std::make_pair(from_node, to_node);
    auto it = current_generation_.find(key);
    if (it != current_generation_.end()) {
        return it->second;
    }
    auto num = next_innovation_++;
    current_generation_[key] = num;
    return num;
}

void InnovationCounter::new_generation() {
    current_generation_.clear();
}

} // namespace evolve
```

- [ ] **Step 4: Update CMakeLists**

Add `src/innovation.cpp` to `libs/evolve/CMakeLists.txt` source list.
Add `neuralnet` to evolve's dependencies: `target_link_libraries(evolve PRIVATE project_warnings PUBLIC neuralnet)`
Add `innovation_test.cpp` to `libs/evolve/tests/CMakeLists.txt`.

- [ ] **Step 5: Build and run tests**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target evolve_tests && ./build/libs/evolve/tests/evolve_tests --gtest_filter='InnovationCounter*'`
Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add libs/evolve/include/evolve/innovation.h libs/evolve/src/innovation.cpp libs/evolve/tests/innovation_test.cpp libs/evolve/CMakeLists.txt libs/evolve/tests/CMakeLists.txt
git commit -m "feat(evolve): add InnovationCounter for NEAT structural mutation tracking"
```

---

### Task 7: NEAT Parameter Mutations (Weight, Bias, Tau)

**Files:**
- Create: `libs/evolve/include/evolve/neat_operators.h`
- Create: `libs/evolve/src/neat_operators.cpp`
- Create: `libs/evolve/tests/neat_operators_test.cpp`
- Modify: `libs/evolve/CMakeLists.txt`
- Modify: `libs/evolve/tests/CMakeLists.txt`

- [ ] **Step 1: Write failing tests for parameter mutations**

```cpp
// libs/evolve/tests/neat_operators_test.cpp
#include <evolve/neat_operators.h>
#include <neuralnet/graph_genome.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

namespace ev = evolve;
namespace nn = neuralnet;

// Helper: make a genome with 2 inputs, 1 CTRNN hidden, 1 output, 3 connections
nn::GraphGenome make_test_genome() {
    nn::GraphGenome g;
    g.nodes = {
        {.id = 0, .role = nn::NodeRole::Input, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 1, .role = nn::NodeRole::Input, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::ReLU, .bias = 0.0f, .tau = 1.0f},
        {.id = 2, .role = nn::NodeRole::Hidden, .type = nn::NodeType::CTRNN,
         .activation = nn::Activation::Tanh, .bias = 0.5f, .tau = 5.0f},
        {.id = 3, .role = nn::NodeRole::Output, .type = nn::NodeType::Stateless,
         .activation = nn::Activation::Tanh, .bias = 0.1f, .tau = 1.0f},
    };
    g.connections = {
        {.from_node = 0, .to_node = 2, .weight = 1.0f, .enabled = true, .innovation = 0},
        {.from_node = 1, .to_node = 2, .weight = -0.5f, .enabled = true, .innovation = 1},
        {.from_node = 2, .to_node = 3, .weight = 0.8f, .enabled = true, .innovation = 2},
    };
    return g;
}

TEST(NeatMutationTest, MutateWeights_AllPerturbed) {
    auto genome = make_test_genome();
    auto original_weights = genome.connections;

    ev::NeatMutationConfig config;
    config.weight_mutate_rate = 1.0f;      // Mutate all
    config.weight_perturb_rate = 1.0f;     // All perturbations (no replacements)
    config.weight_perturb_strength = 0.5f;

    std::mt19937 rng(42);
    ev::mutate_weights(genome, config, rng);

    bool any_changed = false;
    for (std::size_t i = 0; i < genome.connections.size(); ++i) {
        if (genome.connections[i].weight != original_weights[i].weight) {
            any_changed = true;
        }
    }
    EXPECT_TRUE(any_changed);
}

TEST(NeatMutationTest, MutateWeights_ZeroRateChangesNothing) {
    auto genome = make_test_genome();
    auto original = genome.connections;

    ev::NeatMutationConfig config;
    config.weight_mutate_rate = 0.0f;

    std::mt19937 rng(42);
    ev::mutate_weights(genome, config, rng);

    for (std::size_t i = 0; i < genome.connections.size(); ++i) {
        EXPECT_FLOAT_EQ(genome.connections[i].weight, original[i].weight);
    }
}

TEST(NeatMutationTest, MutateBiases_ChangesNonInputNodes) {
    auto genome = make_test_genome();

    ev::NeatMutationConfig config;
    config.bias_mutate_rate = 1.0f;
    config.bias_perturb_strength = 0.5f;

    std::mt19937 rng(42);
    ev::mutate_biases(genome, config, rng);

    // Input node biases should be unchanged (they're ignored anyway)
    EXPECT_FLOAT_EQ(genome.nodes[0].bias, 0.0f);
    EXPECT_FLOAT_EQ(genome.nodes[1].bias, 0.0f);
}

TEST(NeatMutationTest, MutateTau_OnlyCTRNNNodes) {
    auto genome = make_test_genome();
    float original_ctrnn_tau = genome.nodes[2].tau;  // CTRNN hidden
    float original_stateless_tau = genome.nodes[3].tau;  // Stateless output

    ev::NeatMutationConfig config;
    config.tau_mutate_rate = 1.0f;
    config.tau_perturb_strength = 1.0f;
    config.tau_min = 0.1f;
    config.tau_max = 100.0f;

    std::mt19937 rng(42);
    ev::mutate_tau(genome, config, rng);

    // CTRNN node tau should have changed
    EXPECT_NE(genome.nodes[2].tau, original_ctrnn_tau);
    // Stateless node tau should be unchanged
    EXPECT_FLOAT_EQ(genome.nodes[3].tau, original_stateless_tau);
    // Tau should be clamped
    EXPECT_GE(genome.nodes[2].tau, 0.1f);
    EXPECT_LE(genome.nodes[2].tau, 100.0f);
}
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target evolve_tests 2>&1 | head -10`
Expected: Compilation fails — `evolve/neat_operators.h` not found.

- [ ] **Step 3: Create neat_operators.h with config + parameter mutation declarations**

```cpp
// libs/evolve/include/evolve/neat_operators.h
#pragma once

#include <evolve/innovation.h>
#include <neuralnet/graph_genome.h>

#include <random>

namespace evolve {

struct NeatMutationConfig {
    float weight_mutate_rate      = 0.80f;
    float weight_perturb_rate     = 0.90f;
    float weight_perturb_strength = 0.3f;
    float weight_replace_range    = 2.0f;

    float bias_mutate_rate        = 0.40f;
    float bias_perturb_strength   = 0.2f;

    float tau_mutate_rate         = 0.20f;
    float tau_perturb_strength    = 0.1f;
    float tau_min                 = 0.1f;
    float tau_max                 = 100.0f;

    float add_connection_rate     = 0.10f;
    float add_node_rate           = 0.03f;
    float disable_connection_rate = 0.02f;

    float node_type_mutate_rate   = 0.05f;
    float activation_mutate_rate  = 0.05f;
};

struct SpeciationConfig {
    float compatibility_threshold = 3.0f;
    float c_excess    = 1.0f;
    float c_disjoint  = 1.0f;
    float c_weight    = 0.4f;
};

// Parameter mutations
void mutate_weights(neuralnet::GraphGenome& genome, const NeatMutationConfig& config, std::mt19937& rng);
void mutate_biases(neuralnet::GraphGenome& genome, const NeatMutationConfig& config, std::mt19937& rng);
void mutate_tau(neuralnet::GraphGenome& genome, const NeatMutationConfig& config, std::mt19937& rng);

// Structural mutations (Task 8)
void add_connection(neuralnet::GraphGenome& genome, InnovationCounter& counter,
                    const NeatMutationConfig& config, std::mt19937& rng);
void add_node(neuralnet::GraphGenome& genome, InnovationCounter& counter, std::mt19937& rng);
void disable_connection(neuralnet::GraphGenome& genome, std::mt19937& rng);

// Node property mutations (Task 9)
void mutate_node_types(neuralnet::GraphGenome& genome, const NeatMutationConfig& config, std::mt19937& rng);
void mutate_activations(neuralnet::GraphGenome& genome, const NeatMutationConfig& config, std::mt19937& rng);

// Full mutation pass — applies all mutations according to config rates
void mutate(neuralnet::GraphGenome& genome, InnovationCounter& counter,
            const NeatMutationConfig& config, std::mt19937& rng);

// Crossover (Task 10)
[[nodiscard]] neuralnet::GraphGenome crossover(
    const neuralnet::GraphGenome& fitter_parent,
    const neuralnet::GraphGenome& other_parent,
    std::mt19937& rng);

// Speciation (Task 11)
[[nodiscard]] float compatibility_distance(
    const neuralnet::GraphGenome& a,
    const neuralnet::GraphGenome& b,
    const SpeciationConfig& config);

} // namespace evolve
```

- [ ] **Step 4: Implement parameter mutations in neat_operators.cpp**

```cpp
// libs/evolve/src/neat_operators.cpp
#include <evolve/neat_operators.h>

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace evolve {

void mutate_weights(neuralnet::GraphGenome& genome, const NeatMutationConfig& config, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::normal_distribution<float> perturb(0.0f, config.weight_perturb_strength);
    std::uniform_real_distribution<float> replace(-config.weight_replace_range, config.weight_replace_range);

    for (auto& conn : genome.connections) {
        if (prob(rng) < config.weight_mutate_rate) {
            if (prob(rng) < config.weight_perturb_rate) {
                conn.weight += perturb(rng);
            } else {
                conn.weight = replace(rng);
            }
        }
    }
}

void mutate_biases(neuralnet::GraphGenome& genome, const NeatMutationConfig& config, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::normal_distribution<float> perturb(0.0f, config.bias_perturb_strength);

    for (auto& node : genome.nodes) {
        if (node.role == neuralnet::NodeRole::Input) continue;
        if (prob(rng) < config.bias_mutate_rate) {
            node.bias += perturb(rng);
        }
    }
}

void mutate_tau(neuralnet::GraphGenome& genome, const NeatMutationConfig& config, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::normal_distribution<float> perturb(0.0f, config.tau_perturb_strength);

    for (auto& node : genome.nodes) {
        if (node.type != neuralnet::NodeType::CTRNN) continue;
        if (prob(rng) < config.tau_mutate_rate) {
            node.tau += perturb(rng);
            node.tau = std::clamp(node.tau, config.tau_min, config.tau_max);
        }
    }
}

// Stubs for later tasks (so the library compiles)
void add_connection(neuralnet::GraphGenome& /*genome*/, InnovationCounter& /*counter*/,
                    const NeatMutationConfig& /*config*/, std::mt19937& /*rng*/) {}
void add_node(neuralnet::GraphGenome& /*genome*/, InnovationCounter& /*counter*/, std::mt19937& /*rng*/) {}
void disable_connection(neuralnet::GraphGenome& /*genome*/, std::mt19937& /*rng*/) {}
void mutate_node_types(neuralnet::GraphGenome& /*genome*/, const NeatMutationConfig& /*config*/, std::mt19937& /*rng*/) {}
void mutate_activations(neuralnet::GraphGenome& /*genome*/, const NeatMutationConfig& /*config*/, std::mt19937& /*rng*/) {}
void mutate(neuralnet::GraphGenome& /*genome*/, InnovationCounter& /*counter*/,
            const NeatMutationConfig& /*config*/, std::mt19937& /*rng*/) {}
neuralnet::GraphGenome crossover(const neuralnet::GraphGenome& fitter_parent,
                                  const neuralnet::GraphGenome& /*other_parent*/,
                                  std::mt19937& /*rng*/) { return fitter_parent; }
float compatibility_distance(const neuralnet::GraphGenome& /*a*/,
                             const neuralnet::GraphGenome& /*b*/,
                             const SpeciationConfig& /*config*/) { return 0.0f; }

} // namespace evolve
```

- [ ] **Step 5: Update CMakeLists**

Add `src/neat_operators.cpp` to `libs/evolve/CMakeLists.txt` source list.
Add `neat_operators_test.cpp` to `libs/evolve/tests/CMakeLists.txt`.

- [ ] **Step 6: Build and run tests**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target evolve_tests && ./build/libs/evolve/tests/evolve_tests --gtest_filter='NeatMutation*'`
Expected: All 4 parameter mutation tests PASS.

- [ ] **Step 7: Commit**

```bash
git add libs/evolve/include/evolve/neat_operators.h libs/evolve/src/neat_operators.cpp libs/evolve/tests/neat_operators_test.cpp libs/evolve/CMakeLists.txt libs/evolve/tests/CMakeLists.txt
git commit -m "feat(evolve): add NEAT parameter mutations (weight, bias, tau)"
```

---

### Task 8: NEAT Structural Mutations (Add Connection, Add Node, Disable)

**Files:**
- Modify: `libs/evolve/src/neat_operators.cpp`
- Modify: `libs/evolve/tests/neat_operators_test.cpp`

- [ ] **Step 1: Write failing tests for structural mutations**

Append to `libs/evolve/tests/neat_operators_test.cpp`:

```cpp
TEST(NeatMutationTest, AddConnection_IncreasesConnectionCount) {
    auto genome = make_test_genome();  // 3 connections
    ev::InnovationCounter counter;
    // Pre-fill counter to match existing innovations
    counter.get_or_create(0, 2);  // 0
    counter.get_or_create(1, 2);  // 1
    counter.get_or_create(2, 3);  // 2

    ev::NeatMutationConfig config;
    std::mt19937 rng(42);

    auto original_count = genome.connections.size();
    ev::add_connection(genome, counter, config, rng);

    EXPECT_EQ(genome.connections.size(), original_count + 1);
    // New connection should be enabled
    EXPECT_TRUE(genome.connections.back().enabled);
    // New connection should not target an input node
    auto& new_conn = genome.connections.back();
    for (const auto& node : genome.nodes) {
        if (node.id == new_conn.to_node) {
            EXPECT_NE(node.role, nn::NodeRole::Input);
        }
    }
}

TEST(NeatMutationTest, AddNode_SplitsConnection) {
    auto genome = make_test_genome();
    ev::InnovationCounter counter;
    // Pre-fill counter
    counter.get_or_create(0, 2);
    counter.get_or_create(1, 2);
    counter.get_or_create(2, 3);

    auto original_node_count = genome.nodes.size();
    auto original_conn_count = genome.connections.size();

    std::mt19937 rng(42);
    ev::add_node(genome, counter, rng);

    // Should add 1 node and 2 connections
    EXPECT_EQ(genome.nodes.size(), original_node_count + 1);
    EXPECT_EQ(genome.connections.size(), original_conn_count + 2);

    // New node should be Hidden, Stateless, bias=0
    auto& new_node = genome.nodes.back();
    EXPECT_EQ(new_node.role, nn::NodeRole::Hidden);
    EXPECT_EQ(new_node.type, nn::NodeType::Stateless);
    EXPECT_FLOAT_EQ(new_node.bias, 0.0f);
    EXPECT_FLOAT_EQ(new_node.tau, 1.0f);

    // One of the original connections should now be disabled
    int disabled_count = 0;
    for (const auto& conn : genome.connections) {
        if (!conn.enabled) disabled_count++;
    }
    EXPECT_GE(disabled_count, 1);
}

TEST(NeatMutationTest, DisableConnection_DisablesOne) {
    auto genome = make_test_genome();
    std::mt19937 rng(42);

    int enabled_before = 0;
    for (const auto& c : genome.connections) {
        if (c.enabled) enabled_before++;
    }

    ev::disable_connection(genome, rng);

    int enabled_after = 0;
    for (const auto& c : genome.connections) {
        if (c.enabled) enabled_after++;
    }

    EXPECT_EQ(enabled_after, enabled_before - 1);
}
```

- [ ] **Step 2: Run to verify failures (stubs return without doing anything)**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target evolve_tests && ./build/libs/evolve/tests/evolve_tests --gtest_filter='NeatMutation*Add*:NeatMutation*Disable*'`
Expected: Tests FAIL because stubs don't actually mutate.

- [ ] **Step 3: Implement structural mutations**

Replace the stubs in `libs/evolve/src/neat_operators.cpp`:

```cpp
void add_connection(neuralnet::GraphGenome& genome, InnovationCounter& counter,
                    const NeatMutationConfig& /*config*/, std::mt19937& rng) {
    // Build set of existing connections for dedup
    std::unordered_set<uint64_t> existing;
    for (const auto& conn : genome.connections) {
        uint64_t key = (static_cast<uint64_t>(conn.from_node) << 32) | conn.to_node;
        existing.insert(key);
    }

    // Collect valid (from, to) candidates
    std::vector<std::pair<uint32_t, uint32_t>> candidates;
    for (const auto& from : genome.nodes) {
        for (const auto& to : genome.nodes) {
            if (to.role == neuralnet::NodeRole::Input) continue;  // Can't target input
            uint64_t key = (static_cast<uint64_t>(from.id) << 32) | to.id;
            if (existing.count(key) > 0) continue;  // Already exists
            candidates.emplace_back(from.id, to.id);
        }
    }

    if (candidates.empty()) return;  // Fully connected already

    std::uniform_int_distribution<std::size_t> pick(0, candidates.size() - 1);
    auto [from_id, to_id] = candidates[pick(rng)];

    std::uniform_real_distribution<float> weight_dist(-1.0f, 1.0f);
    genome.connections.push_back(neuralnet::ConnectionGene{
        .from_node = from_id,
        .to_node = to_id,
        .weight = weight_dist(rng),
        .enabled = true,
        .innovation = counter.get_or_create(from_id, to_id),
    });
}

void add_node(neuralnet::GraphGenome& genome, InnovationCounter& counter, std::mt19937& rng) {
    // Find enabled connections
    std::vector<std::size_t> enabled_indices;
    for (std::size_t i = 0; i < genome.connections.size(); ++i) {
        if (genome.connections[i].enabled) {
            enabled_indices.push_back(i);
        }
    }
    if (enabled_indices.empty()) return;

    std::uniform_int_distribution<std::size_t> pick(0, enabled_indices.size() - 1);
    auto conn_idx = enabled_indices[pick(rng)];
    auto& conn = genome.connections[conn_idx];

    // Disable the original connection
    conn.enabled = false;

    // Find the next available node ID
    uint32_t max_id = 0;
    for (const auto& node : genome.nodes) {
        max_id = std::max(max_id, node.id);
    }
    uint32_t new_id = max_id + 1;

    // Pick a random activation for the new node
    static constexpr neuralnet::Activation activations[] = {
        neuralnet::Activation::ReLU,
        neuralnet::Activation::Sigmoid,
        neuralnet::Activation::Tanh,
    };
    std::uniform_int_distribution<int> act_pick(0, 2);

    genome.nodes.push_back(neuralnet::NodeGene{
        .id = new_id,
        .role = neuralnet::NodeRole::Hidden,
        .type = neuralnet::NodeType::Stateless,
        .activation = activations[act_pick(rng)],
        .bias = 0.0f,
        .tau = 1.0f,
    });

    // New connections: from -> new_node (weight 1.0), new_node -> to (old weight)
    genome.connections.push_back(neuralnet::ConnectionGene{
        .from_node = conn.from_node,
        .to_node = new_id,
        .weight = 1.0f,
        .enabled = true,
        .innovation = counter.get_or_create(conn.from_node, new_id),
    });
    genome.connections.push_back(neuralnet::ConnectionGene{
        .from_node = new_id,
        .to_node = conn.to_node,
        .weight = conn.weight,
        .enabled = true,
        .innovation = counter.get_or_create(new_id, conn.to_node),
    });
}

void disable_connection(neuralnet::GraphGenome& genome, std::mt19937& rng) {
    std::vector<std::size_t> enabled_indices;
    for (std::size_t i = 0; i < genome.connections.size(); ++i) {
        if (genome.connections[i].enabled) {
            enabled_indices.push_back(i);
        }
    }
    if (enabled_indices.empty()) return;

    std::uniform_int_distribution<std::size_t> pick(0, enabled_indices.size() - 1);
    genome.connections[enabled_indices[pick(rng)]].enabled = false;
}
```

- [ ] **Step 4: Build and run tests**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target evolve_tests && ./build/libs/evolve/tests/evolve_tests --gtest_filter='NeatMutation*'`
Expected: All NeatMutation tests PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/evolve/src/neat_operators.cpp libs/evolve/tests/neat_operators_test.cpp
git commit -m "feat(evolve): add NEAT structural mutations (add connection, add node, disable)"
```

---

### Task 9: NEAT Node Type and Activation Mutations + Full Mutate

**Files:**
- Modify: `libs/evolve/src/neat_operators.cpp`
- Modify: `libs/evolve/tests/neat_operators_test.cpp`

- [ ] **Step 1: Write failing tests**

Append to `libs/evolve/tests/neat_operators_test.cpp`:

```cpp
TEST(NeatMutationTest, MutateNodeType_TogglesCTRNN) {
    auto genome = make_test_genome();

    ev::NeatMutationConfig config;
    config.node_type_mutate_rate = 1.0f;  // Mutate all
    config.tau_min = 0.1f;
    config.tau_max = 100.0f;

    std::mt19937 rng(42);
    ev::mutate_node_types(genome, config, rng);

    // Hidden CTRNN node (index 2) should have toggled to Stateless
    EXPECT_EQ(genome.nodes[2].type, nn::NodeType::Stateless);
    // Output Stateless node (index 3) should have toggled to CTRNN
    EXPECT_EQ(genome.nodes[3].type, nn::NodeType::CTRNN);
    EXPECT_GE(genome.nodes[3].tau, config.tau_min);
    EXPECT_LE(genome.nodes[3].tau, config.tau_max);
    // Input nodes should be unchanged
    EXPECT_EQ(genome.nodes[0].type, nn::NodeType::Stateless);
    EXPECT_EQ(genome.nodes[1].type, nn::NodeType::Stateless);
}

TEST(NeatMutationTest, MutateActivation_ChangesHiddenAndOutput) {
    // Make a genome where all non-input nodes have Tanh
    auto genome = make_test_genome();

    ev::NeatMutationConfig config;
    config.activation_mutate_rate = 1.0f;  // Mutate all

    // Run many times to ensure at least one change happens
    std::mt19937 rng(42);
    bool any_changed = false;
    for (int i = 0; i < 20; ++i) {
        auto g = make_test_genome();
        ev::mutate_activations(g, config, rng);
        if (g.nodes[2].activation != nn::Activation::Tanh ||
            g.nodes[3].activation != nn::Activation::Tanh) {
            any_changed = true;
            break;
        }
    }
    EXPECT_TRUE(any_changed);
}

TEST(NeatMutationTest, FullMutate_DoesNotCrash) {
    auto genome = make_test_genome();
    ev::InnovationCounter counter;
    counter.get_or_create(0, 2);
    counter.get_or_create(1, 2);
    counter.get_or_create(2, 3);

    ev::NeatMutationConfig config;
    std::mt19937 rng(42);

    // Run full mutation many times — should not crash
    for (int i = 0; i < 100; ++i) {
        ev::mutate(genome, counter, config, rng);
    }

    // Genome should still be valid (buildable into a GraphNetwork)
    EXPECT_NO_THROW(nn::GraphNetwork(genome));
}
```

- [ ] **Step 2: Implement node type/activation mutations and full mutate**

Replace stubs in `libs/evolve/src/neat_operators.cpp`:

```cpp
void mutate_node_types(neuralnet::GraphGenome& genome, const NeatMutationConfig& config, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_real_distribution<float> tau_dist(config.tau_min, config.tau_max);

    for (auto& node : genome.nodes) {
        if (node.role == neuralnet::NodeRole::Input) continue;
        if (prob(rng) < config.node_type_mutate_rate) {
            if (node.type == neuralnet::NodeType::Stateless) {
                node.type = neuralnet::NodeType::CTRNN;
                node.tau = tau_dist(rng);
            } else {
                node.type = neuralnet::NodeType::Stateless;
            }
        }
    }
}

void mutate_activations(neuralnet::GraphGenome& genome, const NeatMutationConfig& config, std::mt19937& rng) {
    static constexpr neuralnet::Activation activations[] = {
        neuralnet::Activation::ReLU,
        neuralnet::Activation::Sigmoid,
        neuralnet::Activation::Tanh,
    };
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_int_distribution<int> act_pick(0, 2);

    for (auto& node : genome.nodes) {
        if (node.role == neuralnet::NodeRole::Input) continue;
        if (prob(rng) < config.activation_mutate_rate) {
            node.activation = activations[act_pick(rng)];
        }
    }
}

void mutate(neuralnet::GraphGenome& genome, InnovationCounter& counter,
            const NeatMutationConfig& config, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);

    mutate_weights(genome, config, rng);
    mutate_biases(genome, config, rng);
    mutate_tau(genome, config, rng);

    if (prob(rng) < config.add_connection_rate) {
        add_connection(genome, counter, config, rng);
    }
    if (prob(rng) < config.add_node_rate) {
        add_node(genome, counter, rng);
    }
    if (prob(rng) < config.disable_connection_rate) {
        disable_connection(genome, rng);
    }

    mutate_node_types(genome, config, rng);
    mutate_activations(genome, config, rng);
}
```

- [ ] **Step 3: Build and run tests**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target evolve_tests && ./build/libs/evolve/tests/evolve_tests --gtest_filter='NeatMutation*'`
Expected: All NeatMutation tests PASS.

- [ ] **Step 4: Commit**

```bash
git add libs/evolve/src/neat_operators.cpp libs/evolve/tests/neat_operators_test.cpp
git commit -m "feat(evolve): add node type/activation mutations and full mutate() pass"
```

---

### Task 10: NEAT Crossover

**Files:**
- Modify: `libs/evolve/src/neat_operators.cpp`
- Modify: `libs/evolve/tests/neat_operators_test.cpp`

- [ ] **Step 1: Write failing tests**

Append to `libs/evolve/tests/neat_operators_test.cpp`:

```cpp
TEST(NeatCrossoverTest, MatchingGenes_InheritedFromEitherParent) {
    // Two parents with same topology but different weights
    auto parent_a = make_test_genome();  // innovations 0, 1, 2
    auto parent_b = make_test_genome();
    for (auto& conn : parent_b.connections) {
        conn.weight = conn.weight + 10.0f;  // Make weights clearly different
    }

    std::mt19937 rng(42);
    auto child = ev::crossover(parent_a, parent_b, rng);

    // Same number of connections (all matching)
    EXPECT_EQ(child.connections.size(), 3);

    // Each child weight should come from one parent
    for (std::size_t i = 0; i < child.connections.size(); ++i) {
        float w = child.connections[i].weight;
        EXPECT_TRUE(
            w == parent_a.connections[i].weight ||
            w == parent_b.connections[i].weight
        );
    }
}

TEST(NeatCrossoverTest, DisjointGenes_InheritedFromFitterParent) {
    auto fitter = make_test_genome();   // innovations 0, 1, 2
    auto other = make_test_genome();

    // Add an extra connection only in the fitter parent
    fitter.connections.push_back(nn::ConnectionGene{
        .from_node = 0, .to_node = 3, .weight = 99.0f,
        .enabled = true, .innovation = 10});

    std::mt19937 rng(42);
    auto child = ev::crossover(fitter, other, rng);

    // Child should have 4 connections (3 matching + 1 disjoint from fitter)
    EXPECT_EQ(child.connections.size(), 4);

    // The disjoint gene (innovation 10) should be present
    bool found = false;
    for (const auto& conn : child.connections) {
        if (conn.innovation == 10) {
            found = true;
            EXPECT_FLOAT_EQ(conn.weight, 99.0f);
        }
    }
    EXPECT_TRUE(found);
}

TEST(NeatCrossoverTest, DisabledGene_75PercentChanceStaysDisabled) {
    // Make both parents have a disabled gene at innovation 0
    auto fitter = make_test_genome();
    auto other = make_test_genome();
    fitter.connections[0].enabled = false;
    other.connections[0].enabled = true;  // Disabled in one parent

    int disabled_count = 0;
    int trials = 1000;
    for (int i = 0; i < trials; ++i) {
        std::mt19937 rng(i);
        auto child = ev::crossover(fitter, other, rng);
        if (!child.connections[0].enabled) {
            disabled_count++;
        }
    }

    // Should be approximately 75% disabled (allow 65-85% range)
    float ratio = static_cast<float>(disabled_count) / static_cast<float>(trials);
    EXPECT_NEAR(ratio, 0.75f, 0.10f);
}

TEST(NeatCrossoverTest, NodeProperties_InheritedRandomly) {
    auto fitter = make_test_genome();
    auto other = make_test_genome();

    // Give parents different biases on the hidden node
    fitter.nodes[2].bias = 1.0f;
    other.nodes[2].bias = -1.0f;

    bool got_fitter = false;
    bool got_other = false;
    for (int i = 0; i < 100; ++i) {
        std::mt19937 rng(i);
        auto child = ev::crossover(fitter, other, rng);
        // Find hidden node (id=2)
        for (const auto& node : child.nodes) {
            if (node.id == 2) {
                if (node.bias == 1.0f) got_fitter = true;
                if (node.bias == -1.0f) got_other = true;
            }
        }
    }
    EXPECT_TRUE(got_fitter);
    EXPECT_TRUE(got_other);
}
```

- [ ] **Step 2: Implement crossover**

Replace the crossover stub in `libs/evolve/src/neat_operators.cpp`:

```cpp
neuralnet::GraphGenome crossover(const neuralnet::GraphGenome& fitter_parent,
                                  const neuralnet::GraphGenome& other_parent,
                                  std::mt19937& rng) {
    using namespace neuralnet;
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);

    // Build maps: innovation -> connection for each parent
    std::map<uint32_t, const ConnectionGene*> fitter_conns, other_conns;
    for (const auto& c : fitter_parent.connections) fitter_conns[c.innovation] = &c;
    for (const auto& c : other_parent.connections) other_conns[c.innovation] = &c;

    // Build node maps: id -> NodeGene for each parent
    std::map<uint32_t, const NodeGene*> fitter_nodes, other_nodes;
    for (const auto& n : fitter_parent.nodes) fitter_nodes[n.id] = &n;
    for (const auto& n : other_parent.nodes) other_nodes[n.id] = &n;

    GraphGenome child;
    std::unordered_set<uint32_t> child_node_ids;

    // Collect all innovation numbers
    std::set<uint32_t> all_innovations;
    for (const auto& [innov, _] : fitter_conns) all_innovations.insert(innov);
    for (const auto& [innov, _] : other_conns) all_innovations.insert(innov);

    for (auto innov : all_innovations) {
        auto fit_it = fitter_conns.find(innov);
        auto oth_it = other_conns.find(innov);

        if (fit_it != fitter_conns.end() && oth_it != other_conns.end()) {
            // Matching gene — random inheritance
            const auto* chosen = (prob(rng) < 0.5f) ? fit_it->second : oth_it->second;
            auto conn = *chosen;

            // Disabled gene inheritance: if disabled in either parent, 75% stays disabled
            if (!fit_it->second->enabled || !oth_it->second->enabled) {
                conn.enabled = (prob(rng) >= 0.75f);
            }

            child.connections.push_back(conn);
            child_node_ids.insert(conn.from_node);
            child_node_ids.insert(conn.to_node);
        } else if (fit_it != fitter_conns.end()) {
            // Disjoint/excess from fitter parent — inherit
            child.connections.push_back(*fit_it->second);
            child_node_ids.insert(fit_it->second->from_node);
            child_node_ids.insert(fit_it->second->to_node);
        }
        // Disjoint/excess from other parent — skip
    }

    // Inherit nodes: for nodes in both parents, randomly pick properties
    for (auto id : child_node_ids) {
        auto fit_it = fitter_nodes.find(id);
        auto oth_it = other_nodes.find(id);

        if (fit_it != fitter_nodes.end() && oth_it != other_nodes.end()) {
            // Both parents have this node — random property inheritance
            child.nodes.push_back((prob(rng) < 0.5f) ? *fit_it->second : *oth_it->second);
        } else if (fit_it != fitter_nodes.end()) {
            child.nodes.push_back(*fit_it->second);
        } else if (oth_it != other_nodes.end()) {
            child.nodes.push_back(*oth_it->second);
        }
    }

    // Ensure all input and output nodes from fitter parent are present
    for (const auto& node : fitter_parent.nodes) {
        if ((node.role == NodeRole::Input || node.role == NodeRole::Output)
            && child_node_ids.find(node.id) == child_node_ids.end()) {
            child.nodes.push_back(node);
            child_node_ids.insert(node.id);
        }
    }

    // Sort nodes by ID for consistent ordering
    std::sort(child.nodes.begin(), child.nodes.end(),
              [](const NodeGene& a, const NodeGene& b) { return a.id < b.id; });

    // Orphan cleanup: remove hidden nodes with no enabled connections
    std::unordered_set<uint32_t> connected_nodes;
    for (const auto& conn : child.connections) {
        if (conn.enabled) {
            connected_nodes.insert(conn.from_node);
            connected_nodes.insert(conn.to_node);
        }
    }

    // Remove orphaned hidden nodes
    std::unordered_set<uint32_t> removed_ids;
    child.nodes.erase(
        std::remove_if(child.nodes.begin(), child.nodes.end(),
            [&](const NodeGene& node) {
                if (node.role != NodeRole::Hidden) return false;
                if (connected_nodes.count(node.id) == 0) {
                    removed_ids.insert(node.id);
                    return true;
                }
                return false;
            }),
        child.nodes.end());

    // Remove disabled connections referencing removed nodes
    if (!removed_ids.empty()) {
        child.connections.erase(
            std::remove_if(child.connections.begin(), child.connections.end(),
                [&](const ConnectionGene& conn) {
                    return removed_ids.count(conn.from_node) > 0
                        || removed_ids.count(conn.to_node) > 0;
                }),
            child.connections.end());
    }

    return child;
}
```

Note: you'll need to add `#include <set>` and `#include <map>` to the includes in neat_operators.cpp.

- [ ] **Step 3: Build and run tests**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target evolve_tests && ./build/libs/evolve/tests/evolve_tests --gtest_filter='NeatCrossover*'`
Expected: All 4 crossover tests PASS.

- [ ] **Step 4: Commit**

```bash
git add libs/evolve/src/neat_operators.cpp libs/evolve/tests/neat_operators_test.cpp
git commit -m "feat(evolve): implement NEAT crossover with innovation alignment and orphan cleanup"
```

---

### Task 11: Speciation (Compatibility Distance)

**Files:**
- Modify: `libs/evolve/src/neat_operators.cpp`
- Modify: `libs/evolve/tests/neat_operators_test.cpp`

- [ ] **Step 1: Write failing tests**

Append to `libs/evolve/tests/neat_operators_test.cpp`:

```cpp
TEST(SpeciationTest, IdenticalGenomes_ZeroDistance) {
    auto a = make_test_genome();
    ev::SpeciationConfig config;
    EXPECT_FLOAT_EQ(ev::compatibility_distance(a, a, config), 0.0f);
}

TEST(SpeciationTest, DifferentWeights_NonZeroDistance) {
    auto a = make_test_genome();
    auto b = make_test_genome();
    b.connections[0].weight += 2.0f;

    ev::SpeciationConfig config;
    auto dist = ev::compatibility_distance(a, b, config);
    EXPECT_GT(dist, 0.0f);
}

TEST(SpeciationTest, ExtraConnections_IncreasesDistance) {
    auto a = make_test_genome();
    auto b = make_test_genome();

    // Add excess connections to b
    b.connections.push_back({.from_node = 0, .to_node = 3, .weight = 1.0f,
        .enabled = true, .innovation = 10});
    b.connections.push_back({.from_node = 1, .to_node = 3, .weight = 1.0f,
        .enabled = true, .innovation = 11});

    ev::SpeciationConfig config;
    auto dist = ev::compatibility_distance(a, b, config);

    // Should be > 0 due to excess genes
    EXPECT_GT(dist, 0.0f);
}

TEST(SpeciationTest, SmallGenomes_NNotNormalized) {
    // Both genomes < 20 genes, so N=1
    auto a = make_test_genome();  // 3 connections
    auto b = make_test_genome();
    b.connections.push_back({.from_node = 0, .to_node = 3, .weight = 1.0f,
        .enabled = true, .innovation = 10});

    ev::SpeciationConfig config;
    config.c_excess = 1.0f;
    config.c_disjoint = 1.0f;
    config.c_weight = 0.0f;

    auto dist = ev::compatibility_distance(a, b, config);
    // 1 excess gene, N=1 (both < 20 genes), c_excess=1.0
    // δ = 1.0 * 1 / 1 + 0 + 0 = 1.0
    EXPECT_FLOAT_EQ(dist, 1.0f);
}
```

- [ ] **Step 2: Implement compatibility_distance**

Replace the stub in `libs/evolve/src/neat_operators.cpp`:

```cpp
float compatibility_distance(const neuralnet::GraphGenome& a,
                             const neuralnet::GraphGenome& b,
                             const SpeciationConfig& config) {
    // Build innovation -> weight maps
    std::map<uint32_t, float> a_weights, b_weights;
    for (const auto& c : a.connections) a_weights[c.innovation] = c.weight;
    for (const auto& c : b.connections) b_weights[c.innovation] = c.weight;

    if (a_weights.empty() && b_weights.empty()) return 0.0f;

    // Find max innovation in each
    uint32_t a_max = a_weights.empty() ? 0 : a_weights.rbegin()->first;
    uint32_t b_max = b_weights.empty() ? 0 : b_weights.rbegin()->first;
    uint32_t shared_max = std::min(a_max, b_max);

    uint32_t excess = 0;
    uint32_t disjoint = 0;
    float weight_diff_sum = 0.0f;
    uint32_t matching = 0;

    // Collect all innovations
    std::set<uint32_t> all_innovations;
    for (const auto& [innov, _] : a_weights) all_innovations.insert(innov);
    for (const auto& [innov, _] : b_weights) all_innovations.insert(innov);

    for (auto innov : all_innovations) {
        bool in_a = a_weights.count(innov) > 0;
        bool in_b = b_weights.count(innov) > 0;

        if (in_a && in_b) {
            matching++;
            weight_diff_sum += std::abs(a_weights[innov] - b_weights[innov]);
        } else if (innov > shared_max) {
            excess++;
        } else {
            disjoint++;
        }
    }

    float avg_weight_diff = (matching > 0) ? weight_diff_sum / static_cast<float>(matching) : 0.0f;

    // N = max genome size, or 1 if both small (< 20 genes)
    auto n_a = a.connections.size();
    auto n_b = b.connections.size();
    float n = static_cast<float>(std::max(n_a, n_b));
    if (n_a < 20 && n_b < 20) n = 1.0f;
    if (n < 1.0f) n = 1.0f;

    return (config.c_excess * static_cast<float>(excess) / n)
         + (config.c_disjoint * static_cast<float>(disjoint) / n)
         + (config.c_weight * avg_weight_diff);
}
```

- [ ] **Step 3: Build and run tests**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target evolve_tests && ./build/libs/evolve/tests/evolve_tests --gtest_filter='Speciation*'`
Expected: All 4 speciation tests PASS.

- [ ] **Step 4: Commit**

```bash
git add libs/evolve/src/neat_operators.cpp libs/evolve/tests/neat_operators_test.cpp
git commit -m "feat(evolve): implement NEAT compatibility distance for speciation"
```

---

### Task 12: NeatPopulation

**Files:**
- Create: `libs/evolve/include/evolve/neat_population.h`
- Create: `libs/evolve/src/neat_population.cpp`
- Create: `libs/evolve/tests/neat_population_test.cpp`
- Modify: `libs/evolve/CMakeLists.txt`
- Modify: `libs/evolve/tests/CMakeLists.txt`

- [ ] **Step 1: Write failing tests**

```cpp
// libs/evolve/tests/neat_population_test.cpp
#include <evolve/neat_population.h>
#include <neuralnet/graph_network.h>

#include <gtest/gtest.h>

namespace ev = evolve;
namespace nn = neuralnet;

TEST(NeatPopulationTest, ConstructCreatesPopulation) {
    ev::NeatPopulationConfig config;
    config.population_size = 50;

    std::mt19937 rng(42);
    ev::NeatPopulation pop(3, 2, nn::NodeType::Stateless, nn::Activation::Tanh, config, rng);

    EXPECT_EQ(pop.individuals().size(), 50);
    EXPECT_EQ(pop.generation(), 0);
    EXPECT_GE(pop.num_species(), 1);
}

TEST(NeatPopulationTest, IndividualsAreValidNetworks) {
    ev::NeatPopulationConfig config;
    config.population_size = 20;

    std::mt19937 rng(42);
    ev::NeatPopulation pop(3, 2, nn::NodeType::Stateless, nn::Activation::Tanh, config, rng);

    for (const auto& ind : pop.individuals()) {
        EXPECT_NO_THROW(nn::GraphNetwork(ind.genome));
        nn::GraphNetwork net(ind.genome);
        EXPECT_EQ(net.input_size(), 3);
        EXPECT_EQ(net.output_size(), 2);
    }
}

TEST(NeatPopulationTest, EvolveProducesNextGeneration) {
    ev::NeatPopulationConfig config;
    config.population_size = 30;

    std::mt19937 rng(42);
    ev::NeatPopulation pop(2, 1, nn::NodeType::Stateless, nn::Activation::Tanh, config, rng);

    // Assign random fitness
    for (auto& ind : pop.individuals()) {
        ind.fitness = std::uniform_real_distribution<float>(0.0f, 10.0f)(rng);
    }

    pop.evolve(rng);

    EXPECT_EQ(pop.generation(), 1);
    EXPECT_EQ(pop.individuals().size(), 30);

    // All individuals should still produce valid networks
    for (const auto& ind : pop.individuals()) {
        EXPECT_NO_THROW(nn::GraphNetwork(ind.genome));
    }
}

TEST(NeatPopulationTest, EvolveMultipleGenerations) {
    ev::NeatPopulationConfig config;
    config.population_size = 50;

    std::mt19937 rng(42);
    ev::NeatPopulation pop(3, 2, nn::NodeType::Stateless, nn::Activation::Tanh, config, rng);

    for (int gen = 0; gen < 20; ++gen) {
        // Evaluate: fitness = number of connections (rewards complexity)
        for (auto& ind : pop.individuals()) {
            ind.fitness = static_cast<float>(ind.genome.connections.size());
        }
        pop.evolve(rng);
    }

    EXPECT_EQ(pop.generation(), 20);
    EXPECT_EQ(pop.individuals().size(), 50);

    // After 20 generations selecting for more connections,
    // average should be higher than the starting 6 (3 inputs * 2 outputs)
    float avg_connections = 0;
    for (const auto& ind : pop.individuals()) {
        avg_connections += static_cast<float>(ind.genome.connections.size());
    }
    avg_connections /= static_cast<float>(pop.individuals().size());
    EXPECT_GT(avg_connections, 6.0f);
}

TEST(NeatPopulationTest, SpeciationCreatesMultipleSpecies) {
    ev::NeatPopulationConfig config;
    config.population_size = 100;
    config.speciation.compatibility_threshold = 1.0f;  // Low threshold = more species

    std::mt19937 rng(42);
    ev::NeatPopulation pop(3, 2, nn::NodeType::Stateless, nn::Activation::Tanh, config, rng);

    // Evolve with structural mutations to create diversity
    for (int gen = 0; gen < 10; ++gen) {
        for (auto& ind : pop.individuals()) {
            ind.fitness = std::uniform_real_distribution<float>(0.0f, 10.0f)(rng);
        }
        pop.evolve(rng);
    }

    // With low threshold and structural mutations, should have multiple species
    EXPECT_GT(pop.num_species(), 1);
}
```

- [ ] **Step 2: Run to verify fail**

Expected: Compilation fails — `evolve/neat_population.h` not found.

- [ ] **Step 3: Create neat_population.h**

```cpp
// libs/evolve/include/evolve/neat_population.h
#pragma once

#include <evolve/innovation.h>
#include <evolve/neat_operators.h>
#include <neuralnet/graph_genome.h>
#include <neuralnet/node_types.h>

#include <cstdint>
#include <random>
#include <vector>

namespace evolve {

struct NeatIndividual {
    neuralnet::GraphGenome genome;
    float fitness = 0.0f;
    uint32_t species_id = 0;
};

struct NeatPopulationConfig {
    std::size_t population_size = 150;
    NeatMutationConfig mutation;
    SpeciationConfig speciation;
    std::size_t elitism_per_species = 1;
    float interspecies_mate_rate = 0.001f;
    std::size_t stagnation_limit = 15;
    std::size_t min_species_to_keep = 2;
};

class NeatPopulation {
public:
    NeatPopulation(
        std::size_t num_inputs,
        std::size_t num_outputs,
        neuralnet::NodeType default_output_type,
        neuralnet::Activation default_output_activation,
        const NeatPopulationConfig& config,
        std::mt19937& rng
    );

    [[nodiscard]] std::vector<NeatIndividual>& individuals();
    [[nodiscard]] const std::vector<NeatIndividual>& individuals() const;

    void evolve(std::mt19937& rng);

    [[nodiscard]] std::size_t generation() const noexcept;
    [[nodiscard]] std::size_t num_species() const noexcept;
    [[nodiscard]] InnovationCounter& innovation_counter() noexcept;

private:
    struct SpeciesInfo {
        uint32_t id;
        neuralnet::GraphGenome representative;
        float best_fitness = 0.0f;
        std::size_t stagnation_count = 0;
    };

    void speciate();
    void eliminate_stagnant_species();

    NeatPopulationConfig config_;
    std::vector<NeatIndividual> individuals_;
    std::vector<SpeciesInfo> species_;
    InnovationCounter innovation_counter_;
    std::size_t generation_ = 0;
    uint32_t next_species_id_ = 0;
};

} // namespace evolve
```

- [ ] **Step 4: Implement NeatPopulation**

```cpp
// libs/evolve/src/neat_population.cpp
#include <evolve/neat_population.h>

#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace evolve {

NeatPopulation::NeatPopulation(
    std::size_t num_inputs,
    std::size_t num_outputs,
    neuralnet::NodeType default_output_type,
    neuralnet::Activation default_output_activation,
    const NeatPopulationConfig& config,
    std::mt19937& rng)
    : config_(config) {

    // Seed innovation counter with the initial genome's connection innovations
    // (num_inputs * num_outputs connections, innovations 0..N-1)
    for (std::size_t in = 0; in < num_inputs; ++in) {
        for (std::size_t out = 0; out < num_outputs; ++out) {
            innovation_counter_.get_or_create(
                static_cast<uint32_t>(in),
                static_cast<uint32_t>(num_inputs + out));
        }
    }
    innovation_counter_.new_generation();  // Clear generation cache, counter now at N

    // Create initial population of minimal genomes
    individuals_.reserve(config.population_size);
    for (std::size_t i = 0; i < config.population_size; ++i) {
        auto genome = neuralnet::create_minimal_genome(
            num_inputs, num_outputs, default_output_type, default_output_activation, rng);
        individuals_.push_back(NeatIndividual{.genome = std::move(genome)});
    }

    // Initial speciation — all start in one species
    species_.push_back(SpeciesInfo{
        .id = next_species_id_++,
        .representative = individuals_[0].genome,
    });
    for (auto& ind : individuals_) {
        ind.species_id = species_[0].id;
    }
}

std::vector<NeatIndividual>& NeatPopulation::individuals() { return individuals_; }
const std::vector<NeatIndividual>& NeatPopulation::individuals() const { return individuals_; }
std::size_t NeatPopulation::generation() const noexcept { return generation_; }
std::size_t NeatPopulation::num_species() const noexcept { return species_.size(); }
InnovationCounter& NeatPopulation::innovation_counter() noexcept { return innovation_counter_; }

void NeatPopulation::speciate() {
    // Clear species assignments
    for (auto& ind : individuals_) {
        ind.species_id = UINT32_MAX;  // Unassigned
    }

    for (auto& ind : individuals_) {
        bool found = false;
        for (auto& species : species_) {
            float dist = compatibility_distance(
                ind.genome, species.representative, config_.speciation);
            if (dist < config_.speciation.compatibility_threshold) {
                ind.species_id = species.id;
                found = true;
                break;
            }
        }
        if (!found) {
            species_.push_back(SpeciesInfo{
                .id = next_species_id_++,
                .representative = ind.genome,
            });
            ind.species_id = species_.back().id;
        }
    }

    // Remove empty species and update representatives
    for (auto& species : species_) {
        std::vector<NeatIndividual*> members;
        for (auto& ind : individuals_) {
            if (ind.species_id == species.id) {
                members.push_back(&ind);
            }
        }
        if (!members.empty()) {
            // Update representative to random member
            std::uniform_int_distribution<std::size_t> pick(0, members.size() - 1);
            std::mt19937 temp_rng(species.id);  // Deterministic per species
            species.representative = members[pick(temp_rng)]->genome;
        }
    }

    // Remove species with no members
    species_.erase(
        std::remove_if(species_.begin(), species_.end(),
            [this](const SpeciesInfo& s) {
                return std::none_of(individuals_.begin(), individuals_.end(),
                    [&](const NeatIndividual& ind) { return ind.species_id == s.id; });
            }),
        species_.end());
}

void NeatPopulation::eliminate_stagnant_species() {
    // Update best fitness and stagnation count per species
    for (auto& species : species_) {
        float best = -1e30f;
        for (const auto& ind : individuals_) {
            if (ind.species_id == species.id) {
                best = std::max(best, ind.fitness);
            }
        }
        if (best > species.best_fitness) {
            species.best_fitness = best;
            species.stagnation_count = 0;
        } else {
            species.stagnation_count++;
        }
    }

    // Eliminate stagnant species (but keep minimum)
    if (species_.size() <= config_.min_species_to_keep) return;

    // Sort by stagnation (most stagnant first) so we remove worst first
    std::sort(species_.begin(), species_.end(),
        [](const SpeciesInfo& a, const SpeciesInfo& b) {
            return a.stagnation_count > b.stagnation_count;
        });

    // Remove stagnant species from the front, but keep min_species_to_keep
    while (species_.size() > config_.min_species_to_keep
           && species_.front().stagnation_count >= config_.stagnation_limit) {
        species_.erase(species_.begin());
    }

    if (species_.empty()) return;  // Safety (shouldn't happen)

    // Reassign orphaned individuals to nearest species
    for (auto& ind : individuals_) {
        bool has_species = std::any_of(species_.begin(), species_.end(),
            [&](const SpeciesInfo& s) { return s.id == ind.species_id; });
        if (!has_species) {
            // Assign to first species (they'll be replaced anyway)
            ind.species_id = species_[0].id;
        }
    }
}

void NeatPopulation::evolve(std::mt19937& rng) {
    // Step 1: Speciate
    speciate();

    // Step 2: Stagnation check
    eliminate_stagnant_species();

    // Step 3: Compute adjusted fitness and offspring allocation
    std::unordered_map<uint32_t, std::vector<NeatIndividual*>> species_members;
    for (auto& ind : individuals_) {
        species_members[ind.species_id].push_back(&ind);
    }

    // Sort members within each species by fitness (descending)
    for (auto& [sid, members] : species_members) {
        std::sort(members.begin(), members.end(),
            [](const NeatIndividual* a, const NeatIndividual* b) {
                return a->fitness > b->fitness;
            });
    }

    // Adjusted fitness: individual fitness / species size
    float total_adjusted = 0.0f;
    std::unordered_map<uint32_t, float> species_adjusted_total;
    for (auto& [sid, members] : species_members) {
        float species_total = 0.0f;
        auto species_size = static_cast<float>(members.size());
        for (auto* ind : members) {
            float adjusted = ind->fitness / species_size;
            species_total += adjusted;
        }
        species_adjusted_total[sid] = species_total;
        total_adjusted += species_total;
    }

    // Step 4: Offspring allocation
    std::unordered_map<uint32_t, std::size_t> offspring_count;
    std::size_t total_allocated = 0;
    for (auto& [sid, adj_total] : species_adjusted_total) {
        float proportion = (total_adjusted > 0) ? adj_total / total_adjusted : 1.0f / static_cast<float>(species_.size());
        auto count = static_cast<std::size_t>(std::max(1.0f,
            proportion * static_cast<float>(config_.population_size)));
        offspring_count[sid] = count;
        total_allocated += count;
    }

    // Adjust to exact population size
    while (total_allocated > config_.population_size) {
        // Remove from largest
        auto max_it = std::max_element(offspring_count.begin(), offspring_count.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        if (max_it->second > 1) {
            max_it->second--;
            total_allocated--;
        } else break;
    }
    while (total_allocated < config_.population_size) {
        // Add to best species
        auto best_sid = species_[0].id;
        float best_fit = -1e30f;
        for (const auto& s : species_) {
            if (species_adjusted_total[s.id] > best_fit) {
                best_fit = species_adjusted_total[s.id];
                best_sid = s.id;
            }
        }
        offspring_count[best_sid]++;
        total_allocated++;
    }

    // Step 5: Reproduce
    std::vector<NeatIndividual> new_pop;
    new_pop.reserve(config_.population_size);
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);

    for (auto& [sid, count] : offspring_count) {
        auto& members = species_members[sid];
        if (members.empty()) continue;

        // Elitism
        std::size_t elite_count = std::min(config_.elitism_per_species, members.size());
        elite_count = std::min(elite_count, count);
        for (std::size_t i = 0; i < elite_count; ++i) {
            new_pop.push_back(NeatIndividual{
                .genome = members[i]->genome,
                .fitness = 0.0f,
                .species_id = sid,
            });
        }

        // Fill rest with offspring
        std::uniform_int_distribution<std::size_t> member_pick(0, members.size() - 1);
        for (std::size_t i = elite_count; i < count; ++i) {
            auto* parent_a = members[member_pick(rng)];
            neuralnet::GraphGenome child_genome;

            if (members.size() > 1 || prob(rng) < config_.interspecies_mate_rate) {
                const NeatIndividual* parent_b = nullptr;
                if (prob(rng) < config_.interspecies_mate_rate && individuals_.size() > members.size()) {
                    // Interspecies crossover
                    std::uniform_int_distribution<std::size_t> global_pick(0, individuals_.size() - 1);
                    parent_b = &individuals_[global_pick(rng)];
                } else if (members.size() > 1) {
                    parent_b = members[member_pick(rng)];
                }

                if (parent_b && parent_b != parent_a) {
                    if (parent_a->fitness >= parent_b->fitness) {
                        child_genome = crossover(parent_a->genome, parent_b->genome, rng);
                    } else {
                        child_genome = crossover(parent_b->genome, parent_a->genome, rng);
                    }
                } else {
                    child_genome = parent_a->genome;
                }
            } else {
                child_genome = parent_a->genome;
            }

            mutate(child_genome, innovation_counter_, config_.mutation, rng);

            new_pop.push_back(NeatIndividual{
                .genome = std::move(child_genome),
                .fitness = 0.0f,
                .species_id = sid,
            });
        }
    }

    // Step 6: Replace
    individuals_ = std::move(new_pop);
    innovation_counter_.new_generation();
    generation_++;
}

} // namespace evolve
```

- [ ] **Step 5: Update CMakeLists**

Add `src/neat_population.cpp` to `libs/evolve/CMakeLists.txt` source list.
Add `neat_population_test.cpp` to `libs/evolve/tests/CMakeLists.txt`.

- [ ] **Step 6: Build and run tests**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target evolve_tests && ./build/libs/evolve/tests/evolve_tests --gtest_filter='NeatPopulation*'`
Expected: All 5 NeatPopulation tests PASS.

- [ ] **Step 7: Commit**

```bash
git add libs/evolve/include/evolve/neat_population.h libs/evolve/src/neat_population.cpp libs/evolve/tests/neat_population_test.cpp libs/evolve/CMakeLists.txt libs/evolve/tests/CMakeLists.txt
git commit -m "feat(evolve): implement NeatPopulation with speciation, stagnation, and reproduction"
```

---

### Task 13: Versioned Serialization

**Files:**
- Modify: `libs/neuralnet/include/neuralnet/serialization.h`
- Modify: `libs/neuralnet/src/serialization.cpp`
- Create: `libs/neuralnet/tests/serialization_v2_test.cpp`
- Modify: `libs/neuralnet/tests/CMakeLists.txt`

- [ ] **Step 1: Write failing tests**

```cpp
// libs/neuralnet/tests/serialization_v2_test.cpp
#include <neuralnet/serialization.h>
#include <neuralnet/graph_genome.h>
#include <neuralnet/graph_network.h>

#include <gtest/gtest.h>

#include <sstream>
#include <variant>

namespace nn = neuralnet;

TEST(SerializationV2Test, GraphNetworkRoundTrip) {
    std::mt19937 rng(42);
    auto genome = nn::create_minimal_genome(2, 1, nn::NodeType::CTRNN, nn::Activation::Tanh, rng);
    nn::GraphNetwork net(genome);

    // Run a few forward passes to build up CTRNN state
    net.forward(std::vector<float>{1.0f, 0.5f});
    net.forward(std::vector<float>{0.3f, 0.7f});

    std::stringstream ss;
    nn::save(net, ss);

    ss.seekg(0);
    auto loaded = nn::load(ss);

    ASSERT_TRUE(std::holds_alternative<nn::GraphNetwork>(loaded));
    auto& loaded_net = std::get<nn::GraphNetwork>(loaded);

    EXPECT_EQ(loaded_net.input_size(), 2);
    EXPECT_EQ(loaded_net.output_size(), 1);
    EXPECT_EQ(loaded_net.num_nodes(), net.num_nodes());
    EXPECT_EQ(loaded_net.num_connections(), net.num_connections());

    // Same input should produce same output (CTRNN state was saved)
    auto out_original = net.forward(std::vector<float>{1.0f, 1.0f});
    auto out_loaded = loaded_net.forward(std::vector<float>{1.0f, 1.0f});
    ASSERT_EQ(out_original.size(), out_loaded.size());
    for (std::size_t i = 0; i < out_original.size(); ++i) {
        EXPECT_NEAR(out_original[i], out_loaded[i], 1e-5f);
    }
}

TEST(SerializationV2Test, MLPRoundTrip_VersionedFormat) {
    nn::NetworkTopology topo;
    topo.input_size = 2;
    topo.layers = {{.output_size = 3, .activation = nn::Activation::ReLU},
                   {.output_size = 1, .activation = nn::Activation::Sigmoid}};
    std::vector<float> weights(13, 0.5f);
    nn::Network net(topo, weights);

    std::stringstream ss;
    nn::save(net, ss);

    ss.seekg(0);
    auto loaded = nn::load(ss);

    ASSERT_TRUE(std::holds_alternative<nn::Network>(loaded));
    auto& loaded_net = std::get<nn::Network>(loaded);
    EXPECT_EQ(loaded_net.input_size(), 2);
    EXPECT_EQ(loaded_net.output_size(), 1);
    EXPECT_EQ(loaded_net.get_all_weights(), weights);
}

TEST(SerializationV2Test, LegacyFormatStillLoads) {
    // Save using legacy format
    nn::NetworkTopology topo;
    topo.input_size = 2;
    topo.layers = {{.output_size = 1, .activation = nn::Activation::ReLU}};
    std::vector<float> weights = {1.0f, 1.0f, 0.0f};
    nn::Network net(topo, weights);

    std::stringstream ss;
    nn::serialize(net, ss);  // Legacy format

    ss.seekg(0);
    auto loaded = nn::load(ss);

    ASSERT_TRUE(std::holds_alternative<nn::Network>(loaded));
    auto& loaded_net = std::get<nn::Network>(loaded);
    EXPECT_EQ(loaded_net.get_all_weights(), weights);
}
```

- [ ] **Step 2: Add versioned API declarations to serialization.h**

```cpp
// libs/neuralnet/include/neuralnet/serialization.h
#pragma once

#include <neuralnet/network.h>
#include <neuralnet/graph_network.h>

#include <iosfwd>
#include <variant>

namespace neuralnet {

// === Versioned API (preferred) ===

/// Save a GraphNetwork to a binary stream (versioned format).
void save(const GraphNetwork& net, std::ostream& out);

/// Save an MLP Network to a binary stream (versioned format).
void save(const Network& net, std::ostream& out);

/// Load any network type from a binary stream. Auto-detects format
/// (versioned or legacy MLP).
[[nodiscard]] std::variant<Network, GraphNetwork> load(std::istream& in);

// === Legacy API (deprecated — kept for backward compatibility) ===

/// Serialize a network to a binary stream (legacy format, no version header).
void serialize(const Network& net, std::ostream& out);

/// Deserialize a network from a binary stream (legacy format only).
[[nodiscard]] Network deserialize(std::istream& in);

} // namespace neuralnet
```

- [ ] **Step 3: Implement versioned serialization**

Add to `libs/neuralnet/src/serialization.cpp`, keeping existing code and adding new functions:

```cpp
// Add after existing code:

namespace {

constexpr uint32_t VERSIONED_MAGIC = 0x4E4E504B;  // "NNPK"
constexpr uint16_t FORMAT_VERSION = 1;

enum FormatType : uint8_t {
    FORMAT_MLP = 0,
    FORMAT_GRAPH = 1,
};

enum FeatureFlags : uint32_t {
    FEATURE_CTRNN = 1 << 0,
    FEATURE_INNOVATION = 1 << 1,
};

} // namespace

void save(const GraphNetwork& net, std::ostream& out) {
    const auto& genome = net.genome();

    // Determine feature flags
    uint32_t flags = FEATURE_INNOVATION;
    bool has_ctrnn = false;
    for (const auto& node : genome.nodes) {
        if (node.type == NodeType::CTRNN) {
            has_ctrnn = true;
            break;
        }
    }
    if (has_ctrnn) flags |= FEATURE_CTRNN;

    // Header
    write_val(out, VERSIONED_MAGIC);
    write_val<uint16_t>(out, FORMAT_VERSION);
    write_val<uint32_t>(out, flags);
    write_val<uint8_t>(out, FORMAT_GRAPH);

    // Payload
    write_val<uint32_t>(out, static_cast<uint32_t>(genome.nodes.size()));
    write_val<uint32_t>(out, static_cast<uint32_t>(genome.connections.size()));
    write_val<float>(out, 1.0f);  // dt (default)

    for (const auto& node : genome.nodes) {
        write_val<uint32_t>(out, node.id);
        write_val<uint8_t>(out, static_cast<uint8_t>(node.role));
        write_val<uint8_t>(out, static_cast<uint8_t>(node.type));
        write_val<uint8_t>(out, static_cast<uint8_t>(node.activation));
        write_val<float>(out, node.bias);
        write_val<float>(out, node.tau);
    }

    for (const auto& conn : genome.connections) {
        write_val<uint32_t>(out, conn.from_node);
        write_val<uint32_t>(out, conn.to_node);
        write_val<float>(out, conn.weight);
        write_val<uint8_t>(out, conn.enabled ? 1 : 0);
        write_val<uint32_t>(out, conn.innovation);
    }

    // Save node states if CTRNN
    if (has_ctrnn) {
        auto states = net.get_node_states();
        for (auto s : states) {
            write_val<float>(out, s);
        }
    }
}

void save(const Network& net, std::ostream& out) {
    const auto& topo = net.topology();

    write_val(out, VERSIONED_MAGIC);
    write_val<uint16_t>(out, FORMAT_VERSION);
    write_val<uint32_t>(out, 0);  // No feature flags for MLP
    write_val<uint8_t>(out, FORMAT_MLP);

    // Same payload as legacy format
    write_val<uint32_t>(out, static_cast<uint32_t>(topo.input_size));
    write_val<uint32_t>(out, static_cast<uint32_t>(topo.layers.size()));
    for (const auto& layer_def : topo.layers) {
        write_val<uint32_t>(out, static_cast<uint32_t>(layer_def.output_size));
        write_val<uint32_t>(out, static_cast<uint32_t>(layer_def.activation));
    }
    auto weights = net.get_all_weights();
    write_val<uint32_t>(out, static_cast<uint32_t>(weights.size()));
    out.write(reinterpret_cast<const char*>(weights.data()),
              static_cast<std::streamsize>(weights.size() * sizeof(float)));
}

std::variant<Network, GraphNetwork> load(std::istream& in) {
    auto magic = read_val<uint32_t>(in);

    if (magic == MAGIC) {
        // Legacy MLP format — read the rest using legacy parser logic
        // (magic already consumed, so we read from input_size onward)
        NetworkTopology topo;
        topo.input_size = read_val<uint32_t>(in);
        auto num_layers = read_val<uint32_t>(in);
        topo.layers.resize(num_layers);
        for (auto& layer_def : topo.layers) {
            layer_def.output_size = read_val<uint32_t>(in);
            layer_def.activation = static_cast<Activation>(read_val<uint32_t>(in));
        }
        auto weight_count = read_val<uint32_t>(in);
        std::vector<float> weights(weight_count);
        if (!in.read(reinterpret_cast<char*>(weights.data()),
                     static_cast<std::streamsize>(weight_count * sizeof(float)))) {
            throw std::runtime_error("Unexpected end of stream reading weights");
        }
        return Network(topo, weights);
    }

    if (magic != VERSIONED_MAGIC) {
        throw std::runtime_error("Invalid network file: unrecognized magic number");
    }

    auto version = read_val<uint16_t>(in);
    if (version != FORMAT_VERSION) {
        throw std::runtime_error("Unsupported format version: " + std::to_string(version));
    }

    auto flags = read_val<uint32_t>(in);
    auto format_type = read_val<uint8_t>(in);

    if (format_type == FORMAT_MLP) {
        NetworkTopology topo;
        topo.input_size = read_val<uint32_t>(in);
        auto num_layers = read_val<uint32_t>(in);
        topo.layers.resize(num_layers);
        for (auto& layer_def : topo.layers) {
            layer_def.output_size = read_val<uint32_t>(in);
            layer_def.activation = static_cast<Activation>(read_val<uint32_t>(in));
        }
        auto weight_count = read_val<uint32_t>(in);
        std::vector<float> weights(weight_count);
        if (!in.read(reinterpret_cast<char*>(weights.data()),
                     static_cast<std::streamsize>(weight_count * sizeof(float)))) {
            throw std::runtime_error("Unexpected end of stream reading weights");
        }
        return Network(topo, weights);
    }

    if (format_type == FORMAT_GRAPH) {
        auto num_nodes = read_val<uint32_t>(in);
        auto num_connections = read_val<uint32_t>(in);
        auto dt = read_val<float>(in);

        GraphGenome genome;
        genome.nodes.resize(num_nodes);
        for (auto& node : genome.nodes) {
            node.id = read_val<uint32_t>(in);
            node.role = static_cast<NodeRole>(read_val<uint8_t>(in));
            node.type = static_cast<NodeType>(read_val<uint8_t>(in));
            node.activation = static_cast<Activation>(read_val<uint8_t>(in));
            node.bias = read_val<float>(in);
            node.tau = read_val<float>(in);
        }

        genome.connections.resize(num_connections);
        for (auto& conn : genome.connections) {
            conn.from_node = read_val<uint32_t>(in);
            conn.to_node = read_val<uint32_t>(in);
            conn.weight = read_val<float>(in);
            conn.enabled = read_val<uint8_t>(in) != 0;
            conn.innovation = read_val<uint32_t>(in);
        }

        GraphNetwork net(genome, dt);

        // Restore CTRNN states if present
        if (flags & FEATURE_CTRNN) {
            std::vector<float> states(num_nodes);
            for (uint32_t i = 0; i < num_nodes; ++i) {
                states[i] = read_val<float>(in);
            }
            net.set_node_states(states);
        }

        return net;
    }

    throw std::runtime_error("Unknown format type: " + std::to_string(format_type));
}
```

**Important:** The `write_val` and `read_val` helpers are currently in an anonymous namespace. Move them to a shared location within the file or make them file-scope so both old and new functions can use them. The simplest approach is to keep them in the anonymous namespace at the top of the file — they're already accessible to all functions in the translation unit.

- [ ] **Step 4: Update CMakeLists**

Add `serialization_v2_test.cpp` to `libs/neuralnet/tests/CMakeLists.txt`.

- [ ] **Step 5: Build and run tests**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build --target neuralnet_tests && ./build/libs/neuralnet/tests/neuralnet_tests --gtest_filter='Serialization*'`
Expected: All serialization tests PASS (existing + new).

- [ ] **Step 6: Run ALL tests to verify nothing is broken**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake --build build && ctest --test-dir build --output-on-failure`
Expected: All tests across all libraries PASS.

- [ ] **Step 7: Commit**

```bash
git add libs/neuralnet/include/neuralnet/serialization.h libs/neuralnet/src/serialization.cpp libs/neuralnet/tests/serialization_v2_test.cpp libs/neuralnet/tests/CMakeLists.txt
git commit -m "feat(neuralnet): add versioned serialization for GraphNetwork and MLP"
```

---

### Task 14: Final Integration — Run All Tests + Update evolve CMake dependency

**Files:**
- Verify: all CMakeLists.txt files
- Verify: all tests pass

- [ ] **Step 1: Verify evolve depends on neuralnet**

Check that `libs/evolve/CMakeLists.txt` has: `target_link_libraries(evolve PRIVATE project_warnings PUBLIC neuralnet)`

If not, add it.

- [ ] **Step 2: Full build from clean**

Run: `cd /Users/wldarden/learning/cPlusPlus && cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build`
Expected: Clean build, no warnings (strict `-Werror`).

- [ ] **Step 3: Run all tests**

Run: `cd /Users/wldarden/learning/cPlusPlus && ctest --test-dir build --output-on-failure`
Expected: All tests PASS across neuralnet_tests, evolve_tests, and any other test targets.

- [ ] **Step 4: Verify existing tests still pass**

Run: `cd /Users/wldarden/learning/cPlusPlus && ./build/libs/neuralnet/tests/neuralnet_tests && ./build/libs/evolve/tests/evolve_tests`
Expected: Existing tests (network_test, layer_test, activation_test, serialization_test, genome_test, population_test) all still PASS.

- [ ] **Step 5: Commit any final fixes**

If any adjustments were needed, commit them:
```bash
git add -A
git commit -m "chore: final integration fixes for GraphNetwork + NEAT"
```
