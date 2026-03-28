# GraphNetwork + NEAT Evolution — Design Spec

**Date:** 2026-03-21
**Status:** Draft
**Scope:** `libs/neuralnet` and `libs/evolve` enhancements

## Problem

The neuralnet library currently supports only dense feedforward networks (MLP). This is sufficient for simple projects like NeuroFlyer, but projects requiring evolvable topology, per-neuron memory dynamics, and sparse connectivity (like AntSim) have no library-level support.

## Goals

1. Add a `GraphNetwork` — a NEAT-style graph-based neural network — as a first-class network type alongside the existing MLP (`Network`).
2. Introduce a **node type system** where individual nodes can have different dynamics (stateless, CTRNN, future types). Node types are abstract library concepts, not domain-specific.
3. Add NEAT evolution operators to `libs/evolve` for evolving `GraphNetwork` topology and weights.
4. Implement versioned serialization with feature flags so saved networks are forward-compatible.
5. Do not modify existing MLP code — coexist, don't replace.

## Non-Goals

- Modular/clustered network type (future work, potential converter target)
- GPU acceleration
- Domain-specific node types (pheromone nodes, etc. — apps assign meaning to generic nodes)
- Converting between network types (future work)

---

## Design

### 1. Node Type System

Node types define how an individual neuron computes its output from its inputs and (optionally) its internal state. The library provides node types; consuming apps choose which to use.

#### 1.1 `NodeType` Enum

```cpp
namespace neuralnet {

enum class NodeType : uint8_t {
    Stateless = 0,  // output = activate(weighted_sum + bias)
    CTRNN     = 1,  // leaky integrator with time constant τ
    // Future types added here
};

} // namespace neuralnet
```

#### 1.2 Node Type Behaviors

**Stateless** — What exists today. No internal state. Every tick is independent.

```
output = activate(Σ(w_i * input_i) + bias)
```

**CTRNN** — Continuous-Time Recurrent Neural Network node. Has internal state that persists across ticks. The time constant τ controls how fast the node responds to new input.

```
raw = activate(Σ(w_i * input_i) + bias)
state = state + (dt / τ) * (-state + raw)
output = state
```

Where:
- `state` is per-node persistent state, initialized to 0.0
- `τ` (tau) is the time constant. Large τ = slow response = long memory. Evolvable per-node.
- `dt` is the simulation timestep (typically 1.0 for discrete-time, but configurable for variable timestep sims)
- `activate` is the node's activation function (Tanh recommended for CTRNN nodes, but not enforced)

**Design note:** τ must be > 0. Reasonable initial range for evolution: [0.1, 10.0]. A τ of 1.0 with dt=1.0 means the node reaches ~63% of its target value in one tick — roughly equivalent to a stateless node. Very large τ (100+) creates near-permanent state.

#### 1.3 Node Properties

Each node in a `GraphNetwork` carries:

```cpp
struct NodeGene {
    uint32_t id;                  // Unique node identifier
    NodeRole role;                // Input, Hidden, Output
    NodeType type;                // Stateless, CTRNN, ...
    Activation activation;        // ReLU, Sigmoid, Tanh
    float bias;                   // Evolvable
    float tau;                    // Time constant (only meaningful for CTRNN, ignored for Stateless)
};
```

```cpp
enum class NodeRole : uint8_t {
    Input  = 0,
    Hidden = 1,
    Output = 2,
};
```

Input nodes are pass-through — they have no activation, bias, or dynamics. They just relay external input values into the graph. Their `type`, `activation`, `bias`, and `tau` fields are ignored.

### 2. GraphNetwork

A graph-based neural network where topology is defined by explicit node and connection genes. This is the runtime evaluation structure.

#### 2.1 Connection Gene

```cpp
struct ConnectionGene {
    uint32_t from_node;           // Source node id
    uint32_t to_node;             // Target node id
    float weight;                 // Evolvable
    bool enabled;                 // Can be disabled without removal (NEAT convention)
    uint32_t innovation;          // Global innovation number (for crossover alignment)
};
```

**Innovation numbers** are central to NEAT. When a new connection is created (by mutation), it gets a globally unique innovation number. This allows meaningful crossover between networks with different topologies — matching genes (same innovation number) are aligned, disjoint/excess genes are inherited from the fitter parent.

#### 2.2 Genome

The full genetic representation of a graph network:

```cpp
struct GraphGenome {
    std::vector<NodeGene> nodes;
    std::vector<ConnectionGene> connections;
};
```

This is the evolvable representation. It is distinct from the runtime `GraphNetwork` — a genome is "compiled" into a `GraphNetwork` for evaluation.

#### 2.3 GraphNetwork Class

```cpp
class GraphNetwork {
public:
    /// Build from a genome. Performs topological sort, allocates state buffers.
    explicit GraphNetwork(const GraphGenome& genome);

    /// Run one tick. Input size must match the number of Input nodes.
    /// Returns output values (one per Output node, in node-id order).
    [[nodiscard]] std::vector<float> forward(std::span<const float> input);

    /// Reset all internal node state to zero (for starting a new episode).
    void reset_state();

    /// Access current state of all nodes (for inspection/debugging).
    [[nodiscard]] std::span<const float> get_node_states() const noexcept;

    /// Access the underlying genome.
    [[nodiscard]] const GraphGenome& genome() const noexcept;

    [[nodiscard]] std::size_t input_size() const noexcept;
    [[nodiscard]] std::size_t output_size() const noexcept;
    [[nodiscard]] std::size_t num_nodes() const noexcept;
    [[nodiscard]] std::size_t num_connections() const noexcept;

private:
    GraphGenome genome_;
    std::vector<float> node_states_;     // Persistent state per node (compact, indexed)
    std::vector<float> node_outputs_;    // Output values per node (compact, indexed)
    std::vector<uint32_t> eval_order_;   // Topological sort order (indices, not IDs)
    std::vector<ConnectionGene> recurrent_connections_; // Back-edges, read prev tick
    std::unordered_map<uint32_t, uint32_t> id_to_index_; // Node ID → compact index
    float dt_;
};
```

**Ownership model:** `GraphNetwork` takes a **copy** of the genome at construction. The genome is immutable within the network. To evolve: extract the genome via `genome()`, mutate/crossover it externally, and construct a new `GraphNetwork` from the modified genome.

**Genome validation:** The constructor validates the genome and throws `std::invalid_argument` if:
- There are zero input nodes or zero output nodes
- There are duplicate node IDs
- A connection references a node ID that doesn't exist in the node list
- A connection targets an input node (input nodes are pass-through, cannot receive connections)
```

#### 2.4 Forward Pass Algorithm

**Construction-time graph analysis:**

When a `GraphNetwork` is built from a `GraphGenome`, the constructor performs a **modified DFS** to classify connections and determine evaluation order:

1. Build an adjacency list from all **enabled** connections only (disabled connections are skipped entirely).
2. Run DFS from all input nodes to determine reachability and produce a topological ordering of reachable nodes.
3. **Recurrent edge classification:** After the topological sort is computed, classify each enabled connection: if `from_node` appears at the same position or later than `to_node` in the topological order, the connection is **recurrent** (it would create a cycle if evaluated feedforward). All other connections are feedforward. This position-based classification is simpler and more correct than relying on DFS stack membership alone, as it correctly handles cross-edges and complex cycle topologies.
4. Store recurrent connections separately in a `recurrent_connections_` vector.
5. **Dead node detection:** nodes with no path from any input node (via enabled connections) are marked dead and excluded from `eval_order_`. They are not evaluated.
6. Build an **ID-to-index mapping** (`std::unordered_map<uint32_t, uint32_t>`) so that node IDs (which may be sparse after many add_node mutations) map to compact indices into `node_states_` and `node_outputs_`.

**Self-connections** (from_node == to_node) are allowed and are always classified as recurrent. They provide direct self-feedback, which is useful for CTRNN nodes that need to modulate their own state.

**Per-tick forward pass:**

1. **Set input node outputs** to the provided input values (via the ID-to-index mapping).
2. **Evaluate nodes in topological order** (skipping input nodes):
   a. Sum weighted inputs from **feedforward connections**: `weighted_sum = Σ(conn.weight * node_outputs_[index_of(conn.from_node)])` for all enabled, non-recurrent connections targeting this node.
   b. Sum weighted inputs from **recurrent connections**: these read from `node_outputs_` which still holds the **previous tick's** values (not yet overwritten this tick).
   c. Apply node dynamics based on `NodeType`:
      - **Stateless:** `node_outputs_[idx] = activate(weighted_sum + bias)`
      - **CTRNN:** `raw = activate(weighted_sum + bias); node_states_[idx] += (dt / tau) * (-node_states_[idx] + raw); node_outputs_[idx] = node_states_[idx]`
3. **Collect output node values** (in node-ID order) and return them.

**Update model:** This is an **explicit Euler, sequential update**. Nodes are evaluated in topological order, so a downstream node sees the *current tick's* output from an upstream feedforward node, but the *previous tick's* output from a recurrent connection. This is the standard approach for NEAT implementations and is simpler than simultaneous (synchronous) update.

**Input size mismatch:** If the input span size does not match the number of Input nodes, `forward()` throws `std::invalid_argument`.

#### 2.5 dt Configuration

The timestep `dt` is set at construction time (defaults to 1.0) and can be overridden per-`forward()` call for variable-timestep simulations:

```cpp
explicit GraphNetwork(const GraphGenome& genome, float dt = 1.0f);
[[nodiscard]] std::vector<float> forward(std::span<const float> input, float dt_override);
```

Most users will leave `dt = 1.0` and let τ values alone control timescale.

### 3. NEAT Evolution Operators

These go in `libs/evolve` since they are evolutionary algorithms, not network evaluation. This means `libs/evolve` depends on `libs/neuralnet` for the `GraphGenome`, `NodeGene`, and `ConnectionGene` types. This dependency direction (evolve → neuralnet) is intentional and matches the existing pattern where `evolve` needs to know about network structure to evolve it.

#### 3.1 Innovation Counter

A global counter that assigns unique innovation numbers to new connections. This must be shared across all mutations within a generation to ensure that the same structural mutation in different genomes gets the same innovation number.

```cpp
class InnovationCounter {
public:
    /// Get or assign an innovation number for a connection between two nodes.
    /// If this (from, to) pair was already created this generation, returns the same number.
    uint32_t get_or_create(uint32_t from_node, uint32_t to_node);

    /// Call at the start of each generation to allow re-use tracking.
    void new_generation();

private:
    uint32_t next_innovation_ = 0;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> current_generation_;
};
```

#### 3.2 Mutation Operators

```cpp
struct NeatMutationConfig {
    // Weight mutation
    float weight_mutate_rate    = 0.80f;  // Probability of mutating each weight
    float weight_perturb_rate   = 0.90f;  // Of mutated weights: probability of perturb vs. replace
    float weight_perturb_strength = 0.3f; // Gaussian noise std dev for perturbation
    float weight_replace_range  = 2.0f;   // Uniform range for replacement [-range, range]

    // Bias mutation (same structure as weight)
    float bias_mutate_rate      = 0.40f;
    float bias_perturb_strength = 0.2f;

    // Tau mutation (CTRNN nodes only)
    float tau_mutate_rate       = 0.20f;
    float tau_perturb_strength  = 0.1f;
    float tau_min               = 0.1f;
    float tau_max               = 100.0f;

    // Structural mutation
    float add_connection_rate   = 0.10f;  // Probability of adding a new connection
    float add_node_rate         = 0.03f;  // Probability of adding a new node (splits a connection)
    float disable_connection_rate = 0.02f;

    // Node type mutation
    float node_type_mutate_rate = 0.05f;  // Probability of changing a hidden node's NodeType

    // Activation mutation
    float activation_mutate_rate = 0.05f; // Probability of changing a hidden node's activation
};
```

**Mutation operations:**

- **Mutate weights:** For each connection, with probability `weight_mutate_rate`, either perturb (add Gaussian noise) or replace (uniform random).
- **Mutate biases:** Same pattern as weights, for hidden and output node biases.
- **Mutate tau:** For CTRNN nodes, perturb τ, clamped to `[tau_min, tau_max]`.
- **Add connection:** Pick two unconnected nodes. **Recurrent connections are allowed** — the mutation can create connections that form cycles, including from output nodes back to hidden nodes or self-connections. The only hard constraints are: (1) connections cannot target input nodes (they are pass-through), (2) no duplicate connections (same from/to pair). Create the connection with a small random weight and a new innovation number.
- **Add node:** Pick a random enabled connection. Disable it. Create a new hidden node with: `NodeType::Stateless`, a random activation function, `bias = 0.0`, `tau = 1.0` (inert default). Add two new connections: (old_from → new_node, weight 1.0) and (new_node → old_to, old_weight). This preserves the network's existing behavior initially. The new node starts simple; type/activation mutations can specialize it later.
- **Disable connection:** Pick a random enabled connection and disable it.
- **Mutate node type:** Change a hidden node's `NodeType` (e.g., Stateless ↔ CTRNN). When switching to CTRNN, initialize τ randomly in `[tau_min, tau_max]`. When switching to Stateless, τ is ignored and state is cleared.
- **Mutate activation:** Change a hidden node's activation function to a random one from the available set.

#### 3.3 Crossover

NEAT crossover aligns parent genomes by innovation number:

```cpp
GraphGenome crossover(const GraphGenome& fitter_parent,
                      const GraphGenome& other_parent,
                      std::mt19937& rng);
```

1. **Matching genes** (same innovation number in both parents): randomly inherit from either parent.
2. **Disjoint/excess genes** (in one parent but not the other): inherit from the **fitter parent** only.
3. **Disabled gene inheritance:** If a matching gene is disabled in either parent, there is a 75% chance it remains disabled in the child (standard NEAT convention). This prevents re-enabling connections that were selected against, while leaving some chance for recovery.
4. **Node genes:** For nodes present in both parents, properties (type, activation, bias, tau) are randomly inherited from either parent with equal probability, independent of which parent's connections were chosen. Nodes only present in the fitter parent are included with that parent's properties.
5. **Orphan cleanup:** After crossover, any node that has no incoming AND no outgoing enabled connections (and is not an input/output node) is removed from the child genome. Any disabled connections referencing a removed node are also removed to prevent dangling references.

If fitness is equal, disjoint/excess genes are inherited from both parents randomly.

#### 3.4 Speciation

Species group genetically similar networks to protect innovation. New topologies get time to optimize before competing with established ones.

```cpp
struct SpeciationConfig {
    float compatibility_threshold = 3.0f;
    float c_excess    = 1.0f;   // Coefficient for excess genes
    float c_disjoint  = 1.0f;   // Coefficient for disjoint genes
    float c_weight    = 0.4f;   // Coefficient for weight differences in matching genes
};
```

**Compatibility distance:**

```
δ = (c_excess * E / N) + (c_disjoint * D / N) + (c_weight * W̄)
```

Where E = excess genes, D = disjoint genes, N = number of genes in the larger genome (set N=1 if both genomes have fewer than 20 genes, to avoid penalizing small genomes), W̄ = average weight difference of matching genes.

Two genomes are in the same species if δ < threshold.

#### 3.5 NeatPopulation

Manages a population of `GraphGenome` individuals organized into species. Handles the full generation lifecycle.

```cpp
struct NeatIndividual {
    GraphGenome genome;
    float fitness = 0.0f;
    uint32_t species_id = 0;       // Assigned by speciation
};

struct NeatPopulationConfig {
    std::size_t population_size = 150;
    NeatMutationConfig mutation;
    SpeciationConfig speciation;
    std::size_t elitism_per_species = 1;      // Top N per species pass unchanged
    float interspecies_mate_rate = 0.001f;     // Rare cross-species crossover
    std::size_t stagnation_limit = 15;         // Generations without improvement before species is eliminated
    std::size_t min_species_to_keep = 2;       // Never eliminate below this many species
};

class NeatPopulation {
public:
    /// Create initial population with minimal genomes.
    NeatPopulation(
        std::size_t num_inputs,
        std::size_t num_outputs,
        NodeType default_output_type,
        Activation default_output_activation,
        const NeatPopulationConfig& config,
        std::mt19937& rng
    );

    /// Access individuals for fitness evaluation.
    [[nodiscard]] std::vector<NeatIndividual>& individuals();
    [[nodiscard]] const std::vector<NeatIndividual>& individuals() const;

    /// Run one generation: speciate, compute adjusted fitness, reproduce, mutate.
    void evolve(std::mt19937& rng);

    /// Current generation number.
    [[nodiscard]] std::size_t generation() const noexcept;

    /// Number of current species.
    [[nodiscard]] std::size_t num_species() const noexcept;

    /// Access the innovation counter (needed if caller does manual mutations).
    [[nodiscard]] InnovationCounter& innovation_counter() noexcept;

private:
    NeatPopulationConfig config_;
    std::vector<NeatIndividual> individuals_;
    InnovationCounter innovation_counter_;
    std::size_t generation_ = 0;
    // Species tracking: species ID → (representative genome, best fitness, stagnation count)
};
```

**Generation lifecycle (`evolve()`):**

1. **Speciate:** Assign each individual to a species by comparing it to each species' representative genome using the compatibility distance. If no species is compatible, create a new species. Update each species' representative to a random member.
2. **Stagnation check:** For each species, track whether best fitness improved this generation. If a species has not improved for `stagnation_limit` generations, eliminate it (unless doing so would reduce species count below `min_species_to_keep`).
3. **Adjusted fitness:** Each individual's adjusted fitness = `fitness / species_size`. This prevents large species from dominating and gives small innovative species room to grow.
4. **Offspring allocation:** Each species gets offspring proportional to its total adjusted fitness. Minimum 1 offspring per surviving species.
5. **Reproduction:** Within each species:
   - Top `elitism_per_species` individuals pass unchanged.
   - Remaining slots: select two parents via tournament within the species (or rarely cross-species at `interspecies_mate_rate`), crossover, then mutate.
6. **Replace:** The new individuals become the population. Innovation counter calls `new_generation()`.

**Threading model:** `evolve()` is single-threaded. The typical usage pattern is: (1) evaluate all individuals' fitness in parallel (caller's responsibility), (2) call `evolve()` sequentially. The `InnovationCounter` is not thread-safe and does not need to be, since all mutations happen within `evolve()`.

#### 3.6 Minimal Starting Genome

A NEAT population starts **minimal** — input nodes directly connected to output nodes with random weights, no hidden nodes. Complexity grows from there.

```cpp
GraphGenome create_minimal_genome(
    std::size_t num_inputs,
    std::size_t num_outputs,
    NodeType default_output_type,        // Stateless or CTRNN for output nodes
    Activation default_output_activation, // e.g., Tanh
    InnovationCounter& counter,
    std::mt19937& rng
);
```

**Node ID convention:** Input nodes get IDs `0` through `num_inputs - 1`. Output nodes get IDs `num_inputs` through `num_inputs + num_outputs - 1`. Hidden nodes (added by mutation) get IDs starting at `num_inputs + num_outputs` and incrementing. This convention makes it easy to identify node roles from IDs alone, though the `NodeRole` field is authoritative.

### 4. Serialization

#### 4.1 Versioned Format

All neuralnet serialization gets a version header:

```
[magic: 4 bytes "NNPK"]
[version: uint16]
[feature_flags: uint32]
[format_type: uint8]    // 0 = MLP, 1 = GraphNetwork
[payload...]
```

**Feature flags** (bitmask):
- Bit 0: Has CTRNN nodes (node states need save/restore)
- Bit 1: Has innovation numbers
- Bit 2-31: Reserved for future features

This allows a deserializer to know what's in the file before reading it, and to reject files with unsupported features rather than corrupting data.

#### 4.2 GraphNetwork Payload

```
[num_nodes: uint32]
[num_connections: uint32]
[dt: float32]
For each node:
    [id: uint32] [role: uint8] [type: uint8] [activation: uint8] [bias: float32] [tau: float32]
For each connection:
    [from: uint32] [to: uint32] [weight: float32] [enabled: uint8] [innovation: uint32]
Optional (if feature flag bit 0 set):
    [node_states: num_nodes * float32]   // For resuming mid-episode
```

#### 4.3 Backward Compatibility

Existing MLP serialization (`serialize`/`deserialize` in serialization.h) currently has no version header. Migration plan:

1. The new versioned format is used for **all new saves** (both MLP and GraphNetwork).
2. The deserializer detects old-format files (no "NNPK" magic) and falls back to the legacy MLP parser.
3. Legacy `serialize`/`deserialize` functions are kept but deprecated. New code uses versioned API.

```cpp
// New versioned API
void save(const GraphNetwork& net, std::ostream& out);
void save(const Network& net, std::ostream& out);  // MLP, new versioned format

std::variant<Network, GraphNetwork> load(std::istream& in);  // Auto-detects format

// Legacy (deprecated, kept for backward compat)
void serialize(const Network& net, std::ostream& out);
Network deserialize(std::istream& in);
```

### 5. Library Structure Changes

#### 5.1 New Files

```
libs/neuralnet/
├── include/neuralnet/
│   ├── activation.h          (existing — add future activations here)
│   ├── layer.h               (existing, unchanged)
│   ├── network.h             (existing, unchanged — this is the MLP)
│   ├── serialization.h       (existing — add versioned API, deprecate old)
│   ├── node_types.h          (NEW — NodeType enum, NodeRole enum, NodeGene struct)
│   ├── graph_genome.h        (NEW — GraphGenome, ConnectionGene)
│   └── graph_network.h       (NEW — GraphNetwork class)
├── src/
│   ├── activation.cpp        (existing, unchanged)
│   ├── layer.cpp             (existing, unchanged)
│   ├── network.cpp           (existing, unchanged)
│   ├── serialization.cpp     (existing — add versioned format support)
│   ├── graph_network.cpp     (NEW — GraphNetwork forward pass, topological sort)
│   └── graph_genome.cpp      (NEW — genome utilities)
└── tests/
    ├── network_test.cpp      (existing, unchanged)
    ├── graph_network_test.cpp (NEW)
    ├── node_types_test.cpp   (NEW)
    └── serialization_test.cpp (NEW — test versioned format + backward compat)
```

#### 5.2 Evolve Library Changes

```
libs/evolve/
├── include/evolve/
│   ├── genome.h              (existing — MLP genome, unchanged)
│   ├── population.h          (existing — MLP population, unchanged)
│   ├── neat_operators.h      (NEW — mutation, crossover, speciation)
│   ├── innovation.h          (NEW — InnovationCounter)
│   └── neat_population.h     (NEW — NEAT population management with species)
├── src/
│   ├── genome.cpp            (existing, unchanged)
│   ├── population.cpp        (existing, unchanged)
│   ├── neat_operators.cpp    (NEW)
│   ├── innovation.cpp        (NEW)
│   └── neat_population.cpp   (NEW)
└── tests/
    ├── neat_operators_test.cpp (NEW)
    └── neat_population_test.cpp (NEW)
```

### 6. How AntSim Consumes This

AntSim does **not** modify or extend the library. It configures and uses it:

```cpp
// AntSim creates a minimal starting genome for its colony brain
auto genome = neuralnet::create_minimal_genome(
    50,                              // ~50 ant sensor inputs
    16,                              // ~16 ant action outputs
    neuralnet::NodeType::CTRNN,      // Output nodes are CTRNN (for smooth actions)
    neuralnet::Activation::Tanh,     // Tanh for output range [-1, 1]
    innovation_counter,
    rng
);

// AntSim builds a network from the genome
auto brain = neuralnet::GraphNetwork(genome);

// Each tick, AntSim assembles inputs and reads outputs
std::vector<float> sensory_input = gather_ant_senses(ant, world);
auto actions = brain.forward(sensory_input);
// AntSim knows that actions[0] is "turn", actions[7] is "emit pheromone channel 3", etc.
// The library has no idea — it just computed numbers.
```

AntSim assigns meaning to inputs/outputs. The library provides the computation.

### 7. Future Work (Out of Scope)

These are acknowledged future capabilities, not part of this implementation:

- **ModularNetwork type** — dense modules with sparse inter-module connections. Potential target for NEAT→Modular converter.
- **Network converter** — translate a trained GraphNetwork into an optimized ModularNetwork or MLP.
- **Additional node types** — accumulator nodes, gated memory cells, etc. The NodeType enum is designed for extension.
- **Additional activation functions** — softmax, GELU, etc. Added to the existing Activation enum as needed.
- **Population-level serialization** — saving/loading entire NEAT populations with species information. Note: `InnovationCounter` state would also need serialization to correctly resume training.
- **`forward_into()` overload** — pre-allocated output buffer to avoid per-call `std::vector` allocation, relevant when calling `forward()` thousands of times per tick (e.g., one per ant).
- **Simultaneous (synchronous) node update mode** — alternative to the sequential update where all nodes read from previous tick's state. Different dynamics, sometimes preferred for CTRNNs.
