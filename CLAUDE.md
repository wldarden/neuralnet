# neuralnet — Neural Network Library

- **GitHub:** [wldarden/neuralnet](https://github.com/wldarden/neuralnet)
- **Dependencies:** [evolve](https://github.com/wldarden/evolve) (expected at `../evolve/`)
- **Used by:** NeuroFlyer, AntSim, EcoSim

Shared library providing neural network types for neuroevolution projects. Supports two first-class network types with different trade-offs. Also provides neural-specific NEAT integration types (`NeuralNodeProps`, `NeuralGenome`, `NeuralIndividual`) and a `NeatPolicy` implementation for use with `evolve::NeatPopulation<NeuralNodeProps>`.

## Network Types

### MLP (`Network`)
Dense feedforward layers. Simple, fast, well-understood. Used by NeuroFlyer.

```cpp
neuralnet::NetworkTopology topo;
topo.input_size = 4;
topo.layers = {
    {.output_size = 8, .activation = neuralnet::Activation::ReLU},
    {.output_size = 2, .activation = neuralnet::Activation::Tanh},
};
std::vector<float> weights(total_weight_count, 0.1f);
auto net = neuralnet::Network(topo, weights);
auto output = net.forward(input);
```

### GraphNetwork (NEAT-style)
Arbitrary graph topology with per-node properties. Supports recurrent connections and CTRNN dynamics. Evolvable topology via NEAT operators in `libs/evolve`. Used by AntSim.

```cpp
// NeuralGenome is an alias for evolve::GraphGenome<NeuralNodeProps>
neuralnet::NeuralGenome genome = neuralnet::create_minimal_genome(
    num_inputs, num_outputs, rng);
auto net = neuralnet::GraphNetwork(genome);
auto output = net.forward(input);
```

## Neural NEAT Integration

`neuralnet` provides the node property type, convenience aliases, and a `NeatPolicy` implementation for neural-net-specific NEAT evolution:

```cpp
// Create a policy with default neural mutation config
auto policy = neuralnet::make_neural_neat_policy(neuralnet::NeuralMutationConfig{});

evolve::NeatPopulation<neuralnet::NeuralNodeProps> pop(
    num_inputs, num_outputs, config, policy, rng);

for (neuralnet::NeuralIndividual& ind : pop.individuals()) {
    neuralnet::GraphNetwork net(ind.genome);
    ind.fitness = evaluate(net);
}
pop.evolve(rng);
```

The policy wraps four neural-specific mutation functions that were previously in `evolve`:
- `mutate_biases` — Gaussian noise on non-input node biases
- `mutate_tau` — Gaussian noise on CTRNN time constants, clamped to [tau_min, tau_max]
- `mutate_node_types` — Flip between Stateless and CTRNN
- `mutate_activations` — Random new activation function (ReLU, Sigmoid, Tanh)

## Node Types

Per-neuron dynamics, chosen by the consuming application:

| Type | Behavior | Use case |
|------|----------|----------|
| `Stateless` | `output = activate(sum + bias)` | Reactive, no memory |
| `CTRNN` | Leaky integrator with time constant τ | Temporal memory, smooth transitions |

## Key Headers

| Header | Contents |
|--------|----------|
| `neuralnet/activation.h` | `Activation` enum (ReLU, Sigmoid, Tanh), `activate()` |
| `neuralnet/network.h` | `Network` (MLP), `NetworkTopology`, `LayerDef` |
| `neuralnet/node_types.h` | `NodeType` enum (Stateless, CTRNN) |
| `neuralnet/graph_network.h` | `GraphNetwork` class |
| `neuralnet/neural_node_props.h` | `NeuralNodeProps`, `NeuralGenome`, `NeuralNodeGene`, `NeuralIndividual` |
| `neuralnet/neural_neat_policy.h` | `NeuralMutationConfig`, `make_neural_neat_policy()`, `mutate_biases()`, `mutate_tau()`, `mutate_node_types()`, `mutate_activations()` |
| `neuralnet/serialization.h` | `save()`/`load()` (versioned), `serialize()`/`deserialize()` (legacy) |

## Serialization

Versioned binary format with magic `NNPK`. Auto-detects legacy `NNET` format on load.

```cpp
neuralnet::save(net, output_stream);   // Works for both Network and GraphNetwork
auto loaded = neuralnet::load(input_stream);  // Returns std::variant<Network, GraphNetwork>
```

## Design Principles

- **Library provides abstract building blocks** — no domain-specific concepts (no "pheromone node")
- **Coexistence** — MLP and GraphNetwork are independent types, not a hierarchy
- **Immutable networks** — to evolve, extract genome/weights, mutate, reconstruct
- **Per-neuron properties** — node type, activation, bias, τ are all per-node, not per-layer
- **Decoupled from evolve core** — `NeuralNodeProps` and policy live in neuralnet; `evolve` is a standalone generic library
