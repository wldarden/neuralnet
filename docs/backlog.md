# Neuralnet Library Backlog

## Planned Features

### 1. More Activation Functions
**Priority: High | Effort: Low**

Currently only ReLU, Sigmoid, Tanh. Add:
- **Linear / Identity** — pass-through for skip connections and value forwarding. Without it, every signal gets squished through a nonlinearity.
- **Gaussian** — `exp(-x²)`. Radial basis function behavior. Nets can learn "fire when input is *near* a value" instead of only "above a threshold." Good for spatial reasoning.
- **Sine** — periodic activation. SIREN networks can represent complex periodic patterns with very few nodes. Enables oscillating behaviors (rhythmic dodging, timed shooting, cyclic pheromone patterns).
- **Abs** — `|x|`. Detects magnitude regardless of sign.

Implementation: add to `Activation` enum, add cases to `activate()` in `activation.cpp`. Update `mutate_activations` in neuralnet's neural_neat_policy to include new options. Update serialization if enum values change.

### 2. Connection Delay
**Priority: High | Effort: Medium**

Currently all connections transmit instantly. Add an evolved `delay` parameter (integer ticks) per connection. Signals arrive at different times, enabling temporal pattern recognition — the net can learn rhythms, sequences, and timing without relying solely on CTRNN nodes.

Implementation: ring buffer per delayed connection. During forward pass, write output to buffer; read from buffer[current_tick - delay]. Default delay=0 (instant, backward compat). Delay is an evolvable gene on each connection.

### 3. Gated Nodes (LSTM-like / GRU-like)
**Priority: High | Effort: Medium-High**

CTRNN provides temporal memory via a leaky integrator (single gate). Gated nodes add explicit memory management:
- **Forget gate** — learn to explicitly clear memory
- **Input gate** — learn when to store new information
- **Output gate** — learn when to emit stored information

This is "memory slowly decays" (CTRNN) vs "memory is actively managed" (gated). Ants could learn "I found food here" then explicitly forget when the food is gone. Flyers could learn "I'm in a dodge sequence" and hold that state until clear.

Implementation: new `NodeType::LSTM` or `NodeType::GRU`. Each gated node has multiple internal weights (gate weights). These become evolvable genes. Forward pass computes gate activations, then applies gated update.

### 4. Multiplicative / Attention Connections
**Priority: Medium | Effort: Medium**

Normal connections: `weight * input`. Multiplicative connections: `weight * input_a * input_b`. Two inputs interact — the basis of attention. A node could learn "pay attention to sensor X only when sensor Y is active."

Without this, nets need extra hidden layers to approximate multiplication. With it, conditional behavior is cheap.

Implementation: new connection type that references two source nodes instead of one. Evolved like normal connections but with two source IDs. Forward pass multiplies the two source outputs before applying weight.

### 5. Lateral Inhibition / Competitive Learning
**Priority: Medium | Effort: Medium**

Nodes in the same layer suppress each other — when one fires strongly, neighbors are inhibited. Creates specialization: different nodes naturally learn different input patterns instead of all converging on the same thing.

Implementation: optional per-layer inhibition parameter. After computing layer activations, apply soft winner-take-all: scale each activation by how much it dominates its neighbors. Evolved inhibition strength per layer.

### 6. Dendritic Computation
**Priority: Medium | Effort: High**

Instead of one weight per connection, each connection has a small nonlinear function (a "dendritic segment"). The node sums across dendrites, not across individual inputs. Dramatically increases computational capacity per node while keeping the network small.

Implementation: each connection has a mini-activation (e.g., threshold + gain), applied before the node's main activation. Evolved per-connection parameters. The node sums dendritic outputs rather than raw weighted inputs.

### 7. Hypernetworks (Network of Networks)
**Priority: Low-Medium | Effort: High**

One network generates the weights for another. The "weight-generating" network evolves; the "execution" network is produced from it. Compresses the search space — instead of evolving 1000 weights, evolve a 50-weight hypernetwork that produces 1000 weights with structured patterns.

Implementation: HyperNetwork class wrapping a small Network that outputs a flat weight vector, which is used to construct the execution Network. Evolution operates on the hypernetwork's weights only.

---

## Ideas (Not Planned, But Interesting)

### Neuromodulation (Hebbian-like Lifetime Learning)
Weights modify during an individual's lifetime based on input/output correlations:
```
Δw = η * (A*pre*post + B*pre + C*post + D)
```
Where A, B, C, D, η are evolved. Individuals can learn within their lifetime — adapting without waiting for the next generation. Biological basis: Hebbian learning.

Deprioritized because: adds significant complexity to the forward pass and gene representation. Valuable but the gated nodes (item 3) cover much of the same ground (explicit memory management) with less complexity.

### STDP (Spike-Timing-Dependent Plasticity)
Connections strengthen when pre fires just before post, weaken when post fires first. Enables causal learning. Requires a spiking neuron model which is a bigger departure from the current rate-coded architecture.

### Reward-Modulated Plasticity
Weights only update when a reward signal is present. Combines evolution with reinforcement learning. Requires a reward signal pathway which doesn't exist in the current architecture.

### GELU / Swish Activations
Smooth approximations to ReLU used in modern transformers. Better gradient flow. Less relevant for neuroevolution (no gradients) but could help if we ever add gradient-based fine-tuning.

### Step / Binary Activation
Hard threshold. Some evolved behaviors benefit from crisp on/off decisions. Could be approximated with steep Sigmoid — may not need a dedicated type.

### Self-Organizing Maps
Unsupervised spatial organization of nodes. More relevant for data analysis than for evolved agents. Could be interesting for ant colony spatial memory.
