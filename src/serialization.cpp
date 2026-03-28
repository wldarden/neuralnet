#include <neuralnet/serialization.h>

#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <variant>

namespace neuralnet {

namespace {

constexpr uint32_t MAGIC = 0x4E4E4554;  // "NNET"

constexpr uint32_t VERSIONED_MAGIC = 0x4E4E504B;  // "NNPK"
constexpr uint16_t FORMAT_VERSION = 1;
enum FormatType : uint8_t { FORMAT_MLP = 0, FORMAT_GRAPH = 1 };
enum FeatureFlags : uint32_t { FEATURE_CTRNN = 1 << 0, FEATURE_INNOVATION = 1 << 1 };

template <typename T>
void write_val(std::ostream& out, T val) {
    if (!out.write(reinterpret_cast<const char*>(&val), sizeof(T))) {
        throw std::runtime_error("Failed to write to stream");
    }
}

template <typename T>
T read_val(std::istream& in) {
    T val;
    if (!in.read(reinterpret_cast<char*>(&val), sizeof(T))) {
        throw std::runtime_error("Unexpected end of stream");
    }
    return val;
}

// Shared MLP body parser — used by both legacy and versioned formats.
// Assumes the magic/header bytes have already been consumed.
Network parse_mlp_body(std::istream& in) {
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

GraphNetwork load_versioned_graph(std::istream& in, uint32_t features) {
    // Read node count and connection count
    auto num_nodes = read_val<uint32_t>(in);
    auto num_connections = read_val<uint32_t>(in);

    // Read nodes
    NeuralGenome genome;
    genome.nodes.resize(num_nodes);
    for (auto& node : genome.nodes) {
        node.id = read_val<uint32_t>(in);
        node.role = static_cast<evolve::NodeRole>(read_val<uint8_t>(in));
        node.props.type = static_cast<NodeType>(read_val<uint8_t>(in));
        node.props.activation = static_cast<Activation>(read_val<uint8_t>(in));
        node.props.bias = read_val<float>(in);
        node.props.tau = read_val<float>(in);
    }

    // Read connections
    genome.connections.resize(num_connections);
    for (auto& conn : genome.connections) {
        conn.from_node = read_val<uint32_t>(in);
        conn.to_node = read_val<uint32_t>(in);
        conn.weight = read_val<float>(in);
        conn.enabled = read_val<uint8_t>(in) != 0;
        conn.innovation = read_val<uint32_t>(in);
    }

    // Construct network from genome
    GraphNetwork net(genome);

    // Restore node states if CTRNN flag is set
    if (features & FEATURE_CTRNN) {
        auto state_count = read_val<uint32_t>(in);
        std::vector<float> states(state_count);
        if (state_count > 0) {
            if (!in.read(reinterpret_cast<char*>(states.data()),
                         static_cast<std::streamsize>(state_count * sizeof(float)))) {
                throw std::runtime_error("Unexpected end of stream reading node states");
            }
        }
        net.set_node_states(states);
    }

    return net;
}

} // namespace

// === Legacy API ===

void serialize(const Network& net, std::ostream& out) {
    const auto& topo = net.topology();

    write_val(out, MAGIC);
    write_val<uint32_t>(out, static_cast<uint32_t>(topo.input_size));
    write_val<uint32_t>(out, static_cast<uint32_t>(topo.layers.size()));

    for (const auto& layer_def : topo.layers) {
        write_val<uint32_t>(out, static_cast<uint32_t>(layer_def.output_size));
        write_val<uint32_t>(out, static_cast<uint32_t>(layer_def.activation));
    }

    auto weights = net.get_all_weights();
    write_val<uint32_t>(out, static_cast<uint32_t>(weights.size()));
    if (!out.write(reinterpret_cast<const char*>(weights.data()),
                   static_cast<std::streamsize>(weights.size() * sizeof(float)))) {
        throw std::runtime_error("Failed to write weights to stream");
    }
}

Network deserialize(std::istream& in) {
    auto magic = read_val<uint32_t>(in);
    if (magic != MAGIC) {
        throw std::runtime_error("Invalid network file: bad magic number");
    }
    return parse_mlp_body(in);
}

// === Versioned API ===

void save(const GraphNetwork& net, std::ostream& out) {
    const auto& genome = net.genome();

    // Determine feature flags
    uint32_t features = FEATURE_INNOVATION;
    bool has_ctrnn = false;
    for (const auto& node : genome.nodes) {
        if (node.props.type == NodeType::CTRNN) {
            has_ctrnn = true;
            break;
        }
    }
    if (has_ctrnn) {
        features |= FEATURE_CTRNN;
    }

    // Write header
    write_val(out, VERSIONED_MAGIC);
    write_val(out, FORMAT_VERSION);
    write_val(out, features);
    write_val<uint8_t>(out, FORMAT_GRAPH);

    // Write node count and connection count
    write_val<uint32_t>(out, static_cast<uint32_t>(genome.nodes.size()));
    write_val<uint32_t>(out, static_cast<uint32_t>(genome.connections.size()));

    // Write nodes
    for (const auto& node : genome.nodes) {
        write_val(out, node.id);
        write_val(out, static_cast<uint8_t>(node.role));
        write_val(out, static_cast<uint8_t>(node.props.type));
        write_val(out, static_cast<uint8_t>(node.props.activation));
        write_val(out, node.props.bias);
        write_val(out, node.props.tau);
    }

    // Write connections
    for (const auto& conn : genome.connections) {
        write_val(out, conn.from_node);
        write_val(out, conn.to_node);
        write_val(out, conn.weight);
        write_val<uint8_t>(out, conn.enabled ? 1 : 0);
        write_val(out, conn.innovation);
    }

    // Write node states if CTRNN
    if (has_ctrnn) {
        auto states = net.get_node_states();
        write_val<uint32_t>(out, static_cast<uint32_t>(states.size()));
        if (!out.write(reinterpret_cast<const char*>(states.data()),
                       static_cast<std::streamsize>(states.size() * sizeof(float)))) {
            throw std::runtime_error("Failed to write node states to stream");
        }
    }
}

void save(const Network& net, std::ostream& out) {
    const auto& topo = net.topology();

    // Write versioned header
    write_val(out, VERSIONED_MAGIC);
    write_val(out, FORMAT_VERSION);
    write_val<uint32_t>(out, 0);  // no feature flags for MLP
    write_val<uint8_t>(out, FORMAT_MLP);

    // Write MLP payload (same layout as legacy body)
    write_val<uint32_t>(out, static_cast<uint32_t>(topo.input_size));
    write_val<uint32_t>(out, static_cast<uint32_t>(topo.layers.size()));

    for (const auto& layer_def : topo.layers) {
        write_val<uint32_t>(out, static_cast<uint32_t>(layer_def.output_size));
        write_val<uint32_t>(out, static_cast<uint32_t>(layer_def.activation));
    }

    auto weights = net.get_all_weights();
    write_val<uint32_t>(out, static_cast<uint32_t>(weights.size()));
    if (!out.write(reinterpret_cast<const char*>(weights.data()),
                   static_cast<std::streamsize>(weights.size() * sizeof(float)))) {
        throw std::runtime_error("Failed to write weights to stream");
    }
}

std::variant<Network, GraphNetwork> load(std::istream& in) {
    auto magic = read_val<uint32_t>(in);

    if (magic == MAGIC) {
        return parse_mlp_body(in);
    }

    if (magic != VERSIONED_MAGIC) {
        throw std::runtime_error("Invalid network file: unrecognized magic number");
    }

    auto version = read_val<uint16_t>(in);
    if (version != FORMAT_VERSION) {
        throw std::runtime_error("Unsupported format version: " + std::to_string(version));
    }

    auto features = read_val<uint32_t>(in);
    auto type = read_val<uint8_t>(in);

    if (type == FORMAT_MLP) {
        return parse_mlp_body(in);
    }
    if (type == FORMAT_GRAPH) {
        return load_versioned_graph(in, features);
    }

    throw std::runtime_error("Unknown format type: " + std::to_string(type));
}

} // namespace neuralnet
