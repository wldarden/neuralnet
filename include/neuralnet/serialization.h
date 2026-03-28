#pragma once

#include <neuralnet/network.h>
#include <neuralnet/graph_network.h>

#include <iosfwd>
#include <variant>

namespace neuralnet {

// === Versioned API (preferred) ===
void save(const GraphNetwork& net, std::ostream& out);
void save(const Network& net, std::ostream& out);
[[nodiscard]] std::variant<Network, GraphNetwork> load(std::istream& in);

// === Legacy API (deprecated — kept for backward compatibility) ===
void serialize(const Network& net, std::ostream& out);
[[nodiscard]] Network deserialize(std::istream& in);

} // namespace neuralnet
