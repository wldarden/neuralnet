#pragma once
#include <cstdint>

namespace neuralnet {

enum class NodeType : uint8_t {
    Stateless = 0,
    CTRNN     = 1,
};

} // namespace neuralnet
