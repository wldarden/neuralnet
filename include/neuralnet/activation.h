#pragma once

#include <vector>

namespace neuralnet {

enum class Activation {
    ReLU     = 0,
    Sigmoid  = 1,
    Tanh     = 2,
    Linear   = 3,  // identity pass-through
    Gaussian = 4,  // exp(-x²), radial basis
    Sine     = 5,  // sin(x), periodic
    Abs      = 6,  // |x|, magnitude detection
};

/// Total number of activation functions (for random selection).
inline constexpr int ACTIVATION_COUNT = 7;

/// Apply activation function to a single value.
[[nodiscard]] float activate(Activation func, float x) noexcept;

/// Apply activation function in-place to a vector of values.
void activate_inplace(Activation func, std::vector<float>& values) noexcept;

} // namespace neuralnet
