#include <neuralnet/activation.h>

#include <cmath>

namespace neuralnet {

float activate(Activation func, float x) noexcept {
    switch (func) {
    case Activation::ReLU:
        return x > 0.0f ? x : 0.0f;
    case Activation::Sigmoid:
        return 1.0f / (1.0f + std::exp(-x));
    case Activation::Tanh:
        return std::tanh(x);
    case Activation::Linear:
        return x;
    case Activation::Gaussian:
        return std::exp(-x * x);
    case Activation::Sine:
        return std::sin(x);
    case Activation::Abs:
        return std::abs(x);
    }
    return x; // unreachable, silences warning
}

void activate_inplace(Activation func, std::vector<float>& values) noexcept {
    for (auto& v : values) {
        v = activate(func, v);
    }
}

} // namespace neuralnet
