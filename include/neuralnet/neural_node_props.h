#pragma once
#include <neuralnet/activation.h>
#include <neuralnet/node_types.h>
#include <evolve/graph_gene.h>
#include <evolve/neat_population.h>

namespace neuralnet {

struct NeuralNodeProps {
    Activation activation = Activation::ReLU;
    NodeType type = NodeType::Stateless;
    float bias = 0.0f;
    float tau = 1.0f;
};

using NeuralGenome     = evolve::GraphGenome<NeuralNodeProps>;
using NeuralNodeGene   = evolve::NodeGene<NeuralNodeProps>;
using NeuralIndividual = evolve::NeatIndividual<NeuralNodeProps>;

} // namespace neuralnet
