#pragma once
#include "../gnn/model.hpp"
#include "../host_graph.hpp"
#include "../host_layers.hpp"
#include <algorithm>
#include <stdexcept>

namespace gnn {
using HostModel = Model<layers::HostGCN, layers::HostGraphSAGE>;

struct HostWorkload {
    graph::HostGraphCSC graph;
    layers::HostMatrix input;
    HostModel model;
};

// Existing input/model consistency checks, performed once during workspace setup.
inline void validateWorkload(const HostWorkload& workload) {
    if (workload.model.empty())
        throw std::invalid_argument("Inference model cannot be empty.");
    if (workload.graph.getNumNodes() != workload.input.rows())
        throw std::invalid_argument("Graph node count must match the input row count.");
    if (workload.model.getInputDim() != workload.input.cols())
        throw std::invalid_argument("Model input dimension must match the input column count.");
}
inline std::size_t maximumFeatureWidth(const HostModel& model) {
    std::size_t width = model.getInputDim();
    for (const auto& layer : model.getLayers())
        width = std::max(width, std::visit([](const auto& l) { return l.getOutDim(); }, layer));
    return width;
}
} // namespace gnn
