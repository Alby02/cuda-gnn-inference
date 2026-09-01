#pragma once

#include "../data/graph_csc.hpp"
#include "../gnn/model.hpp"
#include "executor.hpp"

#include <stdexcept>
#include <utility>
#include <variant>

namespace gnn {

template <Executor E>
class InferenceRuntime { // TODO: Check maybe move the model to the constructor and make it work for
                         // CUDA (curenty not working because the model data is in ram and not in
                         // GPU memory)
public:
    using ExecutorType = E;
    using WorkspaceType = typename ExecutorType::WorkspaceType;
    using BufferType = typename WorkspaceType::BufferType;

    explicit InferenceRuntime(ExecutorType executor = {}) : executor_(std::move(executor)) {}

    template <Layer... Layers>
    [[nodiscard]] BufferType run(const graph::GraphCSC& graph, const Model<Layers...>& model,
                                 BufferType input) {
        validate(graph, model, input);
        WorkspaceType workspace{std::move(input)};
        return executeModel(graph, model, workspace);
    }

private:
    template <Layer... Layers>
    [[nodiscard]] BufferType executeModel(const graph::GraphCSC& graph,
                                          const Model<Layers...>& model, WorkspaceType& workspace) {
        for (const auto& layer : model.getLayers()) {
            std::visit(
                [&](const auto& concreteLayer) {
                    forward_layer(concreteLayer, graph, executor_, workspace);
                },
                layer);

            workspace.swapBuffers();
        }

        return workspace.current();
    }

    template <Layer... Layers>
    static void validate(const graph::GraphCSC& graph, const Model<Layers...>& model,
                         const BufferType& input) {
        if (model.empty()) {
            throw std::invalid_argument("Inference model cannot be empty.");
        }
        if (graph.getNumNodes() != input.rows()) {
            throw std::invalid_argument("Graph node count must match the input row count.");
        }
        if (model.getInputDim() != input.cols()) {
            throw std::invalid_argument("Model input dimension must match the input column count.");
        }
    }

    ExecutorType executor_;
};

} // namespace gnn
