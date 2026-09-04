#pragma once

#include "../data/graph_csc.hpp"
#include "../gnn/model.hpp"
#include "executor.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <variant>

namespace gnn {

template <Executor E> class InferenceRuntime {
public:
    using ExecutorType = E;
    using WorkspaceType = typename ExecutorType::WorkspaceType;
    using BufferType = typename WorkspaceType::BufferType;

    explicit InferenceRuntime(ExecutorType executor = {}) : executor_(std::move(executor)) {}

    template <typename Graph, Layer... Layers>
    [[nodiscard]] BufferType run(const Graph& graph, const Model<Layers...>& model,
                                 BufferType input) {
        validate(graph, model, input);
        WorkspaceType workspace{std::move(input), maximumFeatureWidth(model)};
        return std::move(executeModel(graph, model, workspace));
    }

    template <typename Graph, Layer... Layers>
    [[nodiscard]] BufferType run(const Graph& graph, const Model<Layers...>& model,
                                 const BufferType& input, WorkspaceType& workspace) {
        validate(graph, model, input);
        workspace.prepare(input, maximumFeatureWidth(model));
        return executeModel(graph, model, workspace);
    }

private:
    template <typename Graph, Layer... Layers>
    [[nodiscard]] BufferType& executeModel(const Graph& graph, const Model<Layers...>& model,
                                           WorkspaceType& workspace) {
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

    template <typename Graph, Layer... Layers>
    static void validate(const Graph& graph, const Model<Layers...>& model,
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

    template <Layer... Layers>
    static std::size_t maximumFeatureWidth(const Model<Layers...>& model) {
        std::size_t width = model.getInputDim();
        for (const auto& layer : model.getLayers()) {
            width = std::max(
                width,
                std::visit([](const auto& concreteLayer) { return concreteLayer.getOutDim(); },
                           layer));
        }
        return width;
    }

    ExecutorType executor_;
};

} // namespace gnn
