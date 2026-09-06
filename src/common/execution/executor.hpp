#pragma once

#include "../gnn/layer.hpp"
#include "workspace.hpp"

#include <concepts>
#include <random>
#include <vector>
namespace gnn {
namespace layers {
enum class GraphSAGEAggregationType;
}
template <typename E>
concept Executor =
    requires {
        typename E::WorkspaceType;
        typename E::WeightType;
        typename E::BiasType;
    } && Workspace<typename E::WorkspaceType> &&
    requires(E& executor, typename E::WorkspaceType& workspace,
             const typename E::WeightType& weights, const typename E::BiasType& bias,
             gnn::layers::GraphSAGEAggregationType aggType) {
        {
            executor.rowByColumn(workspace.current(), weights, workspace.next())
        } -> std::same_as<void>;
        { executor.add(workspace.current(), weights, workspace.next()) } -> std::same_as<void>;
        { executor.biasAdd(workspace.next(), bias) } -> std::same_as<void>;
        { executor.relu(workspace.next()) } -> std::same_as<void>;
        {
            executor.aggregateNeighbors(workspace.getGraph(), workspace.current(),
                                        workspace.scratch(), aggType)
        } -> std::same_as<void>;
    };

} // namespace gnn
