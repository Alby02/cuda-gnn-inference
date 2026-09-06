#pragma once

#include "../gnn/layer.hpp"
#include "workspace.hpp"

#include <concepts>

namespace gnn {

template <typename E>
concept Executor =
    requires {
        typename E::WorkspaceType;
        typename E::WeightType;
        typename E::BiasType;
    } && Workspace<typename E::WorkspaceType> &&
    requires(E& executor, typename E::WorkspaceType& workspace,
             const typename E::WeightType& weights, const typename E::BiasType& bias) {
        {
            executor.rowByColumn(workspace.current(), weights, workspace.next())
        } -> std::same_as<void>;
        { executor.add(workspace.current(), weights, workspace.next()) } -> std::same_as<void>;
        { executor.biasAdd(workspace.next(), bias) } -> std::same_as<void>;
        { executor.relu(workspace.next()) } -> std::same_as<void>;
    };

} // namespace gnn
