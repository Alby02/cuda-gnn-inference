#pragma once

#include "workspace.hpp"

#include <concepts>

namespace gnn {

template <typename E>
concept Executor =
    requires {
        typename E::WorkspaceType;
        typename E::WeightType;
    } && Workspace<typename E::WorkspaceType> &&
    requires(E& executor, typename E::WorkspaceType& workspace,
             const typename E::WeightType& weights) {
        {
            executor.rowByColumn(workspace.current(), weights, workspace.next())
        } -> std::same_as<void>;
    };

} // namespace gnn
