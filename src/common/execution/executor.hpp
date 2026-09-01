#pragma once

#include "workspace.hpp"

#include <concepts>

namespace gnn {

template <typename E>
concept Executor =
    // First, require the associated type to exist.
    requires {
        typename E::WorkspaceType;
    }

    // Then, require it to satisfy the Workspace concept.
    && Workspace<typename E::WorkspaceType>

    // Finally, use the associated type as a requires-expression argument.
    && requires(E& executor, typename E::WorkspaceType& workspace,
                const typename E::WorkspaceType::BufferType& weights) {
        { executor.rowByColumn(workspace.current(), weights, workspace.next()) } //TODO: Modify to remove current and next
            -> std::same_as<void>;
    };

} // namespace gnn
