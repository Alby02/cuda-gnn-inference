#pragma once
#include "executor.hpp"
#include <utility>
#include <variant>

namespace gnn {
template <Executor E> class InferenceRuntime {
public:
    using WorkspaceType = typename E::WorkspaceType;
    explicit InferenceRuntime(E executor = {}) : executor_(std::move(executor)) {}

    // Prepare once and resetInput outside compute timing before each run.
    void run(WorkspaceType& workspace) {
        for (const auto& layer : workspace.getModel().getLayers()) {
            std::visit(
                [&](const auto& concrete) {
                    forward_layer(concrete, workspace.getGraph(), executor_, workspace);
                },
                layer);
            workspace.swapBuffers();
        }
    }

private:
    E executor_;
};
} // namespace gnn
