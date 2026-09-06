#pragma once
#include <concepts>
#include <cstddef>

namespace gnn {
struct WorkspacePreparation {
    double uploadMs = 0; // Host-to-device preparation time; zero for CPU.
};

template <typename W>
concept Workspace =
    requires(W& workspace, const W& constWorkspace, const typename W::WorkloadType& workload) {
        typename W::BufferType;
        typename W::GraphType;
        typename W::ModelType;
        typename W::HostOutputType;
        { workspace.prepare(workload) } -> std::same_as<WorkspacePreparation>;
        { workspace.resetInput() } -> std::same_as<void>;
        { constWorkspace.getGraph() } -> std::same_as<const typename W::GraphType&>;
        { constWorkspace.getModel() } -> std::same_as<const typename W::ModelType&>;
        // CPU borrows its output; CUDA returns an owning host matrix.
        requires(
            std::same_as<decltype(constWorkspace.getOutput()), const typename W::HostOutputType&> ||
            std::same_as<decltype(constWorkspace.getOutput()), typename W::HostOutputType>);
        { workspace.current() } -> std::same_as<typename W::BufferType&>;
        { constWorkspace.current() } -> std::same_as<const typename W::BufferType&>;
        { workspace.next() } -> std::same_as<typename W::BufferType&>;
        { workspace.scratch() } -> std::same_as<typename W::BufferType&>;
        { workspace.branch() } -> std::same_as<typename W::BufferType&>;
        { workspace.swapBuffers() } -> std::same_as<void>;
        { constWorkspace.capacityBytes() } -> std::same_as<std::size_t>;
    };
} // namespace gnn
