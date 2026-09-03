#pragma once

#include <concepts>

namespace gnn {

template <typename W>
concept Workspace = requires(W& workspace, const W& constWorkspace) {
    typename W::BufferType;

    { workspace.current() } -> std::same_as<typename W::BufferType&>;

    { constWorkspace.current() } -> std::same_as<const typename W::BufferType&>;

    { workspace.next() } -> std::same_as<typename W::BufferType&>;

    { workspace.swapBuffers() } -> std::same_as<void>;
};

} // namespace gnn
