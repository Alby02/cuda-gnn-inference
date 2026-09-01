#pragma once

#include <concepts>

namespace gnn {

template <typename W>
concept Workspace = requires(W& workspace, const W& constWorkspace) {
    typename W::BufferType; // TODO: make a better use of bufferType to make the program work better
                            // with cuda

    { workspace.current() } -> std::same_as<typename W::BufferType&>;

    { constWorkspace.current() } -> std::same_as<const typename W::BufferType&>;

    { workspace.next() } -> std::same_as<typename W::BufferType&>;

    { workspace.swapBuffers() } -> std::same_as<void>;
};

} // namespace gnn
