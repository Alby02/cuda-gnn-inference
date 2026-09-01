#pragma once
#include <concepts>
#include <cstddef>

namespace gnn {

template <typename L>
concept Layer = requires (const L& layer) {
    { layer.getInDim() } -> std::same_as<std::size_t>;
    { layer.getOutDim() } -> std::same_as<std::size_t>;
};

} // namespace gnn
