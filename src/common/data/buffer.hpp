#pragma once

#include <concepts>
#include <cstddef>

namespace gnn {

template <typename B>
concept Buffer = std::movable<B> && std::constructible_from<B, std::size_t> &&
                 requires(B& buffer, const B& constBuffer, std::size_t logicalSize) {
                     typename B::value_type;
                     { buffer.data() } -> std::same_as<typename B::value_type*>;
                     { constBuffer.data() } -> std::same_as<const typename B::value_type*>;
                     { constBuffer.logicalSize() } -> std::same_as<std::size_t>;
                     { constBuffer.physicalSize() } -> std::same_as<std::size_t>;
                     { buffer.setLogicalSize(logicalSize) } -> std::same_as<void>;
                 };

} // namespace gnn
