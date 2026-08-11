#pragma once

#include <concepts>
#include <cstdint>
#include <span>

namespace graph {

template <typename G>
concept Base = requires(const G& g) {
    { g.getNumNodes() } -> std::same_as<std::uint64_t>;
    { g.getNumEdges() } -> std::same_as<std::uint64_t>;
    { g.isDirected() } -> std::same_as<bool>;
    { g.hasEdgeWeights() } -> std::same_as<bool>;
    { g.hasEdgeFeatures() } -> std::same_as<bool>;
};

template <typename G>
concept InNeighbor = Base<G> && requires(const G& g, std::uint64_t node) {
    { g.getInNeighbors(node) } -> std::same_as<std::span<const std::uint64_t>>;
    { g.getInWeights(node) } -> std::same_as<std::span<const float>>;
    { g.inDegree(node) } -> std::same_as<std::uint64_t>;
    { g.template invSqrtDegree<false>(node) } -> std::same_as<float>;
};

} // namespace graph
