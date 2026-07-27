#pragma once

#include "base.hpp"
#include <span>

namespace graph {

class IInNeighbor : public virtual IGraph {
public:
    ~IInNeighbor() override = default;

    [[nodiscard]] virtual std::span<const NodeId> get_in_neighbors(NodeId node) const = 0;
    [[nodiscard]] virtual std::span<const Weight> get_in_weights(NodeId node) const = 0;
    [[nodiscard]] virtual std::size_t in_degree(NodeId node) const = 0;
};

template<typename G>
concept InNeighbor = Graph<G> && requires(const G& g, NodeId node) {
    { g.get_in_neighbors(node) } -> std::same_as<std::span<const NodeId>>;
    { g.get_in_weights(node) } -> std::same_as<std::span<const Weight>>;
    { g.in_degree(node) } -> std::same_as<std::size_t>;
};

} // namespace graph
