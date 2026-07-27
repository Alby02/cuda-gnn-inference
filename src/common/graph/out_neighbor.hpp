#pragma once

#include "base.hpp"
#include <span>

namespace graph {

class IOutNeighbor : public virtual IGraph {
public:
    ~IOutNeighbor() override = default;

    [[nodiscard]] virtual std::span<const NodeId> get_out_neighbors(NodeId node) const = 0;
    [[nodiscard]] virtual std::span<const Weight> get_out_weights(NodeId node) const = 0;
    [[nodiscard]] virtual std::size_t out_degree(NodeId node) const = 0;
};

template<typename G>
concept OutNeighbor = Graph<G> && requires(const G& g, NodeId node) {
    { g.get_out_neighbors(node) } -> std::same_as<std::span<const NodeId>>;
    { g.get_out_weights(node) } -> std::same_as<std::span<const Weight>>;
    { g.out_degree(node) } -> std::same_as<std::size_t>;
};

} // namespace graph
