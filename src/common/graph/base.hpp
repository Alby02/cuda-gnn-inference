#pragma once

#include "types.hpp"
#include <concepts>

namespace graph {

class IGraph {
public:
    virtual ~IGraph() = default;

    [[nodiscard]] virtual std::size_t num_nodes() const noexcept = 0;
    [[nodiscard]] virtual std::size_t num_edges() const noexcept = 0;
    [[nodiscard]] virtual bool is_directed() const noexcept = 0;
    [[nodiscard]] virtual bool has_edge_weights() const noexcept = 0;
};

template<typename G>
concept Graph = requires(const G& g) {
    { g.num_nodes() } -> std::same_as<std::size_t>;
    { g.num_edges() } -> std::same_as<std::size_t>;
    { g.is_directed() } -> std::same_as<bool>;
    { g.has_edge_weights() } -> std::same_as<bool>;
};

} // namespace graph
