#pragma once

#include <concepts>

#include "../graph/base.hpp"
#include "../matrix/dense_matrix.hpp"

namespace gnn {

template <typename E>
concept HasSupportedLayers = requires {
    typename E::SupportedLayers;
};

// Option E Engine Concept:
// 1. Constructed with a GNNModel<typename E::SupportedLayers>
// 2. Provides .run(graph, nodeFeatures) and .run(graph, nodeFeatures, edgeFeatures)
template <typename E, typename G>
concept Engine = graph::InNeighbor<G> && HasSupportedLayers<E> &&
    requires(E& engine, const G& graph,
             const matrix::DenseMatrix<float>& nodeFeatures,
             const matrix::DenseMatrix<float>& edgeFeatures) {
        { engine.run(graph, nodeFeatures) } -> std::same_as<matrix::DenseMatrix<float>>;
        { engine.run(graph, nodeFeatures, edgeFeatures) } -> std::same_as<matrix::DenseMatrix<float>>;
    };

} // namespace gnn
