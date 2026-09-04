#pragma once

// features T-SEQ-02 - T-CON-05: implementing GCN aggregation  

#include "../../data/buffer.hpp"
#include "../../data/matrix.hpp"
#include "../../host_buffer.hpp"
#include "../../host_graph.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gnn::layers {

class GCNAggregationState {
public:
    using HostMatrix = Matrix<HostBuffer<float>>;

    [[nodiscard]] const HostMatrix& aggregate(const graph::HostGraphCSC& graph,
                                              const HostMatrix& input) {
        ensureMetadata(graph);

        const auto n = static_cast<std::size_t>(graph.getNumNodes());
        const auto featureDim = input.cols();
        ensureScratchCapacity(n, featureDim);

        const bool weighted = graph.hasEdgeWeights();

        for (std::size_t v = 0; v < n; ++v) {
            for (std::size_t f = 0; f < featureDim; ++f) {
                scratch_(v, f) = 0.0F;
            }

            const auto neighbors = graph.getInNeighbors(static_cast<std::uint64_t>(v));
            const auto weights = graph.getInWeights(static_cast<std::uint64_t>(v));
            const float invSqrtDv = invSqrtDeg_[v];

            for (std::size_t edge = 0; edge < neighbors.size(); ++edge) {
                const auto u = static_cast<std::size_t>(neighbors[edge]);
                const float w = weighted ? weights[edge] : 1.0F;
                const float alpha = w * invSqrtDv * invSqrtDeg_[u];
                for (std::size_t f = 0; f < featureDim; ++f) {
                    scratch_(v, f) += alpha * input(u, f);
                }
            }

        
            if (hasExplicitSelfLoop_[v] == 0) {
                const float alphaSelf = invSqrtDv * invSqrtDv; // d_u == d_v
                for (std::size_t f = 0; f < featureDim; ++f) {
                    scratch_(v, f) += alphaSelf * input(v, f);
                }
            }
        }

        return scratch_;
    }

private:
    void ensureMetadata(const graph::HostGraphCSC& graph) {
        if (&graph == cachedGraph_) {
            return;
        }

        const auto n = static_cast<std::size_t>(graph.getNumNodes());
        invSqrtDeg_.assign(n, 0.0F);
        hasExplicitSelfLoop_.assign(n, 0);

        const bool weighted = graph.hasEdgeWeights();

        for (std::size_t v = 0; v < n; ++v) {
            const auto neighbors = graph.getInNeighbors(static_cast<std::uint64_t>(v));
            const auto weights = graph.getInWeights(static_cast<std::uint64_t>(v));

            float weightedDegree = 0.0F;
            bool explicitSelf = false;

            for (std::size_t edge = 0; edge < neighbors.size(); ++edge) {
                const float w = weighted ? weights[edge] : 1.0F;
                assert(w > 0.0F &&
                       "GCN aggregation requires finite, strictly positive stored weights (FR-GRAPH-08)");
                weightedDegree += w;
                if (neighbors[edge] == static_cast<std::uint64_t>(v)) {
                    explicitSelf = true;
                }
            }

            if (!explicitSelf) {
                weightedDegree += 1.0F; // self-loop implicito, peso 1
            }

            hasExplicitSelfLoop_[v] = explicitSelf ? 1 : 0;
            invSqrtDeg_[v] = (weightedDegree > 0.0F) ? (1.0F / std::sqrt(weightedDegree)) : 0.0F;
        }

        cachedGraph_ = &graph;
    }

   
    void ensureScratchCapacity(std::size_t rows, std::size_t cols) {
        if (scratch_.physicalSize() < rows * cols) {
            scratch_ = HostMatrix(rows, cols);
        } else {
            scratch_.setShape(rows, cols);
        }
    }

    const graph::HostGraphCSC* cachedGraph_{nullptr};
    std::vector<float> invSqrtDeg_;
    std::vector<std::uint8_t> hasExplicitSelfLoop_;
    HostMatrix scratch_;
};

} //