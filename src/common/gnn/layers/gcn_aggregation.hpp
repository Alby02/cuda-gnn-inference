#pragma once

// features T-SEQ-02 - T-CON-05: implementing GCN aggregation

#include "../../data/buffer.hpp"
#include "../../data/matrix.hpp"
#include "../../host_buffer.hpp"
#include "../../host_graph.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gnn::layers {

class GCNAggregationState {
public:
    using HostMatrix = Matrix<HostBuffer<float>>;

    void prepare(const graph::HostGraphCSC& graph) { prepareMetadata(graph); }

    [[nodiscard]] std::size_t capacityBytes() const noexcept {
        return invSqrtDeg_.capacity() * sizeof(float) +
               hasExplicitSelfLoop_.capacity() * sizeof(std::uint8_t);
    }

    void aggregate(const graph::HostGraphCSC& graph, const HostMatrix& input,
                   HostMatrix& output) const {

        const auto n = static_cast<std::size_t>(graph.getNumNodes());
        const auto featureDim = input.cols();
        output.setShape(n, featureDim);

        const bool weighted = graph.hasEdgeWeights();

        for (std::size_t v = 0; v < n; ++v) {
            for (std::size_t f = 0; f < featureDim; ++f) {
                output(v, f) = 0.0F;
            }

            const auto neighbors = graph.getInNeighbors(static_cast<std::uint64_t>(v));
            const auto weights = graph.getInWeights(static_cast<std::uint64_t>(v));
            const float invSqrtDv = invSqrtDeg_[v];

            for (std::size_t edge = 0; edge < neighbors.size(); ++edge) {
                const auto u = static_cast<std::size_t>(neighbors[edge]);
                const float w = weighted ? weights[edge] : 1.0F;
                const float alpha = w * invSqrtDv * invSqrtDeg_[u];
                for (std::size_t f = 0; f < featureDim; ++f) {
                    output(v, f) += alpha * input(u, f);
                }
            }

            if (hasExplicitSelfLoop_[v] == 0) {
                const float alphaSelf = invSqrtDv * invSqrtDv; // d_u == d_v
                for (std::size_t f = 0; f < featureDim; ++f) {
                    output(v, f) += alphaSelf * input(v, f);
                }
            }
        }
    }

private:
    void prepareMetadata(const graph::HostGraphCSC& graph) {
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
                weightedDegree += w;
                if (neighbors[edge] == static_cast<std::uint64_t>(v)) {
                    explicitSelf = true;
                }
            }

            if (!explicitSelf) {
                weightedDegree += 1.0F;
            }

            hasExplicitSelfLoop_[v] = explicitSelf ? 1 : 0;
            invSqrtDeg_[v] = (weightedDegree > 0.0F) ? (1.0F / std::sqrt(weightedDegree)) : 0.0F;
        }
    }

    std::vector<float> invSqrtDeg_;
    std::vector<std::uint8_t> hasExplicitSelfLoop_;
};

} // namespace gnn::layers

// features T-OMPV-01 and T-OMPV-04

#ifdef _OPENMP

namespace gnn::layers {

class GCNAggregationStateParallel {
public:
    using HostMatrix = Matrix<HostBuffer<float>>;

    void prepare(const graph::HostGraphCSC& graph) { prepareMetadata(graph); }

    [[nodiscard]] std::size_t capacityBytes() const noexcept {
        return invSqrtDeg_.capacity() * sizeof(float) +
               hasExplicitSelfLoop_.capacity() * sizeof(std::uint8_t);
    }

    void aggregate(const graph::HostGraphCSC& graph, const HostMatrix& input,
                   HostMatrix& output) const {

        const auto n = static_cast<std::int64_t>(graph.getNumNodes());
        const auto featureDim = input.cols();
        output.setShape(static_cast<std::size_t>(n), featureDim);

        const bool weighted = graph.hasEdgeWeights();

#pragma omp parallel for schedule(static)
        for (std::int64_t vi = 0; vi < n; ++vi) {
            const auto v = static_cast<std::size_t>(vi);
            for (std::size_t f = 0; f < featureDim; ++f) {
                output(v, f) = 0.0F;
            }

            const auto neighbors = graph.getInNeighbors(static_cast<std::uint64_t>(v));
            const auto weights = graph.getInWeights(static_cast<std::uint64_t>(v));
            const float invSqrtDv = invSqrtDeg_[v];

            for (std::size_t edge = 0; edge < neighbors.size(); ++edge) {
                const auto u = static_cast<std::size_t>(neighbors[edge]);
                const float w = weighted ? weights[edge] : 1.0F;
                const float alpha = w * invSqrtDv * invSqrtDeg_[u];
                for (std::size_t f = 0; f < featureDim; ++f) {
                    output(v, f) += alpha * input(u, f);
                }
            }

            if (hasExplicitSelfLoop_[v] == 0) {
                const float alphaSelf = invSqrtDv * invSqrtDv;
                for (std::size_t f = 0; f < featureDim; ++f) {
                    output(v, f) += alphaSelf * input(v, f);
                }
            }
        }
    }

private:
    // same as the one for the sequential part
    void prepareMetadata(const graph::HostGraphCSC& graph) {
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
                weightedDegree += w;
                if (neighbors[edge] == static_cast<std::uint64_t>(v)) {
                    explicitSelf = true;
                }
            }

            if (!explicitSelf) {
                weightedDegree += 1.0F;
            }

            hasExplicitSelfLoop_[v] = explicitSelf ? 1 : 0;
            invSqrtDeg_[v] = (weightedDegree > 0.0F) ? (1.0F / std::sqrt(weightedDegree)) : 0.0F;
        }
    }

    std::vector<float> invSqrtDeg_;
    std::vector<std::uint8_t> hasExplicitSelfLoop_;
};

} // namespace gnn::layers

#endif // _OPENMP
