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
                weightedDegree += 1.0F; 
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


// features T-OMPV-01 and T-OMPV-04

#include <omp.h>
 
namespace gnn::layers {
 
class GCNAggregationStateParallel {
public:
    using HostMatrix = Matrix<HostBuffer<float>>;
 
    struct OmpConfig {
        int numThreads{omp_get_max_threads()};
        omp_sched_t scheduleKind{omp_sched_static};
        int chunkSize{0};
    };
 
    void setNumThreads(int numThreads) {
        if (numThreads < 1) {
            throw std::invalid_argument("GCNAggregationStateParallel: numThreads must be >= 1");
        }
        config_.numThreads = numThreads;
    }
 
    void setSchedule(omp_sched_t kind, int chunkSize = 0) {
        config_.scheduleKind = kind;
        config_.chunkSize = chunkSize;
    }
 
    [[nodiscard]] const OmpConfig& getConfig() const noexcept { return config_; }
 
    [[nodiscard]] const HostMatrix& aggregate(const graph::HostGraphCSC& graph,
                                              const HostMatrix& input) {
        ensureMetadata(graph);
 
        const auto n = static_cast<std::int64_t>(graph.getNumNodes());
        const auto featureDim = input.cols();
        ensureScratchCapacity(static_cast<std::size_t>(n), featureDim);
 
        const bool weighted = graph.hasEdgeWeights();
 
        omp_set_num_threads(config_.numThreads);
        omp_set_schedule(config_.scheduleKind, config_.chunkSize);
 
#pragma omp parallel for schedule(runtime)
        for (std::int64_t vi = 0; vi < n; ++vi) {
            const auto v = static_cast<std::size_t>(vi);
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
                const float alphaSelf = invSqrtDv * invSqrtDv;
                for (std::size_t f = 0; f < featureDim; ++f) {
                    scratch_(v, f) += alphaSelf * input(v, f);
                }
            }
        }
 
        return scratch_;
    }
 
private:
    //same as the one for the sequential part
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
                       "GCN aggregation requires finite, strictly positive stored weights");
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
    HostMatrix scratch_{0, 0};
    OmpConfig config_;
};
 
} // 
