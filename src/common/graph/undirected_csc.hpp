#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "../matrix/dense_matrix.hpp"
#include "base.hpp"

namespace graph {

class UndirectedCSC {
public:
    UndirectedCSC(std::uint64_t numNodes, std::uint64_t numUndirectedEdges,
                  std::vector<std::uint64_t> colPtr, std::vector<std::uint64_t> rowInd,
                  std::vector<float> weights = {},
                  std::optional<matrix::DenseMatrix<float>> edgeFeatures = std::nullopt)
        : numNodes_(numNodes), numUndirectedEdges_(numUndirectedEdges), colPtr_(std::move(colPtr)),
          rowInd_(std::move(rowInd)), weights_(std::move(weights)),
          edgeFeatures_(std::move(edgeFeatures)) {

        if (colPtr_.size() != numNodes_ + 1) {
            throw std::invalid_argument("colPtr size must be equal to numNodes + 1");
        }
        if (!weights_.empty() && weights_.size() != rowInd_.size()) {
            throw std::invalid_argument("weights size must match rowInd size");
        }
        if (edgeFeatures_.has_value() && edgeFeatures_->rows() != rowInd_.size()) {
            throw std::invalid_argument("edgeFeatures rows must match rowInd size");
        }
    }

    // Graph base properties
    [[nodiscard]] std::uint64_t getNumNodes() const noexcept { return numNodes_; }
    [[nodiscard]] std::uint64_t getNumEdges() const noexcept { return numUndirectedEdges_; }
    [[nodiscard]] std::uint64_t getNumDirectedEntries() const noexcept { return rowInd_.size(); }
    [[nodiscard]] bool isDirected() const noexcept { return false; }
    [[nodiscard]] bool hasEdgeWeights() const noexcept { return !weights_.empty(); }
    [[nodiscard]] bool hasEdgeFeatures() const noexcept {
        return edgeFeatures_.has_value() && !edgeFeatures_->empty();
    }

    // in_Methods
    [[nodiscard]] std::uint64_t inDegree(std::uint64_t node) const noexcept {
        return colPtr_[node + 1] - colPtr_[node];
    }

    [[nodiscard]] std::span<const std::uint64_t> getInNeighbors(std::uint64_t node) const noexcept {
        return std::span<const std::uint64_t>{rowInd_}.subspan(colPtr_[node], inDegree(node));
    }

    [[nodiscard]] std::span<const float> getInWeights(std::uint64_t node) const noexcept {
        if (weights_.empty()) {
            return {};
        }
        return std::span<const float>{weights_}.subspan(colPtr_[node], inDegree(node));
    }

    [[nodiscard]] std::uint64_t getSourceNode(std::uint64_t edgeId) const noexcept {
        return rowInd_[edgeId];
    }

    [[nodiscard]] float getWeight(std::uint64_t edgeId) const noexcept {
        return weights_.empty() ? 1.0f : weights_[edgeId];
    }

    // Direct pointers for CUDA parameter passing
    [[nodiscard]] const std::uint64_t* colPtrData() const noexcept { return colPtr_.data(); }
    [[nodiscard]] const std::uint64_t* rowIndData() const noexcept { return rowInd_.data(); }
    [[nodiscard]] const float* weightsData() const noexcept {
        return weights_.empty() ? nullptr : weights_.data();
    }
    [[nodiscard]] const float* edgeFeatureData() const noexcept {
        return edgeFeatures_.has_value() ? edgeFeatures_->data() : nullptr;
    }
    [[nodiscard]] std::size_t edgeFeatureDim() const noexcept {
        return edgeFeatures_.has_value() ? edgeFeatures_->cols() : 0;
    }

    // GCN Degree Normalization factor for a single node
    template <bool IncludeSelfLoop = false>
    [[nodiscard]] float invSqrtDegree(std::uint64_t node) const {
        constexpr std::uint64_t selfLoopAdd = IncludeSelfLoop ? 1 : 0;
        const float deg = static_cast<float>(inDegree(node) + selfLoopAdd);
        return (deg > 0.0f) ? (1.0f / std::sqrt(deg)) : 0.0f;
    }

private:
    std::uint64_t numNodes_{0};
    std::uint64_t numUndirectedEdges_{0};
    std::vector<std::uint64_t> colPtr_;
    std::vector<std::uint64_t> rowInd_;
    std::vector<float> weights_;
    std::optional<matrix::DenseMatrix<float>> edgeFeatures_;
};

static_assert(InNeighbor<UndirectedCSC>);

} // namespace graph
