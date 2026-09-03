#pragma once

#include "buffer.hpp"

#include <cmath>
#include <concepts>
#include <cstdint>
#include <span>
#include <utility>

namespace graph {

template <gnn::Buffer IndexStorage, gnn::Buffer WeightStorage>
    requires(std::same_as<typename IndexStorage::value_type, std::uint64_t> &&
             std::same_as<typename WeightStorage::value_type, float>)
class GraphCSC {
public:
    GraphCSC(bool isDirected, IndexStorage colPtr, IndexStorage rowInd,
             WeightStorage weights = WeightStorage{0})
        : isDirected_(isDirected), colPtr_(std::move(colPtr)), rowInd_(std::move(rowInd)),
          weights_(std::move(weights)) {};

    // Graph base properties
    [[nodiscard]] std::uint64_t getNumNodes() const noexcept { return colPtr_.logicalSize() - 1; }
    [[nodiscard]] std::uint64_t getNumEdges() const noexcept { return rowInd_.logicalSize(); }
    [[nodiscard]] bool isDirected() const noexcept { return isDirected_; }
    [[nodiscard]] bool hasEdgeWeights() const noexcept { return weights_.logicalSize() != 0; }
    [[nodiscard]] const IndexStorage& colPtrBuffer() const noexcept { return colPtr_; }
    [[nodiscard]] const IndexStorage& rowIndBuffer() const noexcept { return rowInd_; }
    [[nodiscard]] const WeightStorage& weightsBuffer() const noexcept { return weights_; }

    // in_Methods
    [[nodiscard]] std::uint64_t inDegree(std::uint64_t node) const noexcept {
        return colPtr_.data()[node + 1] - colPtr_.data()[node];
    }

    [[nodiscard]] std::span<const std::uint64_t> getInNeighbors(std::uint64_t node) const noexcept {
        return std::span<const std::uint64_t>{rowInd_.data(), rowInd_.logicalSize()}.subspan(
            colPtr_.data()[node], inDegree(node));
    }

    [[nodiscard]] std::span<const float> getInWeights(std::uint64_t node) const noexcept {
        if (!hasEdgeWeights()) {
            return {};
        }
        return std::span<const float>{weights_.data(), weights_.logicalSize()}.subspan(
            colPtr_.data()[node], inDegree(node));
    }

    [[nodiscard]] std::uint64_t getSourceNode(std::uint64_t edgeId) const noexcept {
        return rowInd_.data()[edgeId];
    }

    [[nodiscard]] float getWeight(std::uint64_t edgeId) const noexcept {
        return hasEdgeWeights() ? weights_.data()[edgeId] : 1.0F;
    }

    // GCN Degree Normalization factor for a single node: d_norm[v] = 1 / sqrt(inDegree(v) + (1 for
    // self loop))
    template <bool IncludeSelfLoop = false>
    [[nodiscard]] float invSqrtDegree(std::uint64_t node) const {
        constexpr std::uint64_t selfLoopAdd = IncludeSelfLoop ? 1 : 0;
        const float deg = static_cast<float>(inDegree(node) + selfLoopAdd);
        return (deg > 0.0f) ? (1.0f / std::sqrt(deg)) : 0.0f;
    }

private:
    bool isDirected_;
    IndexStorage colPtr_;
    IndexStorage rowInd_;
    WeightStorage weights_;
};

} // namespace graph
