#pragma once

#include <cmath>
#include <cstdint>
#include <span>
#include <vector>


namespace graph {

class GraphCSC {
public:
    GraphCSC(bool isDirected, std::vector<std::uint64_t> colPtr,
                std::vector<std::uint64_t> rowInd, std::vector<float> weights = {})
        : isDirected_(isDirected),
          colPtr_(std::move(colPtr)),
          rowInd_(std::move(rowInd)), weights_(std::move(weights)){};

    // Graph base properties
    [[nodiscard]] std::uint64_t getNumNodes() const noexcept { return colPtr_.size() - 1; }
    [[nodiscard]] std::uint64_t getNumEdges() const noexcept { return rowInd_.size(); }
    [[nodiscard]] bool isDirected() const noexcept { return isDirected_; }
    [[nodiscard]] bool hasEdgeWeights() const noexcept { return !weights_.empty(); }

    // in_Methods
    [[nodiscard]] std::uint64_t inDegree(std::uint64_t node) const noexcept {
        return colPtr_[node + 1] - colPtr_[node];
    }

    [[nodiscard]] std::span<const std::uint64_t> getInNeighbors(std::uint64_t node) const noexcept {
        return std::span<const std::uint64_t>{rowInd_}.subspan(colPtr_[node], inDegree(node));
    }

    [[nodiscard]] std::span<const float> getInWeights(std::uint64_t node) const noexcept {
        // should not be needed hasEdgeWeights() should be called before this method is called
        /*if (weights_.empty()) {
            return {};
        }*/
        return std::span<const float>{weights_}.subspan(colPtr_[node], inDegree(node));
    }

    [[nodiscard]] std::uint64_t getSourceNode(std::uint64_t edgeId) const noexcept {
        return rowInd_[edgeId];
    }

    [[nodiscard]] float getWeight(std::uint64_t edgeId) const noexcept {
        return weights_.empty() ? 1.0f : weights_[edgeId];
    }

    // Direct pointers for CUDA parameter passing
    /*[[nodiscard]] const std::uint64_t* colPtrData() const noexcept { return colPtr_.data(); }
    [[nodiscard]] const std::uint64_t* rowIndData() const noexcept { return rowInd_.data(); }
    [[nodiscard]] const float* weightsData() const noexcept {
        return weights_.empty() ? nullptr : weights_.data();
    }*/

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
    std::vector<std::uint64_t> colPtr_;
    std::vector<std::uint64_t> rowInd_;
    std::vector<float> weights_;
};

} // namespace graph
