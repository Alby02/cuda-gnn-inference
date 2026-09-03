#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../host_graph.hpp"

namespace graph {

class GraphFactory {
public:
    static HostGraphCSC make(bool isDirected, std::uint64_t numNodes, std::uint64_t numEntries,
                             std::vector<std::uint64_t> colPtr, std::vector<std::uint64_t> rowInd,
                             std::vector<float> weights = {}) {
        validate(isDirected, numNodes, numEntries, colPtr, rowInd, weights);
        return HostGraphCSC(isDirected, gnn::vectorToHostBuffer(std::move(colPtr)),
                            gnn::vectorToHostBuffer(std::move(rowInd)),
                            gnn::vectorToHostBuffer(std::move(weights)));
    }

private:
    static void validate(bool isDirected, std::uint64_t numNodes, std::uint64_t numEntries,
                         const std::vector<std::uint64_t>& colPtr,
                         const std::vector<std::uint64_t>& rowInd,
                         const std::vector<float>& weights) {
        if (numNodes == std::numeric_limits<std::uint64_t>::max() ||
            colPtr.size() != numNodes + 1) {
            throw std::invalid_argument("GraphFactory: node count does not match colPtr");
        }
        if (rowInd.size() != numEntries) {
            throw std::invalid_argument("GraphFactory: edge count does not match rowInd");
        }
        if (colPtr.empty() || colPtr.front() != 0 || colPtr.back() != rowInd.size()) {
            throw std::invalid_argument("GraphFactory: invalid CSC column-pointer boundaries");
        }
        if (!weights.empty() && weights.size() != rowInd.size()) {
            throw std::invalid_argument("GraphFactory: weights size must match rowInd");
        }

        for (std::uint64_t destination = 0; destination < numNodes; ++destination) {
            const auto begin = colPtr[destination];
            const auto end = colPtr[destination + 1];
            if (begin > end || end > rowInd.size()) {
                throw std::invalid_argument(
                    "GraphFactory: column pointers must be nondecreasing and in range");
            }

            std::optional<std::uint64_t> previousSource;
            for (auto edge = begin; edge < end; ++edge) {
                const auto source = rowInd[edge];
                if (source >= numNodes) {
                    throw std::invalid_argument("GraphFactory: row index is outside node range");
                }
                if (previousSource.has_value() && source <= *previousSource) {
                    throw std::invalid_argument(
                        "GraphFactory: source indices must be strictly increasing per column");
                }
                previousSource = source;
                if (!weights.empty() && (!std::isfinite(weights[edge]) || weights[edge] <= 0.0F)) {
                    throw std::invalid_argument(
                        "GraphFactory: edge weights must be finite and positive");
                }
            }
        }

        if (!isDirected) {
            validateUndirectedReciprocity(numNodes, colPtr, rowInd, weights);
        }
    }

    static void validateUndirectedReciprocity(std::uint64_t numNodes,
                                              const std::vector<std::uint64_t>& colPtr,
                                              const std::vector<std::uint64_t>& rowInd,
                                              const std::vector<float>& weights) {
        for (std::uint64_t destination = 0; destination < numNodes; ++destination) {
            for (auto edge = colPtr[destination]; edge < colPtr[destination + 1]; ++edge) {
                const auto source = rowInd[edge];
                const auto reverseBegin = rowInd.begin() + colPtr[source];
                const auto reverseEnd = rowInd.begin() + colPtr[source + 1];
                const auto reverse = std::lower_bound(reverseBegin, reverseEnd, destination);
                if (reverse == reverseEnd || *reverse != destination) {
                    throw std::invalid_argument(
                        "GraphFactory: undirected graphs require reciprocal entries");
                }
                if (!weights.empty()) {
                    const auto reverseEdge = static_cast<std::size_t>(reverse - rowInd.begin());
                    if (weights[edge] != weights[reverseEdge]) {
                        throw std::invalid_argument(
                            "GraphFactory: reciprocal entries must have equal weights");
                    }
                }
            }
        }
    }
};

} // namespace graph
