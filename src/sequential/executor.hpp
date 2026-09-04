#pragma once

#include "cpu_context.hpp"
#include "data/graph_csc.hpp"
#include "execution/executor.hpp"

#include <cstddef>
#include <omp.h>
#include <stdexcept>

namespace gnn {

class SequentialExecutor {
public:
    using WorkspaceType = CpuContext;
    using BufferType = WorkspaceType::BufferType;
    using WeightType = Matrix<HostBuffer<float>>;

    void rowByColumn(const BufferType& left, const BufferType& right, BufferType& output) {
        validateProduct(left, right);
        output.setShape(left.rows(), right.cols());

        for (std::size_t row = 0; row < left.rows(); ++row) {
            for (std::size_t column = 0; column < right.cols(); ++column) {
                float value = 0.0F;
                for (std::size_t inner = 0; inner < left.cols(); ++inner) {
                    value += left(row, inner) * right(inner, column);
                }
                output(row, column) = value;
            }
        }
    }

    void aggregateNeighbors(const auto& graph, const BufferType& in_features,
                            BufferType& out_aggregated, auto agg_type, 
                             const int layer_num, const int * sample[]) {
        const std::size_t num_nodes = graph.getNumNodes();
        const std::size_t feat_dim = in_features.cols();

        out_aggregated = BufferType(num_nodes, feat_dim);

        for (std::size_t v = 0; v < num_nodes; ++v) {
            const auto neighbors = graph.getInNeighbors(static_cast<std::uint64_t>(v));

            if (neighbors.empty()) {
                for (std::size_t d = 0; d < feat_dim; ++d) {
                    out_aggregated(v, d) = 0.0F;
                }
                continue;
            }

            const float inv_degree = 1.0F / static_cast<float>(neighbors.size());

            if (static_cast<int>(agg_type) == 0) { // MEAN
                for (std::size_t d = 0; d < feat_dim; ++d) {
                    float sum = 0.0F;
                    for (const std::uint64_t u : neighbors) {
                        sum += in_features(u, d);
                    }
                    out_aggregated(v, d) = sum * inv_degree;
                }
            } else if (static_cast<int>(agg_type) == 1) { // SUM
                for (std::size_t d = 0; d < feat_dim; ++d) {
                    float sum = 0.0F;
                    for (const std::uint64_t u : neighbors) {
                        sum += in_features(u, d);
                    }
                    out_aggregated(v, d) = sum;
                }
            } else if (static_cast<int>(agg_type) == 2) { // MAX
                for (std::size_t d = 0; d < feat_dim; ++d) {
                    float max_val = in_features(neighbors[0], d);
                    for (std::size_t i = 1; i < neighbors.size(); ++i) {
                        max_val = std::max(max_val, in_features(neighbors[i], d));
                    }
                    out_aggregated(v, d) = max_val;
                }
            }
        }
    }

private:
    static void validateProduct(const BufferType& left, const BufferType& right) {
        if (left.cols() != right.rows()) {
            throw std::invalid_argument(
                "Matrix product requires the left column count to equal the right row count.");
        }
    }
};

static_assert(Executor<SequentialExecutor>);

} // namespace gnn
