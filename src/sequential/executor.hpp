#pragma once

#include "cpu_context.hpp"
#include "data/graph_csc.hpp"
#include "execution/executor.hpp"
#include "gnn/layer.hpp"

#include <cstddef>
#include <stdexcept>

namespace gnn {

class SequentialExecutor {
public:
    using WorkspaceType = CpuContext;
    using BufferType = WorkspaceType::BufferType;
    using WeightType = Matrix<HostBuffer<float>>;

    using BiasType = HostBuffer<float>;

    void rowByColumn(const BufferType& left, const WeightType& right, BufferType& output) const {
        if (left.cols() != right.rows())
            throw std::invalid_argument("Incompatible matrix product shapes.");
        output.setShape(left.rows(), right.cols());
        forRows(left.rows(), [&](std::size_t row) {
            for (std::size_t col = 0; col < right.cols(); ++col) {
                float value = 0;
                for (std::size_t k = 0; k < left.cols(); ++k)
                    value += left(row, k) * right(k, col);
                output(row, col) = value;
            }
        });
    }
    // Elementwise operations permit exact in-place aliasing.
    void add(const BufferType& left, const BufferType& right, BufferType& output) const {
        output.setShape(left.rows(), left.cols());
        forRows(left.size(),
                [&](std::size_t i) { output.data()[i] = left.data()[i] + right.data()[i]; });
    }
    void biasAdd(BufferType& matrix, const BiasType& bias) const {
        forRows(matrix.size(),
                [&](std::size_t i) { matrix.data()[i] += bias.data()[i % matrix.cols()]; });
    }
    void relu(BufferType& matrix) const {
        forRows(matrix.size(), [&](std::size_t i) {
            if (matrix.data()[i] < 0)
                matrix.data()[i] = 0; // Preserve NaN for verification.
        });
    }

    void aggregateNeighbors(const auto& graph, const BufferType& in_features,
                            BufferType& out_aggregated, auto agg_type, const int layer_num,
                            const int* sample[]) {
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
    template <typename F> static void forRows(std::size_t count, F operation) {
        for (std::size_t i = 0; i < count; ++i)
            operation(i);
    }
};

static_assert(Executor<SequentialExecutor>);

} // namespace gnn
