#pragma once

#include "cpu_context.hpp"
#include "data/graph_csc.hpp"
#include "execution/executor.hpp"
#include "gnn/layer.hpp"

#include <cstddef>
#include <limits>
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

    void aggregateGCN(const WorkspaceType::GraphType& graph, const BufferType& input,
                      WorkspaceType::GCNStateType& state, BufferType& output) const {
        state.aggregate(graph, input, output);
    }
    void aggregateNeighbors(const CpuContext::GraphType& graph, const BufferType& input_feats,
                            BufferType& output_feats,
                            gnn::layers::GraphSAGEAggregationType aggType) const {
        const std::uint64_t num_nodes = graph.getNumNodes();
        const std::size_t feat_dim = input_feats.cols();
        output_feats.setShape(num_nodes, feat_dim);

        const auto* col_ptr = graph.colPtrBuffer().data();
        const auto* row_ind = graph.rowIndBuffer().data();
        const auto* weights = graph.weightsBuffer().data();

        for (std::uint64_t u = 0; u < num_nodes; ++u) {
            float* out_ptr = output_feats.data() + u * feat_dim;

            const std::uint64_t start = col_ptr[u];
            const std::uint64_t end = col_ptr[u + 1];
            const std::uint64_t degree = end - start;

            if (degree == 0) {
                std::fill_n(out_ptr, feat_dim, 0.0f);
                continue;
            }

            if (aggType == gnn::layers::GraphSAGEAggregationType::MEAN ||
                aggType == gnn::layers::GraphSAGEAggregationType::SUM) {
                std::fill_n(out_ptr, feat_dim, 0.0f);
                float total_weight = 0;

                for (std::uint64_t i = start; i < end; ++i) {
                    const std::uint64_t v = row_ind[i];
                    const bool mean = aggType == gnn::layers::GraphSAGEAggregationType::MEAN;
                    if (mean && v == u)
                        continue;
                    const float weight = (!mean && graph.hasEdgeWeights()) ? weights[i] : 1.0f;
                    total_weight += weight;
                    const float* in_ptr = input_feats.data() + v * feat_dim;
                    for (std::size_t d = 0; d < feat_dim; ++d) {
                        out_ptr[d] += weight * in_ptr[d];
                    }
                }

                if (aggType == gnn::layers::GraphSAGEAggregationType::MEAN) {
                    const float inv_deg = total_weight > 0 ? 1.0f / total_weight : 0.0f;
                    for (std::size_t d = 0; d < feat_dim; ++d) {
                        out_ptr[d] *= inv_deg;
                    }
                }
            } else if (aggType == gnn::layers::GraphSAGEAggregationType::MAX) {
                std::fill_n(out_ptr, feat_dim, std::numeric_limits<float>::lowest());
                for (std::uint64_t i = start; i < end; ++i) {
                    const std::uint64_t v = row_ind[i];
                    const float* in_ptr = input_feats.data() + v * feat_dim;
                    for (std::size_t d = 0; d < feat_dim; ++d) {
                        out_ptr[d] = std::max(out_ptr[d], in_ptr[d]);
                    }
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
