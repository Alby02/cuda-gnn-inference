#pragma once

#include "cpu_context.hpp"
#include "execution/executor.hpp"

#include <cstddef>
#include <stdexcept>

namespace gnn {

class ParallelExecutor {
public:
    using WorkspaceType = CpuContext;
    using BufferType = WorkspaceType::BufferType;
    using WeightType = Matrix<HostBuffer<float>>;

    void rowByColumn(const BufferType& left, const BufferType& right, BufferType& output) {
        validateProduct(left, right);
        output.setShape(left.rows(), right.cols());

#pragma omp parallel for collapse(2) schedule(static)
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
    }

private:
    static void validateProduct(const BufferType& left, const BufferType& right) {
        if (left.cols() != right.rows()) {
            throw std::invalid_argument(
                "Matrix product requires the left column count to equal the right row count.");
        }
    }
};

static_assert(Executor<ParallelExecutor>);

} // namespace gnn
