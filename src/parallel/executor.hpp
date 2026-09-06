#pragma once

#include "cpu_context.hpp"
#include "execution/executor.hpp"
#include "gnn/layer.hpp"

#include <cstddef>
#include <stdexcept>

namespace gnn {

class ParallelExecutor {
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
    }

private:
    template <typename F> static void forRows(std::size_t count, F operation) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < count; ++i)
            operation(i);
    }
};

static_assert(Executor<ParallelExecutor>);

} // namespace gnn
