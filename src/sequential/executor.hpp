#pragma once

#include "cpu_context.hpp"
#include "execution/executor.hpp"

#include <cstddef>
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
