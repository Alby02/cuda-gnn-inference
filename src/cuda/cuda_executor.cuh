#pragma once

#include "cuda_kernels.cuh"
#include "cuda_workspace.cuh"
#include "data/matrix.hpp"
#include "execution/executor.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace gnn {

class CudaExecutor {
public:
    using WorkspaceType = CudaWorkspace;
    using BufferType = WorkspaceType::BufferType;
    using WeightType = BufferType;

    void rowByColumn(const BufferType& left, const WeightType& right, BufferType& output) {
        if (left.cols() != right.rows()) {
            throw std::invalid_argument(
                "Matrix product requires the left column count to equal the right row count.");
        }

        output.setShape(left.rows(), right.cols());
        cuda::launchRowByColumn(left, right, output);
        checkCuda(cudaDeviceSynchronize(), "CUDA matrix multiplication");
    }
};

static_assert(Executor<CudaExecutor>);

} // namespace gnn
