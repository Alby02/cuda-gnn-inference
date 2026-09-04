#include "cuda_kernels.cuh"
#include "cuda_workspace.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>

namespace gnn::cuda {
namespace {

using MatrixView = Matrix<DeviceBuffer<float>>;

__global__ void rowByColumnKernel(MatrixView left, MatrixView right, MatrixView output) {
    const std::size_t first = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;

    for (std::size_t index = first; index < output.size(); index += stride) {
        const std::size_t row = index / output.cols();
        const std::size_t column = index % output.cols();
        float value = 0.0F;
        for (std::size_t inner = 0; inner < left.cols(); ++inner) {
            value += left(row, inner) * right(inner, column);
        }
        output(row, column) = value;
    }
}

} // namespace

void launchRowByColumn(MatrixView left, MatrixView right, MatrixView output) {
    constexpr unsigned int threadsPerBlock = 256;
    constexpr std::size_t maxBlocks = 65535;
    const std::size_t outputElements = output.size();
    if (outputElements == 0) {
        return;
    }
    const auto requiredBlocks =
        (outputElements + threadsPerBlock - 1) / static_cast<std::size_t>(threadsPerBlock);
    const auto blocks = static_cast<unsigned int>(std::min(requiredBlocks, maxBlocks));

    rowByColumnKernel<<<blocks, threadsPerBlock>>>(left, right, output);
    checkCuda(cudaGetLastError(), "launch CUDA matrix multiplication kernel");
}

} // namespace gnn::cuda
