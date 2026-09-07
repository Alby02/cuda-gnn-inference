#include "cuda_GraphSAGE_kernels.cuh"
#include "cuda_executor.cuh"
#include "cuda_utils.cuh"
#include <algorithm>
namespace gnn {
namespace {
using DeviceMatrix = CudaExecutor::BufferType;

__global__ void linearKernel(const DeviceMatrix left, const DeviceMatrix right, DeviceMatrix out) {
    const std::size_t first = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = std::size_t(blockDim.x) * gridDim.x;
    for (std::size_t i = first; i < out.size(); i += stride) {
        const auto row = i / out.cols(), col = i % out.cols();
        float value = 0;
        for (std::size_t k = 0; k < left.cols(); ++k)
            value += left(row, k) * right(k, col);
        out(row, col) = value;
    }
}

__global__ void addKernel(const DeviceMatrix left, const DeviceMatrix right, DeviceMatrix out) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < out.size();
         i += std::size_t(blockDim.x) * gridDim.x)
        out.data()[i] = left.data()[i] + right.data()[i];
}

__global__ void biasKernel(DeviceMatrix out, const CudaExecutor::BiasType bias) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < out.size();
         i += std::size_t(blockDim.x) * gridDim.x)
        out.data()[i] += bias.data()[i % out.cols()];
}

__global__ void reluKernel(DeviceMatrix out) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < out.size();
         i += std::size_t(blockDim.x) * gridDim.x)
        if (out.data()[i] < 0)
            out.data()[i] = 0;
}

unsigned blocks(std::size_t size, unsigned threads) {
    return static_cast<unsigned>(
        std::min<std::size_t>(size / threads + (size % threads != 0), 65535));
}
} 

void CudaExecutor::rowByColumn(const BufferType& left, const WeightType& right,
                               BufferType& output) const {
    if (left.cols() != right.rows())
        throw std::invalid_argument("Incompatible matrix product shapes.");
    output.setShape(left.rows(), right.cols());
    if (!output.size())
        return;
    linearKernel<<<blocks(output.size(), threads_), threads_>>>(left, right, output);
    checkCuda(cudaGetLastError(), "launch linear kernel");
}

void CudaExecutor::add(const BufferType& left, const BufferType& right, BufferType& output) const {
    output.setShape(left.rows(), left.cols());
    if (!output.size())
        return;
    addKernel<<<blocks(output.size(), threads_), threads_>>>(left, right, output);
    checkCuda(cudaGetLastError(), "launch branch addition kernel");
}

void CudaExecutor::biasAdd(BufferType& output, const BiasType& bias) const {
    if (!output.size())
        return;
    biasKernel<<<blocks(output.size(), threads_), threads_>>>(output, bias);
    checkCuda(cudaGetLastError(), "launch bias kernel");
}

void CudaExecutor::relu(BufferType& output) const {
    if (!output.size())
        return;
    reluKernel<<<blocks(output.size(), threads_), threads_>>>(output);
    checkCuda(cudaGetLastError(), "launch ReLU kernel");
}

void CudaExecutor::aggregateGCN(const WorkspaceType::GraphType& graph, const BufferType& input,
                                WorkspaceType::GCNStateType& state, BufferType& output) const {
    state.aggregate(graph, input, output, threads_);
}

void CudaExecutor::aggregateNeighbors(const WorkspaceType::GraphType& graph,
                                      const BufferType& input, BufferType& output,
                                      layers::GraphSAGEAggregationType aggType, int layer) const {
    output.setShape(input.rows(), input.cols());
    cuda::GraphSAGELaunchConfig config;
    config.aggregateThreadsPerBlock = threads_;
    if (layer >= 0) {
        config.maxSamples = (layer == 0) ? 25 : std::max(10 - layer * 2, 5);
    }
    cuda::launchGraphSAGEAggregate(graph, input, output, aggType, config);
}
}