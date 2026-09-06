#include "cuda_gcn_kernels.cuh"
#include "cuda_utils.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>

namespace gnn::cuda {
namespace {

__global__ void gcnPrepareMetadataKernel(DeviceGraph graph, DeviceBuffer<float> invSqrtDeg,
                                         DeviceBuffer<std::uint8_t> hasExplicitSelfLoop) {
    const std::uint64_t first = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint64_t stride = blockDim.x * gridDim.x;
    const std::uint64_t n = graph.getNumNodes();
    const bool weighted = graph.hasEdgeWeights();

    for (std::uint64_t v = first; v < n; v += stride) {
        const auto neighbors = graph.getInNeighbors(v);
        const auto weights = graph.getInWeights(v);

        float weightedDegree = 0.0F;
        bool explicitSelf = false;
        for (std::size_t e = 0; e < neighbors.size(); ++e) {
            const float w = weighted ? weights[e] : 1.0F;
            weightedDegree += w;
            if (neighbors[e] == v) {
                explicitSelf = true;
            }
        }
        if (!explicitSelf) {
            weightedDegree += 1.0F;
        }

        hasExplicitSelfLoop.data()[v] = explicitSelf ? 1 : 0;
        invSqrtDeg.data()[v] = weightedDegree > 0.0F ? rsqrtf(weightedDegree) : 0.0F;
    }
}

__global__ void gcnAggregateKernel(DeviceGraph graph, const DeviceMatrix input,
                                   DeviceBuffer<float> invSqrtDeg,
                                   DeviceBuffer<std::uint8_t> hasExplicitSelfLoop,
                                   DeviceMatrix output) {
    const std::size_t first = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    const std::size_t featureDim = output.cols();
    const std::size_t total = output.size();
    const bool weighted = graph.hasEdgeWeights();

    for (std::size_t index = first; index < total; index += stride) {
        const std::uint64_t v = static_cast<std::uint64_t>(index / featureDim);
        const std::size_t f = index % featureDim;

        const auto neighbors = graph.getInNeighbors(v);
        const auto weights = graph.getInWeights(v);
        const float invSqrtDv = invSqrtDeg.data()[v];

        float value = 0.0F;
        for (std::size_t e = 0; e < neighbors.size(); ++e) {
            const std::uint64_t u = neighbors[e];
            const float w = weighted ? weights[e] : 1.0F;
            const float alpha = w * invSqrtDv * invSqrtDeg.data()[u];
            value += alpha * input(static_cast<std::size_t>(u), f);
        }

        if (hasExplicitSelfLoop.data()[v] == 0) {
            value += invSqrtDv * invSqrtDv * input(static_cast<std::size_t>(v), f);
        }

        output(static_cast<std::size_t>(v), f) = value;
    }
}

unsigned int blocksFor(std::size_t elements, unsigned int threadsPerBlock, std::size_t maxBlocks) {
    if (elements == 0 || threadsPerBlock == 0) {
        return 0;
    }
    const std::size_t needed = (elements + threadsPerBlock - 1) / threadsPerBlock;
    return static_cast<unsigned int>(std::min(needed, maxBlocks));
}

} // namespace

void launchGcnPrepareMetadata(DeviceGraph graph, DeviceBuffer<float> invSqrtDeg,
                              DeviceBuffer<std::uint8_t> hasExplicitSelfLoop,
                              GcnLaunchConfig config) {
    const auto blocks =
        blocksFor(graph.getNumNodes(), config.metadataThreadsPerBlock, config.maxBlocks);
    if (blocks == 0) {
        return;
    }
    gcnPrepareMetadataKernel<<<blocks, config.metadataThreadsPerBlock>>>(graph, invSqrtDeg,
                                                                         hasExplicitSelfLoop);
    checkCuda(cudaGetLastError(), "launch GCN metadata preparation kernel");
}

void launchGcnAggregate(DeviceGraph graph, DeviceMatrix input, DeviceBuffer<float> invSqrtDeg,
                        DeviceBuffer<std::uint8_t> hasExplicitSelfLoop, DeviceMatrix output,
                        GcnLaunchConfig config) {
    const auto blocks = blocksFor(output.size(), config.aggregateThreadsPerBlock, config.maxBlocks);
    if (blocks == 0) {
        return;
    }
    gcnAggregateKernel<<<blocks, config.aggregateThreadsPerBlock>>>(graph, input, invSqrtDeg,
                                                                    hasExplicitSelfLoop, output);
    checkCuda(cudaGetLastError(), "launch GCN aggregation kernel");
}

} // namespace gnn::cuda
