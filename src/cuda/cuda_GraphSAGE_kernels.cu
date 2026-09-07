#include "cuda_GraphSAGE_kernels.cuh"
#include "cuda_workspace.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstddef>
#include "gnn/layers/graph_sage.hpp"

namespace gnn::cuda {
namespace {

__global__ void graphSAGEAggregateKernel(DeviceGraph graph, DeviceMatrix input,
                                         DeviceMatrix output,
                                         gnn::layers::GraphSAGEAggregationType aggType,
                                         int maxSamples) {
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

        float sumVal = 0.0F;
        float maxVal = -CUDART_INF_F;

        const std::size_t limit = (maxSamples > 0 && neighbors.size() > maxSamples) ? maxSamples : neighbors.size();

        for (std::size_t e = 0; e < limit; ++e) {
            const std::size_t edge_idx = (maxSamples > 0 && neighbors.size() > maxSamples) ? (e * neighbors.size() / maxSamples) : e;
            const std::uint64_t u = neighbors[edge_idx];
            const float w = weighted ? weights[edge_idx] : 1.0F;
            const float val = w * input(static_cast<std::size_t>(u), f);
            
            sumVal += val;
            if (val > maxVal) {
                maxVal = val;
            }
        }

        using AggType = gnn::layers::GraphSAGEAggregationType;
        if (aggType == AggType::MAX) {
            output(static_cast<std::size_t>(v), f) = limit > 0 ? maxVal : 0.0F;
        } else if (aggType == AggType::SUM) {
            output(static_cast<std::size_t>(v), f) = sumVal;
        } else { 
            output(static_cast<std::size_t>(v), f) = limit > 0 ? sumVal / static_cast<float>(limit) : 0.0F;
        }
    }
}

unsigned int blocksFor(std::size_t elements, unsigned int threadsPerBlock, std::size_t maxBlocks) {
    if (elements == 0 || threadsPerBlock == 0) {
        return 0;
    }
    const std::size_t needed = (elements + threadsPerBlock - 1) / threadsPerBlock;
    return static_cast<unsigned int>(std::min(needed, maxBlocks));
}

} 

void launchGraphSAGEAggregate(DeviceGraph graph, DeviceMatrix input, DeviceMatrix output,
                              gnn::layers::GraphSAGEAggregationType aggType,
                              GraphSAGELaunchConfig config) {
    const auto blocks = blocksFor(output.size(), config.aggregateThreadsPerBlock, config.maxBlocks);
    if (blocks == 0) {
        return;
    }
    graphSAGEAggregateKernel<<<blocks, config.aggregateThreadsPerBlock>>>(
        graph, input, output, aggType, config.maxSamples);
    checkCuda(cudaGetLastError(), "launch GraphSAGE aggregation kernel");
}

}