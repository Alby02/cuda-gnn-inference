#include "cuda_GraphSAGE_kernels.cuh"
#include "cuda_workspace.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#include "gnn/layers/graph_sage.hpp"
#include <algorithm>
#include <cstddef>

namespace gnn::cuda {
namespace {

__global__ void graphSAGEAggregateKernel(DeviceGraph graph, DeviceMatrix input, DeviceMatrix output,
                                         gnn::layers::GraphSAGEAggregationType aggType) {
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

        const bool mean = aggType == layers::GraphSAGEAggregationType::MEAN;
        float totalWeight = 0.0F;
        float sumVal = 0.0F;
        float maxVal = -CUDART_INF_F;

        for (std::size_t e = 0; e < neighbors.size(); ++e) {
            const std::uint64_t u = neighbors[e];
            if (mean && u == v)
                continue;
            const float w = mean && weighted ? weights[e] : 1.0F;
            totalWeight += w;
            const float val = w * input(static_cast<std::size_t>(u), f);

            sumVal += val;
            if (val > maxVal) {
                maxVal = val;
            }
        }

        // if (static_cast<int>(aggType) == 1) {
        //     output(static_cast<std::size_t>(v), f) = neighbors.size() > 0 ? maxVal : 0.0F;
        // } else if (static_cast<int>(aggType) == 2) {
        //     output(static_cast<std::size_t>(v), f) = sumVal;
        // } else {
        //     output(static_cast<std::size_t>(v), f) = totalWeight > 0 ? sumVal / totalWeight :
        //     0.0F;
        // }
        using AggType = gnn::layers::GraphSAGEAggregationType;
        if (aggType == AggType::MAX) {
            output(static_cast<std::size_t>(v), f) = neighbors.size() > 0 ? maxVal : 0.0F;
        } else if (aggType == AggType::SUM) {
            output(static_cast<std::size_t>(v), f) = sumVal;
        } else { // MEAN
            output(static_cast<std::size_t>(v), f) = totalWeight > 0 ? sumVal / totalWeight : 0.0F;
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

} // namespace

void launchGraphSAGEAggregate(DeviceGraph graph, DeviceMatrix input, DeviceMatrix output,
                              gnn::layers::GraphSAGEAggregationType aggType,
                              GraphSAGELaunchConfig config) {
    const auto blocks = blocksFor(output.size(), config.aggregateThreadsPerBlock, config.maxBlocks);
    if (blocks == 0) {
        return;
    }
    graphSAGEAggregateKernel<<<blocks, config.aggregateThreadsPerBlock>>>(graph, input, output,
                                                                          aggType);
    checkCuda(cudaGetLastError(), "launch GraphSAGE aggregation kernel");
}

} // namespace gnn::cuda