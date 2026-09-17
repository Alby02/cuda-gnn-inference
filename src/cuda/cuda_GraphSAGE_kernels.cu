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
                                         gnn::layers::GraphSAGEAggregationType aggType,
                                         int maxSamples) {
    const std::size_t first = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    const std::size_t featureDim = output.cols();
    const std::size_t total = output.size();
    const bool weighted = graph.hasEdgeWeights();
    const std::size_t maximumSamples =
        maxSamples > 0 ? static_cast<std::size_t>(maxSamples) : 0;

    for (std::size_t index = first; index < total; index += stride) {
        const std::uint64_t v = static_cast<std::uint64_t>(index / featureDim);
        const std::size_t f = index % featureDim;

        const auto neighbors = graph.getInNeighbors(v);
        const auto weights = graph.getInWeights(v);

        float sumVal = 0.0F;
        float maxVal = -CUDART_INF_F;
        std::size_t contributingNeighbors = 0;

        const bool sampling = maximumSamples > 0 && neighbors.size() > maximumSamples;
        const std::size_t limit = sampling ? maximumSamples : neighbors.size();

        for (std::size_t e = 0; e < limit; ++e) {
            const std::size_t edge_idx = sampling ? (e * neighbors.size() / maximumSamples) : e;
            const std::uint64_t u = neighbors[edge_idx];
            const float inputValue = input(static_cast<std::size_t>(u), f);

            using AggType = gnn::layers::GraphSAGEAggregationType;
            if (aggType == AggType::MEAN) {
                // Match SAGEConv and the CPU executors: the root/self branch is
                // handled separately, and neighbor means do not use edge weights.
                if (u == v) {
                    continue;
                }
                sumVal += inputValue;
            } else if (aggType == AggType::SUM) {
                const float weight = weighted ? weights[edge_idx] : 1.0F;
                sumVal += weight * inputValue;
            } else if (inputValue > maxVal) {
                // MAX is over raw neighbor features, as on the CPU backends.
                maxVal = inputValue;
            }
            ++contributingNeighbors;
        }

        using AggType = gnn::layers::GraphSAGEAggregationType;
        if (aggType == AggType::MAX) {
            output(static_cast<std::size_t>(v), f) = contributingNeighbors > 0 ? maxVal : 0.0F;
        } else if (aggType == AggType::SUM) {
            output(static_cast<std::size_t>(v), f) = sumVal;
        } else {
            output(static_cast<std::size_t>(v), f) =
                contributingNeighbors > 0 ? sumVal / static_cast<float>(contributingNeighbors)
                                          : 0.0F;
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
    graphSAGEAggregateKernel<<<blocks, config.aggregateThreadsPerBlock>>>(
        graph, input, output, aggType, config.maxSamples);
    checkCuda(cudaGetLastError(), "launch GraphSAGE aggregation kernel");
}

} // namespace gnn::cuda
