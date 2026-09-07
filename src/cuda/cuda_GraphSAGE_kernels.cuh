#pragma once

#include "data/graph_csc.hpp"
#include "data/matrix.hpp"
#include "device_buffer.cuh"
#include "gnn/layers/graph_sage.hpp"

#include <cstddef>
#include <cstdint>

namespace gnn::cuda {

using DeviceGraph = graph::GraphCSC<DeviceBuffer<std::uint64_t>, DeviceBuffer<float>>;
using DeviceMatrix = Matrix<DeviceBuffer<float>>;

struct GraphSAGELaunchConfig {
    unsigned int aggregateThreadsPerBlock{256};
    std::size_t maxBlocks{65535};
    int maxSamples{0};
};

void launchGraphSAGEAggregate(DeviceGraph graph, DeviceMatrix input, DeviceMatrix output,
                              gnn::layers::GraphSAGEAggregationType aggType,
                              GraphSAGELaunchConfig config = {});

}