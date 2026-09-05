#pragma once

#include "data/graph_csc.hpp"
#include "data/matrix.hpp"
#include "device_buffer.cuh"

#include <cstddef>
#include <cstdint>

namespace gnn::cuda {

using DeviceGraph = graph::GraphCSC<DeviceBuffer<std::uint64_t>, DeviceBuffer<float>>;
using DeviceMatrix = Matrix<DeviceBuffer<float>>;


struct GcnLaunchConfig {
    unsigned int metadataThreadsPerBlock{128};
    unsigned int aggregateThreadsPerBlock{256};
    std::size_t maxBlocks{65535};
};

// One thread per destination node
void launchGcnPrepareMetadata(DeviceGraph graph, DeviceBuffer<float> invSqrtDeg,
                              DeviceBuffer<std::uint8_t> hasExplicitSelfLoop,
                              GcnLaunchConfig config = {});


void launchGcnAggregate(DeviceGraph graph, DeviceMatrix input, DeviceBuffer<float> invSqrtDeg,
                        DeviceBuffer<std::uint8_t> hasExplicitSelfLoop, DeviceMatrix output,
                        GcnLaunchConfig config = {});

} // 
