#pragma once

#include "cuda_GraphSAGE_kernels.cuh"
#include "cuda_workspace.cuh"
#include "data/graph_csc.hpp"
#include "data/matrix.hpp"
#include "device_buffer.cuh"
#include "../common/gnn/layer.hpp"
#include "gnn/layers/graph_sage.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gnn::layers {

class CudaGraphSAGEAggregationState {
public:
    using DeviceGraph = cuda::DeviceGraph;
    using DeviceMatrix = cuda::DeviceMatrix;

    CudaGraphSAGEAggregationState() = default;
    CudaGraphSAGEAggregationState(const CudaGraphSAGEAggregationState&) = delete;
    CudaGraphSAGEAggregationState& operator=(const CudaGraphSAGEAggregationState&) = delete;
    CudaGraphSAGEAggregationState(CudaGraphSAGEAggregationState&&) = delete;
    CudaGraphSAGEAggregationState& operator=(CudaGraphSAGEAggregationState&&) = delete;

    ~CudaGraphSAGEAggregationState() {
        freeIfOwned(scratch_.data());
    }

    [[nodiscard]] cuda::GraphSAGELaunchConfig& launchConfig() noexcept { return config_; }
    [[nodiscard]] const cuda::GraphSAGELaunchConfig& launchConfig() const noexcept { return config_; }

    [[nodiscard]] const DeviceMatrix& aggregate(const DeviceGraph& graph,
                                                const DeviceMatrix& input,
                                                GraphSAGEAggregationType aggType) {
        const auto n = static_cast<std::size_t>(graph.getNumNodes());
        ensureScratchCapacity(n, input.cols());

        cuda::launchGraphSAGEAggregate(graph, input, scratch_, aggType, config_);
        checkCuda(cudaDeviceSynchronize(), "GraphSAGE CUDA aggregation");

        return scratch_;
    }

private:
    static void freeIfOwned(void* pointer) {
        if (pointer != nullptr) {
            static_cast<void>(cudaFree(pointer));
        }
    }

    void ensureScratchCapacity(std::size_t rows, std::size_t cols) {
        const std::size_t required = rows * cols;
        if (scratch_.physicalSize() < required) {
            freeIfOwned(scratch_.data());
            float* raw = nullptr;
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&raw), required * sizeof(float)),
                      "cudaMalloc GraphSAGE aggregation scratch buffer");
            scratch_ = DeviceMatrix{rows, cols, cuda::DeviceBuffer<float>{raw, required, required}};
        } else {
            scratch_.setShape(rows, cols);
        }
    }

    DeviceMatrix scratch_{};
    cuda::GraphSAGELaunchConfig config_;
};

} // namespace gnn::layers