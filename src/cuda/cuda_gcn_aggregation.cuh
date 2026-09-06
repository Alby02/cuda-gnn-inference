#pragma once

#include "cuda_gcn_kernels.cuh"
#include "cuda_utils.cuh"
#include "data/graph_csc.hpp"
#include "data/matrix.hpp"
#include "device_buffer.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gnn::layers {

// CUDA counterpart of GCNAggregationState / GCNAggregationStateParallel
class CudaGCNAggregationState {
public:
    using DeviceGraph = cuda::DeviceGraph;
    using DeviceMatrix = cuda::DeviceMatrix;

    CudaGCNAggregationState() = default;
    CudaGCNAggregationState(const CudaGCNAggregationState&) = delete;
    CudaGCNAggregationState& operator=(const CudaGCNAggregationState&) = delete;
    CudaGCNAggregationState(CudaGCNAggregationState&&) = delete;
    CudaGCNAggregationState& operator=(CudaGCNAggregationState&&) = delete;

    ~CudaGCNAggregationState() {
        freeIfOwned(invSqrtDeg_.data());
        freeIfOwned(hasExplicitSelfLoop_.data());
    }

    void prepare(const DeviceGraph& graph) {
        const auto n = static_cast<std::size_t>(graph.getNumNodes());
        allocate(invSqrtDeg_, n);
        allocate(hasExplicitSelfLoop_, n);
        cuda::launchGcnPrepareMetadata(graph, invSqrtDeg_, hasExplicitSelfLoop_);
        checkCuda(cudaDeviceSynchronize(), "GCN CUDA metadata preparation");
    }

    void aggregate(const DeviceGraph& graph, const DeviceMatrix& input, DeviceMatrix& output,
                   unsigned threads) const {
        output.setShape(input.rows(), input.cols());
        cuda::GcnLaunchConfig config;
        config.aggregateThreadsPerBlock = threads;
        cuda::launchGcnAggregate(graph, input, invSqrtDeg_, hasExplicitSelfLoop_, output, config);
    }

    [[nodiscard]] std::size_t capacityBytes() const noexcept {
        return invSqrtDeg_.physicalSize() * sizeof(float) +
               hasExplicitSelfLoop_.physicalSize() * sizeof(std::uint8_t);
    }

private:
    static void freeIfOwned(void* pointer) {
        if (pointer != nullptr) {
            static_cast<void>(cudaFree(pointer));
        }
    }

    template <typename T> void allocate(cuda::DeviceBuffer<T>& buffer, std::size_t count) {
        if (buffer.physicalSize() >= count) {
            buffer.setLogicalSize(count);
            return;
        }
        T* raw = nullptr;
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&raw), count * sizeof(T)),
                  "cudaMalloc GCN aggregation state buffer");
        freeIfOwned(buffer.data());
        buffer = cuda::DeviceBuffer<T>{raw, count, count};
    }

    cuda::DeviceBuffer<float> invSqrtDeg_;
    cuda::DeviceBuffer<std::uint8_t> hasExplicitSelfLoop_;
};

} // namespace gnn::layers