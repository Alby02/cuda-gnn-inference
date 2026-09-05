#pragma once

#include "cuda_gcn_kernels.cuh"
#include "cuda_workspace.cuh" 
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
        freeIfOwned(scratch_.data());
    }


    [[nodiscard]] cuda::GcnLaunchConfig& launchConfig() noexcept { return config_; }
    [[nodiscard]] const cuda::GcnLaunchConfig& launchConfig() const noexcept { return config_; }

    [[nodiscard]] const DeviceMatrix& aggregate(const DeviceGraph& graph,
                                                const DeviceMatrix& input) {
        ensureMetadata(graph);

        const auto n = static_cast<std::size_t>(graph.getNumNodes());
        ensureScratchCapacity(n, input.cols());

        cuda::launchGcnAggregate(graph, input, invSqrtDeg_, hasExplicitSelfLoop_, scratch_,
                                 config_);
        checkCuda(cudaDeviceSynchronize(), "GCN CUDA aggregation");

        return scratch_;
    }

private:
    static void freeIfOwned(void* pointer) {
        if (pointer != nullptr) {
            static_cast<void>(cudaFree(pointer));
        }
    }

    void ensureMetadata(const DeviceGraph& graph) {
        if (graph.colPtrBuffer().data() == cachedColPtr_) {
            return;
        }

        const auto n = static_cast<std::size_t>(graph.getNumNodes());
        allocate(invSqrtDeg_, n);
        allocate(hasExplicitSelfLoop_, n);

        cuda::launchGcnPrepareMetadata(graph, invSqrtDeg_, hasExplicitSelfLoop_, config_);
        checkCuda(cudaDeviceSynchronize(), "GCN CUDA metadata preparation");

        cachedColPtr_ = graph.colPtrBuffer().data();
    }

    template <typename T> void allocate(cuda::DeviceBuffer<T>& buffer, std::size_t count) {
        if (buffer.physicalSize() >= count) {
            buffer.setLogicalSize(count);
            return;
        }
        freeIfOwned(buffer.data());
        T* raw = nullptr;
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&raw), count * sizeof(T)),
                  "cudaMalloc GCN aggregation state buffer");
        buffer = cuda::DeviceBuffer<T>{raw, count, count};
    }

    void ensureScratchCapacity(std::size_t rows, std::size_t cols) {
        const std::size_t required = rows * cols;
        if (scratch_.physicalSize() < required) {
            freeIfOwned(scratch_.data());
            float* raw = nullptr;
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&raw), required * sizeof(float)),
                      "cudaMalloc GCN aggregation scratch buffer");
            scratch_ = DeviceMatrix{rows, cols, cuda::DeviceBuffer<float>{raw, required, required}};
        } else {
            scratch_.setShape(rows, cols);
        }
    }

    const std::uint64_t* cachedColPtr_{nullptr};
    cuda::DeviceBuffer<float> invSqrtDeg_;
    cuda::DeviceBuffer<std::uint8_t> hasExplicitSelfLoop_;
    DeviceMatrix scratch_{};
    cuda::GcnLaunchConfig config_;
};

} // 