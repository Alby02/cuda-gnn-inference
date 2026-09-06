#pragma once
#include "cuda_gcn_aggregation.cuh"
#include "cuda_workspace.cuh"
#include "execution/executor.hpp"
#include <memory>
namespace gnn {
class CudaExecutor {
public:
    using WorkspaceType = CudaWorkspace;
    using BufferType = WorkspaceType::BufferType;
    using WeightType = BufferType;
    using BiasType = cuda::DeviceBuffer<float>;
    CudaExecutor(unsigned threads = 256) : threads_(threads) {}
    void rowByColumn(const BufferType& left, const WeightType& right, BufferType& output) const;
    void add(const BufferType& left, const BufferType& right, BufferType& output) const;
    void biasAdd(BufferType& output, const BiasType& bias) const;
    void relu(BufferType& output) const;
    [[nodiscard]] layers::CudaGCNAggregationState& gcnState() noexcept { return *gcnState_; }

private:
    unsigned threads_;
    // The aggregation state is non-movable; its owner moves with the executor into the runtime.
    std::unique_ptr<layers::CudaGCNAggregationState> gcnState_ =
        std::make_unique<layers::CudaGCNAggregationState>();
};
static_assert(Executor<CudaExecutor>);
} // namespace gnn
