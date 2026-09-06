#pragma once
#include "cuda_workspace.cuh"
#include "execution/executor.hpp"
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
    void aggregateGCN(const WorkspaceType::GraphType& graph, const BufferType& input,
                      WorkspaceType::GCNStateType& state, BufferType& output) const;
    void aggregateNeighbors(const WorkspaceType::GraphType& graph, const BufferType& input,
                            BufferType& output, layers::GraphSAGEAggregationType aggType) const;
    [[nodiscard]] unsigned threads() const noexcept { return threads_; }

private:
    unsigned threads_;
};
static_assert(Executor<CudaExecutor>);
} // namespace gnn
