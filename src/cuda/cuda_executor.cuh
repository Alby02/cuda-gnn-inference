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

    void aggregateNeighbors(const decltype(std::declval<WorkspaceType&>().getGraph())& graph,
                            const BufferType& current,
                            BufferType& scratch,
                            gnn::layers::GraphSAGEAggregationType aggType) const;

    [[nodiscard]] unsigned threads() const noexcept { return threads_; }
    [[nodiscard]] layers::CudaGCNAggregationState& gcnState() noexcept { return *gcnState_; }

private:
    unsigned threads_;
    std::unique_ptr<layers::CudaGCNAggregationState> gcnState_ =
        std::make_unique<layers::CudaGCNAggregationState>();
};

static_assert(Executor<CudaExecutor>);

} // namespace gnn