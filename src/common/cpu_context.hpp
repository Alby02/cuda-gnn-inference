#pragma once
#include "data/workload.hpp"
#include "execution/workspace.hpp"
#include "gnn/layers/gcn_aggregation.hpp"
#include "host_buffer.hpp"
#include <algorithm>

namespace gnn {
template <typename AggregationState> class CpuContextBase {
public:
    using GCNStateType = AggregationState;
    using BufferType = Matrix<HostBuffer<float>>;
    using WorkloadType = HostWorkload;
    using GraphType = graph::HostGraphCSC;
    using ModelType = HostModel;
    using HostOutputType = BufferType;

    // The caller keeps the workload alive until this context is no longer used.
    WorkspacePreparation prepare(const WorkloadType& workload) {
        validateWorkload(workload);
        workload_ = &workload;
        gcnState_.prepare(workload.graph);
        const auto& input = workload.input;
        const auto physicalColumns = maximumFeatureWidth(workload.model);
        const auto capacity = input.rows() * physicalColumns;
        if (capacity > current_.physicalSize()) {
            BufferType a(input.rows(), physicalColumns), b(input.rows(), physicalColumns);
            BufferType scratch(input.rows(), physicalColumns),
                branch(input.rows(), physicalColumns);
            std::copy_n(input.data(), input.size(), a.data());
            current_ = std::move(a);
            next_ = std::move(b);
            scratch_ = std::move(scratch);
            branch_ = std::move(branch);
        } else if (input.data() != current_.data()) {
            std::copy_n(input.data(), input.size(), current_.data());
        }
        current_.setShape(input.rows(), input.cols());
        next_.setShape(input.rows(), 0);
        scratch_.setShape(input.rows(), 0);
        branch_.setShape(input.rows(), 0);
        return {};
    }
    // Input restoration is outside compute timing and never allocates.
    void resetInput() {
        const auto& input = workload_->input;
        if (input.data() != current_.data())
            std::copy_n(input.data(), input.size(), current_.data());
        current_.setShape(input.rows(), input.cols());
    }
    [[nodiscard]] GCNStateType& getGCNState() noexcept { return gcnState_; }
    [[nodiscard]] const GraphType& getGraph() const noexcept { return workload_->graph; }
    [[nodiscard]] const ModelType& getModel() const noexcept { return workload_->model; }
    // Borrowed until the next reset, run, prepare, or destruction.
    [[nodiscard]] const HostOutputType& getOutput() const noexcept { return current_; }
    [[nodiscard]] BufferType& current() noexcept { return current_; }
    [[nodiscard]] const BufferType& current() const noexcept { return current_; }
    [[nodiscard]] BufferType& next() noexcept { return next_; }
    [[nodiscard]] BufferType& scratch() noexcept { return scratch_; }
    [[nodiscard]] BufferType& branch() noexcept { return branch_; }
    [[nodiscard]] std::size_t capacityBytes() const noexcept {
        return 4 * current_.physicalSize() * sizeof(float) + gcnState_.capacityBytes();
    }
    void swapBuffers() noexcept { std::swap(current_, next_); }

private:
    const WorkloadType* workload_ = nullptr;
    BufferType current_, next_, scratch_, branch_;
    GCNStateType gcnState_;
};
using CpuContext = CpuContextBase<layers::GCNAggregationState>;
static_assert(Workspace<CpuContext>);
#ifdef _OPENMP
using ParallelCpuContext = CpuContextBase<layers::GCNAggregationStateParallel>;
static_assert(Workspace<ParallelCpuContext>);
#endif
} // namespace gnn
