#pragma once

#include "cuda_utils.cuh"
#include "data/graph_csc.hpp"
#include "data/matrix.hpp"
#include "data/workload.hpp"
#include "device_buffer.cuh"
#include "execution/workspace.hpp"
#include "gnn/model.hpp"
#include "host_buffer.hpp"
#include "host_graph.hpp"
#include "host_layers.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <concepts>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace gnn {

class CudaWorkspace {
public:
    using BufferType = Matrix<cuda::DeviceBuffer<float>>;
    using GraphType = graph::GraphCSC<cuda::DeviceBuffer<std::uint64_t>, cuda::DeviceBuffer<float>>;
    using GCNType = layers::GCNLayer<BufferType, cuda::DeviceBuffer<float>>;
    using GraphSAGEType = layers::GraphSAGELayer<BufferType, cuda::DeviceBuffer<float>>;
    using ModelType = Model<GCNType, GraphSAGEType>;

    using WorkloadType = HostWorkload;
    using HostOutputType = layers::HostMatrix;

    WorkspacePreparation prepare(const WorkloadType& workload) {
        validateWorkload(workload);
        // Discard the previous workload and its device allocations on re-preparation.
        model_.reset();
        graph_.reset();
        allocations_.clear();
        const auto begin = std::chrono::steady_clock::now();
        graph_.emplace(uploadGraph(workload.graph));
        model_.emplace(uploadModel(workload.model));
        input_ = uploadMatrix(workload.input);
        checkCuda(cudaDeviceSynchronize(), "complete CUDA workload upload");
        const double uploadMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                .count();
        const auto width = maximumFeatureWidth(workload.model);
        current_ = makeMatrix(input_.rows(), width);
        next_ = makeMatrix(input_.rows(), width);
        scratch_ = makeMatrix(input_.rows(), width);
        branch_ = makeMatrix(input_.rows(), width);
        resetInput();
        next_.setShape(input_.rows(), 0);
        scratch_.setShape(input_.rows(), 0);
        branch_.setShape(input_.rows(), 0);
        return {uploadMs};
    }
    // Device-to-device restoration; no allocation or host upload per iteration.
    void resetInput() {
        if (input_.size() != 0) {
            checkCuda(cudaMemcpy(current_.data(), input_.data(), input_.size() * sizeof(float),
                                 cudaMemcpyDeviceToDevice),
                      "reset CUDA input");
            checkCuda(cudaDeviceSynchronize(), "complete CUDA input reset outside compute timing");
        }
        current_.setShape(input_.rows(), input_.cols());
    }
    [[nodiscard]] const GraphType& getGraph() const noexcept { return *graph_; }
    [[nodiscard]] const ModelType& getModel() const noexcept { return *model_; }
    // Owning host result; performs a device-to-host copy outside compute timing.
    [[nodiscard]] HostOutputType getOutput() const { return downloadMatrix(current_); }
    [[nodiscard]] BufferType& scratch() noexcept { return scratch_; }
    [[nodiscard]] BufferType& branch() noexcept { return branch_; }
    [[nodiscard]] std::size_t capacityBytes() const noexcept {
        std::size_t total = 0;
        for (const auto& allocation : allocations_)
            total += allocation.bytes();
        return total;
    }

    [[nodiscard]] BufferType& current() noexcept { return current_; }
    [[nodiscard]] const BufferType& current() const noexcept { return current_; }
    [[nodiscard]] BufferType& next() noexcept { return next_; }

    void swapBuffers() noexcept { std::swap(current_, next_); }

private:
    template <typename T>
        requires std::is_arithmetic_v<T>
    [[nodiscard]] cuda::DeviceBuffer<T> allocateBuffer(std::size_t physicalSize) {
        cuda::Allocation allocation(physicalSize * sizeof(T));
        auto* data = static_cast<T*>(allocation.data());
        allocations_.push_back(std::move(allocation));
        return {data, physicalSize, physicalSize};
    }

    template <typename T>
    [[nodiscard]] cuda::DeviceBuffer<T> uploadBuffer(const HostBuffer<T>& host) {
        auto device = allocateBuffer<T>(host.physicalSize());
        device.setLogicalSize(host.logicalSize());
        if (host.physicalSize() != 0) {
            checkCuda(cudaMemcpy(device.data(), host.data(), host.physicalSize() * sizeof(T),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy host buffer to workspace");
        }
        return device;
    }

    [[nodiscard]] BufferType uploadMatrix(const Matrix<HostBuffer<float>>& host) {
        auto storage = allocateBuffer<float>(host.physicalSize());
        storage.setLogicalSize(host.size());
        if (host.physicalSize() != 0) {
            checkCuda(cudaMemcpy(storage.data(), host.data(), host.physicalSize() * sizeof(float),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy host matrix to workspace");
        }
        return {host.rows(), host.cols(), storage};
    }

    [[nodiscard]] Matrix<HostBuffer<float>> downloadMatrix(const BufferType& device) const {
        Matrix<HostBuffer<float>> host{device.rows(), device.cols()};
        if (device.size() != 0) {
            checkCuda(cudaMemcpy(host.data(), device.data(), device.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy workspace matrix to host");
        }
        return host;
    }

    [[nodiscard]] GraphType uploadGraph(const graph::HostGraphCSC& host) {
        return GraphType{host.isDirected(), uploadBuffer(host.colPtrBuffer()),
                         uploadBuffer(host.rowIndBuffer()), uploadBuffer(host.weightsBuffer())};
    }

    [[nodiscard]] ModelType
    uploadModel(const Model<layers::HostGCN, layers::HostGraphSAGE>& hostModel) {
        std::vector<ModelType::LayerVariant> deviceLayers;
        deviceLayers.reserve(hostModel.numLayers());

        for (const auto& layer : hostModel.getLayers()) {
            std::visit(
                [&](const auto& hostLayer) {
                    using HostLayer = std::decay_t<decltype(hostLayer)>;
                    if constexpr (std::same_as<HostLayer, layers::HostGCN>) {
                        deviceLayers.emplace_back(GCNType{uploadMatrix(hostLayer.getWNeigh()),
                                                          uploadBuffer(hostLayer.getBias()),
                                                          hostLayer.getActType()});
                    } else {
                        deviceLayers.emplace_back(GraphSAGEType{
                            uploadMatrix(hostLayer.getWNeigh()), uploadMatrix(hostLayer.getWSelf()),
                            uploadBuffer(hostLayer.getBias()), hostLayer.getAggType(),
                            hostLayer.getActType()});
                    }
                },
                layer);
        }

        return ModelType{std::move(deviceLayers)};
    }

    [[nodiscard]] BufferType makeMatrix(std::size_t rows, std::size_t physicalColumns) {
        auto storage = allocateBuffer<float>(rows * physicalColumns);
        storage.setLogicalSize(rows * physicalColumns);
        return {rows, physicalColumns, storage};
    }

    std::vector<cuda::Allocation> allocations_;
    std::optional<GraphType> graph_;
    std::optional<ModelType> model_;
    BufferType input_{};
    BufferType current_{};
    BufferType next_{};
    BufferType scratch_{};
    BufferType branch_{};
};

using CudaContext = CudaWorkspace;

static_assert(Workspace<CudaWorkspace>);
static_assert(std::is_trivially_copyable_v<CudaWorkspace::BufferType>);
static_assert(std::is_trivially_copyable_v<CudaWorkspace::GraphType>);
static_assert(std::is_trivially_copyable_v<CudaWorkspace::GCNType>);
static_assert(std::is_trivially_copyable_v<CudaWorkspace::GraphSAGEType>);

} // namespace gnn
