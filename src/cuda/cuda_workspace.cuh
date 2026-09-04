#pragma once

#include "data/graph_csc.hpp"
#include "data/matrix.hpp"
#include "device_buffer.cuh"
#include "execution/workspace.hpp"
#include "host_buffer.hpp"
#include "host_graph.hpp"
#include "host_layers.hpp"
#include "gnn/model.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <concepts>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace gnn {

inline void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

class CudaWorkspace {
public:
    using BufferType = Matrix<cuda::DeviceBuffer<float>>;
    using GraphType =
        graph::GraphCSC<cuda::DeviceBuffer<std::uint64_t>, cuda::DeviceBuffer<float>>;
    using GCNType = layers::GCNLayer<BufferType, cuda::DeviceBuffer<float>>;
    using GraphSAGEType = layers::GraphSAGELayer<BufferType, cuda::DeviceBuffer<float>>;
    using ModelType = Model<GCNType, GraphSAGEType>;

    CudaWorkspace() = default;
    CudaWorkspace(const CudaWorkspace&) = delete;
    CudaWorkspace& operator=(const CudaWorkspace&) = delete;
    CudaWorkspace(CudaWorkspace&&) = delete;
    CudaWorkspace& operator=(CudaWorkspace&&) = delete;

    ~CudaWorkspace() {
        for (void* allocation : allocations_) {
            static_cast<void>(cudaFree(allocation));
        }
    }

    template <typename T>
        requires std::is_arithmetic_v<T>
    [[nodiscard]] cuda::DeviceBuffer<T> allocateBuffer(std::size_t physicalSize) {
        T* data = nullptr;
        if (physicalSize != 0) {
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), physicalSize * sizeof(T)),
                      "cudaMalloc workspace buffer");
            allocations_.push_back(data);
        }
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

    template <typename T>
    [[nodiscard]] HostBuffer<T> downloadBuffer(const cuda::DeviceBuffer<T>& device) const {
        HostBuffer<T> host{device.physicalSize()};
        host.setLogicalSize(device.logicalSize());
        if (device.physicalSize() != 0) {
            checkCuda(cudaMemcpy(host.data(), device.data(), device.physicalSize() * sizeof(T),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy workspace buffer to host");
        }
        return host;
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
                        deviceLayers.emplace_back(GCNType{
                            uploadMatrix(hostLayer.getWNeigh()), uploadBuffer(hostLayer.getBias()),
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

    void prepare(const BufferType& input, std::size_t physicalColumns) {
        current_ = makeMatrix(input.rows(), physicalColumns);
        next_ = makeMatrix(input.rows(), physicalColumns);
        if (input.size() != 0) {
            checkCuda(cudaMemcpy(current_.data(), input.data(), input.size() * sizeof(float),
                                 cudaMemcpyDeviceToDevice),
                      "cudaMemcpy input into CUDA workspace");
        }
        current_.setShape(input.rows(), input.cols());
        next_.setShape(input.rows(), 0);
    }

    [[nodiscard]] BufferType& current() noexcept { return current_; }
    [[nodiscard]] const BufferType& current() const noexcept { return current_; }
    [[nodiscard]] BufferType& next() noexcept { return next_; }

    void swapBuffers() noexcept { std::swap(current_, next_); }

private:
    [[nodiscard]] BufferType makeMatrix(std::size_t rows, std::size_t physicalColumns) {
        auto storage = allocateBuffer<float>(rows * physicalColumns);
        storage.setLogicalSize(rows * physicalColumns);
        return {rows, physicalColumns, storage};
    }

    std::vector<void*> allocations_;
    BufferType current_{};
    BufferType next_{};
};

using CudaContext = CudaWorkspace;

static_assert(Workspace<CudaWorkspace>);
static_assert(std::is_trivially_copyable_v<CudaWorkspace::BufferType>);
static_assert(std::is_trivially_copyable_v<CudaWorkspace::GraphType>);
static_assert(std::is_trivially_copyable_v<CudaWorkspace::GCNType>);
static_assert(std::is_trivially_copyable_v<CudaWorkspace::GraphSAGEType>);

} // namespace gnn
