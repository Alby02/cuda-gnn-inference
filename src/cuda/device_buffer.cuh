#pragma once

#include "data/buffer.hpp"
#include <cstddef>
#include <type_traits>

namespace gnn::cuda {

template <typename T>
    requires std::is_arithmetic_v<T>
class DeviceBuffer {
public:
    using value_type = T;

    GNN_HOST_DEVICE DeviceBuffer() noexcept {}
    GNN_HOST_DEVICE DeviceBuffer(T* data, std::size_t logicalSize,
                                 std::size_t physicalSize) noexcept
        : data_(data), logicalSize_(logicalSize), physicalSize_(physicalSize) {}

    [[nodiscard]] GNN_HOST_DEVICE T* data() noexcept { return data_; }
    [[nodiscard]] GNN_HOST_DEVICE const T* data() const noexcept { return data_; }
    [[nodiscard]] GNN_HOST_DEVICE std::size_t logicalSize() const noexcept {
        return logicalSize_;
    }
    [[nodiscard]] GNN_HOST_DEVICE std::size_t physicalSize() const noexcept {
        return physicalSize_;
    }

    GNN_HOST_DEVICE void setLogicalSize(std::size_t logicalSize) noexcept {
        logicalSize_ = logicalSize;
    }

private:
    T* data_{nullptr};
    std::size_t logicalSize_{0};
    std::size_t physicalSize_{0};
};

static_assert(Buffer<DeviceBuffer<float>>);

} // namespace gnn::cuda
