#pragma once

#include "data/buffer.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace gnn {

template <typename T>
    requires std::is_arithmetic_v<T>
class HostBuffer {
public:
    using value_type = T;

    HostBuffer() noexcept = default;

    explicit HostBuffer(std::size_t physicalSize)
        : data_(physicalSize == 0 ? nullptr : std::make_unique<T[]>(physicalSize)),
          dataView_(data_.get()), logicalSize_(physicalSize), physicalSize_(physicalSize) {}

    HostBuffer(const HostBuffer& other) : HostBuffer(other.physicalSize_) {
        logicalSize_ = other.logicalSize_;
        if (physicalSize_ != 0) {
            std::copy_n(other.data_.get(), physicalSize_, data_.get());
        }
    }

    HostBuffer& operator=(const HostBuffer& other) {
        if (this != &other) {
            HostBuffer copy{other};
            swap(copy);
        }
        return *this;
    }

    HostBuffer(HostBuffer&& other) noexcept { swap(other); }

    HostBuffer& operator=(HostBuffer&& other) noexcept {
        if (this != &other) {
            HostBuffer moved{std::move(other)};
            swap(moved);
        }
        return *this;
    }

    [[nodiscard]] GNN_HOST_DEVICE T* data() noexcept { return dataView_; }
    [[nodiscard]] GNN_HOST_DEVICE const T* data() const noexcept { return dataView_; }
    [[nodiscard]] GNN_HOST_DEVICE std::size_t logicalSize() const noexcept { return logicalSize_; }
    [[nodiscard]] GNN_HOST_DEVICE std::size_t physicalSize() const noexcept {
        return physicalSize_;
    }

    GNN_HOST_DEVICE void setLogicalSize(std::size_t logicalSize) {
#if !defined(__CUDA_ARCH__)
        if (logicalSize > physicalSize_) {
            throw std::length_error("HostBuffer logical size exceeds its physical size.");
        }
#endif
        logicalSize_ = logicalSize;
    }

private:
    void swap(HostBuffer& other) noexcept {
        data_.swap(other.data_);
        std::swap(dataView_, other.dataView_);
        std::swap(logicalSize_, other.logicalSize_);
        std::swap(physicalSize_, other.physicalSize_);
    }

    std::unique_ptr<T[]> data_;
    T* dataView_{nullptr};
    std::size_t logicalSize_{0};
    std::size_t physicalSize_{0};
};

static_assert(Buffer<HostBuffer<float>>);

template <typename T> [[nodiscard]] HostBuffer<T> vectorToHostBuffer(std::vector<T> values) {
    HostBuffer<T> buffer{values.size()};
    std::move(values.begin(), values.end(), buffer.data());
    return buffer;
}

} // namespace gnn
