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

    explicit HostBuffer(std::size_t physicalSize)
        : data_(physicalSize == 0 ? nullptr : std::make_unique<T[]>(physicalSize)),
          logicalSize_(physicalSize), physicalSize_(physicalSize) {}

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

    HostBuffer(HostBuffer&&) noexcept = default;
    HostBuffer& operator=(HostBuffer&&) noexcept = default;

    [[nodiscard]] T* data() noexcept { return data_.get(); }
    [[nodiscard]] const T* data() const noexcept { return data_.get(); }
    [[nodiscard]] std::size_t logicalSize() const noexcept { return logicalSize_; }
    [[nodiscard]] std::size_t physicalSize() const noexcept { return physicalSize_; }

    void setLogicalSize(std::size_t logicalSize) {
        if (logicalSize > physicalSize_) {
            throw std::length_error("HostBuffer logical size exceeds its physical size.");
        }
        logicalSize_ = logicalSize;
    }

private:
    void swap(HostBuffer& other) noexcept {
        data_.swap(other.data_);
        std::swap(logicalSize_, other.logicalSize_);
        std::swap(physicalSize_, other.physicalSize_);
    }

    std::unique_ptr<T[]> data_;
    std::size_t logicalSize_;
    std::size_t physicalSize_;
};

static_assert(Buffer<HostBuffer<float>>);

template <typename T> [[nodiscard]] HostBuffer<T> vectorToHostBuffer(std::vector<T> values) {
    HostBuffer<T> buffer{values.size()};
    std::move(values.begin(), values.end(), buffer.data());
    return buffer;
}

} // namespace gnn
