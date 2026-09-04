#pragma once

#include "../cuda_compat.hpp"

#include <concepts>
#include <cstddef>

namespace gnn {

template <typename B>
concept Buffer = std::movable<B> &&
                 requires(B& buffer, const B& constBuffer, std::size_t logicalSize) {
                     typename B::value_type;
                     { buffer.data() } -> std::same_as<typename B::value_type*>;
                     { constBuffer.data() } -> std::same_as<const typename B::value_type*>;
                     { constBuffer.logicalSize() } -> std::same_as<std::size_t>;
                     { constBuffer.physicalSize() } -> std::same_as<std::size_t>;
                     { buffer.setLogicalSize(logicalSize) } -> std::same_as<void>;
                 };

template <typename T> class BufferSpan {
public:
    GNN_HOST_DEVICE BufferSpan() noexcept {}
    GNN_HOST_DEVICE BufferSpan(T* data, std::size_t size) noexcept : data_(data), size_(size) {}

    [[nodiscard]] GNN_HOST_DEVICE T* data() const noexcept { return data_; }
    [[nodiscard]] GNN_HOST_DEVICE std::size_t size() const noexcept { return size_; }
    [[nodiscard]] GNN_HOST_DEVICE bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] GNN_HOST_DEVICE T& operator[](std::size_t index) const noexcept {
        return data_[index];
    }
    [[nodiscard]] GNN_HOST_DEVICE T* begin() const noexcept { return data_; }
    [[nodiscard]] GNN_HOST_DEVICE T* end() const noexcept {
        return size_ == 0 ? data_ : data_ + size_;
    }

private:
    T* data_{nullptr};
    std::size_t size_{0};
};

} // namespace gnn
