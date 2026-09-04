#pragma once

#include "buffer.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace gnn {

template <typename Storage>
    requires(Buffer<Storage> && std::is_arithmetic_v<typename Storage::value_type>)
class Matrix {
public:
    using ValueType = typename Storage::value_type;

    Matrix() = default;

    Matrix(std::size_t rows, std::size_t columns)
        requires std::constructible_from<Storage, std::size_t>
        : rows_(rows), columns_(columns), buffer_(elementCount(rows, columns)) {}

    Matrix(std::size_t rows, std::size_t columns, Storage buffer)
        : rows_(rows), columns_(columns), buffer_(std::move(buffer)) {
        buffer_.setLogicalSize(elementCount(rows, columns));
    }

    [[nodiscard]] GNN_HOST_DEVICE std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] GNN_HOST_DEVICE std::size_t cols() const noexcept { return columns_; }
    [[nodiscard]] GNN_HOST_DEVICE std::size_t size() const noexcept {
        return buffer_.logicalSize();
    }
    [[nodiscard]] GNN_HOST_DEVICE std::size_t physicalSize() const noexcept {
        return buffer_.physicalSize();
    }
    [[nodiscard]] GNN_HOST_DEVICE bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] GNN_HOST_DEVICE ValueType* data() noexcept { return buffer_.data(); }
    [[nodiscard]] GNN_HOST_DEVICE const ValueType* data() const noexcept {
        return buffer_.data();
    }

    GNN_HOST_DEVICE void setShape(std::size_t rows, std::size_t columns) {
        buffer_.setLogicalSize(elementCount(rows, columns));
        rows_ = rows;
        columns_ = columns;
    }

    [[nodiscard]] GNN_HOST_DEVICE ValueType& operator()(std::size_t row,
                                                        std::size_t column) noexcept {
        return buffer_.data()[row * columns_ + column];
    }

    [[nodiscard]] GNN_HOST_DEVICE const ValueType&
    operator()(std::size_t row, std::size_t column) const noexcept {
        return buffer_.data()[row * columns_ + column];
    }

private:
    [[nodiscard]] GNN_HOST_DEVICE static std::size_t elementCount(std::size_t rows,
                                                                  std::size_t columns) {
#if !defined(__CUDA_ARCH__)
        if (rows != 0 && columns > std::numeric_limits<std::size_t>::max() / rows) {
            throw std::overflow_error("Matrix dimensions overflow size_t.");
        }
#endif
        return rows * columns;
    }

    std::size_t rows_{0};
    std::size_t columns_{0};
    Storage buffer_;
};

} // namespace gnn
