#pragma once

#include "buffer.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace gnn {

template <typename Storage>
    requires(Buffer<Storage> && std::is_arithmetic_v<typename Storage::value_type>)
class Matrix {
public:
    using ValueType = typename Storage::value_type;

    Matrix(std::size_t rows, std::size_t columns)
        : rows_(rows), columns_(columns), buffer_(elementCount(rows, columns)) {}

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return columns_; }
    [[nodiscard]] std::size_t size() const noexcept { return buffer_.logicalSize(); }
    [[nodiscard]] std::size_t physicalSize() const noexcept { return buffer_.physicalSize(); }
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] ValueType* data() noexcept { return buffer_.data(); }
    [[nodiscard]] const ValueType* data() const noexcept { return buffer_.data(); }

    void setShape(std::size_t rows, std::size_t columns) {
        buffer_.setLogicalSize(elementCount(rows, columns));
        rows_ = rows;
        columns_ = columns;
    }

    [[nodiscard]] ValueType& operator()(std::size_t row, std::size_t column) noexcept {
        return buffer_.data()[row * columns_ + column];
    }

    [[nodiscard]] const ValueType& operator()(std::size_t row, std::size_t column) const noexcept {
        return buffer_.data()[row * columns_ + column];
    }

private:
    [[nodiscard]] static std::size_t elementCount(std::size_t rows, std::size_t columns) {
        if (rows != 0 && columns > std::numeric_limits<std::size_t>::max() / rows) {
            throw std::overflow_error("Matrix dimensions overflow size_t.");
        }
        return rows * columns;
    }

    std::size_t rows_;
    std::size_t columns_;
    Storage buffer_;
};

} // namespace gnn
