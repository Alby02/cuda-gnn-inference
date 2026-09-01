#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace gnn {

template <typename T = float>
    requires(std::is_arithmetic_v<T>)
class Matrix {
public:
    Matrix() = default; // TODO: Remove later

    Matrix(std::size_t rows, std::size_t cols, T initialValue = T{})
        : rows_(rows), cols_(cols), data_(rows * cols, initialValue) {}

    Matrix(std::size_t rows, std::size_t cols, std::vector<T> data)
        : rows_(rows), cols_(cols), data_(std::move(data)) {
        if (data_.size() != rows_ * cols_) {
            throw std::invalid_argument("DenseMatrix dimension mismatch with data size.");
        }
    }

    explicit Matrix(const std::vector<std::vector<T>>& grid) {
        if (grid.empty()) {
            return;
        }
        rows_ = grid.size();
        cols_ = grid.front().size();
        data_.reserve(rows_ * cols_);
        for (const auto& r : grid) {
            if (r.size() != cols_) {
                throw std::invalid_argument(
                    "DenseMatrix constructor requires all rows to have equal length.");
            }
            data_.insert(data_.end(), r.begin(), r.end());
        }
    }

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }

    // TODO: Make a proper implementation
    void resize(std::size_t rows, std::size_t cols, T initialValue = T{}) {
        rows_ = rows;
        cols_ = cols;
        data_.assign(rows * cols, initialValue);
    }

    // Unchecked indexing (high performance execution)
    T& operator()(std::size_t r, std::size_t c) noexcept { return data_[r * cols_ + c]; }
    const T& operator()(std::size_t r, std::size_t c) const noexcept {
        return data_[r * cols_ + c];
    }

    // Subscript operator [] returns span over row r (enables mat[r][c] 2D syntax)
    std::span<T> operator[](std::size_t r) noexcept {
        return std::span<T>(data_).subspan(r * cols_, cols_);
    }
    std::span<const T> operator[](std::size_t r) const noexcept {
        return std::span<const T>(data_).subspan(r * cols_, cols_);
    }

    std::span<T> row(std::size_t r) noexcept {
        return std::span<T>(data_).subspan(r * cols_, cols_);
    }
    std::span<const T> row(std::size_t r) const noexcept {
        return std::span<const T>(data_).subspan(r * cols_, cols_);
    }

    // Checked indexing (bounds checking safety)
    T& at(std::size_t r, std::size_t c) {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("DenseMatrix::at out of bounds");
        }
        return data_[r * cols_ + c];
    }
    const T& at(std::size_t r, std::size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("DenseMatrix::at out of bounds");
        }
        return data_[r * cols_ + c];
    }

private:
    std::size_t rows_{0};
    std::size_t cols_{0};
    std::vector<T> data_;
};

} // namespace gnn
