#pragma once

#include "data/matrix.hpp"
#include "execution/workspace.hpp"
#include "host_buffer.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace gnn {

class CpuContext {
public:
    using BufferType = Matrix<HostBuffer<float>>;

    CpuContext(BufferType input, std::size_t physicalColumns)
        : current_(input.rows(), physicalColumns), next_(input.rows(), physicalColumns) {
        if (input.empty()) {
            throw std::invalid_argument("CPU context input cannot be empty.");
        }
        std::copy_n(input.data(), input.size(), current_.data());
        current_.setShape(input.rows(), input.cols());
        next_.setShape(input.rows(), 0);
    }

    [[nodiscard]] BufferType& current() noexcept { return current_; }
    [[nodiscard]] const BufferType& current() const noexcept { return current_; }
    [[nodiscard]] BufferType& next() noexcept { return next_; }

    void swapBuffers() noexcept { std::swap(current_, next_); }

private:
    BufferType current_;
    BufferType next_;
};

static_assert(Workspace<CpuContext>);

} // namespace gnn
