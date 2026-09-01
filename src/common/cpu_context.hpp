#pragma once

#include "data/matrix.hpp"
#include "execution/workspace.hpp"

#include <stdexcept>
#include <utility>

namespace gnn {

class CpuContext {
public:
    using BufferType = Matrix<float>;

    explicit CpuContext(BufferType input) : current_(std::move(input)) {
        if (current_.empty()) {
            throw std::invalid_argument("CPU context input cannot be empty.");
        }
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
