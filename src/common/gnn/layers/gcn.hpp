#pragma once

#include <span>
#include <stdexcept>
#include <vector>

#include "../../data/graph_csc.hpp"
#include "../../data/matrix.hpp"
#include "../../execution/executor.hpp"


namespace gnn::layers {

class GCN {
public:

    enum class ActivationType {
        NONE,
        RELU
    };

    GCN(Matrix<float> W_neigh, std::vector<float> bias = {},
             ActivationType act_type = ActivationType::NONE)
        : W_neigh_(std::move(W_neigh)), bias_(std::move(bias)), act_type_(act_type) {
        if (W_neigh_.empty()) {
            throw std::invalid_argument("W_neigh cannot be empty");
        }
        in_dim_ = W_neigh_.rows();
        out_dim_ = W_neigh_.cols();
        if (!bias_.empty() && bias_.size() != out_dim_) {
            throw std::invalid_argument("bias dimension must match out_dim");
        }
    }

    [[nodiscard]] std::size_t getInDim() const noexcept { return in_dim_; }
    [[nodiscard]] std::size_t getOutDim() const noexcept { return out_dim_; }
    [[nodiscard]] ActivationType getActType() const noexcept { return act_type_; }
    [[nodiscard]] const Matrix<float>& getWNeigh() const noexcept { return W_neigh_; }
    [[nodiscard]] bool hasBias() const noexcept { return !bias_.empty(); }
    [[nodiscard]] std::span<const float> getBias() const noexcept { return bias_; }

private:
    Matrix<float> W_neigh_;
    std::vector<float> bias_;
    ActivationType act_type_{ActivationType::NONE};
    std::size_t in_dim_{0};
    std::size_t out_dim_{0};
};

template <Executor E>
void forward_layer(const GCN& layer,
                   const graph::GraphCSC&,
                   E& executor,
                   typename E::WorkspaceType& workspace) {
    executor.rowByColumn(workspace.current(), layer.getWNeigh(), workspace.next());
}

} // namespace gnn::layers
