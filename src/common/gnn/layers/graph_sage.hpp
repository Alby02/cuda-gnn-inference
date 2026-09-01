#pragma once

#include <span>
#include <stdexcept>
#include <vector>

#include "../../data/graph_csc.hpp"
#include "../../data/matrix.hpp"
#include "../../execution/executor.hpp"

namespace gnn::layers {


class GraphSAGE {
public:
    enum class AggregationType {
        MEAN,
        SUM,
        MAX
    };

    enum class ActivationType {
        NONE,
        RELU
    };

    GraphSAGE(Matrix<float> W_neigh,
                   Matrix<float> W_self = {0, 0},
                   std::vector<float> bias = {}, AggregationType agg_type = AggregationType::MEAN,
                   ActivationType act_type = ActivationType::NONE)
        : W_neigh_(std::move(W_neigh)), W_self_(std::move(W_self)), bias_(std::move(bias)),
          agg_type_(agg_type), act_type_(act_type) {
        if (W_neigh_.empty()) {
            throw std::invalid_argument("W_neigh cannot be empty");
        }
        if (!W_self_.empty() && (W_self_.rows() != W_neigh_.rows() || W_self_.cols() != W_neigh_.cols())) {
            throw std::invalid_argument("W_self dimensions must match W_neigh dimensions");
        }
        if (!bias_.empty() && bias_.size() != W_neigh_.cols()) {
            throw std::invalid_argument("bias dimension must match out_dim");
        }
    }

    [[nodiscard]] std::size_t getInDim() const noexcept { return W_neigh_.rows(); }
    [[nodiscard]] std::size_t getOutDim() const noexcept { return W_neigh_.cols(); }
    [[nodiscard]] AggregationType getAggType() const noexcept { return agg_type_; }
    [[nodiscard]] ActivationType getActType() const noexcept { return act_type_; }
    [[nodiscard]] const Matrix<float>& getWNeigh() const noexcept { return W_neigh_; }
    [[nodiscard]] bool hasWSelf() const noexcept { return !W_self_.empty(); }
    [[nodiscard]] const Matrix<float>& getWSelf() const { return W_self_; }
    [[nodiscard]] bool hasBias() const noexcept { return !bias_.empty(); }
    [[nodiscard]] std::span<const float> getBias() const noexcept { return bias_; }
    [[nodiscard]] bool requiresEdgeFeatures() const noexcept { return false; }

private:
    Matrix<float> W_neigh_;
    Matrix<float> W_self_;
    std::vector<float> bias_;
    AggregationType agg_type_;
    ActivationType act_type_;
};

template <Executor E>
void forward_layer(const GraphSAGE& layer,
                   const graph::GraphCSC&,
                   E& executor,
                   typename E::WorkspaceType& workspace) {
    executor.rowByColumn(workspace.current(), layer.getWNeigh(), workspace.next());
}

} // namespace gnn::layers