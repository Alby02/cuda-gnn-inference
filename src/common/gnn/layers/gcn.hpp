#pragma once

#include "../../data/buffer.hpp"
#include "../../data/matrix.hpp"
#include "../../execution/executor.hpp"
#include "gcn_aggregation.hpp" 

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace gnn::layers {

enum class GCNActivationType { NONE, RELU };

template <typename WeightMatrix, typename BiasStorage>
    requires Buffer<BiasStorage>
class GCNLayer {
public:
    using ActivationType = GCNActivationType;
    using WeightType = WeightMatrix;
    using BiasType = BiasStorage;

    GCNLayer(WeightMatrix neighborWeights, BiasStorage bias = BiasStorage{},
             ActivationType activation = ActivationType::NONE)
        : neighborWeights_(std::move(neighborWeights)), bias_(std::move(bias)),
          activation_(activation) {
        validate();
    }

    [[nodiscard]] GNN_HOST_DEVICE std::size_t getInDim() const noexcept {
        return neighborWeights_.rows();
    }
    [[nodiscard]] GNN_HOST_DEVICE std::size_t getOutDim() const noexcept {
        return neighborWeights_.cols();
    }
    [[nodiscard]] GNN_HOST_DEVICE ActivationType getActType() const noexcept {
        return activation_;
    }
    [[nodiscard]] GNN_HOST_DEVICE const WeightMatrix& getWNeigh() const noexcept {
        return neighborWeights_;
    }
    [[nodiscard]] GNN_HOST_DEVICE bool hasBias() const noexcept {
        return bias_.logicalSize() != 0;
    }
    [[nodiscard]] GNN_HOST_DEVICE const BiasStorage& getBias() const noexcept { return bias_; }

private:
    void validate() const {
        if (neighborWeights_.empty()) {
            throw std::invalid_argument("W_neigh cannot be empty");
        }
        if (hasBias() && bias_.logicalSize() != getOutDim()) {
            throw std::invalid_argument("bias dimension must match out_dim");
        }
    }

    WeightMatrix neighborWeights_;
    BiasStorage bias_;
    ActivationType activation_{ActivationType::NONE};
};

namespace detail {

template <typename MatrixT, typename BiasStorage>
void applyBiasAndActivation(MatrixT& output, const BiasStorage& bias, bool hasBias,
                            GCNActivationType activation) {
    for (std::size_t row = 0; row < output.rows(); ++row) {
        for (std::size_t col = 0; col < output.cols(); ++col) {
            float value = output(row, col);
            if (hasBias) {
                value += bias.data()[col];
            }
            if (activation == GCNActivationType::RELU) {
                value = value > 0.0F ? value : 0.0F;
            }
            output(row, col) = value;
        }
    }
}
}

template <Executor E, typename WeightMatrix, typename BiasStorage, typename Graph>
void forward_layer(const GCNLayer<WeightMatrix, BiasStorage>& layer, const Graph& graph, E& executor,
                   typename E::WorkspaceType& workspace) {
    const auto& aggregated = executor.gcnState().aggregate(graph, workspace.current());
    executor.rowByColumn(workspace.current(), layer.getWNeigh(), workspace.next());
    detail::applyBiasAndActivation(workspace.next(), layer.getBias(), layer.hasBias(), layer.getActType());
}

} // 
