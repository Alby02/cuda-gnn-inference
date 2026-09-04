#pragma once

#include "../../data/buffer.hpp"
#include "../../data/matrix.hpp"
#include "../../execution/executor.hpp"

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

template <Executor E, typename WeightMatrix, typename BiasStorage, typename Graph>
void forward_layer(const GCNLayer<WeightMatrix, BiasStorage>& layer, const Graph&, E& executor,
                   typename E::WorkspaceType& workspace) {
    executor.rowByColumn(workspace.current(), layer.getWNeigh(), workspace.next());
}

} // namespace gnn::layers
