#pragma once

#include "../../data/buffer.hpp"
#include "../../data/matrix.hpp"
#include "../../execution/executor.hpp"

#include <stdexcept>
#include <utility>

namespace gnn::layers {

enum class GraphSAGEAggregationType { MEAN, SUM, MAX };
enum class GraphSAGEActivationType { NONE, RELU };

template <typename WeightMatrix, typename BiasStorage>
    requires Buffer<BiasStorage>
class GraphSAGELayer {
public:
    using AggregationType = GraphSAGEAggregationType;
    using ActivationType = GraphSAGEActivationType;
    using WeightType = WeightMatrix;
    using BiasType = BiasStorage;

    GraphSAGELayer(WeightMatrix neighborWeights, WeightMatrix selfWeights = WeightMatrix{0, 0},
                   BiasStorage bias = BiasStorage{0},
                   AggregationType aggregation = AggregationType::MEAN,
                   ActivationType activation = ActivationType::NONE)
        : neighborWeights_(std::move(neighborWeights)), selfWeights_(std::move(selfWeights)),
          bias_(std::move(bias)), aggregation_(aggregation), activation_(activation) {
        validate();
    }

    [[nodiscard]] std::size_t getInDim() const noexcept { return neighborWeights_.rows(); }
    [[nodiscard]] std::size_t getOutDim() const noexcept { return neighborWeights_.cols(); }
    [[nodiscard]] AggregationType getAggType() const noexcept { return aggregation_; }
    [[nodiscard]] ActivationType getActType() const noexcept { return activation_; }
    [[nodiscard]] const WeightMatrix& getWNeigh() const noexcept { return neighborWeights_; }
    [[nodiscard]] bool hasWSelf() const noexcept { return !selfWeights_.empty(); }
    [[nodiscard]] const WeightMatrix& getWSelf() const noexcept { return selfWeights_; }
    [[nodiscard]] bool hasBias() const noexcept { return bias_.logicalSize() != 0; }
    [[nodiscard]] const BiasStorage& getBias() const noexcept { return bias_; }
    [[nodiscard]] bool requiresEdgeFeatures() const noexcept { return false; }

private:
    void validate() const {
        if (neighborWeights_.empty()) {
            throw std::invalid_argument("W_neigh cannot be empty");
        }
        if (!selfWeights_.empty() && (selfWeights_.rows() != neighborWeights_.rows() ||
                                      selfWeights_.cols() != neighborWeights_.cols())) {
            throw std::invalid_argument("W_self dimensions must match W_neigh dimensions");
        }
        if (hasBias() && bias_.logicalSize() != neighborWeights_.cols()) {
            throw std::invalid_argument("bias dimension must match out_dim");
        }
    }

    WeightMatrix neighborWeights_;
    WeightMatrix selfWeights_;
    BiasStorage bias_;
    AggregationType aggregation_;
    ActivationType activation_;
};

template <Executor E, typename WeightMatrix, typename BiasStorage, typename Graph>
void forward_layer(const GraphSAGELayer<WeightMatrix, BiasStorage>& layer, const Graph&,
                   E& executor, typename E::WorkspaceType& workspace) {
    executor.rowByColumn(workspace.current(), layer.getWNeigh(), workspace.next());
}

} // namespace gnn::layers
