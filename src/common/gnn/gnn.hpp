#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "layer.hpp"

namespace gnn {

template <typename LayerVariant = AllKnownLayers> class GNNModel {
public:
    GNNModel() = default;

    explicit GNNModel(std::vector<LayerVariant> layers) : layers_(std::move(layers)) {
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            if (layerRequiresEdgeFeatures(layers_.at(i))) {
                requires_edge_features_ = true;
            }
            if (i > 0 && getLayerOutDim(layers_.at(i - 1)) != getLayerInDim(layers_.at(i))) {
                throw std::invalid_argument("Layer dimension mismatch in GNNModel pipeline.");
            }
        }
    }

    [[nodiscard]] std::size_t numLayers() const noexcept { return layers_.size(); }
    [[nodiscard]] bool empty() const noexcept { return layers_.empty(); }

    [[nodiscard]] const LayerVariant& getLayer(std::size_t idx) const { return layers_.at(idx); }

    [[nodiscard]] std::size_t getInputDim() const {
        if (layers_.empty()) {
            return 0;
        }
        return getLayerInDim(layers_.front());
    }

    [[nodiscard]] std::size_t getOutputDim() const {
        if (layers_.empty()) {
            return 0;
        }
        return getLayerOutDim(layers_.back());
    }

    [[nodiscard]] bool requiresEdgeFeatures() const noexcept { return requires_edge_features_; }

    [[nodiscard]] const std::vector<LayerVariant>& getLayers() const noexcept { return layers_; }

private:
    std::vector<LayerVariant> layers_;
    bool requires_edge_features_{false};
};

} // namespace gnn
