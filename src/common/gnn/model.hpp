#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include "layer.hpp"

namespace gnn {

template <Layer... Layers> class Model {
public:
    using LayerVariant = std::variant<Layers...>;

    explicit Model(std::vector<LayerVariant> layers) : layers_(std::move(layers)) {
        for (std::size_t i = 1; i < layers_.size(); ++i) {
            if (getLayerOutDim(layers_[i - 1]) != getLayerInDim(layers_[i])) {
                throw std::invalid_argument("Layer dimension mismatch in GNNModel pipeline.");
            }
        }
    }

    [[nodiscard]] std::size_t numLayers() const noexcept { return layers_.size(); }
    [[nodiscard]] bool empty() const noexcept { return layers_.empty(); }

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

    [[nodiscard]] const std::vector<LayerVariant>& getLayers() const noexcept { return layers_; }

private:
    [[nodiscard]] static std::size_t getLayerInDim(const LayerVariant& layer) {
        return std::visit([](const auto& concreteLayer) { return concreteLayer.getInDim(); },
                          layer);
    }

    [[nodiscard]] static std::size_t getLayerOutDim(const LayerVariant& layer) {
        return std::visit([](const auto& concreteLayer) { return concreteLayer.getOutDim(); },
                          layer);
    }

    std::vector<LayerVariant> layers_;
};

} // namespace gnn
