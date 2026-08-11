#pragma once

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "engine.hpp"
#include "gnn.hpp"
#include "layer.hpp"

namespace gnn {

// Builder pattern for constructing multi-layer GNN models.
// Restricts layer addition to types present in BOTH BuilderAllowedLayers AND Engine::SupportedLayers.
template <typename Engine, typename BuilderAllowedLayers = AllKnownLayers>
    requires HasSupportedLayers<Engine>
class GNNBuilder {
public:
    using EngineSupportedLayers = typename Engine::SupportedLayers;

    template <typename LayerT>
    static constexpr bool is_allowed_v =
        detail::is_in_variant_v<LayerT, BuilderAllowedLayers> &&
        detail::is_in_variant_v<LayerT, EngineSupportedLayers>;

    GNNBuilder() = default;

    template <typename LayerT>
        requires is_allowed_v<LayerT>
    GNNBuilder& addLayer(LayerT layer) {
        if (!layers_.empty()) {
            if (getLayerOutDim(layers_.back()) != layer.getInDim()) {
                throw std::invalid_argument("Layer dimension mismatch in GNNBuilder pipeline.");
            }
        }
        layers_.push_back(EngineSupportedLayers(std::move(layer)));
        return *this;
    }

    GNNBuilder& addGCNLayer(matrix::DenseMatrix<float> wNeigh, std::vector<float> bias = {},
                            ActivationType actType = ActivationType::NONE)
        requires is_allowed_v<GCNLayer>
    {
        return addLayer(GCNLayer(std::move(wNeigh), std::move(bias), actType));
    }

    GNNBuilder& addGraphSAGELayer(matrix::DenseMatrix<float> wNeigh,
                                  std::optional<matrix::DenseMatrix<float>> wSelf = std::nullopt,
                                  std::vector<float> bias = {},
                                  AggregationType aggType = AggregationType::MEAN,
                                  ActivationType actType = ActivationType::NONE)
        requires is_allowed_v<GraphSAGELayer>
    {
        return addLayer(GraphSAGELayer(std::move(wNeigh), std::move(wSelf), std::move(bias),
                                       aggType, actType));
    }

    GNNBuilder& addEGATLayer(matrix::DenseMatrix<float> wNode, matrix::DenseMatrix<float> wEdge,
                             matrix::DenseMatrix<float> wAttn, std::size_t numHeads = 1,
                             ActivationType actType = ActivationType::NONE)
        requires is_allowed_v<EGATLayer>
    {
        return addLayer(EGATLayer(std::move(wNode), std::move(wEdge), std::move(wAttn), numHeads, actType));
    }

    [[nodiscard]] GNNModel<EngineSupportedLayers> build() {
        return GNNModel<EngineSupportedLayers>(std::move(layers_));
    }

private:
    std::vector<EngineSupportedLayers> layers_;
};

} // namespace gnn