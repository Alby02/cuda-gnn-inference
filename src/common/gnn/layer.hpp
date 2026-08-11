#pragma once

#include "../matrix/dense_matrix.hpp"
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace gnn {

enum class AggregationType { SUM, MEAN, GCN_NORM, MAX };

enum class ActivationType { NONE, RELU, SIGMOID };

class GCNLayer {
public:
    GCNLayer(matrix::DenseMatrix<float> W_neigh, std::vector<float> bias = {},
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
    [[nodiscard]] AggregationType getAggType() const noexcept { return AggregationType::GCN_NORM; }
    [[nodiscard]] ActivationType getActType() const noexcept { return act_type_; }
    [[nodiscard]] const matrix::DenseMatrix<float>& getWNeigh() const noexcept { return W_neigh_; }
    [[nodiscard]] bool hasBias() const noexcept { return !bias_.empty(); }
    [[nodiscard]] std::span<const float> getBias() const noexcept { return bias_; }
    [[nodiscard]] bool requiresEdgeFeatures() const noexcept { return false; }

private:
    matrix::DenseMatrix<float> W_neigh_;
    std::vector<float> bias_;
    ActivationType act_type_{ActivationType::NONE};
    std::size_t in_dim_{0};
    std::size_t out_dim_{0};
};

class GraphSAGELayer {
public:
    GraphSAGELayer(matrix::DenseMatrix<float> W_neigh,
                   std::optional<matrix::DenseMatrix<float>> W_self = std::nullopt,
                   std::vector<float> bias = {}, AggregationType agg_type = AggregationType::MEAN,
                   ActivationType act_type = ActivationType::NONE)
        : W_neigh_(std::move(W_neigh)), W_self_(std::move(W_self)), bias_(std::move(bias)),
          agg_type_(agg_type), act_type_(act_type) {
        if (W_neigh_.empty()) {
            throw std::invalid_argument("W_neigh cannot be empty");
        }
        in_dim_ = W_neigh_.rows();
        out_dim_ = W_neigh_.cols();
        if (W_self_.has_value() && (W_self_->rows() != in_dim_ || W_self_->cols() != out_dim_)) {
            throw std::invalid_argument("W_self dimensions must match W_neigh dimensions");
        }
        if (!bias_.empty() && bias_.size() != out_dim_) {
            throw std::invalid_argument("bias dimension must match out_dim");
        }
    }

    [[nodiscard]] std::size_t getInDim() const noexcept { return in_dim_; }
    [[nodiscard]] std::size_t getOutDim() const noexcept { return out_dim_; }
    [[nodiscard]] AggregationType getAggType() const noexcept { return agg_type_; }
    [[nodiscard]] ActivationType getActType() const noexcept { return act_type_; }
    [[nodiscard]] const matrix::DenseMatrix<float>& getWNeigh() const noexcept { return W_neigh_; }
    [[nodiscard]] bool hasWSelf() const noexcept {
        return W_self_.has_value() && !W_self_->empty();
    }
    [[nodiscard]] const matrix::DenseMatrix<float>& getWSelf() const { return W_self_.value(); }
    [[nodiscard]] bool hasBias() const noexcept { return !bias_.empty(); }
    [[nodiscard]] std::span<const float> getBias() const noexcept { return bias_; }
    [[nodiscard]] bool requiresEdgeFeatures() const noexcept { return false; }

private:
    matrix::DenseMatrix<float> W_neigh_;
    std::optional<matrix::DenseMatrix<float>> W_self_;
    std::vector<float> bias_;
    AggregationType agg_type_{AggregationType::MEAN};
    ActivationType act_type_{ActivationType::NONE};
    std::size_t in_dim_{0};
    std::size_t out_dim_{0};
};

class EGATLayer {
public:
    EGATLayer(matrix::DenseMatrix<float> W_node, matrix::DenseMatrix<float> W_edge,
              matrix::DenseMatrix<float> W_attn, std::size_t num_heads = 1,
              ActivationType act_type = ActivationType::NONE)
        : W_node_(std::move(W_node)), W_edge_(std::move(W_edge)), W_attn_(std::move(W_attn)),
          num_heads_(num_heads), act_type_(act_type) {
        if (W_node_.empty()) {
            throw std::invalid_argument("W_node cannot be empty");
        }
        in_dim_ = W_node_.rows();
        out_dim_ = W_node_.cols();
    }

    [[nodiscard]] std::size_t getInDim() const noexcept { return in_dim_; }
    [[nodiscard]] std::size_t getOutDim() const noexcept { return out_dim_; }
    [[nodiscard]] std::size_t getNumHeads() const noexcept { return num_heads_; }
    [[nodiscard]] ActivationType getActType() const noexcept { return act_type_; }
    [[nodiscard]] const matrix::DenseMatrix<float>& getWNode() const noexcept { return W_node_; }
    [[nodiscard]] const matrix::DenseMatrix<float>& getWEdge() const noexcept { return W_edge_; }
    [[nodiscard]] const matrix::DenseMatrix<float>& getWAttn() const noexcept { return W_attn_; }
    [[nodiscard]] bool requiresEdgeFeatures() const noexcept { return true; }

private:
    matrix::DenseMatrix<float> W_node_;
    matrix::DenseMatrix<float> W_edge_;
    matrix::DenseMatrix<float> W_attn_;
    std::size_t num_heads_{1};
    ActivationType act_type_{ActivationType::NONE};
    std::size_t in_dim_{0};
    std::size_t out_dim_{0};
};

using AllKnownLayers = std::variant<GCNLayer, GraphSAGELayer, EGATLayer>;

namespace detail {

template <typename T, typename Variant> struct is_in_variant : std::false_type {};

template <typename T, typename... Types>
struct is_in_variant<T, std::variant<Types...>> : std::disjunction<std::is_same<T, Types>...> {};

template <typename T, typename Variant>
inline constexpr bool is_in_variant_v = is_in_variant<T, Variant>::value;

} // namespace detail

inline std::size_t getLayerInDim(const auto& layerVariant) {
    return std::visit([](const auto& l) { return l.getInDim(); }, layerVariant);
}

inline std::size_t getLayerOutDim(const auto& layerVariant) {
    return std::visit([](const auto& l) { return l.getOutDim(); }, layerVariant);
}

inline bool layerRequiresEdgeFeatures(const auto& layerVariant) {
    return std::visit([](const auto& l) { return l.requiresEdgeFeatures(); }, layerVariant);
}

} // namespace gnn
