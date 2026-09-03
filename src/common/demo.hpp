#pragma once

#include "data/graph_csc.hpp"
#include "data/loader.hpp"
#include "data/matrix.hpp"
#include "gnn/model.hpp"
#include "host_layers.hpp"

#include <algorithm>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace gnn::demo {

using CpuDemoModel = Model<layers::HostGCN, layers::HostGraphSAGE>;

struct CpuDemo {
    graph::HostGraphCSC graph;
    Matrix<HostBuffer<float>> input;
    CpuDemoModel model;
};

[[nodiscard]] inline Matrix<HostBuffer<float>> makeMatrix(std::size_t rows, std::size_t columns,
                                                          std::initializer_list<float> values) {
    Matrix<HostBuffer<float>> matrix{rows, columns};
    if (values.size() != matrix.size()) {
        throw std::invalid_argument("Demo matrix values do not match its shape.");
    }
    std::copy(values.begin(), values.end(), matrix.data());
    return matrix;
}

[[nodiscard]] inline CpuDemo makeCpuDemo() {
    auto graph = graph::GraphFactory::make(false, 2, 2, {0, 1, 2}, {1, 0});

    auto input = makeMatrix(2, 3, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    auto firstGcnWeights =
        makeMatrix(3, 4, {1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F});
    auto graphSageWeights =
        makeMatrix(4, 3, {1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.5F, 0.5F, 0.5F});
    auto secondGcnWeights = makeMatrix(3, 2, {1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F});

    std::vector<CpuDemoModel::LayerVariant> modelLayers;
    modelLayers.emplace_back(layers::HostGCN{std::move(firstGcnWeights)});
    modelLayers.emplace_back(layers::HostGraphSAGE{std::move(graphSageWeights)});
    modelLayers.emplace_back(layers::HostGCN{std::move(secondGcnWeights)});

    return CpuDemo{
        std::move(graph),
        std::move(input),
        CpuDemoModel{std::move(modelLayers)},
    };
}

[[nodiscard]] inline CpuDemo loadCpuDemo(const std::string& graphPath,
                                         const std::string& featurePath) {
    auto graph = graph::GraphLoader::load(graphPath);
    auto input = graph::GraphLoader::loadDenseMatrix(featurePath);

    if (input.empty()) {
        throw std::invalid_argument("The feature matrix cannot be empty.");
    }
    if (graph.getNumNodes() != input.rows()) {
        throw std::invalid_argument(
            "The loaded graph node count must match the feature-matrix row count.");
    }

    // The data tools do not produce trained model parameters yet. Build a small projection
    // layer so any generated/downloaded feature width can exercise the current runtime.
    Matrix<HostBuffer<float>> projectionWeights{input.cols(), 1};
    std::fill_n(projectionWeights.data(), projectionWeights.size(),
                1.0F / static_cast<float>(input.cols()));
    std::vector<CpuDemoModel::LayerVariant> modelLayers;
    modelLayers.emplace_back(layers::HostGCN{std::move(projectionWeights)});

    return CpuDemo{
        std::move(graph),
        std::move(input),
        CpuDemoModel{std::move(modelLayers)},
    };
}

inline void printMatrix(const Matrix<HostBuffer<float>>& matrix, std::ostream& output,
                        std::size_t maxRows = 10, std::size_t maxColumns = 8) {
    const auto rowsToPrint = std::min(matrix.rows(), maxRows);
    const auto columnsToPrint = std::min(matrix.cols(), maxColumns);
    for (std::size_t row = 0; row < rowsToPrint; ++row) {
        output << "  [";
        for (std::size_t column = 0; column < columnsToPrint; ++column) {
            if (column != 0) {
                output << ", ";
            }
            output << matrix(row, column);
        }
        if (columnsToPrint < matrix.cols()) {
            output << ", ...";
        }
        output << "]\n";
    }
    if (rowsToPrint < matrix.rows()) {
        output << "  ... (" << matrix.rows() - rowsToPrint << " more rows)\n";
    }
}

} // namespace gnn::demo
