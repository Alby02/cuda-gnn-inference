#pragma once

#include "data/graph_csc.hpp"
#include "data/matrix.hpp"
#include "gnn/layers/gcn.hpp"
#include "gnn/layers/graph_sage.hpp"
#include "gnn/model.hpp"

#include <ostream>
#include <utility>
#include <variant>
#include <vector>

namespace gnn::demo {

using CpuDemoModel = Model<layers::GCN, layers::GraphSAGE>;

struct CpuDemo {
    graph::GraphCSC graph;
    Matrix<float> input;
    CpuDemoModel model;
};

[[nodiscard]] inline CpuDemo makeCpuDemo() {
    graph::GraphCSC graph{
        false,
        {0, 1, 2},
        {1, 0},
    };

    Matrix<float> input{std::vector<std::vector<float>>{
        {1.0F, 2.0F, 3.0F},
        {4.0F, 5.0F, 6.0F},
    }};

    Matrix<float> firstGcnWeights{std::vector<std::vector<float>>{
        {1.0F, 0.0F, 0.0F, 1.0F},
        {0.0F, 1.0F, 0.0F, 1.0F},
        {0.0F, 0.0F, 1.0F, 1.0F},
    }};

    Matrix<float> graphSageWeights{std::vector<std::vector<float>>{
        {1.0F, 0.0F, 0.0F},
        {0.0F, 1.0F, 0.0F},
        {0.0F, 0.0F, 1.0F},
        {0.5F, 0.5F, 0.5F},
    }};

    Matrix<float> secondGcnWeights{std::vector<std::vector<float>>{
        {1.0F, 0.0F},
        {0.0F, 1.0F},
        {1.0F, 1.0F},
    }};

    std::vector<CpuDemoModel::LayerVariant> modelLayers;
    modelLayers.emplace_back(layers::GCN{std::move(firstGcnWeights)});
    modelLayers.emplace_back(layers::GraphSAGE{std::move(graphSageWeights)});
    modelLayers.emplace_back(layers::GCN{std::move(secondGcnWeights)});

    return CpuDemo{
        std::move(graph),
        std::move(input),
        CpuDemoModel{std::move(modelLayers)},
    };
}

inline void printMatrix(const Matrix<float>& matrix, std::ostream& output) {
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        output << "  [";
        for (std::size_t column = 0; column < matrix.cols(); ++column) {
            if (column != 0) {
                output << ", ";
            }
            output << matrix(row, column);
        }
        output << "]\n";
    }
}

} // namespace gnn::demo
