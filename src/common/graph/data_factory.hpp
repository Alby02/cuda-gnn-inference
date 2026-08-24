#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include "../matrix/dense_matrix.hpp"
#include "directed_csc.hpp"
#include "undirected_csc.hpp"

using namespace std;

namespace graph {

// Creation of a single type that can hold both a DirectedCSC or an UndirectedCSC 

using GraphVariant = std::variant<DirectedCSC, UndirectedCSC>;

class GraphFactory { //decides which graph type to instantiate and builds it (works with CSC data)
public:

    static GraphVariant makeDirected( // Builds a DirectedCSC
        uint64_t numNodes, uint64_t numEdges, vector<uint64_t> colPtr,
        vector<uint64_t> rowInd, vector<float> weights = {},
        optional<matrix::DenseMatrix<float>> edgeFeatures = nullopt) {
        return GraphVariant(in_place_type<DirectedCSC>, numNodes, numEdges, move(colPtr),
                             move(rowInd), move(weights), move(edgeFeatures));
    }


    static GraphVariant makeUndirected( // Builds an UndirectedCSC 
        uint64_t numNodes, uint64_t numDirectedEntries, vector<std::uint64_t> colPtr, 
        vector<uint64_t> rowInd, vector<float> weights = {}, optional<matrix::DenseMatrix<float>> edgeFeatures = nullopt) {
        if (numDirectedEntries % 2 != 0) {
            throw runtime_error(
                "GraphFactory error: graph not correctly symmetrized ");
        }
        const uint64_t numUndirectedEdges = numDirectedEntries / 2;
        return GraphVariant(in_place_type<UndirectedCSC>, numNodes, numUndirectedEdges,
                             move(colPtr), move(rowInd), move(weights), move(edgeFeatures));
    }

    // given the isDirected flag and the raw CSC data, builds the correct concrete type and returns it in GraphVariant
    static GraphVariant make(bool isDirected, uint64_t numNodes, uint64_t numEdgesOrDirectedEntries,
                              vector<uint64_t> colPtr, vector<uint64_t> rowInd, vector<float> weights = {},
                              optional<matrix::DenseMatrix<float>> edgeFeatures = nullopt) {
        if (isDirected) {
            return makeDirected(numNodes, numEdgesOrDirectedEntries, move(colPtr),
                                move(rowInd), move(weights), move(edgeFeatures));
        }
        return makeUndirected(numNodes, numEdgesOrDirectedEntries, move(colPtr),
                               move(rowInd), move(weights), move(edgeFeatures));
    }
};

} 