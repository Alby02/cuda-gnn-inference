#pragma once

#include <cstdint>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../matrix/dense_matrix.hpp"
#include "directed_csc.hpp"
#include "data_factory.hpp" //decide which graph type to build and builds it
#include "undirected_csc.hpp"

using namespace std;

namespace graph {

#pragma pack(push, 1) //ignore padding to make python and C++ formats compatible (binary graph header and python little endian)
struct GraphHeader {
    uint64_t num_nodes{0};
    uint64_t num_edges{0};
    uint8_t is_directed{0};
    uint8_t has_weights{0};
    uint8_t has_edge_features{0};
    uint64_t edge_feature_dim{0};
};
#pragma pack(pop)

// Check on formats
static_assert(sizeof(GraphHeader) == 27, "GraphHeader must match Python '<QQBBBQ' layout");


class GraphLoader { //convertion bytes of .bin_graph / .bin_matrix into in-memory data
public:
    // loads a .bin_graph file and returns a GraphVariant (DirectedCSC or UndirectedCSC, decided by header.is_directed)
    static GraphVariant load(const string& filepath) {
        auto raw = readBinaryGraph(filepath);

        return GraphFactory::make(raw.header.is_directed != 0, raw.header.num_nodes,
                                   raw.header.num_edges, move(raw.colPtr),
                                   move(raw.rowInd), move(raw.weights),
                                   move(raw.edgeFeatures));
    }

    
    // Loads a dense matrix produced by the script export_dense_matrix.py.
    static matrix::DenseMatrix<float> loadDenseMatrix(const std::string& filepath) {
        ifstream file(filepath, ios::binary);
        if (!file.is_open()) {
            throw runtime_error("GraphLoader-loadDenseMatrix: could not open " + filepath);
        }

        uint64_t rows = 0;
        uint64_t cols = 0;
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        if (!file) {
            throw runtime_error(
                "GraphLoader - loadDenseMatrix: error in reading header" );
        }

        vector<float> data(static_cast<size_t>(rows * cols));
        if (!data.empty()) {
            file.read(reinterpret_cast<char*>(data.data()), static_cast<streamsize>(data.size() * sizeof(float)));
        }
        if (!file) {
            throw runtime_error("GraphLoader - loadDenseMatrix: corrupt file");
        }

        return matrix::DenseMatrix<float>(static_cast<size_t>(rows),
                                          static_cast<size_t>(cols), move(data));
    }

private:
    // struct used internally to store raw data of a .bin_graph file from readBinaryGraph() to load()
    struct LoadedRawData {
        GraphHeader header;
        vector<uint64_t> colPtr;
        vector<uint64_t> rowInd;
        vector<float> weights;
        optional<matrix::DenseMatrix<float>> edgeFeatures;
    };

   
    // byte parsing the header and CSC arrays out of a .bin_graph file
    static LoadedRawData readBinaryGraph(const string& filepath) {
        ifstream file(filepath, ios::binary);
        if (!file.is_open()) {
            throw runtime_error("GraphLoader: could not open file ");
        }

        //reads fixed-size 27-byte header
        GraphHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(GraphHeader)); 
        if (!file) {
            throw runtime_error("GraphLoader: error reading binary header.");
        }

        //reads colPtr (lenght is standard)
        vector<uint64_t> colPtr(header.num_nodes + 1); 
        file.read(reinterpret_cast<char*>(colPtr.data()), colPtr.size() * sizeof(uint64_t)); 

        //reads rowInd (lenght is standard)
        vector<uint64_t> rowInd(header.num_edges); 
        file.read(reinterpret_cast<char*>(rowInd.data()), rowInd.size() * sizeof(uint64_t)); 

        //reads edge weights if present
        vector<float> weights; 
        if (header.has_weights) {
            weights.resize(header.num_edges);
            file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float)); 
        }

        //read edge features if present and store them matching export_graph_csc.py's edge_features call
        optional<matrix::DenseMatrix<float>> edgeFeatures = nullopt;
        if (header.has_edge_features && header.edge_feature_dim > 0) {
            vector<float> edgeFeatsData(header.num_edges * header.edge_feature_dim);
            file.read(reinterpret_cast<char*>(edgeFeatsData.data()), edgeFeatsData.size() * sizeof(float));
            edgeFeatures = matrix::DenseMatrix<float>(
                static_cast<size_t>(header.num_edges),
                static_cast<size_t>(header.edge_feature_dim), move(edgeFeatsData));
        }

        //check for reading past the end or for an I/O error
        if (!file) { 
            throw runtime_error(
                "GraphLoader: error while reading data.");
        }

        return {header, move(colPtr), move(rowInd), move(weights), move(edgeFeatures)};
    }
};

} 