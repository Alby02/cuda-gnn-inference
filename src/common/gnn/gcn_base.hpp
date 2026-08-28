
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>
#include <cassert>

#include "../graph/base.hpp"

using namespace std;

namespace gnn {

// Feature T-CON-05: immutable metadata required by the GCN aggregation rule
struct GCNPreparedMetadata {
    vector<float> invSqrtDeg;              
    vector<uint8_t> hasExplicitSelfLoop; 
    vector<float> selfLoopWeight;           // support explicit weight if present, else implicit 1
    uint64_t numExplicitSelfLoops{0};
    uint64_t effectiveMessageCount{0};      //stored entries + added implicit self-loops
    [[nodiscard]] size_t numNodes() const noexcept { return invSqrtDeg.size(); }
};

template <graph::InNeighbor G>
[[nodiscard]] GCNPreparedMetadata prepareGCNMetadata(const G& graph) {
    const uint64_t n = graph.getNumNodes();
    GCNPreparedMetadata meta;
    meta.invSqrtDeg.assign(n, 0.0f);
    meta.hasExplicitSelfLoop.assign(n, 0);
    meta.selfLoopWeight.assign(n, 1.0f);

    const bool weighted = graph.hasEdgeWeights();

    for (uint64_t v = 0; v < n; ++v) {
        const auto neighbors = graph.getInNeighbors(v);
        const auto weights = graph.getInWeights(v);

        float weightedDegree = 0.0f;
        bool explicitSelf = false;
        float explicitSelfWeight = 0.0f;

        
       //sums weighted degree and detects an explicit self-loop
        for (size_t i = 0; i < neighbors.size(); ++i) {
            const float w = weighted ? weights[i] : 1.0f;

            //ONLY AS A WARNING FOR FUTURE USERS - TO HANDLE AFTER COMPLETION OF THE PROJECT
            assert(w > 0.0f && "GCN aggregation requires finite and positive stored weights");

            weightedDegree += w;
            if (neighbors[i] == v) {
                explicitSelf = true;
                explicitSelfWeight = w;
            }
        }

        meta.effectiveMessageCount += neighbors.size();

        if (explicitSelf) {
            meta.selfLoopWeight[v] = explicitSelfWeight;
            ++meta.numExplicitSelfLoops;
        } else {
            weightedDegree += 1.0f; // add 1 weight contribution if not explicit
            meta.selfLoopWeight[v] = 1.0f;
            ++meta.effectiveMessageCount;
        }

        meta.hasExplicitSelfLoop[v] = explicitSelf ? 1 : 0;
        meta.invSqrtDeg[v] = (weightedDegree > 0.0f) ? (1.0f / sqrt(weightedDegree)) : 0.0f;
    }

    return meta;
}

} 