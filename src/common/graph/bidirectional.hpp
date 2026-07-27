#pragma once

#include "in_neighbor.hpp"
#include "out_neighbor.hpp"

namespace graph {

class IBiDirectional : public IInNeighbor, public IOutNeighbor {
public:
    ~IBiDirectional() override = default;
};

template<typename G>
concept BiDirectional = InNeighbor<G> && OutNeighbor<G>;

} // namespace graph
