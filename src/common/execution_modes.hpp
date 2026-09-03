#pragma once

#include <string_view>

namespace gnn {

int run_sequential(std::string_view graphPath = {}, std::string_view featurePath = {});
int run_parallel(std::string_view graphPath = {}, std::string_view featurePath = {});
int run_cuda(std::string_view graphPath = {}, std::string_view featurePath = {});

} // namespace gnn
