#pragma once

#include "cli.hpp"

namespace gnn {

int run_sequential(const RunOptions& options);
int run_parallel(const RunOptions& options);
int run_cuda(const RunOptions& options);

} // namespace gnn
