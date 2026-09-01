#include "demo.hpp"
#include "execution/runtime.hpp"
#include "execution_modes.hpp"
#include "executor.hpp"

#include <iostream>
#include <utility>

namespace gnn {

int run_sequential() {
    auto workload = demo::makeCpuDemo();
    InferenceRuntime<SequentialExecutor> runtime;

    const Matrix<float> result =
        runtime.run(workload.graph, workload.model, std::move(workload.input));

    std::cout << "Sequential GCN -> GraphSAGE -> GCN output:\n";
    demo::printMatrix(result, std::cout);
    return 0;
}

} // namespace gnn
