#pragma once
#include <string>

namespace gnn {
struct RunOptions {
    std::string backend = "sequential"; // Execution mode: sequential, parallel (OpenMP), or cuda.

    std::string graph;    // Graph topology binary path.
    std::string features; // Node-feature matrix path, paired with graph.
    std::string
        model; // Model description with weight and bias paths; required with graph and features.
    std::string output;     // Benchmark CSV path; empty disables CSV export.
    std::string embeddings; // Final output matrix path; empty disables binary export.

    int threads = 1;     // Number of OpenMP threads.
    int blockSize = 256; // CUDA threads per block.

    int warmups = 1;      // Unrecorded iterations before collecting measurements.
    int repetitions = 10; // Number of measured iterations.

    bool help = false; // Print CLI usage and exit without executing the workload.
};
inline RunOptions parseOptions(int argc, char** argv) {
    RunOptions o;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            o.help = true;
            return o;
        }
        if (!key.starts_with("--")) {
            o.help = true;
            return o;
        }
        const std::string value = argv[++i];
        if (key == "--backend")
            o.backend = value;
        else if (key == "--graph")
            o.graph = value;
        else if (key == "--features")
            o.features = value;
        else if (key == "--model")
            o.model = value;
        else if (key == "--output")
            o.output = value;
        else if (key == "--embeddings")
            o.embeddings = value;
        else if (key == "--threads")
            o.threads = std::stoi(value);
        else if (key == "--block-size")
            o.blockSize = std::stoi(value);
        else if (key == "--warmups")
            o.warmups = std::stoi(value);
        else if (key == "--repetitions")
            o.repetitions = std::stoi(value);
    }
    if (argc == 1) {
        o.help = true;
        return o;
    }
    // Partial input sets use the complete built-in demo instead.
    if (o.graph.empty() || o.features.empty() || o.model.empty()) {
        o.graph.clear();
        o.features.clear();
        o.model.clear();
    }
    return o;
}
} // namespace gnn
