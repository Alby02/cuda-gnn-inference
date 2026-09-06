#pragma once
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gnn::bench {
using Clock = std::chrono::steady_clock;
inline double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
class CpuTimer {
public:
    void start() { start_ = Clock::now(); }
    double stop() const { return milliseconds(start_); }

private:
    Clock::time_point start_;
};
struct Sample {
    double resetMs, computeMs, endToEndMs;
};
struct Measurements {
    std::vector<Sample> samples;
    double meanMs = 0, stddevMs = 0;
};
// Timer.stop must synchronize asynchronous compute. Reset is measured separately.
template <typename Reset, typename Compute, typename Timer>
Measurements measure(int warmups, int repetitions, Reset reset, Compute compute, Timer& timer) {
    Measurements result;
    result.samples.reserve(repetitions);
    for (int i = 0; i < warmups; ++i) {
        reset();
        timer.start();
        compute();
        static_cast<void>(timer.stop());
    }
    for (int i = 0; i < repetitions; ++i) {
        const auto begin = Clock::now();
        reset();
        const double resetMs = milliseconds(begin);
        timer.start();
        compute();
        const double computeMs = timer.stop();
        result.samples.push_back({resetMs, computeMs, milliseconds(begin)});
        result.meanMs += computeMs;
    }
    result.meanMs /= repetitions;
    for (const auto& s : result.samples)
        result.stddevMs += (s.computeMs - result.meanMs) * (s.computeMs - result.meanMs);
    result.stddevMs = std::sqrt(result.stddevMs / repetitions); // Population standard deviation.
    return result;
}
inline std::string number(double value) {
    std::ostringstream s;
    s << std::setprecision(17) << value;
    return s.str();
}
using Record = std::map<std::string, std::string>;
inline constexpr const char* fields =
    "sample,load_ms,setup_ms,upload_ms,download_ms,reset_ms,compute_ms,end_to_end_ms,"
    "mean_ms,stddev_ms,workspace_bytes";
inline void writeResults(const std::string& path, Record record, const Measurements& measurements) {
    if (path.empty())
        return;
    std::ofstream file(path, std::ios::trunc);
    file << fields << '\n';
    for (std::size_t i = 0; i < measurements.samples.size(); ++i) {
        const auto& s = measurements.samples[i];
        record["sample"] = std::to_string(i);
        record["reset_ms"] = number(s.resetMs);
        record["compute_ms"] = number(s.computeMs);
        record["end_to_end_ms"] = number(s.endToEndMs);
        record["mean_ms"] = number(measurements.meanMs);
        record["stddev_ms"] = number(measurements.stddevMs);
        std::istringstream names(fields);
        std::string name;
        bool first = true;
        while (std::getline(names, name, ',')) {
            if (!first)
                file << ',';
            first = false;
            file << '"';
            for (char c : record[name]) {
                if (c == '"')
                    file << '"';
                file << c;
            }
            file << '"';
        }
        file << '\n';
    }
    if (!file)
        throw std::runtime_error("Cannot write benchmark CSV: " + path);
}
} // namespace gnn::bench
