#pragma once
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace gnn {
inline void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
namespace cuda {
// Sole owner of a device allocation. DeviceBuffer and Matrix<DeviceBuffer<T>> are borrowed handles.
class Allocation {
public:
    explicit Allocation(std::size_t bytes = 0) : bytes_(bytes) {
        if (bytes)
            checkCuda(cudaMalloc(&data_, bytes), "cudaMalloc allocation");
    }
    ~Allocation() {
        if (data_)
            static_cast<void>(cudaFree(data_));
    }
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    Allocation(Allocation&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}
    Allocation& operator=(Allocation&& other) noexcept {
        if (this != &other) {
            Allocation temp(std::move(other));
            std::swap(data_, temp.data_);
            std::swap(bytes_, temp.bytes_);
        }
        return *this;
    }
    void* data() const noexcept { return data_; }
    std::size_t bytes() const noexcept { return bytes_; }

private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};
class Event {
public:
    Event() { checkCuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~Event() {
        if (event_)
            static_cast<void>(cudaEventDestroy(event_));
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&& other) noexcept : event_(std::exchange(other.event_, nullptr)) {}
    Event& operator=(Event&& other) noexcept {
        if (this != &other) {
            Event temp(std::move(other));
            std::swap(event_, temp.event_);
        }
        return *this;
    }
    cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};
class Timer {
public:
    void start() { checkCuda(cudaEventRecord(start_.get()), "record compute start"); }
    double stop() {
        checkCuda(cudaEventRecord(stop_.get()), "record compute stop");
        checkCuda(cudaEventSynchronize(stop_.get()), "synchronize compute stop");
        float ms = 0;
        checkCuda(cudaEventElapsedTime(&ms, start_.get(), stop_.get()), "cudaEventElapsedTime");
        return ms;
    }

private:
    Event start_, stop_;
};
} // namespace cuda
} // namespace gnn
