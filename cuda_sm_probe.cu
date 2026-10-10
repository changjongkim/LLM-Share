// How many streaming multiprocessors a CUDA device gives to a kernel that
// only computes. Prints the count that the runtime reports, the time of a
// kernel of N blocks of 1,024 threads for N = 1..32 (a block of this size
// occupies one SM, so the time steps up when N exceeds the SMs), and the
// rate of a kernel with many small blocks.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void spin(std::uint32_t* out, std::uint32_t iterations) {
  std::uint32_t x = blockIdx.x * 2654435761u + threadIdx.x + 1u;
  for (std::uint32_t i = 0; i < iterations; ++i) {
    x = x * 1664525u + 1013904223u;
    x ^= x >> 13;
  }
  if (threadIdx.x == 0) out[blockIdx.x] = x;
}

static double run_ms(std::uint32_t* out, int blocks, int threads, std::uint32_t iterations, int runs) {
  std::vector<double> ms;
  for (int r = 0; r < runs; ++r) {
    const auto begin = std::chrono::steady_clock::now();
    spin<<<blocks, threads>>>(out, iterations);
    if (cudaDeviceSynchronize() != cudaSuccess) std::exit(4);
    ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

int main() {
  if (cudaSetDevice(0) != cudaSuccess) return 3;
  int sms = 0, threads_per_sm = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  cudaDeviceGetAttribute(&threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
  std::uint32_t* out = nullptr;
  if (cudaMalloc(&out, 65536 * sizeof(std::uint32_t)) != cudaSuccess) return 3;
  run_ms(out, 64, 1024, 1000, 2);
  std::printf("reported_sms=%d max_threads_per_sm=%d\n", sms, threads_per_sm);
  for (int blocks = 1; blocks <= 32; ++blocks) {
    std::printf("STEP blocks=%d ms=%.2f\n", blocks, run_ms(out, blocks, 1024, 2000000, 5));
  }
  const int blocks = 8192;
  const std::uint32_t iterations = 200000;
  const double ms = run_ms(out, blocks, 256, iterations, 7);
  std::printf("RATE blocks=%d threads=256 ms=%.2f giga_iterations_per_s=%.2f\n", blocks, ms,
              double(blocks) * 256 * iterations / ms / 1e6);
  return 0;
}
