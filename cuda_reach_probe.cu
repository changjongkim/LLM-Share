// How fast the GPU reads a buffer, by the kind of memory behind it.
//
//   cuda_reach_probe KIND SIZE_MIB PASSES
//
//   KIND  device      cudaMalloc
//         host_huge   private anonymous host memory, 2 MiB pages
//         host_small  private anonymous host memory, 4 KiB pages
//
// The host kinds are pageable memory that the kernel below reads through the
// host page tables, as the engine does with GGML_CUDA_HOST_PTR. Three
// kernels read the same buffer:
//   full    every word once; threads read adjacent chunks, as a kernel that
//           scans the rows of a cache does
//   page    one word of every 4 KiB page
//   block   one word of every 2 MiB block
// Each runs PASSES times after a warm-up pass; the line reports the median
// and the fastest pass. A slower full pass with equal page and block passes
// is a limit on bytes; slower page or block passes are a cost per
// translation.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/mman.h>

namespace {

#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

constexpr unsigned kThreads = 256;

// Thread t of all threads sums `chunk` adjacent words starting at t * chunk.
__global__ void read_full(const uint32_t* data, uint64_t chunk,
                          unsigned long long* out) {
  const uint64_t thread =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint32_t* first = data + thread * chunk;
  uint32_t sum = 0;
  for (uint64_t i = 0; i < chunk; ++i) {
    sum += first[i];
  }
  atomicAdd(out + (blockIdx.x & 1023U), static_cast<unsigned long long>(sum));
}

// Thread t reads the first word of unit t, a unit being `stride` words.
__global__ void read_stride(const uint32_t* data, uint64_t stride,
                            uint64_t units, unsigned long long* out) {
  const uint64_t thread =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (thread < units) {
    atomicAdd(out + (blockIdx.x & 1023U),
              static_cast<unsigned long long>(data[thread * stride]));
  }
}

using Clock = std::chrono::steady_clock;

struct Timing {
  double median_ms;
  double fastest_ms;
};

template <typename Launch>
Timing timed(int passes, Launch launch) {
  launch();
  cudaDeviceSynchronize();
  std::vector<double> samples;
  for (int pass = 0; pass < passes; ++pass) {
    const auto start = Clock::now();
    launch();
    if (cudaDeviceSynchronize() != cudaSuccess) {
      return {-1.0, -1.0};
    }
    samples.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - start)
            .count());
  }
  std::sort(samples.begin(), samples.end());
  return {samples[samples.size() / 2], samples.front()};
}

int run(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: cuda_reach_probe KIND SIZE_MIB PASSES\n");
    return 2;
  }
  const std::string kind = argv[1];
  const uint64_t bytes = std::strtoull(argv[2], nullptr, 10) << 20U;
  const int passes = std::atoi(argv[3]);
  const uint64_t huge = 2ULL << 20U;
  if (bytes == 0 || bytes % huge != 0 || passes < 1) {
    std::fprintf(stderr, "SIZE_MIB must be a positive multiple of 2\n");
    return 2;
  }

  uint32_t* data = nullptr;
  if (kind == "device") {
    void* memory = nullptr;
    if (cudaMalloc(&memory, bytes) != cudaSuccess ||
        cudaMemset(memory, 1, bytes) != cudaSuccess) {
      std::fprintf(stderr, "cannot allocate device memory\n");
      return 3;
    }
    data = static_cast<uint32_t*>(memory);
  } else if (kind == "host_huge" || kind == "host_small") {
    void* reserved = mmap(nullptr, bytes + huge, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (reserved == MAP_FAILED) {
      std::fprintf(stderr, "cannot map host memory\n");
      return 3;
    }
    char* base = reinterpret_cast<char*>(
        (reinterpret_cast<uintptr_t>(reserved) + huge - 1) & ~(huge - 1));
    madvise(base, bytes, kind == "host_huge" ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    if (madvise(base, bytes, MADV_POPULATE_WRITE) != 0) {
      std::fprintf(stderr, "cannot populate host memory\n");
      return 3;
    }
    std::memset(base, 1, bytes);
    data = reinterpret_cast<uint32_t*>(base);
  } else {
    std::fprintf(stderr, "unknown kind %s\n", kind.c_str());
    return 2;
  }

  unsigned long long* out = nullptr;
  if (cudaMalloc(&out, 1024 * sizeof(unsigned long long)) != cudaSuccess ||
      cudaMemset(out, 0, 1024 * sizeof(unsigned long long)) != cudaSuccess) {
    std::fprintf(stderr, "cannot allocate the result\n");
    return 3;
  }

  const uint64_t words = bytes / sizeof(uint32_t);
  // 256 words per thread: a block of threads covers 256 KiB.
  const uint64_t chunk = 256;
  const unsigned full_blocks = static_cast<unsigned>(words / chunk / kThreads);
  const Timing full = timed(passes, [&] {
    read_full<<<full_blocks, kThreads>>>(data, chunk, out);
  });
  const uint64_t pages = bytes / 4096;
  const Timing page = timed(passes, [&] {
    read_stride<<<static_cast<unsigned>((pages + kThreads - 1) / kThreads),
                  kThreads>>>(data, 4096 / sizeof(uint32_t), pages, out);
  });
  const uint64_t blocks = bytes / huge;
  const Timing block = timed(passes, [&] {
    read_stride<<<static_cast<unsigned>((blocks + kThreads - 1) / kThreads),
                  kThreads>>>(data, huge / sizeof(uint32_t), blocks, out);
  });
  if (full.median_ms < 0 || page.median_ms < 0 || block.median_ms < 0) {
    std::fprintf(stderr, "a kernel failed\n");
    return 4;
  }
  std::printf(
      "RESULT kind=%s size_mib=%llu passes=%d full_ms=%.3f full_fastest_ms=%.3f "
      "full_gib_per_s=%.1f page_ms=%.3f page_fastest_ms=%.3f ns_per_page=%.2f "
      "block_ms=%.3f block_fastest_ms=%.3f\n",
      kind.c_str(), static_cast<unsigned long long>(bytes >> 20U), passes,
      full.median_ms, full.fastest_ms,
      static_cast<double>(bytes) / (1ULL << 30U) / (full.median_ms / 1000.0),
      page.median_ms, page.fastest_ms,
      page.median_ms * 1e6 / static_cast<double>(pages), block.median_ms,
      block.fastest_ms);
  return 0;
}

}  // namespace

int main(int argc, char** argv) { return run(argc, argv); }
