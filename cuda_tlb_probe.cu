// Does the reach of the GPU's address translation explain a loss of speed
// with a buffer in host memory?
//
//   cuda_tlb_probe KIND SIZE_MIB STRIDE_KIB [FILE]
//
// KIND   device      device memory (cudaMalloc)
//        host_huge   private anonymous host memory in 2 MiB pages
//        host_small  private anonymous host memory in 4 KiB pages
//        file        a read-only private mapping of FILE, as an agent maps a
//                    published prefix (the page size is that of its tmpfs)
//
// The buffer is filled and read on the CPU first, so that every page is
// present and marked accessed. Two kernels then read it on the GPU:
//
//   scan    every thread reads consecutive words of its own share
//   gather  every thread reads words at addresses that jump STRIDE_KIB (or a
//           pseudo-random multiple of it) through the whole buffer, so that
//           almost every read is on another page
//
// and one writes it:
//
//   scatter the same addresses as gather, written (not for KIND file)
//
// Each kernel runs REPEAT times; the best time is reported as words per
// second, with the sum that proves the reads happened.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

const int kRepeat = 5;
const unsigned kThreads = 256;

__global__ void scan(const std::uint32_t* data, std::uint64_t words,
                     std::uint64_t reads_per_thread, unsigned long long* out) {
  const std::uint64_t thread = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t threads = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  const std::uint64_t share = words / threads;
  const std::uint64_t first = thread * share;
  unsigned long long sum = 0;
  for (std::uint64_t i = 0; i < reads_per_thread; ++i) {
    sum += data[first + i % share];
  }
  atomicAdd(out, sum);
}

// Address of read i of a thread: a multiplicative walk over the strides of
// the buffer, different for every thread.
__device__ std::uint64_t address(std::uint64_t thread, std::uint64_t i,
                                 std::uint64_t strides, std::uint64_t stride_words) {
  const std::uint64_t slot = (thread * 2654435761ULL + i * 40503ULL + thread * i) % strides;
  return slot * stride_words + (thread + i) % stride_words;
}

__global__ void gather(const std::uint32_t* data, std::uint64_t strides,
                       std::uint64_t stride_words, std::uint64_t reads_per_thread,
                       unsigned long long* out) {
  const std::uint64_t thread = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned long long sum = 0;
  for (std::uint64_t i = 0; i < reads_per_thread; ++i) {
    sum += data[address(thread, i, strides, stride_words)];
  }
  atomicAdd(out, sum);
}

__global__ void scatter(std::uint32_t* data, std::uint64_t strides,
                        std::uint64_t stride_words, std::uint64_t writes_per_thread) {
  const std::uint64_t thread = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (std::uint64_t i = 0; i < writes_per_thread; ++i) {
    data[address(thread, i, strides, stride_words)] = static_cast<std::uint32_t>(thread + i);
  }
}

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

int run(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: cuda_tlb_probe KIND SIZE_MIB STRIDE_KIB [FILE]\n");
    return 2;
  }
  const std::string kind = argv[1];
  const std::uint64_t bytes = std::strtoull(argv[2], nullptr, 10) << 20;
  const std::uint64_t stride = std::strtoull(argv[3], nullptr, 10) << 10;
  const std::uint64_t words = bytes / sizeof(std::uint32_t);
  const std::uint64_t stride_words = stride / sizeof(std::uint32_t);
  const std::uint64_t strides = bytes / stride;
  if (bytes == 0 || stride < sizeof(std::uint32_t) || strides == 0) return 2;

  if (cudaSetDevice(0) != cudaSuccess) return 3;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);

  std::uint32_t* data = nullptr;
  bool writable = true;
  if (kind == "device") {
    if (cudaMalloc(&data, bytes) != cudaSuccess) return 3;
    if (cudaMemset(data, 1, bytes) != cudaSuccess) return 3;
  } else if (kind == "host_huge" || kind == "host_small") {
    const std::uint64_t huge = 2u << 20;
    void* reserved = mmap(nullptr, bytes + huge, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (reserved == MAP_FAILED) return 3;
    data = reinterpret_cast<std::uint32_t*>(
        (reinterpret_cast<std::uintptr_t>(reserved) + huge - 1) & ~(huge - 1));
    madvise(data, bytes, kind == "host_huge" ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    std::memset(data, 1, bytes);
  } else if (kind == "file") {
    if (argc < 5) return 2;
    const int fd = open(argv[4], O_RDONLY);
    if (fd < 0) return 3;
    void* mapped = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) return 3;
    data = static_cast<std::uint32_t*>(mapped);
    writable = false;
  } else {
    return 2;
  }
  if (kind != "device") {
    // A CPU read of every page: the device faults on entries without the
    // accessed flag.
    volatile std::uint32_t seen = 0;
    for (std::uint64_t at = 0; at < bytes; at += 4096) {
      seen = seen + reinterpret_cast<const volatile unsigned char*>(data)[at];
    }
  }

  unsigned long long* out = nullptr;
  if (cudaMalloc(&out, sizeof(*out)) != cudaSuccess) return 3;
  const unsigned blocks = static_cast<unsigned>(sms) * 16;
  const std::uint64_t threads = static_cast<std::uint64_t>(blocks) * kThreads;
  const std::uint64_t reads = 2048;

  double best[3] = {0.0, 0.0, 0.0};
  unsigned long long sums[2] = {0, 0};
  for (int pass = 0; pass < 3; ++pass) {
    if (pass == 2 && !writable) break;
    for (int repeat = 0; repeat <= kRepeat; ++repeat) {
      cudaMemset(out, 0, sizeof(*out));
      cudaDeviceSynchronize();
      const auto start = std::chrono::steady_clock::now();
      if (pass == 0) {
        scan<<<blocks, kThreads>>>(data, words, reads, out);
      } else if (pass == 1) {
        gather<<<blocks, kThreads>>>(data, strides, stride_words, reads, out);
      } else {
        scatter<<<blocks, kThreads>>>(data, strides, stride_words, reads);
      }
      const cudaError_t status = cudaDeviceSynchronize();
      const double elapsed = seconds_since(start);
      if (status != cudaSuccess) {
        std::fprintf(stderr, "kernel %d failed: %s\n", pass, cudaGetErrorName(status));
        return 4;
      }
      if (repeat == 0) continue;  // the first run pays for the first touch
      const double rate = static_cast<double>(threads * reads) / elapsed;
      if (rate > best[pass]) best[pass] = rate;
      if (pass < 2) cudaMemcpy(&sums[pass], out, sizeof(*out), cudaMemcpyDeviceToHost);
    }
  }
  std::printf("RESULT kind=%s sms=%d size_mib=%llu stride_kib=%llu threads=%llu "
              "scan_mwords_s=%.1f gather_mwords_s=%.1f scatter_mwords_s=%.1f "
              "scan_sum=%llu gather_sum=%llu\n",
              kind.c_str(), sms, static_cast<unsigned long long>(bytes >> 20),
              static_cast<unsigned long long>(stride >> 10),
              static_cast<unsigned long long>(threads), best[0] / 1e6, best[1] / 1e6,
              best[2] / 1e6, sums[0], sums[1]);
  std::fflush(stdout);
  _exit(0);
}

}  // namespace

int main(int argc, char** argv) { return run(argc, argv); }
