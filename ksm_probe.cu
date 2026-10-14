// Kernel samepage merging (KSM) on memory that the GPU of a unified-memory
// device reads and writes through the host page tables.
//
//   ksm_probe SIZE_MIB COPIES {cuda|nocuda}
//
// COPIES buffers of SIZE_MIB anonymous memory with the same content are
// marked mergeable; the caller runs ksmd. The probe reports how long the
// merge takes (pages that ksmd counts as merged for this process). With cuda,
// the process initializes CUDA first and, after a line on stdin (the caller
// has stopped ksmd), reads one buffer on the GPU, reads a second from the CPU
// and then on the GPU, and writes one word per page of a third on the GPU,
// with the pages of the process that are mapped to a merged page after each
// step (the KSM line of smaps). It then prints DONE and waits for another
// line, so that the caller can unmerge everything while the process still
// maps it, and exits.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

__global__ void sum_words(const std::uint64_t* data, std::uint64_t words,
                          unsigned long long* out) {
  unsigned long long local = 0;
  const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += stride) {
    local += data[i];
  }
  atomicAdd(out, local);
}

__global__ void touch_pages(std::uint64_t* data, std::uint64_t pages) {
  const std::uint64_t page = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (page < pages) data[page * 512] += 1;
}

long smaps_kib(const char* field) {
  std::ifstream in("/proc/self/smaps_rollup");
  std::string line;
  const std::size_t length = std::strlen(field);
  while (std::getline(in, line)) {
    if (line.compare(0, length, field) == 0 && line.size() > length && line[length] == ':') {
      return std::strtol(line.c_str() + length + 1, nullptr, 10);
    }
  }
  return -1;
}

long ksmd_merged_pages() {
  std::ifstream in("/proc/self/ksm_stat");
  std::string line;
  const char* key = "ksm_merging_pages ";
  while (std::getline(in, line)) {
    if (line.compare(0, std::strlen(key), key) == 0) {
      return std::strtol(line.c_str() + std::strlen(key), nullptr, 10);
    }
  }
  return -1;
}

long mapped_merged_pages() { return smaps_kib("KSM") / 4; }

std::uint64_t* map_filled(std::uint64_t bytes) {
  void* mapped = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapped == MAP_FAILED) return nullptr;
  auto* words = static_cast<std::uint64_t*>(mapped);
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  for (std::uint64_t i = 0; i < bytes / 8; ++i) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    words[i] = state;
  }
  return words;
}

std::uint64_t cpu_sum(const std::uint64_t* words, std::uint64_t count) {
  std::uint64_t sum = 0;
  for (std::uint64_t i = 0; i < count; ++i) sum += words[i];
  return sum;
}

double gpu_sum(const std::uint64_t* words, std::uint64_t count, std::uint64_t expected,
               bool* right) {
  unsigned long long* out = nullptr;
  if (cudaMalloc(&out, sizeof(*out)) != cudaSuccess) return -1;
  if (cudaMemset(out, 0, sizeof(*out)) != cudaSuccess) return -1;
  const auto start = Clock::now();
  sum_words<<<1024, 256>>>(words, count, out);
  if (cudaDeviceSynchronize() != cudaSuccess) return -2;
  const double took = seconds_since(start);
  unsigned long long value = 0;
  if (cudaMemcpy(&value, out, sizeof(value), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
  cudaFree(out);
  *right = value == expected;
  return took;
}

void wait_line() {
  std::string line;
  std::getline(std::cin, line);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s SIZE_MIB COPIES cuda|nocuda\n", argv[0]);
    return 2;
  }
  const std::uint64_t bytes = std::strtoull(argv[1], nullptr, 10) << 20;
  const int copies = std::atoi(argv[2]);
  const bool cuda = std::string(argv[3]) == "cuda";
  if (copies < 3 || bytes == 0) return 2;
  if (cuda && cudaFree(nullptr) != cudaSuccess) return 3;
  std::vector<std::uint64_t*> buffers;
  for (int c = 0; c < copies; ++c) {
    std::uint64_t* words = map_filled(bytes);
    if (words == nullptr || madvise(words, bytes, MADV_MERGEABLE) != 0) return 3;
    buffers.push_back(words);
  }
  const std::uint64_t expected = cpu_sum(buffers[0], bytes / 8);
  const long want = static_cast<long>(copies) * static_cast<long>(bytes / 4096);
  std::printf("READY pages=%ld\n", want);
  std::fflush(stdout);
  const auto start = Clock::now();
  long merged = 0;
  while (seconds_since(start) < 900) {
    merged = ksmd_merged_pages();
    if (merged >= want * 99 / 100) break;
    usleep(100000);
  }
  const double merge_s = seconds_since(start);
  std::printf("MERGE cuda=%d size_mib=%llu copies=%d merged_pages=%ld of=%ld merge_s=%.2f "
              "pages_per_s=%.0f\n",
              cuda ? 1 : 0, static_cast<unsigned long long>(bytes >> 20), copies, merged, want,
              merge_s, merged / merge_s);
  std::printf("MERGED\n");
  std::fflush(stdout);
  wait_line();
  if (cuda) {
    const long m0 = mapped_merged_pages();
    bool right = false;
    const double read_s = gpu_sum(buffers[0], bytes / 8, expected, &right);
    const long m1 = mapped_merged_pages();
    const auto touch_start = Clock::now();
    const volatile char* view = reinterpret_cast<const volatile char*>(buffers[1]);
    for (std::uint64_t offset = 0; offset < bytes; offset += 4096) (void)view[offset];
    const double cpu_s = seconds_since(touch_start);
    bool right2 = false;
    const double read2_s = gpu_sum(buffers[1], bytes / 8, expected, &right2);
    const long m2 = mapped_merged_pages();
    const std::uint64_t pages = bytes / 4096;
    const auto write_start = Clock::now();
    touch_pages<<<static_cast<unsigned>((pages + 255) / 256), 256>>>(buffers[2], pages);
    const cudaError_t written = cudaDeviceSynchronize();
    const double write_s = seconds_since(write_start);
    const long m3 = mapped_merged_pages();
    std::printf("GPU merged_before=%ld gpu_read_s=%.4f right=%d merged_after_gpu_read=%ld "
                "cpu_read_s=%.4f gpu_read_after_cpu_s=%.4f right2=%d merged_after_cpu_gpu_read=%ld "
                "gpu_write=%s gpu_write_s=%.4f written_pages=%llu merged_after_gpu_write=%ld\n",
                m0, read_s, right ? 1 : 0, m1, cpu_s, read2_s, right2 ? 1 : 0, m2,
                cudaGetErrorName(written), write_s, static_cast<unsigned long long>(pages), m3);
  }
  std::printf("DONE\n");
  std::fflush(stdout);
  wait_line();
  return 0;
}
