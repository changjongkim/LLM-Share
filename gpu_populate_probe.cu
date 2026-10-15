// K2 of PLAN-20261015: whether a call of the kernel prepares host memory for
// the GPU as the CPU pass of STATOR does. STATOR reads one byte of every page
// of a mapped prefix from the CPU before the GPU reads it, since the GPU faults
// on an entry whose accessed flag is clear, and it puts memory behind its private
// rows with MADV_POPULATE_WRITE before the GPU writes them.
//
//   gpu_populate_probe ro FILE SIZE_MIB {none|touch|populate|willneed}
//   gpu_populate_probe rw SIZE_MIB {none|touch|populate}
//
// ro  maps SIZE_MIB of FILE private and read-only at a 2 MiB-aligned address
//     (as an agent maps the prefix), prepares it, and reads every word on the GPU.
//       none      nothing: the GPU faults on every entry
//       touch     a CPU read of one byte per 4 KiB page (STATOR)
//       populate  MADV_POPULATE_READ
//       willneed  MADV_WILLNEED
// rw  maps SIZE_MIB of private anonymous memory advised against huge pages (as
//     the private extent of an agent grows), prepares it, and writes one word
//     per 4 KiB page on the GPU.
//       none      nothing: the GPU faults on every page
//       touch     a CPU write of one byte per page
//       populate  MADV_POPULATE_WRITE (STATOR)
// Output: PREPARE mode= step= prepare_ms= gpu_ms= sum_ok=
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

#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kMiB = 1ULL << 20;
constexpr std::uint64_t kPage = 4096;

__global__ void sum_words(const std::uint64_t* data, std::uint64_t words, unsigned long long* out) {
  const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  unsigned long long local = 0;
  for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < words;
       i += stride) {
    local += data[i];
  }
  atomicAdd(out, local);
}

__global__ void write_pages(std::uint64_t* data, std::uint64_t pages) {
  const std::uint64_t page = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (page < pages) data[page * (kPage / 8)] = page;
}

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool ok(cudaError_t error, const char* what) {
  if (error == cudaSuccess) return true;
  std::printf("ERROR %s: %s\n", what, cudaGetErrorString(error));
  return false;
}

char* reserve(std::uint64_t bytes) {
  void* r = mmap(nullptr, bytes + 2 * kMiB, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (r == MAP_FAILED) return nullptr;
  return reinterpret_cast<char*>((reinterpret_cast<std::uintptr_t>(r) + 2 * kMiB - 1) & ~(2 * kMiB - 1));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (!((mode == "ro" && argc == 5) || (mode == "rw" && argc == 4))) {
    std::fprintf(stderr, "usage: gpu_populate_probe ro FILE SIZE_MIB STEP | rw SIZE_MIB STEP\n");
    return 2;
  }
  const std::uint64_t bytes = std::strtoull(argv[mode == "ro" ? 3 : 2], nullptr, 10) * kMiB;
  const std::string step = argv[mode == "ro" ? 4 : 3];
  if (!ok(cudaFree(nullptr), "init")) return 1;
  unsigned long long* sum = nullptr;
  if (!ok(cudaMalloc(&sum, sizeof(*sum)), "malloc") || !ok(cudaMemset(sum, 0, sizeof(*sum)), "memset")) return 1;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);

  char* base = reserve(bytes);
  if (base == nullptr) return 1;
  double prepare_ms = 0;
  double gpu_ms = 0;
  bool sum_ok = true;
  if (mode == "ro") {
    const int fd = open(argv[2], O_RDONLY);
    if (fd < 0 || mmap(base, bytes, PROT_READ, MAP_PRIVATE | MAP_NORESERVE | MAP_FIXED, fd, 0) == MAP_FAILED) {
      std::perror("map");
      return 1;
    }
    close(fd);
    // the CPU sum of the file for the check of the GPU read
    unsigned long long expected = 0;
    {
      const auto* words = reinterpret_cast<const std::uint64_t*>(base);
      // read through a second mapping, so that this mapping keeps the state of the step
      const int fd2 = open(argv[2], O_RDONLY);
      void* other = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd2, 0);
      if (other == MAP_FAILED) return 1;
      const auto* w2 = static_cast<const std::uint64_t*>(other);
      for (std::uint64_t i = 0; i < bytes / 8; ++i) expected += w2[i];
      munmap(other, bytes);
      close(fd2);
      (void)words;
    }
    const auto started = Clock::now();
    if (step == "touch") {
      const volatile char* view = base;
      for (std::uint64_t offset = 0; offset < bytes; offset += kPage) (void)view[offset];
    } else if (step == "populate") {
      if (madvise(base, bytes, MADV_POPULATE_READ) != 0) { std::perror("populate"); return 1; }
    } else if (step == "willneed") {
      if (madvise(base, bytes, MADV_WILLNEED) != 0) { std::perror("willneed"); return 1; }
    } else if (step != "none") {
      return 2;
    }
    prepare_ms = ms_since(started);
    const auto gpu_started = Clock::now();
    sum_words<<<sms * 8, 256>>>(reinterpret_cast<const std::uint64_t*>(base), bytes / 8, sum);
    if (!ok(cudaDeviceSynchronize(), "read")) return 1;
    gpu_ms = ms_since(gpu_started);
    unsigned long long got = 0;
    if (!ok(cudaMemcpy(&got, sum, sizeof(got), cudaMemcpyDeviceToHost), "copy")) return 1;
    sum_ok = got == expected;
  } else {
    if (mmap(base, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == MAP_FAILED ||
        madvise(base, bytes, MADV_NOHUGEPAGE) != 0) {
      std::perror("map");
      return 1;
    }
    const auto started = Clock::now();
    if (step == "touch") {
      volatile char* view = base;
      for (std::uint64_t offset = 0; offset < bytes; offset += kPage) view[offset] = 0;
    } else if (step == "populate") {
      if (madvise(base, bytes, MADV_POPULATE_WRITE) != 0) { std::perror("populate"); return 1; }
    } else if (step != "none") {
      return 2;
    }
    prepare_ms = ms_since(started);
    const std::uint64_t pages = bytes / kPage;
    const auto gpu_started = Clock::now();
    write_pages<<<static_cast<unsigned>((pages + 255) / 256), 256>>>(reinterpret_cast<std::uint64_t*>(base), pages);
    if (!ok(cudaDeviceSynchronize(), "write")) return 1;
    gpu_ms = ms_since(gpu_started);
    const auto* words = reinterpret_cast<const std::uint64_t*>(base);
    for (std::uint64_t page = 0; page < pages; ++page) sum_ok = sum_ok && words[page * (kPage / 8)] == page;
  }
  std::printf("PREPARE mode=%s step=%s prepare_ms=%.2f gpu_ms=%.2f sum_ok=%d\n", mode.c_str(), step.c_str(),
              prepare_ms, gpu_ms, sum_ok ? 1 : 0);
  return 0;
}
