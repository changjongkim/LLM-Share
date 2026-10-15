// Where the GPU loses speed when it reads a KV cache from host memory with the
// access pattern of attention. A row of the cache holds every KV head of one
// cell (1 KiB for 4 heads, 2 KiB for 8 heads of 128 f16 values); a decoding step
// reads, for each head, the 256 bytes of that head in every row. On the whole
// GPU of Thor, eight agents of Llama-3.1-8B (8 KV heads) on STATOR generate at
// 0.912 of a copy in device memory and a private cache in host memory at
// 0.944-0.951, where Qwen2.5-7B (4 KV heads) loses nothing.
//
//   gpu_attn_probe KIND SIZE_MIB ROW_BYTES ROUNDS [FILE]
//
// KIND is where the buffer lives:
//   device     cudaMalloc
//   host_4k    anonymous memory advised against huge pages (4 KiB entries)
//   host_thp   anonymous memory advised for huge pages (2 MiB entries)
//   tlb_2m     anonymous hugetlb memory with 2 MiB pages
//   tlb_1g     anonymous hugetlb memory with 1 GiB pages
//   file_ro    FILE (on a tmpfs with 2 MiB pages) mapped private and read-only,
//              as an agent of STATOR maps the prefix
// Host memory is written and then read from the CPU before the GPU reads it,
// so that the GPU takes no fault. Each round times three kernels over the
// buffer: `heads` reads the 256-byte slice of every head in every row, one
// head after the other per block (the pattern of attention); `rows` reads whole
// rows in order; `random` reads one 256-byte slice at a pseudo-random row.
// Output, one line per round:
//   ATTN kind= size_mib= row_bytes= round= heads_gbs= rows_gbs= random_gbs=
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

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif

namespace {

constexpr std::uint64_t kMiB = 1ULL << 20;
constexpr std::uint64_t kSlice = 256;  // bytes of one head in a row (128 f16 values)

// One warp reads one 256-byte slice: 32 threads x 8 bytes.
__global__ void read_heads(const std::uint64_t* data, std::uint64_t rows, std::uint64_t row_words,
                           std::uint64_t heads, unsigned long long* sink) {
  const std::uint64_t lane = threadIdx.x % 32;
  const std::uint64_t warp = (static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const std::uint64_t warps = static_cast<std::uint64_t>(gridDim.x) * blockDim.x / 32;
  const std::uint64_t slice_words = kSlice / 8;
  unsigned long long local = 0;
  // warps sweep head by head; within a head, consecutive warps take consecutive rows
  for (std::uint64_t task = warp; task < rows * heads; task += warps) {
    const std::uint64_t head = task / rows;
    const std::uint64_t row = task % rows;
    local += data[row * row_words + head * slice_words + lane];
  }
  if (local == 0x1234567) atomicAdd(sink, local);
}

__global__ void read_rows(const std::uint64_t* data, std::uint64_t words, unsigned long long* sink) {
  const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  unsigned long long local = 0;
  for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < words;
       i += stride) {
    local += data[i];
  }
  if (local == 0x1234567) atomicAdd(sink, local);
}

__global__ void read_random(const std::uint64_t* data, std::uint64_t rows, std::uint64_t row_words,
                            std::uint64_t heads, std::uint64_t tasks, unsigned long long* sink) {
  const std::uint64_t lane = threadIdx.x % 32;
  const std::uint64_t warp = (static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const std::uint64_t warps = static_cast<std::uint64_t>(gridDim.x) * blockDim.x / 32;
  unsigned long long local = 0;
  for (std::uint64_t task = warp; task < tasks; task += warps) {
    std::uint64_t x = task * 0x9E3779B97F4A7C15ULL;
    x ^= x >> 31;
    const std::uint64_t row = x % rows;
    const std::uint64_t head = (x >> 40) % heads;
    local += data[row * row_words + head * (kSlice / 8) + lane];
  }
  if (local == 0x1234567) atomicAdd(sink, local);
}

bool ok(cudaError_t error, const char* what) {
  if (error == cudaSuccess) return true;
  std::printf("ERROR %s: %s\n", what, cudaGetErrorString(error));
  return false;
}

char* host_buffer(const std::string& kind, std::uint64_t bytes, const char* file) {
  void* p = MAP_FAILED;
  if (kind == "host_4k" || kind == "host_thp") {
    p = mmap(nullptr, bytes + 2 * kMiB, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    char* aligned = reinterpret_cast<char*>((reinterpret_cast<std::uintptr_t>(p) + 2 * kMiB - 1) & ~(2 * kMiB - 1));
    if (madvise(aligned, bytes, kind == "host_4k" ? MADV_NOHUGEPAGE : MADV_HUGEPAGE) != 0) return nullptr;
    return aligned;
  }
  if (kind == "tlb_2m" || kind == "tlb_1g") {
    const int size_flag = kind == "tlb_2m" ? (21 << MAP_HUGE_SHIFT) : (30 << MAP_HUGE_SHIFT);
    p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | size_flag, -1, 0);
    return p == MAP_FAILED ? nullptr : static_cast<char*>(p);
  }
  if (kind == "file_ro" && file != nullptr) {
    // fill the file through a shared mapping, then map it as an agent does
    const int fd = open(file, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0 || ftruncate(fd, static_cast<off_t>(bytes)) != 0) return nullptr;
    void* w = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (w == MAP_FAILED) return nullptr;
    auto* words = static_cast<std::uint64_t*>(w);
    for (std::uint64_t i = 0; i < bytes / 8; ++i) words[i] = i * 0x9E3779B97F4A7C15ULL;
    munmap(w, bytes);
    void* r = mmap(nullptr, bytes + 2 * kMiB, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (r == MAP_FAILED) return nullptr;
    char* base = reinterpret_cast<char*>((reinterpret_cast<std::uintptr_t>(r) + 2 * kMiB - 1) & ~(2 * kMiB - 1));
    if (mmap(base, bytes, PROT_READ, MAP_PRIVATE | MAP_NORESERVE | MAP_FIXED, fd, 0) == MAP_FAILED) return nullptr;
    close(fd);
    return base;
  }
  return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: gpu_attn_probe KIND SIZE_MIB ROW_BYTES ROUNDS [FILE]\n");
    return 2;
  }
  const std::string kind = argv[1];
  const std::uint64_t bytes = std::strtoull(argv[2], nullptr, 10) * kMiB;
  const std::uint64_t row_bytes = std::strtoull(argv[3], nullptr, 10);
  const int rounds = std::atoi(argv[4]);
  const char* file = argc > 5 ? argv[5] : nullptr;
  if (row_bytes < kSlice || row_bytes % kSlice != 0 || bytes % row_bytes != 0) return 2;
  if (!ok(cudaFree(nullptr), "init")) return 1;

  std::uint64_t* data = nullptr;
  if (kind == "device") {
    if (!ok(cudaMalloc(&data, bytes), "malloc")) return 1;
    if (!ok(cudaMemset(data, 1, bytes), "memset")) return 1;
  } else {
    char* host = host_buffer(kind, bytes, file);
    if (host == nullptr) {
      std::printf("ERROR kind=%s: cannot map the buffer\n", kind.c_str());
      return 1;
    }
    if (kind != "file_ro") std::memset(host, 1, bytes);
    // a CPU read of every page: the GPU takes no fault on it
    const volatile char* view = host;
    for (std::uint64_t offset = 0; offset < bytes; offset += 4096) (void)view[offset];
    data = reinterpret_cast<std::uint64_t*>(host);
  }
  unsigned long long* sink = nullptr;
  if (!ok(cudaMalloc(&sink, sizeof(*sink)), "malloc")) return 1;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int blocks = sms * 16;
  const std::uint64_t rows = bytes / row_bytes;
  const std::uint64_t row_words = row_bytes / 8;
  const std::uint64_t heads = row_bytes / kSlice;
  const std::uint64_t random_tasks = rows * heads;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  auto timed = [&](auto launch) -> float {
    cudaEventRecord(start);
    launch();
    cudaEventRecord(stop);
    if (!ok(cudaEventSynchronize(stop), "kernel")) std::exit(1);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
  };
  // one pass of each kernel before timing
  read_heads<<<blocks, 256>>>(data, rows, row_words, heads, sink);
  read_rows<<<blocks, 256>>>(data, bytes / 8, sink);
  if (!ok(cudaDeviceSynchronize(), "warm")) return 1;
  for (int round = 1; round <= rounds; ++round) {
    const float heads_ms = timed([&] { read_heads<<<blocks, 256>>>(data, rows, row_words, heads, sink); });
    const float rows_ms = timed([&] { read_rows<<<blocks, 256>>>(data, bytes / 8, sink); });
    const float random_ms =
        timed([&] { read_random<<<blocks, 256>>>(data, rows, row_words, heads, random_tasks, sink); });
    const double gb = static_cast<double>(bytes) / 1e9;
    std::printf("ATTN kind=%s size_mib=%llu row_bytes=%llu round=%d heads_gbs=%.2f rows_gbs=%.2f random_gbs=%.2f\n",
                kind.c_str(), static_cast<unsigned long long>(bytes / kMiB),
                static_cast<unsigned long long>(row_bytes), round, gb / (heads_ms / 1e3), gb / (rows_ms / 1e3),
                gb / (random_ms / 1e3));
    std::fflush(stdout);
  }
  return 0;
}
