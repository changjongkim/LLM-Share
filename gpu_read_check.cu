// Whether the GPU of a unified-memory device reads the data of a file when it
// faults on the pages of a private, read-only mapping of the file, as an agent
// of STATOR (Map Only) faults on the published prefix.
//
//   gpu_read_check make FILE SIZE_MIB
//   gpu_read_check publish FILE SIZE_MIB HEAD_MIB READY_FILE STOP_FILE
//   gpu_read_check check FILE HEAD_MIB ROUNDS {gpu|cpu} [GO_FILE]
//
// Every 8-byte word of FILE holds a value that is a function of its index
// (expected()), so a reader checks every word without a second copy and, for
// a word that differs, finds the index whose value it read (index_of()).
//
// make     writes FILE from the CPU.
// publish  maps FILE shared and writable, as the publisher of STATOR maps its
//          cache file, puts memory behind every page, writes every word
//          through the GPU, makes the first HEAD_MIB read-only and reads them
//          from the CPU, creates READY_FILE, and then writes the words after
//          HEAD_MIB through the GPU again until STOP_FILE exists, as the
//          publisher continues to generate after it publishes.
// check    ROUNDS times: maps the first HEAD_MIB of FILE private and read-only
//          at a 2 MiB-aligned address, as an agent maps the prefix, reads every
//          page from the CPU first (cpu) or not (gpu), and compares every word
//          on the GPU. The mapping is removed after each round, so that every
//          round faults again. With GO_FILE, the process initializes CUDA and
//          waits for that file before the first round, so that many processes
//          fault together.
//
// Output of check, one line per round and up to four examples:
//   CHECK round= words= differ= zero= pages= gpu_ms=
//   EXAMPLE round= index= value= source=      (source: index whose value it is, or -1)
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kMiB = 1ULL << 20;
constexpr std::uint64_t kHuge = 2 * kMiB;
constexpr std::uint64_t kPage = 4096;
constexpr std::uint64_t kWordsPerPage = kPage / 8;
constexpr std::uint64_t kOffset = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t kFirst = 0xff51afd7ed558ccdULL;
constexpr std::uint64_t kSecond = 0xc4ceb9fe1a85ec53ULL;
constexpr unsigned kExamples = 4;

__host__ __device__ inline std::uint64_t expected(std::uint64_t index) {
  std::uint64_t x = index + kOffset;
  x ^= x >> 33;
  x *= kFirst;
  x ^= x >> 33;
  x *= kSecond;
  x ^= x >> 33;
  return x;
}

std::uint64_t inverse(std::uint64_t odd) {
  std::uint64_t x = odd;
  for (int i = 0; i < 6; ++i) x *= 2 - odd * x;
  return x;
}

std::uint64_t index_of(std::uint64_t value) {
  std::uint64_t x = value;
  x ^= x >> 33;
  x *= inverse(kSecond);
  x ^= x >> 33;
  x *= inverse(kFirst);
  x ^= x >> 33;
  return x - kOffset;
}

struct Report {
  unsigned long long differ;
  unsigned long long zero;
  unsigned int examples;
  unsigned long long index[kExamples];
  unsigned long long value[kExamples];
};

__global__ void check_words(const std::uint64_t* data, std::uint64_t words, Report* report,
                            unsigned int* pages) {
  const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += stride) {
    const std::uint64_t value = data[i];
    if (value == expected(i)) continue;
    atomicAdd(&report->differ, 1ULL);
    if (value == 0) atomicAdd(&report->zero, 1ULL);
    pages[i / kWordsPerPage] = 1;
    const unsigned int slot = atomicAdd(&report->examples, 1U);
    if (slot < kExamples) {
      report->index[slot] = i;
      report->value[slot] = value;
    }
  }
}

__global__ void write_words(std::uint64_t* data, std::uint64_t first, std::uint64_t last) {
  const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t i = first + static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < last; i += stride) {
    data[i] = expected(i);
  }
}

bool ok(cudaError_t error, const char* what) {
  if (error == cudaSuccess) return true;
  std::printf("ERROR %s: %s\n", what, cudaGetErrorString(error));
  return false;
}

int blocks() {
  int count = 0;
  if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) count = 1;
  return count * 8;
}

bool exists(const std::string& path) { return access(path.c_str(), F_OK) == 0; }

void cpu_touch(const void* data, std::uint64_t bytes) {
  const volatile char* view = static_cast<const volatile char*>(data);
  for (std::uint64_t offset = 0; offset < bytes; offset += kPage) (void)view[offset];
}

// A 2 MiB-aligned address with `bytes` of address space reserved after it.
char* reserve(std::uint64_t bytes, void** start) {
  void* reserved = mmap(nullptr, bytes + kHuge, PROT_NONE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (reserved == MAP_FAILED) return nullptr;
  *start = reserved;
  return reinterpret_cast<char*>((reinterpret_cast<std::uintptr_t>(reserved) + kHuge - 1) &
                                 ~(kHuge - 1));
}

int make(const std::string& path, std::uint64_t bytes) {
  const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd < 0) {
    std::perror("open");
    return 1;
  }
  std::vector<std::uint64_t> chunk(kMiB / 8);
  for (std::uint64_t mib = 0; mib < bytes / kMiB; ++mib) {
    for (std::uint64_t i = 0; i < chunk.size(); ++i) chunk[i] = expected(mib * chunk.size() + i);
    if (write(fd, chunk.data(), kMiB) != static_cast<ssize_t>(kMiB)) {
      std::perror("write");
      close(fd);
      return 1;
    }
  }
  close(fd);
  std::printf("MADE bytes=%llu\n", static_cast<unsigned long long>(bytes));
  return 0;
}

int publish(const std::string& path, std::uint64_t bytes, std::uint64_t head,
            const std::string& ready, const std::string& stop) {
  const int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
  if (fd < 0 || ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    std::perror("file");
    return 1;
  }
  void* start = nullptr;
  char* base = reserve(bytes, &start);
  if (base == nullptr ||
      mmap(base, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) == MAP_FAILED) {
    std::perror("mmap");
    return 1;
  }
  close(fd);
  if (madvise(base, bytes, MADV_POPULATE_WRITE) != 0) {
    std::perror("madvise");
    return 1;
  }
  auto* words = reinterpret_cast<std::uint64_t*>(base);
  write_words<<<blocks(), 256>>>(words, 0, bytes / 8);
  if (!ok(cudaDeviceSynchronize(), "publish write")) return 1;
  if (mprotect(base, head, PROT_READ) != 0) {
    std::perror("mprotect");
    return 1;
  }
  cpu_touch(base, head);
  std::uint64_t differ = 0;
  for (std::uint64_t i = 0; i < head / 8; ++i) differ += words[i] != expected(i);
  const int ready_fd = open(ready.c_str(), O_WRONLY | O_CREAT, 0600);
  if (ready_fd >= 0) close(ready_fd);
  std::uint64_t loops = 0;
  while (!exists(stop)) {
    write_words<<<blocks(), 256>>>(words, head / 8, bytes / 8);
    if (!ok(cudaDeviceSynchronize(), "publish tail")) return 1;
    ++loops;
  }
  std::printf("PUBLISH head_differ_cpu=%llu tail_loops=%llu\n",
              static_cast<unsigned long long>(differ), static_cast<unsigned long long>(loops));
  return 0;
}

int check(const std::string& path, std::uint64_t head, int rounds, bool cpu_first,
          const std::string& go) {
  const std::uint64_t words = head / 8;
  const std::uint64_t pages = head / kPage;
  Report* report = nullptr;
  unsigned int* page_flags = nullptr;
  if (!ok(cudaMalloc(&report, sizeof(Report)), "malloc") ||
      !ok(cudaMalloc(&page_flags, pages * sizeof(unsigned int)), "malloc")) {
    return 1;
  }
  std::vector<unsigned int> flags(pages);
  if (!go.empty()) {
    while (!exists(go)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  for (int round = 1; round <= rounds; ++round) {
    const int fd = open(path.c_str(), O_RDONLY);
    void* start = nullptr;
    char* base = reserve(head, &start);
    if (fd < 0 || base == nullptr ||
        mmap(base, head, PROT_READ, MAP_PRIVATE | MAP_NORESERVE | MAP_FIXED, fd, 0) == MAP_FAILED) {
      std::perror("map");
      return 1;
    }
    close(fd);
    if (cpu_first) cpu_touch(base, head);
    if (!ok(cudaMemset(report, 0, sizeof(Report)), "memset") ||
        !ok(cudaMemset(page_flags, 0, pages * sizeof(unsigned int)), "memset")) {
      return 1;
    }
    const auto started = Clock::now();
    check_words<<<blocks(), 256>>>(reinterpret_cast<const std::uint64_t*>(base), words, report,
                                   page_flags);
    if (!ok(cudaDeviceSynchronize(), "check")) return 1;
    const double gpu_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    Report host{};
    if (!ok(cudaMemcpy(&host, report, sizeof(Report), cudaMemcpyDeviceToHost), "copy") ||
        !ok(cudaMemcpy(flags.data(), page_flags, pages * sizeof(unsigned int),
                       cudaMemcpyDeviceToHost), "copy")) {
      return 1;
    }
    std::uint64_t differing_pages = 0;
    for (unsigned int flag : flags) differing_pages += flag;
    std::printf("CHECK round=%d words=%llu differ=%llu zero=%llu pages=%llu gpu_ms=%.1f\n", round,
                static_cast<unsigned long long>(words), host.differ, host.zero,
                static_cast<unsigned long long>(differing_pages), gpu_ms);
    for (unsigned int i = 0; i < kExamples && i < host.examples; ++i) {
      const std::uint64_t source = index_of(host.value[i]);
      std::printf("EXAMPLE round=%d index=%llu value=%016llx source=%lld\n", round, host.index[i],
                  host.value[i], source < words ? static_cast<long long>(source) : -1LL);
    }
    std::fflush(stdout);
    munmap(start, head + kHuge);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "make" && argc == 4) {
    return make(argv[2], std::strtoull(argv[3], nullptr, 10) * kMiB);
  }
  if (mode == "publish" && argc == 7) {
    return publish(argv[2], std::strtoull(argv[3], nullptr, 10) * kMiB,
                   std::strtoull(argv[4], nullptr, 10) * kMiB, argv[5], argv[6]);
  }
  if (mode == "check" && (argc == 6 || argc == 7)) {
    const std::string read = argv[5];
    if (read != "gpu" && read != "cpu") return 2;
    if (!ok(cudaFree(nullptr), "init")) return 1;
    return check(argv[2], std::strtoull(argv[3], nullptr, 10) * kMiB, std::atoi(argv[4]),
                 read == "cpu", argc == 7 ? argv[6] : "");
  }
  std::fprintf(stderr,
               "usage: gpu_read_check make FILE SIZE_MIB\n"
               "       gpu_read_check publish FILE SIZE_MIB HEAD_MIB READY_FILE STOP_FILE\n"
               "       gpu_read_check check FILE HEAD_MIB ROUNDS {gpu|cpu} [GO_FILE]\n");
  return 2;
}
