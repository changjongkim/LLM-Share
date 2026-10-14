// Where a wrong GPU read of a read-only file mapping comes from. Campaign
// 20261014-gpu-read-check-full-v1 (gpu_read_check.cu) found, with 32 processes
// that fault on a file of a tmpfs with 2 MiB pages, two reader rounds in which
// the GPU read words of a 4 KiB page that belong to the first 4 KiB page of the
// same 2 MiB page, at the same offset. This probe repeats that check and, after
// a round with a differing word, reads the same words again from the CPU through
// the same mapping and from the GPU without a new mapping, so that a wrong entry
// of the page table (both read wrong) and a wrong translation on the GPU side
// (the CPU reads right) can be told apart.
//
//   gpu_read_trace make FILE SIZE_MIB
//   gpu_read_trace check FILE HEAD_MIB ROUNDS {gpu|cpu} {huge|nohuge} [GO_FILE]
//
// make    writes FILE from the CPU, as gpu_read_check make.
// check   ROUNDS times: maps the first HEAD_MIB of FILE private and read-only at
//         a 2 MiB-aligned address, as an agent maps the prefix; with nohuge the
//         mapping is advised against huge pages before any access, so that the
//         kernel maps the file in 4 KiB entries. Reads every page from the CPU
//         first (cpu) or not (gpu), and compares every word on the GPU.
//
// Output, one line per round, and per differing round up to four examples:
//   CHECK round= words= differ= zero= pages= gpu_ms= pmd_kib= cpu_differ= again_differ=
//   EXAMPLE round= index= value= source= page_in_block= source_page_in_block= cpu_value_ok=
// pmd_kib is FilePmdMapped of the process after the GPU pass (the file memory
// that the process maps with 2 MiB entries). cpu_differ counts the words that
// differ when the CPU reads the whole mapping after the GPU pass, and
// again_differ those that differ in a second GPU pass over the same mapping;
// both are -1 in a round without a differing word.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
constexpr std::uint64_t kPagesPerBlock = kHuge / kPage;
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

// FilePmdMapped of this process in KiB, or -1.
long pmd_mapped_kib() {
  std::ifstream in("/proc/self/smaps_rollup");
  std::string line;
  const char* key = "FilePmdMapped:";
  while (std::getline(in, line)) {
    if (line.compare(0, std::strlen(key), key) == 0) {
      return std::strtol(line.c_str() + std::strlen(key), nullptr, 10);
    }
  }
  return -1;
}

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

// One GPU pass over the mapping; returns false on a CUDA error.
bool gpu_pass(const char* base, std::uint64_t words, Report* report, unsigned int* page_flags,
              std::uint64_t pages, Report* host, std::vector<unsigned int>* flags, double* ms) {
  if (!ok(cudaMemset(report, 0, sizeof(Report)), "memset") ||
      !ok(cudaMemset(page_flags, 0, pages * sizeof(unsigned int)), "memset")) {
    return false;
  }
  const auto started = Clock::now();
  check_words<<<blocks(), 256>>>(reinterpret_cast<const std::uint64_t*>(base), words, report,
                                 page_flags);
  if (!ok(cudaDeviceSynchronize(), "check")) return false;
  *ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  return ok(cudaMemcpy(host, report, sizeof(Report), cudaMemcpyDeviceToHost), "copy") &&
         ok(cudaMemcpy(flags->data(), page_flags, pages * sizeof(unsigned int),
                       cudaMemcpyDeviceToHost), "copy");
}

int check(const std::string& path, std::uint64_t head, int rounds, bool cpu_first, bool nohuge,
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
    if (nohuge && madvise(base, head, MADV_NOHUGEPAGE) != 0) {
      std::perror("madvise");
      return 1;
    }
    if (cpu_first) cpu_touch(base, head);
    Report host{};
    double gpu_ms = 0;
    if (!gpu_pass(base, words, report, page_flags, pages, &host, &flags, &gpu_ms)) return 1;
    const long pmd_kib = pmd_mapped_kib();
    std::uint64_t differing_pages = 0;
    for (unsigned int flag : flags) differing_pages += flag;
    long long cpu_differ = -1;
    long long again_differ = -1;
    const Report first = host;
    if (first.differ > 0) {
      const auto* data = reinterpret_cast<const volatile std::uint64_t*>(base);
      cpu_differ = 0;
      for (std::uint64_t i = 0; i < words; ++i) cpu_differ += data[i] != expected(i);
      Report again{};
      double again_ms = 0;
      if (!gpu_pass(base, words, report, page_flags, pages, &again, &flags, &again_ms)) return 1;
      again_differ = static_cast<long long>(again.differ);
    }
    std::printf("CHECK round=%d words=%llu differ=%llu zero=%llu pages=%llu gpu_ms=%.1f pmd_kib=%ld "
                "cpu_differ=%lld again_differ=%lld\n",
                round, static_cast<unsigned long long>(words), first.differ, first.zero,
                static_cast<unsigned long long>(differing_pages), gpu_ms, pmd_kib, cpu_differ,
                again_differ);
    for (unsigned int i = 0; i < kExamples && i < first.examples; ++i) {
      const std::uint64_t index = first.index[i];
      const std::uint64_t source = index_of(first.value[i]);
      const auto* data = reinterpret_cast<const volatile std::uint64_t*>(base);
      const bool cpu_value_ok = data[index] == expected(index);
      std::printf("EXAMPLE round=%d index=%llu value=%016llx source=%lld page_in_block=%llu "
                  "source_page_in_block=%lld cpu_value_ok=%d\n",
                  round, static_cast<unsigned long long>(index), first.value[i],
                  source < words ? static_cast<long long>(source) : -1LL,
                  static_cast<unsigned long long>(index / kWordsPerPage % kPagesPerBlock),
                  source < words ? static_cast<long long>(source / kWordsPerPage % kPagesPerBlock)
                                 : -1LL,
                  cpu_value_ok ? 1 : 0);
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
  if (mode == "check" && (argc == 7 || argc == 8)) {
    const std::string read = argv[5];
    const std::string mapping = argv[6];
    if ((read != "gpu" && read != "cpu") || (mapping != "huge" && mapping != "nohuge")) return 2;
    if (!ok(cudaFree(nullptr), "init")) return 1;
    return check(argv[2], std::strtoull(argv[3], nullptr, 10) * kMiB, std::atoi(argv[4]),
                 read == "cpu", mapping == "nohuge", argc == 8 ? argv[7] : "");
  }
  std::fprintf(stderr,
               "usage: gpu_read_trace make FILE SIZE_MIB\n"
               "       gpu_read_trace check FILE HEAD_MIB ROUNDS {gpu|cpu} {huge|nohuge} [GO_FILE]\n");
  return 2;
}
