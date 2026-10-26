// The sharing mechanisms of the kernel when the CPU or the GPU accesses the
// shared pages. Every case is one process that initializes CUDA, so that its
// address space is bound to the SMMU in both cases, and only the accessor of
// the shared pages differs: the CPU (cpu) or a GPU kernel (gpu).
//
//   mech_probe private FILE {cpu|gpu} [read]
//   mech_probe shared  FILE {cpu|gpu}
//   mech_probe fork    SIZE_MIB {cpu|gpu} [read]
//   mech_probe ksm     SIZE_MIB COPIES {cpu|gpu}
//   mech_probe reclaim HOT COLD {cpu|gpu} SECONDS {none|lock}
//
// private  Maps FILE private and writable, reads every word, then writes one
//          word per 4 KiB page. Reports the anonymous memory that each step
//          adds (pages copied from the file).
// shared   Maps FILE shared and writable and reads every word.
// fork     Fills SIZE_MIB of anonymous memory and forks before CUDA is
//          initialized, as a zygote. The child initializes CUDA, reads every
//          word of the inherited memory, then writes one word per page.
//          Reports the private memory that each step adds to the child.
// ksm      COPIES buffers with the same content, SIZE_MIB in total, marked
//          mergeable. Prints READY; the caller runs ksmd until the probe
//          prints MERGED and, after the caller stops ksmd and sends a line,
//          reads every word of every buffer, then writes one word per page,
//          with the pages mapped to a merged page after each step. Prints
//          DONE and waits for a line, so that the caller can unmerge while
//          the process still maps the pages.
// reclaim  Maps HOT and COLD shared and read-only, reads every page of COLD
//          once from the CPU and never again, optionally locks HOT (lock),
//          reads HOT once, prints READY, and reads every word of HOT
//          repeatedly for SECONDS while the caller may ask the kernel to
//          reclaim from the control group of the process. Reports the passes,
//          the mean time of a pass, the time of the lock, and the share of
//          each file that is still in memory at the end (mincore), which shows
//          the pages that reclaim chose.
//
// With "read", private and fork stop after the read.
//
// Each step prints the page faults of the process itself (getrusage) and the
// pages that the threads of the driver fault in for the GPU ("UVM GPU<n> BH",
// minor and major faults in /proc/<pid>/stat) together with their CPU time.
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <dirent.h>
#include <sys/wait.h>
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

long rollup_kib(const char* field) {
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

void wait_line() {
  std::string line;
  std::getline(std::cin, line);
}

struct Faults {
  long minor = 0;
  long major = 0;
  long gpu_pages = 0;  // faulted in by the threads of the driver
  long gpu_ticks = 0;  // their CPU time in clock ticks
};

// The minor and major faults and the CPU time of the threads of the driver
// that serve GPU faults.
void driver_faults(long* pages, long* ticks) {
  *pages = 0;
  *ticks = 0;
  DIR* proc = opendir("/proc");
  if (proc == nullptr) return;
  while (dirent* entry = readdir(proc)) {
    if (entry->d_name[0] < '0' || entry->d_name[0] > '9') continue;
    std::ifstream in(std::string("/proc/") + entry->d_name + "/stat");
    std::string line;
    if (!std::getline(in, line)) continue;
    const std::size_t open = line.find('('), close = line.rfind(')');
    if (open == std::string::npos || close == std::string::npos) continue;
    const std::string name = line.substr(open + 1, close - open - 1);
    if (name.compare(0, 7, "UVM GPU") != 0 || name.size() < 3 ||
        name.compare(name.size() - 3, 3, " BH") != 0) {
      continue;
    }
    // Fields after the name: state(3) ... minflt(10) cminflt majflt(12) cmajflt utime(14) stime.
    std::vector<long> field;
    std::size_t at = close + 2;
    while (at < line.size() && field.size() < 13) {
      const std::size_t end = line.find(' ', at);
      field.push_back(std::strtol(line.c_str() + at, nullptr, 10));
      if (end == std::string::npos) break;
      at = end + 1;
    }
    if (field.size() < 13) continue;
    *pages += field[7] + field[9];
    *ticks += field[11] + field[12];
  }
  closedir(proc);
}

Faults own_faults() {
  rusage usage {};
  getrusage(RUSAGE_SELF, &usage);
  Faults faults {usage.ru_minflt, usage.ru_majflt, 0, 0};
  driver_faults(&faults.gpu_pages, &faults.gpu_ticks);
  return faults;
}

void fill(std::uint64_t* words, std::uint64_t count) {
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  for (std::uint64_t i = 0; i < count; ++i) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    words[i] = state;
  }
}

std::uint64_t* map_filled(std::uint64_t bytes) {
  void* mapped = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapped == MAP_FAILED) return nullptr;
  auto* words = static_cast<std::uint64_t*>(mapped);
  fill(words, bytes / 8);
  return words;
}

std::uint64_t cpu_sum(const std::uint64_t* words, std::uint64_t count) {
  std::uint64_t sum = 0;
  for (std::uint64_t i = 0; i < count; ++i) sum += words[i];
  return sum;
}

// The sum of the words of FILE, read through the file interface and not a
// mapping, so that the expected value does not map any page.
bool file_sum(int fd, std::uint64_t bytes, std::uint64_t* sum) {
  std::vector<std::uint64_t> chunk(1u << 17);
  std::uint64_t total = 0;
  for (std::uint64_t offset = 0; offset < bytes;) {
    const std::size_t want = std::min<std::uint64_t>(chunk.size() * 8, bytes - offset);
    const ssize_t got = pread(fd, chunk.data(), want, static_cast<off_t>(offset));
    if (got <= 0 || got % 8 != 0) return false;
    total += cpu_sum(chunk.data(), static_cast<std::uint64_t>(got) / 8);
    offset += static_cast<std::uint64_t>(got);
  }
  *sum = total;
  return true;
}

// Reads every word with the accessor; returns the seconds or a negative value.
double read_all(const std::uint64_t* words, std::uint64_t bytes, bool gpu,
                std::uint64_t expected, bool* right) {
  const std::uint64_t count = bytes / 8;
  if (!gpu) {
    const auto start = Clock::now();
    const std::uint64_t sum = cpu_sum(words, count);
    const double took = seconds_since(start);
    *right = sum == expected;
    return took;
  }
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

// Writes one word per 4 KiB page with the accessor.
double write_pages(std::uint64_t* words, std::uint64_t bytes, bool gpu) {
  const std::uint64_t pages = bytes / 4096;
  const auto start = Clock::now();
  if (!gpu) {
    volatile std::uint64_t* view = words;
    for (std::uint64_t page = 0; page < pages; ++page) view[page * 512] += 1;
    return seconds_since(start);
  }
  touch_pages<<<static_cast<unsigned>((pages + 255) / 256), 256>>>(words, pages);
  if (cudaDeviceSynchronize() != cudaSuccess) return -2;
  return seconds_since(start);
}

void report(const char* mech, const char* step, bool gpu, double seconds, int right,
            long anon_before, long anon_after, long dirty_before, long dirty_after,
            const Faults& before, const Faults& after) {
  const long tick_ms = 1000 / sysconf(_SC_CLK_TCK);
  std::printf("STEP mech=%s step=%s access=%s seconds=%.4f right=%d anon_mib=%.1f dirty_mib=%.1f "
              "minor=%ld major=%ld gpu_fault_pages=%ld gpu_fault_ms=%ld\n",
              mech, step, gpu ? "gpu" : "cpu", seconds, right, (anon_after - anon_before) / 1024.0,
              (dirty_after - dirty_before) / 1024.0, after.minor - before.minor,
              after.major - before.major, after.gpu_pages - before.gpu_pages,
              (after.gpu_ticks - before.gpu_ticks) * tick_ms);
  std::fflush(stdout);
}

int open_file(const char* path, std::uint64_t* bytes) {
  const int fd = open(path, O_RDONLY);
  struct stat info {};
  if (fd < 0 || fstat(fd, &info) != 0 || info.st_size < 8) return -1;
  *bytes = static_cast<std::uint64_t>(info.st_size) / 4096 * 4096;
  return fd;
}

int file_mapping(const char* path, bool gpu, bool shared, bool read_only) {
  std::uint64_t bytes = 0;
  const int fd = open_file(path, &bytes);
  std::uint64_t expected = 0;
  if (fd < 0 || !file_sum(fd, bytes, &expected)) return 3;
  if (cudaFree(nullptr) != cudaSuccess) return 3;
  void* mapped = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, shared ? MAP_SHARED : MAP_PRIVATE,
                      fd, 0);
  if (mapped == MAP_FAILED) return 3;
  auto* words = static_cast<std::uint64_t*>(mapped);
  const char* mech = shared ? "shared" : "private";

  long anon0 = rollup_kib("Anonymous"), dirty0 = rollup_kib("Private_Dirty");
  Faults f0 = own_faults();
  bool right = false;
  const double read_s = read_all(words, bytes, gpu, expected, &right);
  long anon1 = rollup_kib("Anonymous"), dirty1 = rollup_kib("Private_Dirty");
  Faults f1 = own_faults();
  report(mech, "read", gpu, read_s, right ? 1 : 0, anon0, anon1, dirty0, dirty1, f0, f1);
  if (read_s < 0) return 5;
  if (shared || read_only) return 0;  // a shared write would change the file for every sharer

  const double write_s = write_pages(words, bytes, gpu);
  long anon2 = rollup_kib("Anonymous"), dirty2 = rollup_kib("Private_Dirty");
  Faults f2 = own_faults();
  report(mech, "write", gpu, write_s, write_s < 0 ? 0 : 1, anon1, anon2, dirty1, dirty2, f1, f2);
  return write_s < 0 ? 5 : 0;
}

int fork_zygote(std::uint64_t bytes, bool gpu, bool read_only) {
  std::uint64_t* words = map_filled(bytes);
  if (words == nullptr) return 3;
  const std::uint64_t expected = cpu_sum(words, bytes / 8);
  std::fflush(stdout);
  const pid_t child = fork();
  if (child < 0) return 3;
  if (child == 0) {
    alarm(300);
    if (cudaFree(nullptr) != cudaSuccess) _exit(4);
    long anon0 = rollup_kib("Anonymous"), dirty0 = rollup_kib("Private_Dirty");
    Faults f0 = own_faults();
    bool right = false;
    const double read_s = read_all(words, bytes, gpu, expected, &right);
    long anon1 = rollup_kib("Anonymous"), dirty1 = rollup_kib("Private_Dirty");
    Faults f1 = own_faults();
    report("fork", "read", gpu, read_s, right ? 1 : 0, anon0, anon1, dirty0, dirty1, f0, f1);
    if (read_s < 0) _exit(5);
    if (read_only) _exit(0);
    const double write_s = write_pages(words, bytes, gpu);
    long anon2 = rollup_kib("Anonymous"), dirty2 = rollup_kib("Private_Dirty");
    Faults f2 = own_faults();
    report("fork", "write", gpu, write_s, write_s < 0 ? 0 : 1, anon1, anon2, dirty1, dirty2, f1,
           f2);
    _exit(write_s < 0 ? 5 : 0);
  }
  int status = 0;
  waitpid(child, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : 6;
}

int ksm(std::uint64_t bytes, int copies, bool gpu) {
  if (copies < 2 || bytes % (static_cast<std::uint64_t>(copies) << 12) != 0) return 2;
  if (cudaFree(nullptr) != cudaSuccess) return 3;
  const std::uint64_t each = bytes / static_cast<std::uint64_t>(copies);
  std::vector<std::uint64_t*> buffers;
  for (int c = 0; c < copies; ++c) {
    std::uint64_t* words = map_filled(each);
    if (words == nullptr || madvise(words, each, MADV_MERGEABLE) != 0) return 3;
    buffers.push_back(words);
  }
  const std::uint64_t expected = cpu_sum(buffers[0], each / 8);
  const long want = static_cast<long>(bytes / 4096);
  std::printf("READY pages=%ld\n", want);
  std::fflush(stdout);
  const auto start = Clock::now();
  long merged = 0;
  while (seconds_since(start) < 900) {
    merged = ksmd_merged_pages();
    if (merged >= want * 99 / 100) break;
    usleep(100000);
  }
  std::printf("MERGE merged_pages=%ld of=%ld merge_s=%.2f\n", merged, want, seconds_since(start));
  std::printf("MERGED\n");
  std::fflush(stdout);
  wait_line();

  // The KSM line of smaps counts the pages mapped to a merged page now.
  const auto mapped_merged = [] { return rollup_kib("KSM") / 4; };
  const long m0 = mapped_merged();
  Faults f0 = own_faults();
  double read_s = 0;
  bool all_right = true;
  for (std::uint64_t* words : buffers) {
    bool right = false;
    const double took = read_all(words, each, gpu, expected, &right);
    if (took < 0) return 5;
    read_s += took;
    all_right = all_right && right;
  }
  const long m1 = mapped_merged();
  Faults f1 = own_faults();
  double write_s = 0;
  for (std::uint64_t* words : buffers) {
    const double took = write_pages(words, each, gpu);
    if (took < 0) return 5;
    write_s += took;
  }
  const long m2 = mapped_merged();
  Faults f2 = own_faults();
  std::printf("KSMSTEP access=%s merged_before=%ld read_s=%.4f right=%d merged_after_read=%ld "
              "read_minor=%ld read_gpu_fault_pages=%ld write_s=%.4f merged_after_write=%ld "
              "write_minor=%ld write_gpu_fault_pages=%ld\n",
              gpu ? "gpu" : "cpu", m0, read_s, all_right ? 1 : 0, m1, f1.minor - f0.minor,
              f1.gpu_pages - f0.gpu_pages, write_s, m2, f2.minor - f1.minor,
              f2.gpu_pages - f1.gpu_pages);
  std::printf("DONE\n");
  std::fflush(stdout);
  wait_line();
  return 0;
}

// The share of the pages of a mapping that are in memory.
double resident(const void* mapped, std::uint64_t bytes) {
  std::vector<unsigned char> pages(bytes / 4096);
  if (mincore(const_cast<void*>(mapped), bytes, pages.data()) != 0) return -1;
  std::uint64_t in = 0;
  for (unsigned char page : pages) in += page & 1;
  return static_cast<double>(in) / static_cast<double>(pages.size());
}

int reclaim(const char* hot_path, const char* cold_path, bool gpu, double duration, bool lock) {
  std::uint64_t bytes = 0, cold_bytes = 0;
  const int fd = open_file(hot_path, &bytes);
  const int cold_fd = open_file(cold_path, &cold_bytes);
  std::uint64_t expected = 0, cold_sum = 0;
  if (fd < 0 || cold_fd < 0 || !file_sum(fd, bytes, &expected) ||
      !file_sum(cold_fd, cold_bytes, &cold_sum)) {
    return 3;
  }
  if (cudaFree(nullptr) != cudaSuccess) return 3;
  void* mapped = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
  void* cold = mmap(nullptr, cold_bytes, PROT_READ, MAP_SHARED, cold_fd, 0);
  if (mapped == MAP_FAILED || cold == MAP_FAILED) return 3;
  const auto* words = static_cast<const std::uint64_t*>(mapped);
  const volatile char* cold_view = static_cast<const volatile char*>(cold);
  for (std::uint64_t offset = 0; offset < cold_bytes; offset += 4096) (void)cold_view[offset];
  double lock_s = 0;
  if (lock) {
    const auto start = Clock::now();
    if (mlock(mapped, bytes) != 0) {
      std::printf("RECLAIM_ERROR mlock=%s\n", std::strerror(errno));
      return 4;
    }
    lock_s = seconds_since(start);
  }
  bool right = false;
  if (read_all(words, bytes, gpu, expected, &right) < 0 || !right) return 5;
  std::printf("READY hot_resident=%.4f cold_resident=%.4f\n", resident(mapped, bytes),
              resident(cold, cold_bytes));
  std::fflush(stdout);
  Faults f0 = own_faults();
  const auto start = Clock::now();
  long passes = 0;
  bool all_right = true;
  while (seconds_since(start) < duration) {
    if (read_all(words, bytes, gpu, expected, &right) < 0) return 5;
    all_right = all_right && right;
    ++passes;
  }
  const double took = seconds_since(start);
  Faults f1 = own_faults();
  std::printf("RECLAIM access=%s pin=%s passes=%ld seconds=%.3f pass_ms=%.2f right=%d minor=%ld "
              "major=%ld gpu_fault_pages=%ld gpu_fault_ms=%ld lock_s=%.4f hot_resident=%.4f "
              "cold_resident=%.4f\n",
              gpu ? "gpu" : "cpu", lock ? "lock" : "none", passes, took,
              passes ? took * 1000.0 / passes : 0.0, all_right ? 1 : 0, f1.minor - f0.minor,
              f1.major - f0.major, f1.gpu_pages - f0.gpu_pages,
              (f1.gpu_ticks - f0.gpu_ticks) * (1000 / sysconf(_SC_CLK_TCK)), lock_s,
              resident(mapped, bytes), resident(cold, cold_bytes));
  std::fflush(stdout);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s private|shared|fork|ksm|reclaim ...\n", argv[0]);
    return 2;
  }
  const std::string mech = argv[1];
  const auto is_gpu = [](const char* word) { return std::string(word) == "gpu"; };
  const bool read_only = argc == 5 && std::string(argv[4]) == "read";
  if ((mech == "private" || mech == "shared") && (argc == 4 || read_only)) {
    return file_mapping(argv[2], is_gpu(argv[3]), mech == "shared", read_only);
  }
  if (mech == "fork" && (argc == 4 || read_only)) {
    return fork_zygote(std::strtoull(argv[2], nullptr, 10) << 20, is_gpu(argv[3]), read_only);
  }
  if (mech == "ksm" && argc == 5) {
    return ksm(std::strtoull(argv[2], nullptr, 10) << 20, std::atoi(argv[3]), is_gpu(argv[4]));
  }
  if (mech == "reclaim" && argc == 7) {
    return reclaim(argv[2], argv[3], is_gpu(argv[4]), std::atof(argv[5]),
                   std::string(argv[6]) == "lock");
  }
  std::fprintf(stderr, "bad arguments\n");
  return 2;
}
