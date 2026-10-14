// The kernel's own ways to share memory between processes, on memory that the
// GPU of a unified-memory device reads through the host page tables.
//
//   linux_share_probe fork SIZE_MIB {gpu|cpu}
//   linux_share_probe fork_after_init SIZE_MIB
//   linux_share_probe ksm SIZE_MIB COPIES {cuda|nocuda}
//   linux_share_probe write_file FILE {shared|readonly}
//   linux_share_probe register FILE
//   linux_share_probe hold SECONDS
//
// fork             The parent fills SIZE_MIB of anonymous memory on the CPU
//                  and forks before it initializes CUDA, as a zygote that
//                  loads state once and forks its workers. The child
//                  initializes CUDA and reads the memory on the GPU, directly
//                  (gpu) or after a CPU read of every page (cpu). Reports the
//                  private memory of the child before and after the read.
// fork_after_init  The parent initializes CUDA, fills the memory and reads it
//                  on the GPU, then forks. The child tries to use CUDA.
// ksm              COPIES buffers of SIZE_MIB with the same content are marked
//                  mergeable; the caller runs KSM. Reports how long the merge
//                  takes; then, after a line on stdin (the caller stops ksmd),
//                  (cuda) reads one buffer on the GPU, reads another from the
//                  CPU and then on the GPU, and writes one word per page of a
//                  third on the GPU, with the merged pages of the process
//                  after each step. With nocuda the process never initializes
//                  CUDA (merge time only).
// write_file       A sharer with a defect: maps FILE shared and writable
//                  (shared) or private and read-only (readonly), reads it
//                  from the CPU, and writes the first word of every 2 MiB
//                  block that holds data through the GPU. Reports the error
//                  of the write and whether the file changed.
// register         Maps FILE shared and read-only and registers the mapping
//                  with CUDA as read-only host memory, as a process would to
//                  keep the GPU from faulting on it. Reports the time to
//                  register, to read on the GPU, and to unregister, and the
//                  memory that the registration locks.
// hold             Reads a page of host memory on the GPU and keeps its
//                  context for SECONDS, so that the driver keeps the threads
//                  that serve GPU faults (and their counters) between the
//                  processes of a campaign.
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

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
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

__global__ void write_at(char* base, const std::uint64_t* offsets, std::uint64_t count) {
  const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) *reinterpret_cast<std::uint32_t*>(base + offsets[i]) = 0xdeadbeefU;
}

// Field of /proc/self/smaps_rollup or /proc/self/status in KiB, or -1.
long proc_kib(const char* file, const char* field) {
  std::ifstream in(file);
  std::string line;
  const std::size_t length = std::strlen(field);
  while (std::getline(in, line)) {
    if (line.compare(0, length, field) == 0 && line.size() > length && line[length] == ':') {
      return std::strtol(line.c_str() + length + 1, nullptr, 10);
    }
  }
  return -1;
}

long rollup_kib(const char* field) { return proc_kib("/proc/self/smaps_rollup", field); }

// Pages of this process that KSM has merged, as ksmd last counted them, or -1.
long merged_pages() {
  std::ifstream in("/proc/self/ksm_stat");
  std::string line;
  while (std::getline(in, line)) {
    const char* key = "ksm_merging_pages ";
    if (line.compare(0, std::strlen(key), key) == 0) {
      return std::strtol(line.c_str() + std::strlen(key), nullptr, 10);
    }
  }
  return -1;
}

std::uint64_t* map_filled(std::uint64_t bytes, std::uint64_t seed) {
  void* mapped = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapped == MAP_FAILED) return nullptr;
  auto* words = static_cast<std::uint64_t*>(mapped);
  std::uint64_t state = seed;
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

void cpu_touch(const void* data, std::uint64_t bytes) {
  const volatile char* view = static_cast<const volatile char*>(data);
  for (std::uint64_t offset = 0; offset < bytes; offset += 4096) (void)view[offset];
}

// Sums the words on the GPU; returns the seconds, or a negative value on error.
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

double gpu_touch(std::uint64_t* words, std::uint64_t bytes) {
  const std::uint64_t pages = bytes / 4096;
  const auto start = Clock::now();
  touch_pages<<<static_cast<unsigned>((pages + 255) / 256), 256>>>(words, pages);
  if (cudaDeviceSynchronize() != cudaSuccess) return -2;
  return seconds_since(start);
}

int fork_zygote(std::uint64_t bytes, bool cpu_first) {
  std::uint64_t* words = map_filled(bytes, 0x9e3779b97f4a7c15ULL);
  if (words == nullptr) return 3;
  const std::uint64_t expected = cpu_sum(words, bytes / 8);
  std::fflush(stdout);
  const pid_t child = fork();
  if (child < 0) return 3;
  if (child == 0) {
    alarm(120);
    if (cudaFree(nullptr) != cudaSuccess) {
      std::printf("FORK child_cuda=%s\n", cudaGetErrorString(cudaGetLastError()));
      _exit(4);
    }
    const long before = rollup_kib("Private_Dirty");
    double touch = 0;
    if (cpu_first) {
      const auto start = Clock::now();
      cpu_touch(words, bytes);
      touch = seconds_since(start);
    }
    bool right = false;
    const double took = gpu_sum(words, bytes / 8, expected, &right);
    const long after = rollup_kib("Private_Dirty");
    std::printf("FORK mode=%s size_mib=%llu cpu_read_s=%.4f gpu_read_s=%.4f right=%d "
                "child_private_before_mib=%.1f child_private_after_mib=%.1f\n",
                cpu_first ? "cpu" : "gpu", static_cast<unsigned long long>(bytes >> 20), touch,
                took, right ? 1 : 0, before / 1024.0, after / 1024.0);
    std::fflush(stdout);
    _exit(took < 0 ? 5 : 0);
  }
  int status = 0;
  waitpid(child, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : 6;
}

int fork_after_init(std::uint64_t bytes) {
  if (cudaFree(nullptr) != cudaSuccess) return 3;
  std::uint64_t* words = map_filled(bytes, 0x9e3779b97f4a7c15ULL);
  if (words == nullptr) return 3;
  const std::uint64_t expected = cpu_sum(words, bytes / 8);
  bool right = false;
  const double took = gpu_sum(words, bytes / 8, expected, &right);
  std::printf("FORK_AFTER_INIT parent_gpu_read_s=%.4f right=%d\n", took, right ? 1 : 0);
  std::fflush(stdout);
  const pid_t child = fork();
  if (child < 0) return 3;
  if (child == 0) {
    alarm(60);
    void* probe = nullptr;
    const cudaError_t allocated = cudaMalloc(&probe, 4096);
    bool child_right = false;
    const double child_took =
        allocated == cudaSuccess ? gpu_sum(words, bytes / 8, expected, &child_right) : -1;
    std::printf("FORK_AFTER_INIT child_malloc=%s child_gpu_read_s=%.4f right=%d last=%s\n",
                cudaGetErrorString(allocated), child_took, child_right ? 1 : 0,
                cudaGetErrorString(cudaGetLastError()));
    std::fflush(stdout);
    _exit(0);
  }
  int status = 0;
  waitpid(child, &status, 0);
  if (WIFSIGNALED(status)) std::printf("FORK_AFTER_INIT child_signal=%d\n", WTERMSIG(status));
  return 0;
}

int ksm(std::uint64_t bytes, int copies, bool cuda) {
  if (cuda && cudaFree(nullptr) != cudaSuccess) return 3;
  std::vector<std::uint64_t*> buffers;
  for (int c = 0; c < copies; ++c) {
    std::uint64_t* words = map_filled(bytes, 0x9e3779b97f4a7c15ULL);
    if (words == nullptr || madvise(words, bytes, MADV_MERGEABLE) != 0) return 3;
    buffers.push_back(words);
  }
  const std::uint64_t expected = cpu_sum(buffers[0], bytes / 8);
  const long want = static_cast<long>(copies) * static_cast<long>(bytes / 4096);
  const auto start = Clock::now();
  long merged = 0;
  while (seconds_since(start) < 900) {
    merged = merged_pages();
    if (merged >= want * 99 / 100) break;
    usleep(100000);
  }
  const double merge_s = seconds_since(start);
  std::printf("KSM cuda=%d size_mib=%llu copies=%d merged_pages=%ld of=%ld merge_s=%.2f "
              "pages_per_s=%.0f\n",
              cuda ? 1 : 0, static_cast<unsigned long long>(bytes >> 20), copies, merged, want,
              merge_s, merged / merge_s);
  std::fflush(stdout);
  if (!cuda || copies < 3) return 0;
  // The caller stops ksmd before the GPU steps, so that no merge happens
  // between them.
  std::printf("KSM_READY\n");
  std::fflush(stdout);
  std::string line;
  std::getline(std::cin, line);

  // The KSM line of smaps counts the pages mapped to a merged page now;
  // ksm_stat is only updated when ksmd scans.
  const auto ksm_now = [] { return rollup_kib("KSM") / 4; };
  const long m0 = ksm_now();
  bool right = false;
  const double read_s = gpu_sum(buffers[0], bytes / 8, expected, &right);
  const long m1 = ksm_now();
  const auto touch_start = Clock::now();
  cpu_touch(buffers[1], bytes);
  const double cpu_s = seconds_since(touch_start);
  bool right2 = false;
  const double read2_s = gpu_sum(buffers[1], bytes / 8, expected, &right2);
  const long m2 = ksm_now();
  const double write_s = gpu_touch(buffers[2], bytes);
  const long m3 = ksm_now();
  std::printf("KSM_GPU gpu_read_s=%.4f right=%d merged_before=%ld merged_after_gpu_read=%ld "
              "cpu_read_s=%.4f gpu_read_after_cpu_s=%.4f right=%d merged_after=%ld "
              "gpu_write_s=%.4f merged_after_gpu_write=%ld\n",
              read_s, right ? 1 : 0, m0, m1, cpu_s, read2_s, right2 ? 1 : 0, m2, write_s, m3);
  return 0;
}

int write_file(const char* path, bool shared) {
  const int fd = open(path, shared ? O_RDWR : O_RDONLY);
  struct stat info {};
  if (fd < 0 || fstat(fd, &info) != 0 || info.st_size <= 0) return 3;
  const auto bytes = static_cast<std::uint64_t>(info.st_size);
  // The first word of every 2 MiB block that holds data: every tensor of the
  // cache starts at such a block, so its first row changes.
  std::vector<std::uint64_t> offsets;
  for (std::uint64_t offset = 0; offset + 4 <= bytes; offset += 2u << 20) {
    const off_t data = lseek(fd, static_cast<off_t>(offset), SEEK_DATA);
    if (data == static_cast<off_t>(offset)) offsets.push_back(offset);
  }
  std::vector<std::uint32_t> before(offsets.size());
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    if (pread(fd, &before[i], 4, static_cast<off_t>(offsets[i])) != 4) return 3;
  }
  void* mapped = mmap(nullptr, bytes, shared ? PROT_READ | PROT_WRITE : PROT_READ,
                      shared ? MAP_SHARED : MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) return 3;
  for (const std::uint64_t offset : offsets) cpu_touch(static_cast<char*>(mapped) + offset, 4096);

  std::uint64_t* list = nullptr;
  if (cudaMalloc(&list, offsets.size() * sizeof(std::uint64_t)) != cudaSuccess ||
      cudaMemcpy(list, offsets.data(), offsets.size() * sizeof(std::uint64_t),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    return 3;
  }
  const auto count = static_cast<std::uint64_t>(offsets.size());
  write_at<<<static_cast<unsigned>((count + 255) / 256), 256>>>(static_cast<char*>(mapped), list,
                                                                count);
  const cudaError_t launched = cudaGetLastError();
  const cudaError_t done = cudaDeviceSynchronize();
  std::size_t changed = 0;
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    std::uint32_t now = 0;
    if (pread(fd, &now, 4, static_cast<off_t>(offsets[i])) != 4) return 3;
    if (now != before[i]) ++changed;
  }
  std::printf("WRITER mapping=%s blocks=%llu gpu_write=%s blocks_changed=%zu\n",
              shared ? "shared" : "readonly", static_cast<unsigned long long>(count),
              cudaGetErrorName(launched != cudaSuccess ? launched : done), changed);
  return 0;
}

int register_file(const char* path) {
  const int fd = open(path, O_RDONLY);
  struct stat info {};
  if (fd < 0 || fstat(fd, &info) != 0 || info.st_size <= 0) return 3;
  const std::uint64_t bytes = static_cast<std::uint64_t>(info.st_size) / 8 * 8;
  void* mapped = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
  if (mapped == MAP_FAILED) return 3;
  if (cudaFree(nullptr) != cudaSuccess) return 3;
  const std::uint64_t expected =
      cpu_sum(static_cast<const std::uint64_t*>(mapped), bytes / 8);
  const long locked_before = proc_kib("/proc/self/status", "VmLck");
  const long pinned_before = proc_kib("/proc/self/status", "VmPin");
  auto start = Clock::now();
  const cudaError_t registered =
      cudaHostRegister(mapped, bytes, cudaHostRegisterReadOnly | cudaHostRegisterMapped);
  const double register_s = seconds_since(start);
  const long locked = proc_kib("/proc/self/status", "VmLck");
  const long pinned = proc_kib("/proc/self/status", "VmPin");
  bool right = false;
  const double read_s =
      gpu_sum(static_cast<const std::uint64_t*>(mapped), bytes / 8, expected, &right);
  start = Clock::now();
  const cudaError_t unregistered =
      registered == cudaSuccess ? cudaHostUnregister(mapped) : cudaErrorInvalidValue;
  const double unregister_s = seconds_since(start);
  std::printf("REGISTER size_mib=%.0f register=%s register_s=%.4f gpu_read_s=%.4f right=%d "
              "unregister=%s unregister_s=%.4f locked_mib=%.1f pinned_mib=%.1f\n",
              bytes / 1048576.0, cudaGetErrorName(registered), register_s, read_s, right ? 1 : 0,
              cudaGetErrorName(unregistered), unregister_s, (locked - locked_before) / 1024.0,
              (pinned - pinned_before) / 1024.0);
  return 0;
}

int hold(unsigned seconds) {
  std::uint64_t* words = map_filled(1u << 20, 1);
  bool right = false;
  if (words == nullptr || gpu_sum(words, (1u << 20) / 8, cpu_sum(words, (1u << 20) / 8), &right) < 0) {
    return 3;
  }
  std::printf("HOLD right=%d\n", right ? 1 : 0);
  std::fflush(stdout);
  sleep(seconds);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s fork|fork_after_init|ksm|write_file|register|hold ...\n",
                 argv[0]);
    return 2;
  }
  const std::string mode = argv[1];
  if (mode == "write_file" && argc == 4) {
    return write_file(argv[2], std::string(argv[3]) == "shared");
  }
  if (mode == "register" && argc == 3) return register_file(argv[2]);
  if (mode == "hold" && argc == 3) return hold(static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10)));
  const std::uint64_t bytes = std::strtoull(argv[2], nullptr, 10) << 20;
  if (mode == "fork" && argc == 4) return fork_zygote(bytes, std::string(argv[3]) == "cpu");
  if (mode == "fork_after_init") return fork_after_init(bytes);
  if (mode == "ksm" && argc == 5) {
    return ksm(bytes, std::atoi(argv[3]), std::string(argv[4]) == "cuda");
  }
  std::fprintf(stderr, "bad arguments\n");
  return 2;
}
