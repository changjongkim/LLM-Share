// E1 (review of 2026-10-26): the write attempts of a faulty process of the
// same user against published rows, at the same level for every mechanism.
//
//   default    the cache file of a publisher on tmpfs, published by a change
//              of protection to read-only (the default mode of STATOR)
//   immutable  a memfd that the children inherit, mapped again read-only and
//              sealed against writes, shrinking, growing and further seals
//   device     device memory shared through CUDA VMM (Device)
//
// Attempts of the writer, each in a process of its own:
//   map       GPU write through the mapping that an agent holds
//   raise     raise the protection of that mapping to read-write, then GPU write
//   remap     map the descriptor of the agent again, shared and writable
//   reopen    open the file for writing, map it shared and writable, GPU write
//   truncate  open the file for writing and shrink it to zero bytes
//
//   protect_e1 run MODE ATTEMPT REP   publishes, starts two readers and one writer, prints one CASE line
//   protect_e1 reader|writer ...      started by run
#include <cuda.h>
#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace {

constexpr std::uint64_t kSize = 64ULL << 20U;
constexpr std::uint64_t kWrite = 2ULL << 20U;

__host__ __device__ std::uint32_t pattern(std::uint64_t i) {
  return static_cast<std::uint32_t>(i * 2654435761ULL) ^ 0x5A5A5A5AU;
}

__global__ void count_wrong(const std::uint32_t* data, std::uint64_t words,
                            unsigned long long* wrong) {
  unsigned long long local = 0;
  for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
       i < words; i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
    if (data[i] != pattern(i)) ++local;
  }
  if (local != 0) atomicAdd(wrong, local);
}

__global__ void zero_words(std::uint32_t* data, std::uint64_t words) {
  const std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
  if (i < words) data[i] = 0U;
}

bool ok(CUresult status) { return status == CUDA_SUCCESS; }

std::string cu_name(CUresult status) {
  const char* text = nullptr;
  cuGetErrorName(status, &text);
  return text != nullptr ? text : "?";
}

std::string errno_name(int value) {
  switch (value) {
    case 0: return "ok";
    case EACCES: return "EACCES";
    case EPERM: return "EPERM";
    case EBUSY: return "EBUSY";
    case EINVAL: return "EINVAL";
    case ENOENT: return "ENOENT";
    default: return "errno" + std::to_string(value);
  }
}

void context() {
  CUdevice device = 0;
  CUcontext ctx = nullptr;
  if (!ok(cuInit(0U)) || !ok(cuDeviceGet(&device, 0)) ||
      !ok(cuDevicePrimaryCtxRetain(&ctx, device)) || !ok(cuCtxSetCurrent(ctx)) ||
      cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "protect_e1: no CUDA context\n");
    _exit(3);
  }
}

// The wrong words of the shared rows as the GPU reads them; -1 and the error when the read fails.
std::string gpu_count(const void* data) {
  unsigned long long* wrong = nullptr;
  if (cudaMalloc(&wrong, sizeof(*wrong)) != cudaSuccess) return "-1:cudaMalloc";
  cudaMemset(wrong, 0, sizeof(*wrong));
  count_wrong<<<1024, 256>>>(static_cast<const std::uint32_t*>(data), kSize / 4, wrong);
  cudaError_t status = cudaDeviceSynchronize();
  unsigned long long host = 0;
  if (status == cudaSuccess) status = cudaMemcpy(&host, wrong, sizeof(host), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) return std::string("-1:") + cudaGetErrorName(status);
  return std::to_string(host);
}

std::string gpu_zero(void* data) {
  zero_words<<<static_cast<unsigned>((kWrite / 4 + 255) / 256), 256>>>(
      static_cast<std::uint32_t*>(data), kWrite / 4);
  return cudaGetErrorName(cudaDeviceSynchronize());
}

void touch(const void* data, std::uint64_t size) {
  const volatile char* view = static_cast<const volatile char*>(data);
  for (std::uint64_t at = 0; at < size; at += 4096) (void)view[at];
}

CUresult set_access(CUdeviceptr address, std::uint64_t size, CUmemAccess_flags flags) {
  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = 0;
  access.flags = flags;
  return cuMemSetAccess(address, size, &access, 1U);
}

// The path through which a child opens the published file: the file of the publisher,
// or the inherited memfd of the immutable mode.
std::string path_of(const std::string& mode, const std::string& where) {
  return mode == "immutable" ? "/proc/self/fd/" + where : where;
}

// An agent mapping: read-only and private from a descriptor without write access, read by the CPU once.
void* agent_map(const std::string& mode, const std::string& where, int* fd_out, std::string* failure) {
  if (mode == "device") {
    CUmemGenericAllocationHandle handle = 0;
    CUresult status = cuMemImportFromShareableHandle(
        &handle, reinterpret_cast<void*>(static_cast<uintptr_t>(std::atoi(where.c_str()))),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    CUdeviceptr range = 0;
    if (ok(status)) status = cuMemAddressReserve(&range, kSize, 0U, 0U, 0ULL);
    if (ok(status)) status = cuMemMap(range, kSize, 0U, handle, 0ULL);
    if (ok(status)) status = set_access(range, kSize, CU_MEM_ACCESS_FLAGS_PROT_READ);
    if (!ok(status)) {
      *failure = cu_name(status);
      return nullptr;
    }
    return reinterpret_cast<void*>(range);
  }
  const int fd = open(path_of(mode, where).c_str(), O_RDONLY);
  if (fd < 0) {
    *failure = "open:" + errno_name(errno);
    return nullptr;
  }
  if (mode == "immutable" && (fcntl(fd, F_GET_SEALS) & F_SEAL_WRITE) == 0) {
    *failure = "not_sealed";
    return nullptr;
  }
  void* mapped = mmap(nullptr, kSize, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) {
    *failure = "mmap:" + errno_name(errno);
    return nullptr;
  }
  touch(mapped, kSize);
  if (fd_out != nullptr) {
    *fd_out = fd;
  } else {
    close(fd);
  }
  return mapped;
}

int reader(const std::string& mode, const std::string& where) {
  context();
  std::string failure;
  void* mapped = agent_map(mode, where, nullptr, &failure);
  if (mapped == nullptr) {
    std::printf("BEFORE %s\n", ("-1:" + failure).c_str());
    std::fflush(stdout);
    return 0;
  }
  std::printf("BEFORE %s\n", gpu_count(mapped).c_str());
  std::fflush(stdout);
  char go = 0;
  if (read(STDIN_FILENO, &go, 1) != 1) return 0;
  std::printf("AFTER %s\n", gpu_count(mapped).c_str());
  std::fflush(stdout);
  _exit(0);
}

int writer(const std::string& mode, const std::string& attempt, const std::string& where) {
  context();
  std::string steps;
  const auto step = [&](const std::string& name, const std::string& result) {
    steps += (steps.empty() ? "" : ";") + name + "=" + result;
  };
  if (attempt == "map" || attempt == "raise" || attempt == "remap") {
    std::string failure;
    int fd = -1;
    void* mapped = agent_map(mode, where, &fd, &failure);
    if (mapped == nullptr) {
      step("agent_map", failure);
    } else if (attempt == "map") {
      step("gpu_write", gpu_zero(mapped));
    } else if (attempt == "raise") {
      if (mode == "device") {
        const CUresult raise = set_access(reinterpret_cast<CUdeviceptr>(mapped), kSize,
                                          CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
        step("set_access_rw", cu_name(raise));
        if (ok(raise)) step("gpu_write", gpu_zero(mapped));
      } else {
        const int raised = mprotect(mapped, kSize, PROT_READ | PROT_WRITE);
        step("mprotect_rw", errno_name(raised == 0 ? 0 : errno));
        if (raised == 0) step("gpu_write", gpu_zero(mapped));
      }
    } else if (mode == "device") {
      step("remap", "not_applicable");
    } else {
      void* again = mmap(nullptr, kWrite, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      step("mmap_shared_rw", errno_name(again == MAP_FAILED ? errno : 0));
      if (again != MAP_FAILED) step("gpu_write", gpu_zero(again));
    }
  } else if (mode == "device") {
    step(attempt, "not_applicable");
  } else {
    const int fd = open(path_of(mode, where).c_str(), O_RDWR);
    step("open_rdwr", errno_name(fd < 0 ? errno : 0));
    if (fd >= 0 && attempt == "reopen") {
      std::vector<char> zeros(4096, 0);
      const ssize_t wrote = pwrite(fd, zeros.data(), zeros.size(), static_cast<off_t>(kWrite));
      step("pwrite", errno_name(wrote < 0 ? errno : 0));
      void* again = mmap(nullptr, kWrite, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      step("mmap_shared_rw", errno_name(again == MAP_FAILED ? errno : 0));
      if (again != MAP_FAILED) step("gpu_write", gpu_zero(again));
    } else if (fd >= 0) {
      step("ftruncate_0", errno_name(ftruncate(fd, 0) == 0 ? 0 : errno));
    }
  }
  std::printf("WRITE %s\n", steps.c_str());
  std::fflush(stdout);
  _exit(0);
}

struct Child {
  pid_t pid = -1;
  int out = -1;  // its stdout
  int in = -1;   // its stdin
};

Child start(const std::vector<std::string>& args) {
  // close-on-exec, so that a child keeps only its own stdin and stdout of these pipes
  int out[2], in[2];
  if (pipe2(out, O_CLOEXEC) != 0 || pipe2(in, O_CLOEXEC) != 0) return {};
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, out[1], STDOUT_FILENO);
  posix_spawn_file_actions_adddup2(&actions, in[0], STDIN_FILENO);
  std::vector<char*> argv;
  for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  Child child;
  if (posix_spawn(&child.pid, "/proc/self/exe", &actions, nullptr, argv.data(), environ) != 0) child.pid = -1;
  posix_spawn_file_actions_destroy(&actions);
  close(out[1]);
  close(in[0]);
  child.out = out[0];
  child.in = in[1];
  return child;
}

std::string read_line(int fd) {
  std::string line;
  char c = 0;
  while (read(fd, &c, 1) == 1 && c != '\n') line += c;
  return line;
}

std::string rest(const std::string& line, const std::string& key) {
  return line.rfind(key, 0) == 0 ? line.substr(key.size()) : "missing";
}

int run(const std::string& mode, const std::string& attempt, const std::string& rep) {
  // the published rows
  std::string where;
  std::string file;
  std::vector<std::uint32_t> host(kSize / 4);
  for (std::uint64_t i = 0; i < host.size(); ++i) host[i] = pattern(i);
  if (mode == "default") {
    file = "/dev/shm/protect_e1." + std::to_string(getpid());
    const int fd = open(file.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0 || ftruncate(fd, static_cast<off_t>(kSize)) != 0) return 3;
    void* base = mmap(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) return 3;
    close(fd);
    std::memcpy(base, host.data(), kSize);
    if (mprotect(base, kSize, PROT_READ) != 0) return 3;  // the publish of the default mode
    where = file;
  } else if (mode == "immutable") {
    const int fd = memfd_create("protect_e1", MFD_ALLOW_SEALING);  // inherited by the children
    if (fd < 0 || ftruncate(fd, static_cast<off_t>(kSize)) != 0) return 3;
    char* base = static_cast<char*>(mmap(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (base == MAP_FAILED) return 3;
    std::memcpy(base, host.data(), kSize);
    const int ro = open(("/proc/self/fd/" + std::to_string(fd)).c_str(), O_RDONLY);
    if (ro < 0 || mmap(base, kSize, PROT_READ, MAP_SHARED | MAP_FIXED, ro, 0) == MAP_FAILED) return 3;
    close(ro);
    if (fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL) != 0) {
      std::fprintf(stderr, "protect_e1: seal refused (%s)\n", errno_name(errno).c_str());
      return 3;
    }
    where = std::to_string(fd);
  } else {
    context();
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CUmemGenericAllocationHandle handle = 0;
    CUdeviceptr range = 0;
    int fd = -1;
    if (!ok(cuMemCreate(&handle, kSize, &prop, 0ULL)) ||
        !ok(cuMemAddressReserve(&range, kSize, 0U, 0U, 0ULL)) ||
        !ok(cuMemMap(range, kSize, 0U, handle, 0ULL)) ||
        !ok(set_access(range, kSize, CU_MEM_ACCESS_FLAGS_PROT_READWRITE)) ||
        !ok(cuMemcpyHtoD(range, host.data(), kSize)) ||
        !ok(cuMemExportToShareableHandle(&fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0ULL))) {
      std::fprintf(stderr, "protect_e1: no device memory\n");
      return 3;
    }
    (void)fcntl(fd, F_SETFD, 0);
    where = std::to_string(fd);
  }
  const std::string self = "/proc/self/exe";
  Child readers[2] = {start({self, "reader", mode, where}), start({self, "reader", mode, where})};
  std::string before[2], after[2];
  for (int r = 0; r < 2; ++r) before[r] = rest(read_line(readers[r].out), "BEFORE ");
  Child writer_child = start({self, "writer", mode, attempt, where});
  const std::string written = rest(read_line(writer_child.out), "WRITE ");
  int status = 0;
  waitpid(writer_child.pid, &status, 0);
  for (int r = 0; r < 2; ++r) {
    if (write(readers[r].in, "g", 1) != 1) after[r] = "no_signal";
    after[r] = rest(read_line(readers[r].out), "AFTER ");
    waitpid(readers[r].pid, &status, 0);
  }
  // the content of the file itself, read by the CPU through a new descriptor
  std::string file_wrong = "not_applicable";
  if (mode != "device") {
    const int fd = open(path_of(mode, where).c_str(), O_RDONLY);
    struct stat st {};
    fstat(fd, &st);
    if (static_cast<std::uint64_t>(st.st_size) < kSize) {
      file_wrong = "size=" + std::to_string(st.st_size);
    } else {
      std::vector<std::uint32_t> words(kSize / 4);
      const ssize_t got = pread(fd, words.data(), kSize, 0);
      unsigned long long wrong = 0;
      for (std::uint64_t i = 0; i < words.size(); ++i) wrong += words[i] != pattern(i);
      file_wrong = got == static_cast<ssize_t>(kSize) ? std::to_string(wrong) : "short_read";
    }
    close(fd);
  }
  if (!file.empty()) unlink(file.c_str());
  std::printf("CASE mode=%s attempt=%s rep=%s writer=%s reader_before=%s,%s reader_after=%s,%s file_wrong_words=%s\n",
              mode.c_str(), attempt.c_str(), rep.c_str(), written.c_str(), before[0].c_str(), before[1].c_str(),
              after[0].c_str(), after[1].c_str(), file_wrong.c_str());
  std::fflush(stdout);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  signal(SIGPIPE, SIG_IGN);
  const std::vector<std::string> args(argv + 1, argv + argc);
  if (args.size() == 4 && args[0] == "run") return run(args[1], args[2], args[3]);
  if (args.size() == 3 && args[0] == "reader") return reader(args[1], args[2]);
  if (args.size() == 4 && args[0] == "writer") return writer(args[1], args[2], args[3]);
  std::fprintf(stderr, "usage: protect_e1 run MODE ATTEMPT REP\n");
  return 2;
}
