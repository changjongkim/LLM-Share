// What it costs to hand device memory to another process through the CUDA
// virtual memory management interface, by the size of the allocations.
//
//   cuda_vmm_attach_probe TOTAL_MIB GRANULE_MIB
//
// The parent creates TOTAL_MIB of device memory as allocations of
// GRANULE_MIB each, fills them, makes them read-only and exports one file
// descriptor per allocation (export_ms). It then starts a child that
// inherits the descriptors. The child reserves an address range, imports
// and maps every allocation read-only (attach_ms) and reads one word of
// each to check the contents. One line reports both times.
#include <cuda.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint32_t kValue = 0x5eedf00dU;

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

bool ok(CUresult status, const char* what) {
  if (status != CUDA_SUCCESS) {
    const char* text = nullptr;
    cuGetErrorString(status, &text);
    std::fprintf(stderr, "%s: %s\n", what, text != nullptr ? text : "?");
    return false;
  }
  return true;
}

bool start_context() {
  CUdevice device = 0;
  CUcontext context = nullptr;
  return ok(cuInit(0U), "cuInit") && ok(cuDeviceGet(&device, 0), "cuDeviceGet") &&
         ok(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain") &&
         ok(cuCtxSetCurrent(context), "cuCtxSetCurrent");
}

bool set_access(CUdeviceptr address, size_t bytes, CUmemAccess_flags flags) {
  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = 0;
  access.flags = flags;
  return ok(cuMemSetAccess(address, bytes, &access, 1U), "cuMemSetAccess");
}

int child(size_t granule, int count, char** descriptors) {
  if (!start_context()) return 3;
  const auto start = Clock::now();
  const size_t total = granule * static_cast<size_t>(count);
  CUdeviceptr range = 0;
  if (!ok(cuMemAddressReserve(&range, total, 0U, 0U, 0ULL), "cuMemAddressReserve")) return 3;
  for (int i = 0; i < count; ++i) {
    CUmemGenericAllocationHandle handle = 0;
    const int fd = std::atoi(descriptors[i]);
    if (!ok(cuMemImportFromShareableHandle(
                &handle, reinterpret_cast<void*>(static_cast<uintptr_t>(fd)),
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
            "cuMemImportFromShareableHandle") ||
        !ok(cuMemMap(range + static_cast<size_t>(i) * granule, granule, 0U, handle, 0ULL), "cuMemMap") ||
        !ok(cuMemRelease(handle), "cuMemRelease")) {
      return 3;
    }
    close(fd);
  }
  if (!set_access(range, total, CU_MEM_ACCESS_FLAGS_PROT_READ)) return 3;
  const double attach_ms = ms_since(start);
  int wrong = 0;
  for (int i = 0; i < count; ++i) {
    std::uint32_t word = 0;
    if (!ok(cuMemcpyDtoH(&word, range + static_cast<size_t>(i) * granule, sizeof(word)), "cuMemcpyDtoH")) return 3;
    wrong += word != kValue;
  }
  std::printf("CHILD attach_ms=%.3f wrong=%d\n", attach_ms, wrong);
  return wrong == 0 ? 0 : 4;
}

int parent(const char* self, size_t total_mib, size_t granule_mib) {
  if (!start_context()) return 3;
  const size_t granule = granule_mib << 20U;
  const int count = static_cast<int>(total_mib / granule_mib);
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  CUdeviceptr range = 0;
  if (!ok(cuMemAddressReserve(&range, granule * static_cast<size_t>(count), 0U, 0U, 0ULL), "cuMemAddressReserve")) return 3;
  std::vector<CUmemGenericAllocationHandle> handles(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    if (!ok(cuMemCreate(&handles[static_cast<size_t>(i)], granule, &prop, 0ULL), "cuMemCreate") ||
        !ok(cuMemMap(range + static_cast<size_t>(i) * granule, granule, 0U, handles[static_cast<size_t>(i)], 0ULL), "cuMemMap")) {
      return 3;
    }
  }
  const size_t total = granule * static_cast<size_t>(count);
  if (!set_access(range, total, CU_MEM_ACCESS_FLAGS_PROT_READWRITE) ||
      !ok(cuMemsetD32(range, kValue, total / sizeof(std::uint32_t)), "cuMemsetD32") ||
      !ok(cuCtxSynchronize(), "cuCtxSynchronize")) {
    return 3;
  }

  const auto start = Clock::now();
  if (!set_access(range, total, CU_MEM_ACCESS_FLAGS_PROT_READ)) return 3;
  std::vector<std::string> arguments = {self, "child", std::to_string(granule_mib)};
  for (int i = 0; i < count; ++i) {
    int fd = -1;
    if (!ok(cuMemExportToShareableHandle(&fd, handles[static_cast<size_t>(i)],
                                         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0ULL),
            "cuMemExportToShareableHandle")) {
      return 3;
    }
    // The child receives the descriptor by inheritance.
    fcntl(fd, F_SETFD, fcntl(fd, F_GETFD) & ~FD_CLOEXEC);
    arguments.push_back(std::to_string(fd));
  }
  const double export_ms = ms_since(start);

  std::vector<char*> argv;
  for (auto& argument : arguments) argv.push_back(argument.data());
  argv.push_back(nullptr);
  int pipe_fds[2] = {-1, -1};
  if (pipe(pipe_fds) != 0) return 3;
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, pipe_fds[1], STDOUT_FILENO);
  pid_t pid = -1;
  const int spawned = posix_spawn(&pid, self, &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  close(pipe_fds[1]);
  if (spawned != 0) return 3;
  char line[256] = {};
  const ssize_t got = read(pipe_fds[0], line, sizeof(line) - 1);
  int status = 0;
  waitpid(pid, &status, 0);
  double attach_ms = -1.0;
  int wrong = -1;
  if (got <= 0 || std::sscanf(line, "CHILD attach_ms=%lf wrong=%d", &attach_ms, &wrong) != 2 ||
      !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::fprintf(stderr, "the child failed\n");
    return 4;
  }
  std::printf("RESULT total_mib=%zu granule_mib=%zu handles=%d export_ms=%.3f "
              "attach_ms=%.3f attach_us_per_handle=%.1f wrong=%d\n",
              total_mib, granule_mib, count, export_ms, attach_ms,
              attach_ms * 1000.0 / count, wrong);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 4 && std::strcmp(argv[1], "child") == 0) {
    return child(std::strtoull(argv[2], nullptr, 10) << 20U, argc - 3, argv + 3);
  }
  if (argc != 3) {
    std::fprintf(stderr, "usage: cuda_vmm_attach_probe TOTAL_MIB GRANULE_MIB\n");
    return 2;
  }
  const size_t total = std::strtoull(argv[1], nullptr, 10);
  const size_t granule = std::strtoull(argv[2], nullptr, 10);
  if (granule == 0 || granule % 2 != 0 || total < granule) {
    std::fprintf(stderr, "GRANULE_MIB must be an even number that is at most TOTAL_MIB\n");
    return 2;
  }
  return parent(argv[0], total, granule);
}
