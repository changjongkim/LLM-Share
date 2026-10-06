// Can a sharer change shared state through the GPU?
//
//   cuda_protect_probe host FILE SIZE_MIB
//   cuda_protect_probe vmm SIZE_MIB
//
// host  FILE holds SIZE_MIB of a known pattern (the caller creates it on a
//       tmpfs). The process maps it read-only and private, as an agent maps
//       a published prefix, reads it once on the GPU, and then launches a
//       kernel that writes its first word. Reports the error of the write
//       and whether the file still holds the pattern.
// vmm   The process creates SIZE_MIB of device memory, fills it, makes its
//       own mapping read-only and exports it. A child imports the handle
//       and maps it read-only, then raises its own mapping to read-write
//       and writes the first word. Reports whether the raise and the write
//       succeeded and whether the parent now reads the changed word.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace {

constexpr std::uint32_t kValue = 0x5eedf00dU;

__global__ void sum_words(const std::uint32_t* data, std::uint64_t words,
                          unsigned long long* out) {
  const std::uint64_t thread =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (thread < words) {
    atomicAdd(out, static_cast<unsigned long long>(data[thread] == 0x5eedf00dU));
  }
}

__global__ void write_word(std::uint32_t* data) { data[0] = ~0x5eedf00dU; }

int host(const char* path, std::uint64_t bytes) {
  const int fd = open(path, O_RDONLY);
  if (fd < 0) return 3;
  void* mapped = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) return 3;
  // The GPU cannot set the accessed flag of an entry; a CPU read does.
  const volatile char* view = static_cast<const volatile char*>(mapped);
  for (std::uint64_t offset = 0; offset < bytes; offset += 4096) (void)view[offset];

  unsigned long long* out = nullptr;
  if (cudaMalloc(&out, sizeof(*out)) != cudaSuccess ||
      cudaMemset(out, 0, sizeof(*out)) != cudaSuccess) {
    return 3;
  }
  const std::uint64_t words = bytes / sizeof(std::uint32_t);
  sum_words<<<static_cast<unsigned>((words + 255) / 256), 256>>>(
      static_cast<const std::uint32_t*>(mapped), words, out);
  const cudaError_t read_status = cudaDeviceSynchronize();
  unsigned long long equal = 0;
  if (read_status == cudaSuccess) {
    cudaMemcpy(&equal, out, sizeof(equal), cudaMemcpyDeviceToHost);
  }

  write_word<<<1, 1>>>(static_cast<std::uint32_t*>(mapped));
  const cudaError_t write_status = cudaDeviceSynchronize();

  std::uint32_t now = 0;
  const bool read_back = pread(fd, &now, sizeof(now), 0) == sizeof(now);
  std::printf("RESULT route=host gpu_read=%s words_equal=%llu words=%llu "
              "gpu_write=%s file_intact=%d\n",
              cudaGetErrorName(read_status), equal,
              static_cast<unsigned long long>(words),
              cudaGetErrorName(write_status), read_back && now == kValue);
  std::fflush(stdout);
  // The context is in an error state after a failed write; leave at once.
  _exit(0);
}

bool ok(CUresult status) { return status == CUDA_SUCCESS; }

bool start_context() {
  CUdevice device = 0;
  CUcontext context = nullptr;
  return ok(cuInit(0U)) && ok(cuDeviceGet(&device, 0)) &&
         ok(cuDevicePrimaryCtxRetain(&context, device)) && ok(cuCtxSetCurrent(context));
}

CUresult set_access(CUdeviceptr address, size_t bytes, CUmemAccess_flags flags) {
  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = 0;
  access.flags = flags;
  return cuMemSetAccess(address, bytes, &access, 1U);
}

const char* name_of(CUresult status) {
  const char* text = nullptr;
  cuGetErrorName(status, &text);
  return text != nullptr ? text : "?";
}

int vmm_child(size_t bytes, int fd) {
  if (!start_context()) return 3;
  CUmemGenericAllocationHandle handle = 0;
  CUdeviceptr range = 0;
  if (!ok(cuMemImportFromShareableHandle(
          &handle, reinterpret_cast<void*>(static_cast<uintptr_t>(fd)),
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)) ||
      !ok(cuMemAddressReserve(&range, bytes, 0U, 0U, 0ULL)) ||
      !ok(cuMemMap(range, bytes, 0U, handle, 0ULL)) ||
      !ok(set_access(range, bytes, CU_MEM_ACCESS_FLAGS_PROT_READ))) {
    return 3;
  }
  // Without the raise the write is refused (run_vmm_routes.sh); with it:
  const CUresult raise = set_access(range, bytes, CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
  CUresult write = cuMemsetD32(range, ~kValue, 1U);
  if (ok(write)) write = cuCtxSynchronize();
  std::printf("CHILD raise=%s write=%s\n", name_of(raise), name_of(write));
  std::fflush(stdout);
  _exit(0);
}

int vmm(const char* self, size_t bytes) {
  if (!start_context()) return 3;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  CUmemGenericAllocationHandle handle = 0;
  CUdeviceptr range = 0;
  int fd = -1;
  if (!ok(cuMemCreate(&handle, bytes, &prop, 0ULL)) ||
      !ok(cuMemAddressReserve(&range, bytes, 0U, 0U, 0ULL)) ||
      !ok(cuMemMap(range, bytes, 0U, handle, 0ULL)) ||
      !ok(set_access(range, bytes, CU_MEM_ACCESS_FLAGS_PROT_READWRITE)) ||
      !ok(cuMemsetD32(range, kValue, bytes / sizeof(std::uint32_t))) ||
      !ok(cuCtxSynchronize()) ||
      !ok(set_access(range, bytes, CU_MEM_ACCESS_FLAGS_PROT_READ)) ||
      !ok(cuMemExportToShareableHandle(&fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0ULL))) {
    std::fprintf(stderr, "cannot create or export device memory\n");
    return 3;
  }
  fcntl(fd, F_SETFD, fcntl(fd, F_GETFD) & ~FD_CLOEXEC);
  std::vector<std::string> arguments = {self, "vmm_child", std::to_string(bytes >> 20U), std::to_string(fd)};
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
  char raise[64] = "none";
  char write[64] = "none";
  if (got > 0) std::sscanf(line, "CHILD raise=%63s write=%63s", raise, write);
  std::uint32_t now = 0;
  const CUresult read_back = cuMemcpyDtoH(&now, range, sizeof(now));
  std::printf("RESULT route=vmm importer_raise=%s importer_write=%s "
              "exporter_read=%s exporter_intact=%d\n",
              raise, write, name_of(read_back), ok(read_back) && now == kValue);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 4 && std::strcmp(argv[1], "host") == 0) {
    return host(argv[2], std::strtoull(argv[3], nullptr, 10) << 20U);
  }
  if (argc == 3 && std::strcmp(argv[1], "vmm") == 0) {
    return vmm(argv[0], std::strtoull(argv[2], nullptr, 10) << 20U);
  }
  if (argc == 4 && std::strcmp(argv[1], "vmm_child") == 0) {
    return vmm_child(std::strtoull(argv[2], nullptr, 10) << 20U, std::atoi(argv[3]));
  }
  std::fprintf(stderr, "usage: cuda_protect_probe host FILE SIZE_MIB | vmm SIZE_MIB\n");
  return 2;
}
