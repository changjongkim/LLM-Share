// Does device memory that one process shares with another stay usable while
// a third process uses the GPU of the same MIG instance? The producer
// allocates memory, fills it, shares it, and then stays idle. The consumer,
// a separately executed process, reads the shared memory with the GPU again
// and again for a number of seconds. A bystander, a third process, keeps the
// GPU busy in a MIG instance of choice, or is absent.
//
// usage: cuda_share_load_probe ROUTE PRODUCER_MIG CONSUMER_MIG BYSTANDER SECONDS
//   ROUTE      ipc: cudaMalloc and cudaIpcGetMemHandle
//              vmm: cuMemCreate and cuMemExportToShareableHandle, mapped
//                   read-only by the consumer
//              own: nothing is shared; the consumer reads memory that it
//                   mapped itself with the virtual memory interface
//   BYSTANDER  a MIG UUID, or none
#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr std::uint32_t kValue = 0x5eedf00dU;
constexpr std::size_t kBytes = std::size_t{64U} << 20U;

using Clock = std::chrono::steady_clock;

struct Request {
  int route = 0;  // 0 ipc, 1 vmm, 2 own
  double seconds = 0.0;
  cudaIpcMemHandle_t handle{};
};
struct Reply {
  char stage[16] = "none";
  int error = 0;        // a cudaError_t or CUresult, by stage
  int driver = 0;       // 1 when error is a CUresult
  long iterations = 0;
  long wrong = 0;
};

bool read_all(const int fd, void* const data, const std::size_t bytes) {
  std::size_t done = 0U;
  while (done < bytes) {
    const ssize_t got = ::read(fd, static_cast<char*>(data) + done, bytes - done);
    if (got <= 0) return false;
    done += static_cast<std::size_t>(got);
  }
  return true;
}

bool send_request(const int socket, const Request& request, const int fd) {
  char control[CMSG_SPACE(sizeof(int))] = {};
  iovec part{const_cast<Request*>(&request), sizeof(request)};
  msghdr message{};
  message.msg_iov = &part;
  message.msg_iovlen = 1U;
  if (fd >= 0) {
    message.msg_control = control;
    message.msg_controllen = sizeof(control);
    cmsghdr* const header = CMSG_FIRSTHDR(&message);
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(header), &fd, sizeof(int));
  }
  return ::sendmsg(socket, &message, 0) == static_cast<ssize_t>(sizeof(request));
}

bool receive_request(const int socket, Request* const request, int* const fd) {
  char control[CMSG_SPACE(sizeof(int))] = {};
  iovec part{request, sizeof(*request)};
  msghdr message{};
  message.msg_iov = &part;
  message.msg_iovlen = 1U;
  message.msg_control = control;
  message.msg_controllen = sizeof(control);
  if (::recvmsg(socket, &message, 0) != static_cast<ssize_t>(sizeof(*request))) {
    return false;
  }
  const cmsghdr* const header = CMSG_FIRSTHDR(&message);
  *fd = -1;
  if (header != nullptr && header->cmsg_type == SCM_RIGHTS) {
    std::memcpy(fd, CMSG_DATA(header), sizeof(int));
  }
  return true;
}

CUmemAllocationProp properties(const bool shareable) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  if (shareable) prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  return prop;
}

CUresult set_access(const CUdeviceptr address, const CUmemAccess_flags flags) {
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = 0;
  access.flags = flags;
  return cuMemSetAccess(address, kBytes, &access, 1U);
}

// A new allocation of the virtual memory interface, mapped and filled.
CUresult map_new(const bool shareable, CUdeviceptr* const range,
                 CUmemGenericAllocationHandle* const handle) {
  const CUmemAllocationProp prop = properties(shareable);
  CUresult status = cuMemCreate(handle, kBytes, &prop, 0ULL);
  if (status == CUDA_SUCCESS) status = cuMemAddressReserve(range, kBytes, 0U, 0U, 0ULL);
  if (status == CUDA_SUCCESS) status = cuMemMap(*range, kBytes, 0U, *handle, 0ULL);
  if (status == CUDA_SUCCESS) status = set_access(*range, CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
  if (status == CUDA_SUCCESS) {
    status = cuMemsetD32(*range, kValue, kBytes / sizeof(std::uint32_t));
  }
  return status;
}

int run_bystander(const double seconds) {
  if (cudaSetDevice(0) != cudaSuccess) return 3;
  void* memory = nullptr;
  if (cudaMalloc(&memory, std::size_t{256U} << 20U) != cudaSuccess) return 4;
  // Tell the parent that the GPU is in use from here on.
  const char ready = 1;
  (void)!::write(STDOUT_FILENO, &ready, 1);
  const auto start = Clock::now();
  int value = 0;
  while (std::chrono::duration<double>(Clock::now() - start).count() < seconds) {
    if (cudaMemset(memory, ++value, std::size_t{256U} << 20U) != cudaSuccess) return 5;
    if (cudaDeviceSynchronize() != cudaSuccess) return 5;
  }
  return 0;
}

int run_consumer(const int socket) {
  Request request;
  int fd = -1;
  if (!receive_request(socket, &request, &fd)) return 4;
  Reply reply;
  const auto fail_runtime = [&reply](const char* const stage, const cudaError_t error) {
    std::snprintf(reply.stage, sizeof(reply.stage), "%s", stage);
    reply.error = static_cast<int>(error);
    reply.driver = 0;
  };
  const auto fail_driver = [&reply](const char* const stage, const CUresult error) {
    std::snprintf(reply.stage, sizeof(reply.stage), "%s", stage);
    reply.error = static_cast<int>(error);
    reply.driver = 1;
  };
  void* shared = nullptr;
  void* mine = nullptr;
  cudaError_t status = cudaSetDevice(0);
  if (status == cudaSuccess) status = cudaMalloc(&mine, kBytes);
  if (status != cudaSuccess) {
    fail_runtime("context", status);
  } else if (request.route == 0) {
    status = cudaIpcOpenMemHandle(&shared, request.handle, cudaIpcMemLazyEnablePeerAccess);
    if (status != cudaSuccess) fail_runtime("open", status);
  } else if (request.route == 1) {
    CUmemGenericAllocationHandle handle = 0U;
    CUdeviceptr range = 0U;
    CUresult result = cuMemImportFromShareableHandle(
        &handle, reinterpret_cast<void*>(static_cast<std::uintptr_t>(fd)),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    if (result == CUDA_SUCCESS) result = cuMemAddressReserve(&range, kBytes, 0U, 0U, 0ULL);
    if (result == CUDA_SUCCESS) result = cuMemMap(range, kBytes, 0U, handle, 0ULL);
    if (result == CUDA_SUCCESS) result = set_access(range, CU_MEM_ACCESS_FLAGS_PROT_READ);
    if (result != CUDA_SUCCESS) fail_driver("import", result);
    shared = reinterpret_cast<void*>(range);
  } else {
    CUmemGenericAllocationHandle handle = 0U;
    CUdeviceptr range = 0U;
    const CUresult result = map_new(false, &range, &handle);
    if (result != CUDA_SUCCESS) fail_driver("map", result);
    shared = reinterpret_cast<void*>(range);
  }
  if (std::strcmp(reply.stage, "none") == 0) {
    // The GPU reads the shared memory into memory of this process; two words
    // of the copy are checked on the host.
    const auto start = Clock::now();
    while (std::chrono::duration<double>(Clock::now() - start).count() < request.seconds) {
      std::uint32_t first = 0U;
      std::uint32_t last = 0U;
      status = cudaMemcpy(mine, shared, kBytes, cudaMemcpyDeviceToDevice);
      if (status == cudaSuccess) {
        status = cudaMemcpy(&first, mine, sizeof(first), cudaMemcpyDeviceToHost);
      }
      if (status == cudaSuccess) {
        status = cudaMemcpy(&last, static_cast<char*>(mine) + kBytes - sizeof(last),
                            sizeof(last), cudaMemcpyDeviceToHost);
      }
      if (status != cudaSuccess) {
        fail_runtime("read", status);
        break;
      }
      ++reply.iterations;
      if (first != kValue || last != kValue) ++reply.wrong;
    }
    if (std::strcmp(reply.stage, "none") == 0) {
      std::snprintf(reply.stage, sizeof(reply.stage), "done");
    }
  }
  return ::write(socket, &reply, sizeof(reply)) == static_cast<ssize_t>(sizeof(reply))
             ? 0
             : 5;
}

// Executes this program again in a role, in a MIG instance.
pid_t start(const char* const mig, const char* const role, const std::string& argument,
            const int keep_fd, const int output_fd) {
  const pid_t child = ::fork();
  if (child != 0) return child;
  if (output_fd >= 0) ::dup2(output_fd, STDOUT_FILENO);
  ::setenv("CUDA_VISIBLE_DEVICES", mig, 1);
  char self[4096];
  const ssize_t length = ::readlink("/proc/self/exe", self, sizeof(self) - 1U);
  if (length <= 0) ::_exit(127);
  self[length] = '\0';
  (void)keep_fd;
  ::execl(self, self, role, argument.c_str(), static_cast<char*>(nullptr));
  ::_exit(127);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::strcmp(argv[1], "--consumer") == 0) {
    return run_consumer(std::atoi(argv[2]));
  }
  if (argc == 3 && std::strcmp(argv[1], "--bystander") == 0) {
    return run_bystander(std::atof(argv[2]));
  }
  if (argc != 6) {
    std::fprintf(stderr,
                 "usage: %s ROUTE PRODUCER_MIG CONSUMER_MIG BYSTANDER SECONDS\n", argv[0]);
    return 2;
  }
  const std::string route_name = argv[1];
  const int route = route_name == "ipc" ? 0 : route_name == "vmm" ? 1 : route_name == "own" ? 2 : -1;
  if (route < 0) return 2;
  const bool with_bystander = std::strcmp(argv[4], "none") != 0;
  const double seconds = std::atof(argv[5]);
  const bool mps = ::getenv("CUDA_MPS_PIPE_DIRECTORY") != nullptr;

  // Both helpers are started before this process touches the GPU.
  int sockets[2];
  int ready_pipe[2];
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0 || ::pipe(ready_pipe) != 0) return 2;
  pid_t bystander = -1;
  if (with_bystander) {
    bystander = start(argv[4], "--bystander", std::to_string(seconds + 4.0), -1, ready_pipe[1]);
  }
  ::close(ready_pipe[1]);
  const pid_t consumer =
      start(argv[3], "--consumer", std::to_string(sockets[1]), sockets[1], -1);
  ::close(sockets[1]);
  ::setenv("CUDA_VISIBLE_DEVICES", argv[2], 1);

  const char* result = "FAILED";
  std::string stage = "none";
  std::string error = "none";
  long iterations = 0;
  long wrong = 0;
  Request request;
  request.route = route;
  request.seconds = seconds;
  int fd = -1;
  bool produced = true;
  if (route == 0) {
    void* memory = nullptr;
    cudaError_t status = cudaSetDevice(0);
    if (status == cudaSuccess) status = cudaMalloc(&memory, kBytes);
    if (status == cudaSuccess) {
      status = static_cast<cudaError_t>(
          cuMemsetD32(reinterpret_cast<CUdeviceptr>(memory), kValue,
                      kBytes / sizeof(std::uint32_t)) == CUDA_SUCCESS
              ? cudaSuccess
              : cudaErrorUnknown);
    }
    if (status == cudaSuccess) status = cudaIpcGetMemHandle(&request.handle, memory);
    if (status != cudaSuccess) {
      produced = false;
      stage = "produce";
      error = cudaGetErrorName(status);
    }
  } else if (route == 1) {
    CUmemGenericAllocationHandle handle = 0U;
    CUdeviceptr range = 0U;
    CUresult status = cudaSetDevice(0) == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    if (status == CUDA_SUCCESS) status = map_new(true, &range, &handle);
    if (status == CUDA_SUCCESS) {
      status = cuMemExportToShareableHandle(&fd, handle,
                                            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0ULL);
    }
    if (status != CUDA_SUCCESS) {
      const char* name = nullptr;
      (void)cuGetErrorName(status, &name);
      produced = false;
      stage = "produce";
      error = name != nullptr ? name : "unknown";
    }
  }
  if (produced && with_bystander) {
    // Wait until the bystander uses the GPU.
    char ready = 0;
    if (::read(ready_pipe[0], &ready, 1) != 1) {
      produced = false;
      stage = "bystander";
      error = "not_ready";
    }
  }
  if (produced) {
    Reply reply;
    if (!send_request(sockets[0], request, fd) ||
        !read_all(sockets[0], &reply, sizeof(reply))) {
      stage = "consumer";
      error = "no_reply";
    } else {
      stage = reply.stage;
      iterations = reply.iterations;
      wrong = reply.wrong;
      if (std::strcmp(reply.stage, "done") == 0 && reply.wrong == 0 && reply.iterations > 0) {
        result = "PASS";
      } else if (reply.driver != 0) {
        const char* name = nullptr;
        (void)cuGetErrorName(static_cast<CUresult>(reply.error), &name);
        error = name != nullptr ? name : "unknown";
      } else if (reply.error != 0) {
        error = cudaGetErrorName(static_cast<cudaError_t>(reply.error));
      } else {
        error = "wrong_value";
      }
    }
  }
  ::close(sockets[0]);
  int status = 0;
  ::waitpid(consumer, &status, 0);
  int bystander_exit = -1;
  if (bystander > 0) {
    if (!produced) ::kill(bystander, SIGKILL);
    ::waitpid(bystander, &status, 0);
    bystander_exit = WIFEXITED(status) ? WEXITSTATUS(status) : 128;
  }
  std::printf("probe=share_load route=%s producer=%.12s consumer=%.12s bystander=%.12s "
              "mps=%d seconds=%.0f result=%s stage=%s error=%s iterations=%ld wrong=%ld "
              "bystander_exit=%d\n",
              argv[1], argv[2], argv[3], argv[4], mps ? 1 : 0, seconds, result,
              stage.c_str(), error.c_str(), iterations, wrong, bystander_exit);
  return 0;
}
