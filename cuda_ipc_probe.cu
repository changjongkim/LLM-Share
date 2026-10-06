// Can two processes share a device allocation through CUDA IPC in a given
// placement? The producer allocates device memory and exports a handle; the
// consumer, a separately executed process, opens it and reads a value.
// Unlike the Portal baseline this probe allows both processes in the same
// MIG instance, and both inherit the MPS environment of the caller.
//
// usage: cuda_ipc_probe PRODUCER_MIG CONSUMER_MIG
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr std::uint32_t kValue = 0x5eedf00dU;

struct Reply {
  int open_error = -1;
  int read_error = -1;
  std::uint32_t value = 0U;
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

int run_consumer() {
  cudaIpcMemHandle_t handle{};
  Reply reply;
  if (cudaSetDevice(0) != cudaSuccess) return 3;
  if (!read_all(STDIN_FILENO, &handle, sizeof(handle))) return 4;
  void* memory = nullptr;
  reply.open_error = static_cast<int>(
      cudaIpcOpenMemHandle(&memory, handle, cudaIpcMemLazyEnablePeerAccess));
  if (reply.open_error == static_cast<int>(cudaSuccess)) {
    reply.read_error = static_cast<int>(cudaMemcpy(
        &reply.value, memory, sizeof(reply.value), cudaMemcpyDeviceToHost));
    (void)cudaIpcCloseMemHandle(memory);
  }
  return ::write(STDOUT_FILENO, &reply, sizeof(reply)) ==
                 static_cast<ssize_t>(sizeof(reply))
             ? 0
             : 5;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--consumer") == 0) return run_consumer();
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s PRODUCER_MIG CONSUMER_MIG\n", argv[0]);
    return 2;
  }
  const bool mps = ::getenv("CUDA_MPS_PIPE_DIRECTORY") != nullptr;
  int to_child[2];
  int from_child[2];
  if (::pipe(to_child) != 0 || ::pipe(from_child) != 0) return 2;
  const pid_t child = ::fork();
  if (child < 0) return 2;
  if (child == 0) {
    ::dup2(to_child[0], STDIN_FILENO);
    ::dup2(from_child[1], STDOUT_FILENO);
    for (const int end : {to_child[0], to_child[1], from_child[0], from_child[1]}) {
      ::close(end);
    }
    ::setenv("CUDA_VISIBLE_DEVICES", argv[2], 1);
    char self[4096];
    const ssize_t length = ::readlink("/proc/self/exe", self, sizeof(self) - 1U);
    if (length <= 0) ::_exit(127);
    self[length] = '\0';
    ::execl(self, self, "--consumer", static_cast<char*>(nullptr));
    ::_exit(127);
  }
  ::close(to_child[0]);
  ::close(from_child[1]);
  ::setenv("CUDA_VISIBLE_DEVICES", argv[1], 1);

  const char* result = "FAILED";
  const char* stage = "none";
  std::string error = "none";
  void* memory = nullptr;
  cudaIpcMemHandle_t handle{};
  cudaError_t status = cudaSetDevice(0);
  if (status == cudaSuccess) status = cudaMalloc(&memory, std::size_t{64U} << 20U);
  if (status == cudaSuccess) {
    status = cudaMemcpy(memory, &kValue, sizeof(kValue), cudaMemcpyHostToDevice);
  }
  if (status != cudaSuccess) {
    stage = "allocate";
    error = cudaGetErrorName(status);
  } else if ((status = cudaIpcGetMemHandle(&handle, memory)) != cudaSuccess) {
    result = "UNSUPPORTED";
    stage = "export";
    error = cudaGetErrorName(status);
  } else {
    Reply reply;
    const bool sent = ::write(to_child[1], &handle, sizeof(handle)) ==
                      static_cast<ssize_t>(sizeof(handle));
    if (!sent || !read_all(from_child[0], &reply, sizeof(reply))) {
      stage = "consumer";
      error = "no_reply";
    } else if (reply.open_error != static_cast<int>(cudaSuccess)) {
      result = "UNSUPPORTED";
      stage = "open";
      error = cudaGetErrorName(static_cast<cudaError_t>(reply.open_error));
    } else if (reply.read_error != static_cast<int>(cudaSuccess) ||
               reply.value != kValue) {
      stage = "read";
      error = reply.read_error != static_cast<int>(cudaSuccess)
                  ? cudaGetErrorName(static_cast<cudaError_t>(reply.read_error))
                  : "wrong_value";
    } else {
      result = "PASS";
    }
  }
  ::close(to_child[1]);
  int child_status = 0;
  ::waitpid(child, &child_status, 0);
  std::printf("probe=cuda_ipc producer=%.12s consumer=%.12s mps=%d result=%s "
              "stage=%s error=%s\n",
              argv[1], argv[2], mps ? 1 : 0, result, stage, error.c_str());
  return 0;
}
