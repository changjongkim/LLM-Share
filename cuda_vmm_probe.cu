// Can a tensor be composed from a range that two processes share and a range
// that is private, in device memory, in a given placement? This is the
// device-memory counterpart of a mapped prefix with a private tail: the
// producer creates a physical allocation with the CUDA virtual-memory API,
// fills it, and exports it as a file descriptor; the consumer, a separately
// executed process, imports it, maps it at the start of a reserved address
// range, maps an allocation of its own right after it, and reads across the
// seam. The consumer then asks for read-only access to the shared range and
// tries to write it.
//
// usage: cuda_vmm_probe PRODUCER_MIG CONSUMER_MIG
#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr std::uint32_t kValue = 0x5eedf00dU;
constexpr std::uint32_t kOverwrite = 0x0badc0deU;
constexpr std::size_t kBytes = std::size_t{64U} << 20U;

struct Reply {
  int import_error = -1;
  int map_error = -1;
  int readonly_error = -1;
  int private_error = -1;
  int read_error = -1;
  int write_error = -1;
  std::uint32_t first = 0U;
  std::uint32_t before_seam = 0U;
  std::uint32_t after_seam = 0U;
};

std::string name_of(const int result) {
  const char* name = nullptr;
  cuGetErrorName(static_cast<CUresult>(result), &name);
  return name != nullptr ? name : "unknown";
}

CUresult make_context() {
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUresult status = cuInit(0U);
  if (status == CUDA_SUCCESS) status = cuDeviceGet(&device, 0);
  if (status == CUDA_SUCCESS) status = cuDevicePrimaryCtxRetain(&context, device);
  if (status == CUDA_SUCCESS) status = cuCtxSetCurrent(context);
  return status;
}

CUmemAllocationProp properties(const bool shareable) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  if (shareable) {
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  }
  return prop;
}

CUresult set_access(const CUdeviceptr address, const std::size_t bytes,
                    const CUmemAccess_flags flags) {
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = 0;
  access.flags = flags;
  return cuMemSetAccess(address, bytes, &access, 1U);
}

bool send_descriptor(const int socket, const int descriptor) {
  char payload = 'd';
  iovec part{&payload, 1U};
  char control[CMSG_SPACE(sizeof(int))] = {};
  msghdr message{};
  message.msg_iov = &part;
  message.msg_iovlen = 1U;
  message.msg_control = control;
  message.msg_controllen = sizeof(control);
  cmsghdr* header = CMSG_FIRSTHDR(&message);
  header->cmsg_level = SOL_SOCKET;
  header->cmsg_type = SCM_RIGHTS;
  header->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(header), &descriptor, sizeof(int));
  return ::sendmsg(socket, &message, 0) == 1;
}

int receive_descriptor(const int socket) {
  char payload = 0;
  iovec part{&payload, 1U};
  char control[CMSG_SPACE(sizeof(int))] = {};
  msghdr message{};
  message.msg_iov = &part;
  message.msg_iovlen = 1U;
  message.msg_control = control;
  message.msg_controllen = sizeof(control);
  if (::recvmsg(socket, &message, 0) != 1) return -1;
  const cmsghdr* header = CMSG_FIRSTHDR(&message);
  if (header == nullptr || header->cmsg_type != SCM_RIGHTS) return -1;
  int descriptor = -1;
  std::memcpy(&descriptor, CMSG_DATA(header), sizeof(int));
  return descriptor;
}

bool read_all(const int fd, void* const data, const std::size_t bytes) {
  std::size_t done = 0U;
  while (done < bytes) {
    const ssize_t got = ::read(fd, static_cast<char*>(data) + done, bytes - done);
    if (got <= 0) return false;
    done += static_cast<std::size_t>(got);
  }
  return true;
}

int run_consumer(const int socket) {
  Reply reply;
  if (make_context() != CUDA_SUCCESS) return 3;
  const int descriptor = receive_descriptor(socket);
  if (descriptor < 0) return 4;
  CUmemGenericAllocationHandle shared = 0U;
  reply.import_error = static_cast<int>(cuMemImportFromShareableHandle(
      &shared, reinterpret_cast<void*>(static_cast<std::uintptr_t>(descriptor)),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  CUdeviceptr range = 0U;
  if (reply.import_error == static_cast<int>(CUDA_SUCCESS)) {
    CUresult status = cuMemAddressReserve(&range, 2U * kBytes, 0U, 0U, 0ULL);
    if (status == CUDA_SUCCESS) status = cuMemMap(range, kBytes, 0U, shared, 0ULL);
    if (status == CUDA_SUCCESS) {
      status = set_access(range, kBytes, CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
    }
    reply.map_error = static_cast<int>(status);
  }
  if (reply.map_error == static_cast<int>(CUDA_SUCCESS)) {
    // The private range, right after the shared one.
    const CUmemAllocationProp prop = properties(false);
    CUmemGenericAllocationHandle own = 0U;
    CUresult status = cuMemCreate(&own, kBytes, &prop, 0ULL);
    if (status == CUDA_SUCCESS) {
      status = cuMemMap(range + kBytes, kBytes, 0U, own, 0ULL);
    }
    if (status == CUDA_SUCCESS) {
      status = set_access(range + kBytes, kBytes,
                          CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
    }
    if (status == CUDA_SUCCESS) {
      status = cuMemsetD32(range + kBytes, ~kValue, kBytes / sizeof(std::uint32_t));
    }
    reply.private_error = static_cast<int>(status);
  }
  if (reply.private_error == static_cast<int>(CUDA_SUCCESS)) {
    // One device read over the seam between the two ranges.
    std::uint32_t seam[2] = {0U, 0U};
    CUresult status = cuMemcpyDtoH(&reply.first, range, sizeof(reply.first));
    if (status == CUDA_SUCCESS) {
      status = cuMemcpyDtoH(seam, range + kBytes - sizeof(std::uint32_t),
                            sizeof(seam));
    }
    reply.read_error = static_cast<int>(status);
    reply.before_seam = seam[0];
    reply.after_seam = seam[1];
  }
  // The composition is reported before the write below, which may end this
  // process.
  if (::write(STDOUT_FILENO, &reply, sizeof(reply)) !=
      static_cast<ssize_t>(sizeof(reply))) {
    return 5;
  }
  if (reply.read_error != static_cast<int>(CUDA_SUCCESS)) return 0;
  // Read-only access to the shared range, then a write to it.
  reply.readonly_error = static_cast<int>(
      set_access(range, kBytes, CU_MEM_ACCESS_FLAGS_PROT_READ));
  reply.write_error = static_cast<int>(cuMemsetD32(range, kOverwrite, 1U));
  if (reply.write_error == static_cast<int>(CUDA_SUCCESS)) {
    reply.write_error = static_cast<int>(cuCtxSynchronize());
  }
  return ::write(STDOUT_FILENO, &reply, sizeof(reply)) ==
                 static_cast<ssize_t>(sizeof(reply))
             ? 0
             : 5;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::strcmp(argv[1], "--consumer") == 0) {
    return run_consumer(std::atoi(argv[2]));
  }
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s PRODUCER_MIG CONSUMER_MIG\n", argv[0]);
    return 2;
  }
  const bool mps = ::getenv("CUDA_MPS_PIPE_DIRECTORY") != nullptr;
  int sockets[2];
  int from_child[2];
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0 ||
      ::pipe(from_child) != 0) {
    return 2;
  }
  // The child is started before this process touches the GPU.
  const pid_t child = ::fork();
  if (child < 0) return 2;
  if (child == 0) {
    ::dup2(from_child[1], STDOUT_FILENO);
    ::close(from_child[0]);
    ::close(from_child[1]);
    ::close(sockets[0]);
    ::setenv("CUDA_VISIBLE_DEVICES", argv[2], 1);
    char self[4096];
    const ssize_t length = ::readlink("/proc/self/exe", self, sizeof(self) - 1U);
    if (length <= 0) ::_exit(127);
    self[length] = '\0';
    const std::string socket = std::to_string(sockets[1]);
    ::execl(self, self, "--consumer", socket.c_str(), static_cast<char*>(nullptr));
    ::_exit(127);
  }
  ::close(sockets[1]);
  ::close(from_child[1]);
  ::setenv("CUDA_VISIBLE_DEVICES", argv[1], 1);

  const char* result = "FAILED";
  const char* stage = "none";
  std::string error = "none";
  const char* readonly = "-";
  int consumer_changed = -1;
  std::size_t granularity = 0U;
  const CUmemAllocationProp prop = properties(true);
  CUmemGenericAllocationHandle handle = 0U;
  CUdeviceptr range = 0U;
  int descriptor = -1;
  CUresult status = make_context();
  if (status == CUDA_SUCCESS) {
    status = cuMemGetAllocationGranularity(&granularity, &prop,
                                           CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  }
  if (status == CUDA_SUCCESS) status = cuMemCreate(&handle, kBytes, &prop, 0ULL);
  if (status == CUDA_SUCCESS) status = cuMemAddressReserve(&range, kBytes, 0U, 0U, 0ULL);
  if (status == CUDA_SUCCESS) status = cuMemMap(range, kBytes, 0U, handle, 0ULL);
  if (status == CUDA_SUCCESS) {
    status = set_access(range, kBytes, CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
  }
  if (status == CUDA_SUCCESS) {
    status = cuMemsetD32(range, kValue, kBytes / sizeof(std::uint32_t));
  }
  if (status != CUDA_SUCCESS) {
    result = "UNSUPPORTED";
    stage = "allocate";
    error = name_of(static_cast<int>(status));
  } else if ((status = cuMemExportToShareableHandle(
                  &descriptor, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                  0ULL)) != CUDA_SUCCESS) {
    result = "UNSUPPORTED";
    stage = "export";
    error = name_of(static_cast<int>(status));
  } else {
    Reply reply;
    if (!send_descriptor(sockets[0], descriptor) ||
        !read_all(from_child[0], &reply, sizeof(reply))) {
      stage = "consumer";
      error = "no_reply";
    } else if (reply.import_error != static_cast<int>(CUDA_SUCCESS)) {
      result = "UNSUPPORTED";
      stage = "import";
      error = name_of(reply.import_error);
    } else if (reply.map_error != static_cast<int>(CUDA_SUCCESS)) {
      result = "UNSUPPORTED";
      stage = "map";
      error = name_of(reply.map_error);
    } else if (reply.private_error != static_cast<int>(CUDA_SUCCESS)) {
      stage = "private";
      error = name_of(reply.private_error);
    } else if (reply.read_error != static_cast<int>(CUDA_SUCCESS) ||
               reply.first != kValue || reply.before_seam != kValue ||
               reply.after_seam != ~kValue) {
      stage = "read";
      error = reply.read_error != static_cast<int>(CUDA_SUCCESS)
                  ? name_of(reply.read_error)
                  : "wrong_value";
    } else {
      result = "PASS";
      // The second reply follows the consumer's write to the shared range.
      Reply after;
      const bool answered = read_all(from_child[0], &after, sizeof(after));
      // Did the consumer's write reach the producer's memory?
      std::uint32_t now = 0U;
      (void)cuMemcpyDtoH(&now, range, sizeof(now));
      consumer_changed = now == kValue ? 0 : 1;
      if (!answered) {
        readonly = consumer_changed == 0 ? "enforced_consumer_ended" : "not_enforced";
      } else if (after.readonly_error != static_cast<int>(CUDA_SUCCESS)) {
        readonly = "refused";
      } else if (after.write_error != static_cast<int>(CUDA_SUCCESS) ||
                 consumer_changed == 0) {
        readonly = "enforced";
      } else {
        readonly = "not_enforced";
      }
    }
  }
  ::close(sockets[0]);
  int child_status = 0;
  ::waitpid(child, &child_status, 0);
  std::printf("probe=cuda_vmm producer=%.12s consumer=%.12s mps=%d result=%s "
              "stage=%s error=%s granularity=%zu readonly=%s "
              "consumer_changed_shared=%d\n",
              argv[1], argv[2], mps ? 1 : 0, result, stage, error.c_str(),
              granularity, readonly, consumer_changed);
  return 0;
}
