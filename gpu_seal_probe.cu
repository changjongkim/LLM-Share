// K1 of PLAN-20261015: memfd seals, the immutable shared memory of the kernel,
// against the read-only mappings of STATOR. A publisher of STATOR keeps writing
// its own rows after the prefix in the same cache file and makes the rows of the
// prefix read-only in its own mapping (mprotect); an agent opens the file
// read-only. A seal applies to the whole file. This probe records what the
// seals allow, in one process, and then how the GPU reads a sealed buffer.
//
//   gpu_seal_probe SIZE_MIB
//
// Steps (one line each, SEAL step= result= errno=):
//   write_while_mapped   F_SEAL_WRITE while a shared writable mapping exists
//   future_write         F_SEAL_FUTURE_WRITE while that mapping exists
//   old_mapping_write    a CPU write through the old writable mapping after it
//   new_shared_write     a new shared writable mapping after it
//   private_write        a new private writable mapping after it (copy-on-write)
//   write_after_unmap    F_SEAL_WRITE after the writable mapping is removed
//   gpu_read             the GPU reads a shared read-only mapping of the sealed
//                        buffer after a CPU read of every page (sum checked)
#include <cuda_runtime.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef F_SEAL_FUTURE_WRITE
#define F_SEAL_FUTURE_WRITE 0x0010
#endif

namespace {

constexpr std::uint64_t kMiB = 1ULL << 20;

__global__ void sum_words(const std::uint64_t* data, std::uint64_t words, unsigned long long* out) {
  const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  unsigned long long local = 0;
  for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < words;
       i += stride) {
    local += data[i];
  }
  atomicAdd(out, local);
}

void report(const char* step, bool done) {
  std::printf("SEAL step=%s result=%s errno=%d\n", step, done ? "allowed" : "refused", done ? 0 : errno);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: gpu_seal_probe SIZE_MIB\n");
    return 2;
  }
  const std::uint64_t bytes = std::strtoull(argv[1], nullptr, 10) * kMiB;
  const int fd = memfd_create("stator-seal", MFD_ALLOW_SEALING);
  if (fd < 0 || ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    std::perror("memfd");
    return 1;
  }
  void* rw = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (rw == MAP_FAILED) return 1;
  auto* words = static_cast<std::uint64_t*>(rw);
  unsigned long long expected = 0;
  for (std::uint64_t i = 0; i < bytes / 8; ++i) {
    words[i] = i * 0x9E3779B97F4A7C15ULL;
    expected += words[i];
  }
  report("write_while_mapped", fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE) == 0);
  report("future_write", fcntl(fd, F_ADD_SEALS, F_SEAL_FUTURE_WRITE) == 0);
  {
    const std::uint64_t old = words[0];
    words[0] = old + 1;  // the old writable mapping
    const bool changed = words[0] == old + 1;
    words[0] = old;
    std::printf("SEAL step=old_mapping_write result=%s errno=0\n", changed ? "allowed" : "refused");
  }
  {
    void* w = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    report("new_shared_write", w != MAP_FAILED);
    if (w != MAP_FAILED) munmap(w, bytes);
  }
  {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    bool done = p != MAP_FAILED;
    if (done) {
      static_cast<std::uint64_t*>(p)[0] ^= 1;  // a private copy of one page
      munmap(p, bytes);
    }
    report("private_write", done);
  }
  munmap(rw, bytes);
  report("write_after_unmap", fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW) == 0);

  void* ro = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
  if (ro == MAP_FAILED) return 1;
  const volatile char* view = static_cast<const char*>(ro);
  for (std::uint64_t offset = 0; offset < bytes; offset += 4096) (void)view[offset];
  unsigned long long* sum = nullptr;
  if (cudaMalloc(&sum, sizeof(*sum)) != cudaSuccess || cudaMemset(sum, 0, sizeof(*sum)) != cudaSuccess) return 1;
  const auto started = std::chrono::steady_clock::now();
  sum_words<<<160, 256>>>(static_cast<const std::uint64_t*>(ro), bytes / 8, sum);
  const cudaError_t error = cudaDeviceSynchronize();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  unsigned long long got = 0;
  cudaMemcpy(&got, sum, sizeof(got), cudaMemcpyDeviceToHost);
  std::printf("SEAL step=gpu_read result=%s errno=0 gpu_ms=%.2f sum_ok=%d\n",
              error == cudaSuccess ? "allowed" : cudaGetErrorString(error), ms, got == expected ? 1 : 0);
  return 0;
}
