// Prototype only: after a protection change to read-only that ends inside a
// 2 MiB block, the kernel drops the huge mapping of that block. This shim puts
// its entries back from the CPU: a read of every page of the read-only part,
// and a populate for write of the rest of the block.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
static int (*real_mprotect)(void *, size_t, int);
int mprotect(void *addr, size_t len, int prot) {
  if (!real_mprotect) real_mprotect = (int (*)(void *, size_t, int)) dlsym(RTLD_NEXT, "mprotect");
  int rc = real_mprotect(addr, len, prot);
  const uintptr_t huge = (uintptr_t) 2 << 20;
  if (rc == 0 && prot == PROT_READ && len >= ((size_t) 1 << 20) && ((uintptr_t) addr % huge) == 0) {
    const uintptr_t end = (uintptr_t) addr + len;
    const uintptr_t lo = end & ~(huge - 1);
    if (end != lo) {
      for (uintptr_t p = lo; p < end; p += 4096) (void) *(volatile const char *) p;
      madvise((void *) end, lo + huge - end, MADV_POPULATE_WRITE);
    }
  }
  return rc;
}
