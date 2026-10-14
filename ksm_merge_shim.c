// Preloaded into a process (LD_PRELOAD): asks KSM to consider all anonymous
// memory of the process for merging, as systemd's MemoryKSM= does for a
// service (prctl PR_SET_MEMORY_MERGE, Linux 6.4). The flag passes to the
// children that the process spawns, which inherit LD_PRELOAD.
#include <stdio.h>
#include <sys/prctl.h>

#ifndef PR_SET_MEMORY_MERGE
#define PR_SET_MEMORY_MERGE 67
#endif

__attribute__((constructor)) static void ksm_merge_all(void) {
  if (prctl(PR_SET_MEMORY_MERGE, 1, 0, 0, 0) != 0) perror("ksm_merge_shim: prctl");
}
