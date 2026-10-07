// Publishes a model file as a shared mapping with 2 MiB pages, read from
// storage without a copy in the page cache.
//
//   weights_publish SOURCE TARGET [THREADS [CHUNK_MIB]]
//
// TARGET is a file on a tmpfs mounted with huge=always (or on hugetlbfs).
// The program sizes it like SOURCE, maps it shared and reads SOURCE into
// the mapping with O_DIRECT, THREADS readers on chunks of CHUNK_MIB. Direct
// reads put the blocks of the device into the pages of the mapping; no page
// of the page cache is filled, so the file is held once while it is loaded
// and afterwards. A copy with cp holds it in the page cache as well until
// the kernel reclaims those pages. The end of the file that is not a
// multiple of the block size is read without O_DIRECT.
//
// Prints one line: the bytes, the time, the rate and the largest drop of
// MemAvailable while loading.
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { kBlock = 4096 };

struct job {
  int fd;
  char* base;
  uint64_t direct_bytes;
  uint64_t chunk;
  uint64_t next;
  int failed;
  pthread_mutex_t lock;
};

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static long available_kib(void) {
  FILE* info = fopen("/proc/meminfo", "r");
  char line[128];
  long value = -1;
  if (info == NULL) return -1;
  while (fgets(line, sizeof(line), info) != NULL) {
    if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) break;
  }
  fclose(info);
  return value;
}

static void* reader(void* argument) {
  struct job* job = argument;
  for (;;) {
    pthread_mutex_lock(&job->lock);
    const uint64_t offset = job->next;
    job->next += job->chunk;
    pthread_mutex_unlock(&job->lock);
    if (offset >= job->direct_bytes || job->failed) return NULL;
    uint64_t left = job->direct_bytes - offset;
    if (left > job->chunk) left = job->chunk;
    uint64_t done = 0;
    while (done < left) {
      const ssize_t got = pread(job->fd, job->base + offset + done, left - done,
                                (off_t)(offset + done));
      if (got <= 0) {
        job->failed = 1;
        return NULL;
      }
      done += (uint64_t)got;
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    fprintf(stderr, "usage: weights_publish SOURCE TARGET [THREADS [CHUNK_MIB]]\n");
    return 2;
  }
  const int threads = argc > 3 ? atoi(argv[3]) : 8;
  const uint64_t chunk = (uint64_t)(argc > 4 ? atoi(argv[4]) : 32) << 20;
  if (threads < 1 || threads > 64 || chunk < kBlock) return 2;

  const int source = open(argv[1], O_RDONLY | O_DIRECT);
  struct stat status;
  if (source < 0 || fstat(source, &status) != 0) {
    perror(argv[1]);
    return 1;
  }
  const uint64_t bytes = (uint64_t)status.st_size;
  const int target = open(argv[2], O_RDWR | O_CREAT | O_TRUNC, 0444);
  if (target < 0 || ftruncate(target, (off_t)bytes) != 0) {
    perror(argv[2]);
    return 1;
  }
  char* base = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, target, 0);
  if (base == MAP_FAILED) {
    perror("mmap");
    return 1;
  }

  const long before = available_kib();
  long lowest = before;
  const double start = now_s();
  struct job job = {.fd = source,
                    .base = base,
                    .direct_bytes = bytes / kBlock * kBlock,
                    .chunk = chunk / kBlock * kBlock,
                    .next = 0,
                    .failed = 0};
  pthread_mutex_init(&job.lock, NULL);
  pthread_t workers[64];
  for (int i = 0; i < threads; ++i) {
    if (pthread_create(&workers[i], NULL, reader, &job) != 0) return 1;
  }
  // The largest drop of the available memory while the readers run.
  while (job.next < job.direct_bytes && !job.failed) {
    const long sample = available_kib();
    if (sample < lowest) lowest = sample;
    usleep(20000);
  }
  for (int i = 0; i < threads; ++i) pthread_join(workers[i], NULL);
  if (job.failed) {
    fprintf(stderr, "weights_publish: a direct read failed: %s\n", strerror(errno));
    return 1;
  }
  if (job.direct_bytes < bytes) {
    const int tail = open(argv[1], O_RDONLY);
    if (tail < 0 || pread(tail, base + job.direct_bytes, bytes - job.direct_bytes,
                          (off_t)job.direct_bytes) !=
                        (ssize_t)(bytes - job.direct_bytes)) {
      perror("tail");
      return 1;
    }
    close(tail);
  }
  const double elapsed = now_s() - start;
  const long after = available_kib();
  if (after < lowest) lowest = after;
  printf("PUBLISHED bytes=%llu seconds=%.3f gib_per_s=%.2f threads=%d chunk_mib=%llu "
         "peak_drop_mib=%ld\n",
         (unsigned long long)bytes, elapsed,
         elapsed > 0 ? (double)bytes / 1073741824.0 / elapsed : 0.0, threads,
         (unsigned long long)(chunk >> 20), (before - lowest) / 1024);
  munmap(base, bytes);
  close(target);
  close(source);
  return 0;
}
