/* Counts tracepoints and page faults over the whole system until SIGINT or
 * SIGTERM, with counting events of perf_event_open (no buffer, no loss). The
 * kernel of this device has no kprobes, no function tracer and no hist
 * triggers, so tracepoints are the paths of the kernel that can be counted.
 *
 *   tp_count GROUP:EVENT ...      (as root: the events are system-wide)
 *
 * The names `faults-minor` and `faults-major` count the software events of
 * minor and major page faults. Output at exit, one line per event:
 *   TP name=GROUP:EVENT count=N
 */
#define _GNU_SOURCE
#include <linux/perf_event.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#define MAX_EVENTS 32
#define MAX_CPUS 64

static volatile sig_atomic_t stop;

static void on_signal(int sig) {
  (void)sig;
  stop = 1;
}

static long tracepoint_id(const char *name) {
  char path[512];
  char group[128];
  const char *colon = strchr(name, ':');
  if (colon == NULL || (size_t)(colon - name) >= sizeof(group)) return -1;
  memcpy(group, name, (size_t)(colon - name));
  group[colon - name] = '\0';
  snprintf(path, sizeof(path), "/sys/kernel/tracing/events/%s/%s/id", group, colon + 1);
  FILE *f = fopen(path, "r");
  if (f == NULL) return -1;
  long id = -1;
  if (fscanf(f, "%ld", &id) != 1) id = -1;
  fclose(f);
  return id;
}

int main(int argc, char **argv) {
  if (argc < 2 || argc - 1 > MAX_EVENTS) {
    fprintf(stderr, "usage: tp_count GROUP:EVENT ... (at most %d)\n", MAX_EVENTS);
    return 2;
  }
  const long cpus = sysconf(_SC_NPROCESSORS_ONLN);
  if (cpus <= 0 || cpus > MAX_CPUS) return 2;
  static int fds[MAX_EVENTS][MAX_CPUS];
  for (int e = 1; e < argc; ++e) {
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.size = sizeof(attr);
    if (strcmp(argv[e], "faults-minor") == 0 || strcmp(argv[e], "faults-major") == 0) {
      attr.type = PERF_TYPE_SOFTWARE;
      attr.config = strcmp(argv[e], "faults-minor") == 0 ? PERF_COUNT_SW_PAGE_FAULTS_MIN
                                                          : PERF_COUNT_SW_PAGE_FAULTS_MAJ;
    } else {
      const long id = tracepoint_id(argv[e]);
      if (id < 0) {
        fprintf(stderr, "no tracepoint %s\n", argv[e]);
        return 1;
      }
      attr.type = PERF_TYPE_TRACEPOINT;
      attr.config = (uint64_t)id;
    }
    attr.disabled = 1;
    for (long c = 0; c < cpus; ++c) {
      fds[e - 1][c] = (int)syscall(SYS_perf_event_open, &attr, -1, (int)c, -1, 0);
      if (fds[e - 1][c] < 0) {
        perror(argv[e]);
        return 1;
      }
    }
  }
  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);
  for (int e = 1; e < argc; ++e) {
    for (long c = 0; c < cpus; ++c) ioctl(fds[e - 1][c], PERF_EVENT_IOC_ENABLE, 0);
  }
  printf("TP_READY\n");
  fflush(stdout);
  while (!stop) pause();
  for (int e = 1; e < argc; ++e) {
    uint64_t total = 0;
    for (long c = 0; c < cpus; ++c) {
      uint64_t value = 0;
      if (read(fds[e - 1][c], &value, sizeof(value)) == (ssize_t)sizeof(value)) total += value;
      close(fds[e - 1][c]);
    }
    printf("TP name=%s count=%llu\n", argv[e], (unsigned long long)total);
  }
  return 0;
}
