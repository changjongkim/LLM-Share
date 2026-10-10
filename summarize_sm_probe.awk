# One row per instance from the log of run_sm_probe.sh: the count of SMs that
# the runtime reports, the largest number of 1,024-thread blocks whose kernel
# takes at most 1.5 of the time of one block (the smallest and the largest
# over the repetitions), the mean rate of the compute-only kernel, and that
# rate relative to the first instance.
/^BEGIN_SM_PROBE / {
  split($3, a, "="); mig = a[2]
  if (!(mig in seen)) { seen[mig] = 1; order[++n] = mig }
  first = 0; flat = 0
}
/^reported_sms=/ { split($1, a, "="); reported[mig] = a[2] }
/^STEP / {
  split($2, b, "="); split($3, m, "=")
  if (b[2] == 1) first = m[2]
  if (m[2] <= 1.5 * first && b[2] == flat + 1) flat = b[2]
}
/^RATE / { split($5, r, "="); rate[mig] += r[2]; runs[mig]++ }
/^END_SM_PROBE/ {
  if (!(mig in low) || flat < low[mig]) low[mig] = flat
  if (flat > high[mig]) high[mig] = flat
}
END {
  print "instance,runs,reported_sms,blocks_in_parallel_min,blocks_in_parallel_max,giga_iterations_per_s,rate_vs_first"
  for (i = 1; i <= n; i++) {
    k = order[i]
    printf "%s,%d,%d,%d,%d,%.2f,%.4f\n", k, runs[k], reported[k], low[k], high[k], rate[k] / runs[k],
           (rate[k] / runs[k]) / (rate[order[1]] / runs[order[1]])
  }
}
