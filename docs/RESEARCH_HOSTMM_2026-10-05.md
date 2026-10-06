# HostMM: the host memory manager as the owner of GPU-visible memory on Thor

**Evidence cutoff:** 2026-10-05
**Platform:** NVIDIA Jetson AGX Thor, JetPack 7.2, kernel 6.8.12-1021-tegra,
open GPU kernel modules 595.78 (`uvm_ats_mode=1`, `uvm_disable_hmm=Y`),
`CONFIG_ARM_SMMU_V3_SVA=y`, MIG `2g.0gb` (12 SMs) + `1g.0gb` (6 SMs)
**Code and artifacts:** `thor_hostmm/`
**Prior-art audit:** `SOTA_HOSTMM_2026-10-05.md`
**Status:** measured and implemented; novelty verdicts in Section 9 are scoped
to the audited literature

## 1. Question and claim

On Thor a CUDA kernel can dereference ordinary pageable host memory, and the
GPU reaches that memory through the host page tables (ARM SMMUv3 shared
virtual addressing). Its MIG instances have no memory of their own. The
question is which services of the host memory manager already govern
GPU-visible memory on such a device, what they cost, and whether the costly
ones can be made cheap without changing the kernel or the driver.

The claim, with the scope that the evidence supports:

> For pageable host memory on a GPU that follows the host page tables, the
> host memory manager already accounts the memory (cgroup), revokes GPU
> access to it (`mprotect`), and gives a forked child a consistent
> copy-on-write image of it while a GPU kernel keeps writing. These services
> are correct, but every per-page event waits for a secondary-TLB
> invalidation in the IOMMU, for every page of the process, which makes a
> fork snapshot of GPU state about 160 times slower than it has to be.
> Replacing per-page copy-on-write by one copy and one remap per extent
> removes that cost in user space, and the kernel's copy-on-write remains
> the safety net. The same two facts, host page tables and the price of
> per-page events, let tenants in different MIG instances share one sealed
> copy of a model and diverge privately (Section 8), and remove most of the
> protection cost of GPU Portals when its objects use huge pages
> (Section 6.3).

Non-claims are listed in Section 9. In particular nothing here applies to
`cudaMalloc`, pinned, or managed allocations, no individual technique is
claimed as new, the per-page mechanism is intended kernel behaviour that was
known in words and not measured, and a fork snapshot is not faster than an
application-specific stop-and-copy.

## 2. What the host memory manager governs: four allocation planes

`devplane_probe` allocates 256 MiB in each plane, has a GPU kernel write it,
and applies three host operations. Both MIG instances give the same answers
(`results/20261005-hostmm-matrix-v1/allocation_planes.txt`).

| Plane | Charged to the memory cgroup | After `fork`, the child | GPU write after `mprotect(PROT_NONE)` |
|---|---|---|---|
| pageable (`mmap`, `malloc`) | yes, 256 MiB | holds a frozen image | `cudaErrorIllegalAddress` |
| pinned (`cudaMallocHost`) | yes, 257 MiB | sees the parent's later GPU writes | succeeds |
| managed (`cudaMallocManaged`) | yes, 257 MiB | has no mapping | succeeds |
| device (`cudaMalloc`) | **no, 0 MiB** | not CPU-addressable | not applicable |

Three facts follow.

1. Only pageable memory obeys all three host rules. It is the only plane in
   which the host memory manager is the sole authority.
2. `mprotect` on pinned or managed memory returns success and the GPU keeps
   writing. A design that relies on `mprotect` for revocation, as GPU Portals
   does, is sound only for pageable memory.
3. A device allocation is invisible to the memory cgroup. A 768 MiB
   `cudaMalloc` succeeds under `memory.max=512M` in both MIG instances while
   `memory.current` stays at 43 MB, and each instance reports all 131.86 GB
   of DRAM as its total. This was the open gate G0 of
   `RESEARCH_ZEROGB_2026-10-01.md`; it could not be run then because MIG was
   disabled. With MIG enabled the hole is still there.

## 3. What the services cost

`run_hostmm_matrix.sh` measures 34 cases in fresh processes: five repetitions
on each MIG instance at 256 MiB and 1 GiB (ten runs per cell), and three at
4 GiB for five cases. All 107 correctness and snapshot-integrity flag groups
pass. Costs are linear in size: between 256 MiB and 4 GiB the per-GiB figure
of first touch, kernel copy-on-write, extent privatization, and steady-state
writes stays within 16%. `fork` and `mprotect` have a fixed part that is
visible at 256 MiB.

Mean and 95% confidence half-width at 1 GiB, in milliseconds, both MIG
instances pooled:

| Operation on 1 GiB that a GPU kernel uses | ms | +/- |
|---|---:|---:|
| GPU write, steady state | 6.6 | 1.2 |
| GPU read, steady state | 20.8 | 3.0 |
| GPU first write to untouched memory, 4 KiB pages | 4628.3 | 132.5 |
| GPU first read of untouched memory, 4 KiB pages | 4319.7 | 422.6 |
| GPU first write, transparent huge pages | 4533.4 | 204.1 |
| GPU first write, shmem (memfd) | 785.0 | 78.0 |
| GPU first write, hugetlb | 106.6 | 22.8 |
| CPU populates first, 1 thread | 216.3 | 0.6 |
| CPU populates first, 12 threads | 163.8 | 35.8 |
| CPU populates first, 12 threads, huge pages | 12.9 | 1.6 |
| then the GPU first write | 6.8 | 1.2 |
| fork, 4 KiB pages | 23.9 | 2.2 |
| fork, huge pages | 1.4 | 0.2 |
| GPU read after fork | 21.9 | 3.0 |
| GPU write after fork (kernel CoW), 4 KiB | 3138.6 | 26.1 |
| GPU write after fork (kernel CoW), huge pages | 3178.7 | 19.5 |
| GPU write after fork (kernel CoW), hugetlb | 280.3 | 15.3 |
| GPU write after the child has exited | 2715.5 | 16.2 |
| CPU CoW break, 12 threads, CUDA process | 1636.0 | 253.5 |
| CPU CoW break, 12 threads, CUDA process, huge pages | 1834.5 | 290.9 |
| CPU CoW break, 12 threads, hugetlb | 74.2 | 11.5 |
| CPU write faults after the child has exited | 1716.8 | 317.3 |
| CPU CoW break, 12 threads, before CUDA is initialized | 244.3 | 71.6 |
| CPU CoW break, 1 thread, before CUDA is initialized | 457.1 | 8.1 |
| extent privatization, 4 KiB, 32 MiB extents | 143.2 | 1.5 |
| extent privatization, 4 KiB, one extent | 102.4 | 1.8 |
| extent privatization, huge pages, 32 MiB extents | 32.0 | 0.5 |
| extent privatization, huge pages, one extent | 19.2 | 0.2 |
| extent privatization, huge pages, one extent, 1 thread | 75.7 | 0.7 |
| GPU write after privatization | 6.8 | 1.2 |
| `mprotect` revoke / grant, anonymous 4 KiB | 7.4 / 17.7 | 1.0 / 3.1 |
| `mprotect` revoke / grant, memfd 4 KiB | 3.55 / 3.50 | 0.01 |
| `mprotect` revoke / grant, hugetlb memfd | 0.03 / 0.02 | 0.00 |
| GPU write after revoke and grant, memfd or hugetlb memfd | 6.6 | 1.2 |
| `MADV_DONTNEED` or `munmap` of GPU-used memory | 96 to 101 | 7.4 |

Reading the table:

- **Copy-on-write is correct for the GPU.** In every fork case the child's
  image stayed intact while the parent's kernel overwrote the memory, on both
  MIG instances.
- **Per-page events are three orders of magnitude more expensive than range
  operations.** A GPU write after fork costs 3,139 ms/GiB, about 12 us per
  4 KiB page. An `mprotect` over the same memory costs 3.5 to 18 ms/GiB, 13 to
  68 ns per page.
- **Huge pages do not rescue copy-on-write.** The kernel splits a shared
  transparent huge page on a write fault and copies 4 KiB at a time
  (3,179 ms). hugetlb pages are copied whole and bring the cost to 280 ms, but
  they need a statically reserved pool of twice the state. With a pool of 600
  pages for a 512-page region the snapshot holder did not survive, which is
  the documented hugetlb behaviour when the pool cannot supply the copy: the
  page is taken from the child. With 1,100 pages the image stayed intact.
- **Doing the copy-on-write from the CPU does not help** (1,636 ms with twelve
  threads) and neither does waiting for the child to exit (2,716 ms): the
  cost is not the copy.
- **Extent privatization costs 19.2 ms/GiB**: 163 times less than kernel
  copy-on-write, 14.6 times less than hugetlb copy-on-write, 3.9 times less
  than hugetlb with a CPU pre-break, with no reserved pool and no privilege.
- **First touch** from the GPU costs 4.6 s/GiB and 13 ms/GiB if the CPU
  populates huge pages first. This part is known for other host-page-table
  GPUs (Section 9) and is listed for completeness.

### 3.1 Where the per-page cost comes from

`notifier_scope_probe` forks and then breaks copy-on-write from the CPU with
twelve threads on 1 GiB regions (`results/20261005-side-probes-v1/`, five
repetitions on each MIG instance, mean and 95% confidence half-width):

| Region | Child during the break | ms per GiB | +/- |
|---|---|---:|---:|
| before CUDA is initialized | alive, pages are copied | 292.7 | 51.5 |
| CUDA initialized, region never passed to a kernel | alive | 1,704.9 | 297.1 |
| CUDA initialized, region written by a kernel | alive | 1,712.9 | 202.9 |
| the never-used region after a second `fork` | alive | 1,671.4 | 224.1 |
| before CUDA is initialized | gone, nothing to copy | 213.2 | 26.1 |
| CUDA initialized, region never passed to a kernel | gone | 1,637.6 | 168.7 |

A region that the GPU never touched pays the full cost, 5.8 times the cost
before initialization. A break that copies nothing, because the child has
already gone, pays it as well, 7.7 times. The cost is therefore a property of
the process, not of GPU-mapped ranges, and not of copying.

A kernel profile shows what it is (`results/20261005-smmu-trace-v1/`,
`perf record -g -e cpu-clock`, one thread, three passes). Without CUDA the
break takes 465 ms/GiB and the profile is page-table and copy work
(`ptep_clear_flush` 22%, `copy_page` 13%, `clear_page` 11%). In a CUDA process
it takes 2,908 ms/GiB and **73.5% of all samples are in
`arm_smmu_cmdq_issue_cmdlist`**, reached through

```text
do_wp_page -> ptep_clear_flush
  -> __mmu_notifier_arch_invalidate_secondary_tlbs
  -> arm_smmu_mm_arch_invalidate_secondary_tlbs
       -> arm_smmu_tlb_inv_range_asid -> arm_smmu_cmdq_issue_cmdlist   51.2%
       -> arm_smmu_atc_inv_domain     -> arm_smmu_cmdq_issue_cmdlist   22.2%
```

The same function dominates when the fault comes from the GPU
(`results/20261005-gpu-fault-trace-v1/`, system-wide profile, 1 GiB). One
kernel thread of the UVM driver serves GPU faults, so 90% of all samples are
idle CPUs. Of the samples that are not idle:

| Case | ms per GiB in this run | Share in `arm_smmu_cmdq_issue_cmdlist` | Reached from `uvm_ats_service_faults` through |
|---|---:|---:|---|
| steady GPU write, the control | 5.0 | 0.7% | no fault is served |
| first GPU write to untouched memory | 4,742 | 63.2% | `uvm_migrate_pageable -> migrate_vma_setup -> migrate_vma_collect_pmd -> ptep_clear_flush` |
| first GPU write after `fork` | 3,099 | 51.8% | `uvm_populate_pageable_vma -> handle_mm_fault -> do_wp_page -> ptep_clear_flush` |

The first-touch row needs a note. An anonymous page fault flushes nothing by
itself. The driver populates the faulting pages and then runs a migration
pass over them, and that pass clears every page-table entry with a flush.
Populating from the CPU first avoids the GPU fault altogether, which is why
it costs 13 ms/GiB instead of 4.6 s/GiB.

Once a process is bound to the GPU for shared virtual addressing, every flush
of a single page anywhere in its address space makes the IOMMU driver submit
a TLB invalidation and an ATS invalidation to the SMMU and wait for each of
them. The surcharge is 5 to 9 us per page from the CPU (twelve threads and
one thread) and the whole GPU fault costs 12 us per page. Threads do not
help much: 2,908 ms with one thread and 1,705 ms with twelve. This explains
every slow row of the table: copy-on-write from either side, the write
faults after the child has exited, and first touch from the GPU. A range
operation makes one notifier call for the whole range.

Three qualifications.

- **This is intended kernel behaviour.** Since Linux 6.6 the architecture
  flush functions notify secondary TLBs, so that a device never keeps a stale
  translation, and an IOMMU maintainer wrote in 2023 that `fork`, `mmap`, and
  `munmap` become slower once shared virtual addressing is enabled
  (`SOTA_HOSTMM_2026-10-05.md`, Section 11). What the audit did not find is a
  measurement, a profile, or a GPU case.
- **The number of submissions per page in this kernel is not known.**
  Upstream 6.8 submits twice per flushed page, once for the TLB and once for
  ATS. The driver source of the Tegra kernel is not installed on the device,
  and mainline 7.3 doubles the TLB submission on this SoC for an erratum.
  The figures above are totals and are not the cost of one SMMU command.
- **The path is not specific to Linux 6.8.** In mainline 7.3-rc6 source a
  copy-on-write fault still handles one page, its flush still reaches the
  secondary-TLB notifier, and the SMMUv3 callback still submits synchronously
  per call. Only 6.8.12-tegra was measured; the magnitude on a newer kernel
  is not.

The consequence reaches beyond snapshots: any CUDA process on this platform
that forks, or whose pages the kernel migrates, compacts, or reclaims one at a
time, pays the same per-page cost on all of its memory. Section 6.1 measures
one such case and its repair.

### 3.2 The tax on ordinary memory management

`run_mm_tax.sh` measures twelve memory-management operations on memory that
no kernel ever touches, in a fresh process before and after CUDA is
initialized (`results/20261005-mm-tax-v1/`, six paired repetitions that
alternate between the MIG instances, 1 GiB, and 128 MiB for the operations
that make one call per page). The ratio is the geometric mean of the paired
ratios.

| Operation | Threads | Runs | Before CUDA (ms/GiB) | After CUDA (ms/GiB) | Ratio | 95% CI | Added per flush (us) |
|---|---:|---:|---:|---:|---:|---|---:|
| copy-on-write break of every page, child alive | 1 | 6 | 465.8 | 2862.5 | 6.15x | [5.98, 6.33] | 9.14 |
| copy-on-write break of every page, child alive | 2 | 6 | 245.4 | 1768.9 | 7.21x | [7.01, 7.42] | 5.81 |
| copy-on-write break of every page, child alive | 4 | 6 | 140.1 | 1101.6 | 7.87x | [7.71, 8.03] | 3.67 |
| copy-on-write break of every page, child alive | 8 | 6 | 223.3 | 1203.0 | 5.60x | [2.92, 10.75] | 3.74 |
| copy-on-write break of every page, child alive | 12 | 6 | 308.8 | 1628.2 | 5.35x | [4.01, 7.14] | 5.03 |
| write fault on every page after the child has gone | 1 | 6 | 298.3 | 2674.7 | 8.97x | [8.78, 9.16] | 9.06 |
| write fault on every page after the child has gone | 12 | 6 | 211.2 | 1714.4 | 8.09x | [5.78, 11.33] | 5.73 |
| `mprotect` of one page, per call | 1 | 6 | 1477.9 | 6318.9 | 4.28x | [4.02, 4.55] | 9.23 |
| `MADV_DONTNEED` of one page, per call | 1 | 6 | 459.0 | 2823.2 | 6.16x | [5.86, 6.47] | 9.02 |
| split of a huge page mapping, per huge page | 1 | 6 | 12.6 | 20.7 | 1.64x | [1.57, 1.71] | 15.71 |
| `MADV_COLLAPSE` into huge pages, per huge page | 1 | 6 | 368.4 | 354.8 | 0.96x | [0.91, 1.02] |  |
| `fork`, 4 KiB pages | 1 | 6 | 21.5 | 19.0 | 0.89x | [0.81, 0.97] |  |
| `fork`, transparent huge pages | 1 | 6 | 0.7 | 1.4 | 1.89x | [1.71, 2.08] |  |
| `mprotect` of the range, two calls | 1 | 6 | 30.5 | 22.3 | 0.72x | [0.61, 0.86] |  |
| `MADV_DONTNEED` of the range, one call | 1 | 6 | 108.4 | 81.2 | 0.75x | [0.67, 0.84] |  |
| `munmap` of the range | 1 | 6 | 109.4 | 87.9 | 0.80x | [0.69, 0.93] |  |
| `mremap` of the range | 1 | 6 | 0.4 | 3.8 | 8.67x | [7.34, 10.24] |  |

Reading the table:

- **Operations that flush one page at a time become 4.3 to 9.0 times
  slower.** The added cost per flush is the same for four different
  operations with one thread: 9.0 to 9.2 us for a copy-on-write break, a
  write fault that copies nothing, `mprotect` of one page, and
  `MADV_DONTNEED` of one page.
- **Threads do not buy it back.** After initialization a copy-on-write break
  costs 2,862, 1,769, and 1,102 ms/GiB with one, two, and four threads, and
  then rises again to 1,203 and 1,628 ms/GiB with eight and twelve. The best
  rate is about 240,000 single-page flushes per second for the whole
  process.
- **Operations that cover a range in one call are not taxed.** `fork` of
  4 KiB pages and `mprotect`, `MADV_DONTNEED`, and `munmap` of the range are
  not slower; they are 11 to 28% faster after initialization, which was not
  investigated. `MADV_COLLAPSE` is unchanged.
- **Three operations pay a small fixed amount**: `mremap` of 1 GiB 3.3 ms
  per call, `fork` of huge pages 0.6 ms, and the split of a huge mapping
  16 us.

Thus, the rule for a process that uses the GPU on this platform is
mechanical: an operation pays about 9 us for every separate single-page
flush it causes, in the whole address space, and nothing measurable for
flushing a range. Everything in Sections 4 to 8 that is fast is a way of
turning the first kind of operation into the second.

## 4. Mechanism

### 4.1 Extent privatization

To make an extent exclusively owned without per-page copy-on-write
(`extent_cow.cpp`, `gpu_snapshot.cpp`):

1. copy the extent into a staging mapping with CPU threads; reading shared
   pages faults nothing;
2. `mremap(staging, MREMAP_FIXED, extent)`: one range invalidation replaces
   the mapping, and the old pages stay alive for whoever still maps them.

Copy-then-remap is the user-space copy-on-write of RUMA and AnKer. What is
specific here is the reason to use it, the IOMMU invalidation per page, and
two details that this cost forces:

- A staging area that is populated in advance must be excluded from `fork`
  with `MADV_DONTFORK` and re-included with `MADV_DOFORK` after it is in
  place. Otherwise the fork shares the staging pages with the child and the
  copy into them faults page by page. The first version of the runtime made
  this mistake and took 1,673 ms instead of 32 ms for 2 GiB.
- The copy of all extents runs on one team of threads. Starting threads per
  32 MiB extent cost more than the copy.

### 4.2 Snapshot sessions

`SnapshotSession::begin` forks a consumer that owns the frozen image and
privatizes extents according to a policy:

| Policy | Privatizes at the snapshot |
|---|---|
| `kernel_cow` | nothing; the kernel resolves every later write per page |
| `privatize_all` | every registered byte |
| `privatize_declared` | the regions the application declared mutable |
| `privatize_learned` | the extents that earlier snapshots saw written |

Correctness does not depend on the policy or on a correct prediction. The
kernel write-protects every private page at fork, for the CPU and for the
GPU, and copies whatever is written while the child lives. A wrong prediction
costs time and cannot corrupt the image. This is stronger than interception
with validated speculation, which must detect a misprediction and retry.

### 4.3 Learning the write set without dirty tracking

This kernel has neither `userfaultfd` nor soft-dirty bits, so nothing reports
which pages a GPU kernel wrote. The runtime uses copy-on-write itself as the
detector:

- The last 4 KiB page of a privatized extent, the sentinel, is left shared
  with the child. If the GPU writes the extent, the kernel copies the
  sentinel and `/proc/self/pagemap` reports it exclusively mapped. One
  per-page event per extent is the whole cost.
- An extent that was not privatized reports its written pages the same way,
  and their count is the number of pages that took the slow path.
- An extent whose sentinel was written n times in a row is privatized without
  a sentinel for the next 2^n - 1 snapshots (at most 15), so a stable writer
  pays for the check rarely. A silent sentinel demotes the extent.
- The sharing state is read on a background thread while the child is kept
  alive, so ending a snapshot does not pause the workload.

### 4.4 Tests

`gpu_snapshot_test` (CPU-only, because copy-on-write and the pagemap sharing
state do not depend on who writes) checks, with and without huge pages and
with and without spare staging: that every policy hands the consumer the
pre-snapshot image and leaves the live state correct; that the learned policy
converges to the written extents, counts kernel copies exactly, and follows a
moved write set; the backoff schedule; the non-blocking end; a consumer that
fails or is killed; and rejection of invalid configurations. It found one
defect before any GPU run: a consumer that died took the parent down with
`SIGPIPE`; the channel is now a socket pair written with `MSG_NOSIGNAL`.

## 5. End-to-end snapshots

### 5.1 Campaign

`run_snapshot_campaign.sh` takes periodic snapshots of a running GPU workload
and measures what the workload loses. Six repetitions on each MIG instance
give twelve runs per cell; the strategy order rotates with the repetition.
The campaign has 276 runs and 1,004 snapshot images
(`results/20261005-snapshot-v1/`). Every image is checked against the content
that the state had at the snapshot, and the live state is checked at the end
of the run. All 276 runs are correct. In four statevector runs the circuit
ended before the third snapshot could start; time lost is reported per
snapshot taken.

Workloads:

- **dense**: a kernel rewrites all 2 GiB of state in every step;
- **rotating**: a kernel rewrites one of eight 256 MiB blocks per step;
- **sparse**: a kernel reads 8 GiB of weights and rewrites 256 MiB of state
  in every step; only the state is declared mutable;
- **statevector**: a 28-qubit simulation (2 GiB) built from the gate kernels
  of the published circuit suite, with the VQC, QSVM, and VQE circuits
  rotating over the repetitions. Images are compared bit for bit with a
  replay of the same gate prefix.

Strategies:

- `stop_copy_all`, `stop_copy_declared`: the workload stops and twelve CPU
  threads copy all registered memory, or the declared-mutable part, into a
  shadow buffer that is allocated once. This is what an application-specific
  checkpointer does;
- `kernel_cow`: `fork`, and the kernel resolves every later write;
- `privatize_all`, `privatize_declared`, `privatize_learned`: `fork`, then
  extent privatization under the policies of Section 4.2.

Metrics: the duration of the snapshot call, during which the workload cannot
launch a kernel; the slowest step; and the time lost per snapshot, which is
the run time beyond the same number of undisturbed steps, divided by the
number of snapshots. The last metric includes everything that happens after
the call returns.

One asymmetry favours stop-and-copy. A fork snapshot's consumer reads its
whole image and then holds it for 200 ms while the workload runs, so its
memory traffic is part of the time lost. The stop-and-copy image is checked
after the run, so nothing runs beside the workload.

### 5.2 Results

Mean and 95% confidence half-width over twelve runs.

**dense writer, 2 GiB state**

| Strategy | Runs | Step (ms) | Snapshot call (ms) | First / last call (ms) | Slowest step (ms) | Time lost per snapshot (ms) | Pages left to kernel CoW |
|---|---:|---:|---:|---:|---:|---:|---:|
| `stop_copy_all` | 12 (12 correct) | 24.7 | 19.2 +/- 0.2 | 19.1 / 19.1 | 44.3 +/- 1.9 | 19.1 +/- 0.2 | 0 |
| `stop_copy_declared` | 12 (12 correct) | 24.7 | 19.2 +/- 0.2 | 19.6 / 19.1 | 44.5 +/- 1.9 | 19.1 +/- 0.2 | 0 |
| `kernel_cow` | 12 (12 correct) | 24.7 | 33.8 +/- 0.4 | 3.3 / 45.6 | 6375.5 +/- 30.9 | 6465.6 +/- 27.7 | 2097152 |
| `privatize_all` | 12 (12 correct) | 24.7 | 35.3 +/- 0.2 | 43.3 / 32.6 | 77.5 +/- 1.6 | 52.9 +/- 1.1 | 0 |
| `privatize_declared` | 12 (12 correct) | 24.7 | 35.2 +/- 0.1 | 43.1 / 32.6 | 77.2 +/- 1.5 | 52.9 +/- 1.5 | 0 |
| `privatize_learned` | 12 (12 correct) | 24.7 | 46.6 +/- 0.5 | 56.7 / 37.3 | 104.0 +/- 3.7 | 75.6 +/- 2.5 | 0 |

**rotating writer, 2 GiB state in 8 blocks**

| Strategy | Runs | Step (ms) | Snapshot call (ms) | First / last call (ms) | Slowest step (ms) | Time lost per snapshot (ms) | Pages left to kernel CoW |
|---|---:|---:|---:|---:|---:|---:|---:|
| `stop_copy_all` | 12 (12 correct) | 3.1 | 19.1 +/- 0.4 | 19.0 / 19.5 | 22.9 +/- 1.2 | 19.1 +/- 0.4 | 0 |
| `stop_copy_declared` | 12 (12 correct) | 3.1 | 19.2 +/- 0.4 | 19.3 / 19.6 | 23.1 +/- 1.3 | 19.3 +/- 0.4 | 0 |
| `kernel_cow` | 12 (12 correct) | 3.1 | 33.0 +/- 1.5 | 3.3 / 44.6 | 851.3 +/- 5.4 | 5538.2 +/- 28.9 | 1313000 |
| `privatize_all` | 12 (12 correct) | 3.1 | 34.8 +/- 0.2 | 42.3 / 32.2 | 48.6 +/- 0.5 | 132.9 +/- 14.6 | 0 |
| `privatize_declared` | 12 (12 correct) | 3.1 | 34.7 +/- 0.2 | 42.3 / 32.2 | 48.8 +/- 0.4 | 111.3 +/- 16.0 | 0 |
| `privatize_learned` | 12 (12 correct) | 3.1 | 46.1 +/- 0.6 | 56.3 / 37.1 | 69.3 +/- 2.4 | 135.5 +/- 20.4 | 0 |

**8 GiB read-only weights + 256 MiB state**

| Strategy | Runs | Step (ms) | Snapshot call (ms) | First / last call (ms) | Slowest step (ms) | Time lost per snapshot (ms) | Pages left to kernel CoW |
|---|---:|---:|---:|---:|---:|---:|---:|
| `stop_copy_all` | 12 (12 correct) | 18.4 | 77.9 +/- 0.4 | 78.5 / 77.7 | 97.3 +/- 4.4 | 77.8 +/- 0.3 | 0 |
| `stop_copy_declared` | 12 (12 correct) | 18.3 | 2.9 +/- 0.0 | 2.9 / 2.9 | 21.3 +/- 4.4 | 2.9 +/- 0.1 | 0 |
| `kernel_cow` | 12 (12 correct) | 18.4 | 12.1 +/- 0.3 | 7.5 / 13.9 | 856.3 +/- 6.7 | 860.7 +/- 5.3 | 262144 |
| `privatize_all` | 12 (12 correct) | 18.4 | 133.4 +/- 0.4 | 165.4 / 123.3 | 194.2 +/- 3.6 | 179.8 +/- 9.5 | 0 |
| `privatize_declared` | 12 (12 correct) | 18.3 | 11.9 +/- 0.1 | 13.1 / 11.6 | 32.7 +/- 4.2 | 21.0 +/- 2.0 | 0 |
| `privatize_learned` | 12 (12 correct) | 18.4 | 78.6 +/- 0.8 | 213.4 / 26.7 | 274.1 +/- 9.9 | 116.2 +/- 4.0 | 0 |

**statevector, 28 qubits (2 GiB), VQC/QSVM/VQE**

| Strategy | Runs | Step (ms) | Snapshot call (ms) | First / last call (ms) | Slowest step (ms) | Time lost per snapshot (ms) | Pages left to kernel CoW |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none` | 12 (12 correct) | 19.7 | 0.0 +/- 0.0 | 0.0 / 0.0 | 21.2 +/- 2.0 | 0.0 +/- 0.0 | 0 |
| `stop_copy` | 12 (12 correct) | 19.6 | 19.2 +/- 0.2 | 19.2 / 19.2 | 39.8 +/- 1.9 | 26.3 +/- 2.6 | 0 |
| `kernel_cow` | 12 (12 correct) | 19.6 | 29.5 +/- 1.2 | 29.5 / 29.5 | 6521.0 +/- 78.9 | 6541.9 +/- 52.1 | 1572864 |
| `privatize_all` | 12 (12 correct) | 19.6 | 35.9 +/- 0.5 | 35.9 / 35.9 | 69.2 +/- 3.1 | 59.0 +/- 2.6 | 0 |
| `privatize_learned` | 12 (12 correct) | 19.6 | 49.8 +/- 0.7 | 49.8 / 49.8 | 90.5 +/- 4.1 | 84.5 +/- 3.8 | 0 |

Paired ratios (same repetition and MIG instance, geometric mean, 95%
confidence interval of the log-ratio). A ratio above one means that the first
strategy loses more.

| Workload | Comparison (first / second) | Metric | Runs | Ratio | 95% CI |
|---|---|---|---:|---:|---|
| dense | `kernel_cow` / `privatize_all` | slowest step | 12 | 82.33x | [80.61, 84.09] |
| dense | `kernel_cow` / `privatize_all` | time lost per snapshot | 12 | 122.20x | [119.41, 125.05] |
| dense | `kernel_cow` / `privatize_learned` | slowest step | 12 | 61.39x | [59.44, 63.40] |
| dense | `kernel_cow` / `privatize_learned` | time lost per snapshot | 12 | 85.67x | [82.82, 88.62] |
| dense | `stop_copy_all` / `privatize_learned` | slowest step | 12 | 0.43x | [0.40, 0.45] |
| dense | `stop_copy_all` / `privatize_learned` | time lost per snapshot | 12 | 0.25x | [0.24, 0.26] |
| dense | `stop_copy_all` / `privatize_all` | slowest step | 12 | 0.57x | [0.56, 0.58] |
| dense | `stop_copy_all` / `privatize_all` | time lost per snapshot | 12 | 0.36x | [0.35, 0.37] |
| dense | `stop_copy_declared` / `privatize_declared` | slowest step | 12 | 0.58x | [0.56, 0.59] |
| dense | `stop_copy_declared` / `privatize_declared` | time lost per snapshot | 12 | 0.36x | [0.35, 0.37] |
| dense | `privatize_declared` / `privatize_learned` | slowest step | 12 | 0.74x | [0.71, 0.78] |
| dense | `privatize_declared` / `privatize_learned` | time lost per snapshot | 12 | 0.70x | [0.69, 0.71] |
| rotating | `kernel_cow` / `privatize_all` | slowest step | 12 | 17.52x | [17.31, 17.73] |
| rotating | `kernel_cow` / `privatize_all` | time lost per snapshot | 12 | 42.27x | [37.70, 47.39] |
| rotating | `kernel_cow` / `privatize_learned` | slowest step | 12 | 12.30x | [11.87, 12.74] |
| rotating | `kernel_cow` / `privatize_learned` | time lost per snapshot | 12 | 42.03x | [35.83, 49.29] |
| rotating | `stop_copy_all` / `privatize_learned` | slowest step | 12 | 0.33x | [0.31, 0.35] |
| rotating | `stop_copy_all` / `privatize_learned` | time lost per snapshot | 12 | 0.15x | [0.12, 0.17] |
| rotating | `stop_copy_all` / `privatize_all` | slowest step | 12 | 0.47x | [0.45, 0.49] |
| rotating | `stop_copy_all` / `privatize_all` | time lost per snapshot | 12 | 0.15x | [0.13, 0.16] |
| rotating | `stop_copy_declared` / `privatize_declared` | slowest step | 12 | 0.47x | [0.45, 0.50] |
| rotating | `stop_copy_declared` / `privatize_declared` | time lost per snapshot | 12 | 0.18x | [0.15, 0.20] |
| rotating | `privatize_declared` / `privatize_learned` | slowest step | 12 | 0.71x | [0.68, 0.73] |
| rotating | `privatize_declared` / `privatize_learned` | time lost per snapshot | 12 | 0.83x | [0.70, 0.98] |
| sparse | `kernel_cow` / `privatize_all` | slowest step | 12 | 4.41x | [4.32, 4.51] |
| sparse | `kernel_cow` / `privatize_all` | time lost per snapshot | 12 | 4.80x | [4.56, 5.06] |
| sparse | `kernel_cow` / `privatize_learned` | slowest step | 12 | 3.13x | [3.02, 3.24] |
| sparse | `kernel_cow` / `privatize_learned` | time lost per snapshot | 12 | 7.42x | [7.16, 7.68] |
| sparse | `stop_copy_all` / `privatize_learned` | slowest step | 12 | 0.35x | [0.34, 0.37] |
| sparse | `stop_copy_all` / `privatize_learned` | time lost per snapshot | 12 | 0.67x | [0.65, 0.70] |
| sparse | `stop_copy_all` / `privatize_all` | slowest step | 12 | 0.50x | [0.48, 0.52] |
| sparse | `stop_copy_all` / `privatize_all` | time lost per snapshot | 12 | 0.43x | [0.41, 0.46] |
| sparse | `stop_copy_declared` / `privatize_declared` | slowest step | 12 | 0.63x | [0.58, 0.69] |
| sparse | `stop_copy_declared` / `privatize_declared` | time lost per snapshot | 12 | 0.14x | [0.13, 0.15] |
| sparse | `privatize_declared` / `privatize_learned` | slowest step | 12 | 0.12x | [0.10, 0.13] |
| sparse | `privatize_declared` / `privatize_learned` | time lost per snapshot | 12 | 0.18x | [0.17, 0.19] |
| qsim | `kernel_cow` / `privatize_all` | slowest step | 12 | 94.47x | [89.70, 99.50] |
| qsim | `kernel_cow` / `privatize_all` | time lost per snapshot | 12 | 111.09x | [106.43, 115.96] |
| qsim | `kernel_cow` / `privatize_learned` | slowest step | 12 | 72.21x | [68.43, 76.19] |
| qsim | `kernel_cow` / `privatize_learned` | time lost per snapshot | 12 | 77.64x | [74.13, 81.32] |
| qsim | `stop_copy` / `privatize_all` | slowest step | 12 | 0.58x | [0.54, 0.61] |
| qsim | `stop_copy` / `privatize_all` | time lost per snapshot | 12 | 0.44x | [0.39, 0.50] |
| qsim | `stop_copy` / `privatize_learned` | slowest step | 12 | 0.44x | [0.41, 0.47] |
| qsim | `stop_copy` / `privatize_learned` | time lost per snapshot | 12 | 0.31x | [0.27, 0.35] |

### 5.3 Reading the results

- **A fork snapshot with kernel copy-on-write is not usable.** The call
  returns in 3 to 46 ms, and the workload then stalls for 6.5 s per snapshot
  with 2 GiB of dense state, 5.5 s with a rotating writer, and 0.86 s when
  only 256 MiB are written. This is the per-page cost of Section 3.1 at
  2.8 to 3.4 s per GiB written.
- **Extent privatization removes that cost.** The time lost per snapshot is
  122 times lower for the dense writer, 111 times lower for the statevector
  simulation, 42 times lower for the rotating writer, and 4.8 times lower for
  the sparse model even when all 8.25 GiB are copied. With the mutable region
  declared, the sparse model loses 21.0 ms instead of 860.7 ms. No page took
  the per-page path under any privatizing policy.
- **The learned policy converges without any declaration.** On the sparse
  model its first call copies everything and takes 213 ms; its last call
  takes 26.7 ms, against 77.7 ms for a copy of everything and 11.6 ms for the
  policy that was told the mutable region. For dense state it costs 43% more
  than `privatize_all` (75.6 against 52.9 ms): the sentinel pages are
  per-page events, and there is nothing to learn.
- **An application-specific stop-and-copy is cheaper at these sizes.** It
  loses 19.1 ms against 52.9 ms for dense state, 26.3 against 59.0 ms for the
  statevector, and 2.9 against 21.0 ms for the declared sparse state. Both
  mechanisms copy the same bytes. The fork snapshot also forks, zeroes fresh
  pages for the copy, and after the call shares the memory bus with its
  consumer and with the refill of its staging area. The rotating writer shows
  the last part most clearly: its call takes 34.8 ms and it loses 132.9 ms,
  because its 3 ms steps are bound by memory bandwidth. Over four snapshots
  the learned policy also loses more than a copy of everything (116.2 against
  77.8 ms), because its first call dominates; Section 5.4 measures more
  snapshots at a larger size.

Thus, the measured result is not a faster checkpoint than a hand-written one.
It is that the fork snapshot, which needs no knowledge of the state layout
and whose image is a whole process, moves from seconds to tens of
milliseconds, within a factor of 2.2 to 7.3 of a hand-written copy in this
campaign.

### 5.4 Scale check

`run_scale_check.sh` repeats two cases at a larger size on the 2g instance,
three repetitions each (`results/20261005-scale-v1/`, 27 runs, all images
and final states correct): the sparse model with 32 GiB of weights and 1 GiB
of state over eight snapshots, and a 30-qubit statevector (8 GiB) over two.
The learned policy runs without spare staging so that its first, full copy
peaks at twice the model.

| Workload | Strategy | Runs | Step (ms) | Snapshot call, first / last (ms) | Slowest step (ms) | Time lost per snapshot (ms) |
|---|---|---:|---:|---:|---:|---:|
| 33 GiB sparse model | `stop_copy_all` | 3 | 49.9 | 312.0 / 313.4 | 367.0 | 312.6 +/- 24.3 |
| 33 GiB sparse model | `stop_copy_declared` | 3 | 50.7 | 9.6 / 9.5 | 58.6 | 7.5 +/- 5.2 |
| 33 GiB sparse model | `kernel_cow` | 3 | 51.6 | 22.7 / 47.6 | 3,285.2 | 3,281.9 +/- 42.5 |
| 33 GiB sparse model | `privatize_declared` | 3 | 50.0 | 42.2 / 38.5 | 98.2 | 66.3 +/- 23.7 |
| 33 GiB sparse model | `privatize_learned` | 3 | 50.5 | 829.4 / 83.3 | 886.4 | 244.4 +/- 14.4 |
| 30-qubit statevector | `none` | 3 | 72.5 | | 74.9 | |
| 30-qubit statevector | `stop_copy` | 3 | 71.9 | 79.2 / 79.2 | 153.0 | 92.2 +/- 6.5 |
| 30-qubit statevector | `kernel_cow` | 3 | 71.9 | 62.3 / 62.3 | 25,287.3 | 25,471.8 +/- 565.1 |
| 30-qubit statevector | `privatize_all` | 3 | 72.0 | 140.0 / 140.0 | 278.4 | 275.4 +/- 31.6 |

- **Kernel copy-on-write scales with the bytes written**: 25.5 s per
  snapshot of 8 GiB and 3.3 s per snapshot with 1 GiB of mutable state.
  Extent privatization loses 275 ms and 66 ms, 92 and 50 times less.
- **At 33 GiB the learned policy overtakes a copy of everything.** Its first
  call copies the whole model in 829 ms and its later calls take 83 ms,
  against 313 ms for every stop-and-copy of everything. Over eight snapshots
  it loses 244 ms per snapshot against 313 ms, and it needs no second copy
  of the model between snapshots. At 8 GiB and four snapshots
  (Section 5.3) it was still behind.
- **A checkpointer that knows the mutable region stays ahead**: 7.5 ms
  against 66.3 ms for the fork snapshot that was told the same region, and
  92 ms against 275 ms for the statevector.

## 6. Consequences outside snapshots

### 6.1 The latent cost of `fork` followed by `exec`, and its repair

A CUDA process that starts a helper program forks. The child calls `exec` at
once and shares nothing afterwards, but every private page of the parent has
been write-protected by the `fork`, and its next write takes a fault that
flushes one page. `heal_probe` measures this for 1 GiB of state that a GPU
kernel writes (`results/20261005-side-probes-v1/`, five repetitions on each
MIG instance):

| | ms | +/- |
|---|---:|---:|
| GPU write, steady state | 6.6 | 1.2 |
| first GPU write after the child has gone | 2,703.5 | 17.6 |
| one privatization pass over the state after the child has gone | 20.6 | 0.2 |
| first GPU write after that pass | 6.8 | 1.2 |

One call that starts a helper program costs 2.7 s per GiB of GPU-written
state at the next kernel, with no sharing left to justify it. One
privatization pass removes the cost at 20.6 ms per GiB, 131 times less,
because it replaces 262,144 single-page flushes by one remap.

The cost can also be avoided instead of repaired. `fork_guard_probe` starts
`/bin/true` from a CUDA process with 1 GiB of GPU-written state in four ways
(`results/20261005-mm-tax-v1/fork_guard_summary.csv`, six repetitions that
alternate between the MIG instances):

| Helper started by | Runs (correct) | Start of the helper, including the repair (ms) | Next GPU write of 1 GiB (ms) |
|---|---:|---:|---:|
| `fork` and `exec` | 6 (6) | 4.0 | 2680.0 |
| `MADV_DONTFORK` around the `fork` | 6 (6) | 3.0 | 6.8 |
| `posix_spawn` | 6 (6) | 1.0 | 6.6 |
| `fork` and `exec`, then one privatization pass | 6 (6) | 24.7 | 6.8 |

`posix_spawn` does not copy the address space, and `MADV_DONTFORK` keeps the
state out of the copy, so neither write-protects it. Both leave the next
kernel at its steady cost. `MADV_DONTFORK` for this purpose is existing
practice for RDMA-registered memory, where its reason is pinning. A program
that must call `fork` itself, and whose child needs the state, is left with
a repair. The kernel's own bulk repair, `MADV_POPULATE_WRITE`, takes the
per-page path: 2,675 ms/GiB with one thread and 1,714 ms/GiB with twelve in
a CUDA process (Section 3.2, write fault after the child has gone), against
20.6 ms/GiB for the privatization pass.

### 6.2 Registering memory does not take it away from the host

`cudaMallocHost` memory is outside the host's revocation and copy-on-write
(Section 2). `registered_probe` asks whether `cudaHostRegister` moves
ordinary pageable memory into the same class. It does not. On both MIG
instances, for private and for shared anonymous mappings, a registered
region behaves like an unregistered one: a GPU write after
`mprotect(PROT_NONE)` fails with `cudaErrorIllegalAddress`, a forked child
of a private mapping keeps a frozen image, and a forked child of a shared
mapping sees the parent's writes (16 cases,
`results/20261005-side-probes-v1/side_probes_summary.txt`).

Thus, a tenant cannot escape `mprotect` revocation of a pageable object by
registering it. This closes a question that GPU Portals left open about its
revocation primitive.

### 6.3 Portal objects on huge pages

GPU Portals revokes and grants access to an object with `mprotect` and
mapping calls on a `memfd`. Section 3 measured that the same call on a
hugetlbfs `memfd` is about a hundred times cheaper (0.03 against 3.5 ms per
GiB), because the object has 512 times fewer page-table entries.
`portal_hugetlb_variant.cu` rebuilds the published Portal benchmarks with
one changed flag (`MFD_HUGETLB` at object creation); no pinned source is
edited.

**Statevector pipeline** (`results/20261005-portal-hugetlb-v1/`, the
published 24-qubit pipeline, 128 MiB object, two handoffs per round, 30
rounds, six paired repetitions with balanced mode order):

| Objects | Mode | Runs | Round (ms) | p99 (ms) | Forward + return handoff (ms) | Gates (ms) |
|---|---|---:|---:|---:|---:|---:|
| 4 KiB shmem | copy | 6 | 17.859 | 19.619 | 14.653 | 3.207 |
| 4 KiB shmem | portal | 6 | 8.054 | 9.204 | 5.084 | 2.970 |
| 4 KiB shmem | cross_mig_shared_unprotected | 6 | 2.993 | 3.380 | 0.000 | 2.993 |
| 2 MiB hugetlbfs | copy | 6 | 17.498 | 19.455 | 14.443 | 3.054 |
| 2 MiB hugetlbfs | portal | 6 | 3.125 | 3.631 | 0.310 | 2.814 |
| 2 MiB hugetlbfs | cross_mig_shared_unprotected | 6 | 2.865 | 3.209 | 0.000 | 2.865 |

| Comparison | Scope | Metric | Runs | Ratio | 95% CI |
|---|---|---|---:|---:|---|
| base / huge | copy | mean | 6 | 1.021x | [1.006, 1.035] |
| base / huge | copy | p99 | 6 | 1.008x | [0.998, 1.019] |
| base / huge | portal | mean | 6 | 2.582x | [2.380, 2.800] |
| base / huge | portal | p99 | 6 | 2.551x | [2.268, 2.869] |
| base / huge | cross_mig_shared_unprotected | mean | 6 | 1.045x | [0.943, 1.158] |
| base / huge | cross_mig_shared_unprotected | p99 | 6 | 1.057x | [0.922, 1.212] |
| base / huge | portal | handoff | 6 | 18.569x | [10.410, 33.120] |
| copy / portal | base | mean | 6 | 2.218x | [2.125, 2.316] |
| portal / unprotected | base | mean | 6 | 2.692x | [2.557, 2.834] |
| portal / unprotected | base | p99 | 6 | 2.721x | [2.587, 2.862] |
| copy / portal | huge | mean | 6 | 5.611x | [5.204, 6.049] |
| portal / unprotected | huge | mean | 6 | 1.090x | [0.989, 1.201] |
| portal / unprotected | huge | p99 | 6 | 1.128x | [0.960, 1.325] |

**Authority campaigns** (`results/20261005-portal-hugetlb-suite-v1/`, the
published VQC, QSVM, and VQE circuits at 22 qubits with their cross-principal
policies and the authority scheduler, both backends, six paired
repetitions):

| Backend | Circuit | Portal, 4 KiB (ms) | Portal, 2 MiB (ms) | Unprotected, 4 KiB (ms) | Portal 4 KiB / 2 MiB | Portal / unprotected, 4 KiB | Portal / unprotected, 2 MiB |
|---|---|---:|---:|---:|---|---|---|
| in-place kernels | VQC | 18.04 | 18.10 | 16.72 | 1.003x [0.876, 1.149] | 1.079x [1.022, 1.139] | 1.098x [0.948, 1.272] |
| in-place kernels | QSVM | 11.15 | 10.23 | 10.54 | 1.090x [0.971, 1.223] | 1.058x [0.984, 1.136] | 0.990x [0.931, 1.052] |
| in-place kernels | VQE | 18.28 | 17.07 | 16.88 | 1.071x [1.000, 1.147] | 1.082x [1.038, 1.129] | 1.016x [0.964, 1.070] |
| cuStateVec | VQC | 15.66 | 14.74 | 14.58 | 1.063x [1.003, 1.127] | 1.074x [1.044, 1.106] | 0.988x [0.957, 1.021] |
| cuStateVec | QSVM | 10.68 | 9.54 | 9.48 | 1.119x [1.076, 1.163] | 1.127x [1.076, 1.180] | 1.013x [0.973, 1.054] |
| cuStateVec | VQE | 15.84 | 14.60 | 14.38 | 1.085x [1.047, 1.124] | 1.101x [1.072, 1.132] | 1.017x [0.995, 1.040] |

Reading the two campaigns:

- **The 4 KiB rows reproduce the published pipeline** (Portal 8.05 ms, copy
  17.86 ms, unprotected 2.99 ms here; 7.78, 17.57, and 2.99 ms in the Portal
  record), so both page sizes are compared from the published baseline.
- **Huge-page objects remove most of the handoff.** The two handoffs of a
  round fall from 5.08 ms to 0.31 ms, and a Portal round from 8.05 ms to
  3.13 ms: 2.58 times faster, 95% CI [2.38, 2.80].
- **The price of protection falls from 2.69 to 1.09 times.** Portal over
  unprotected sharing is 2.69 times [2.56, 2.83] on 4 KiB pages and 1.09
  times [0.99, 1.20] on huge pages. The second interval contains one: with
  six repetitions the remaining price is not distinguishable from zero. The
  speedup over the copy baseline rises from 2.22 to 5.61 times.
- **Where handoffs are a small part of the work, the gain is small.** In the
  authority campaigns a circuit makes two handoffs in 10 to 18 ms. With
  cuStateVec the Portal circuit becomes 6 to 12% faster, and the price of
  protection, 7 to 13% on 4 KiB pages, is no longer distinguishable from
  zero for any of the three circuits. With the in-place kernels QSVM and VQE
  move in the same direction, VQC does not move, and no interval excludes
  one.
- **Modes without permission changes do not benefit**: copy and unprotected
  sharing change by at most 5%, and only the copy mean of the pipeline
  (1.02 times) excludes one.

The cost is a hugetlb pool: it is reserved by a privileged user, it is
static, the reserved memory is unavailable to everything else, and object
sizes become multiples of 2 MiB. Object creation fails when the pool is
exhausted, so a deployment needs the 4 KiB path as a fallback.
Section 6.4 tests transparent huge pages of shmem as an alternative without
a pool.

Thus, for the statevector pipeline the 2.61 times protection cost that the
Portal record reports as its largest is a property of 4 KiB page tables, not
of the revoke-before-grant protocol.

That a permission change or a remap costs per page-table entry, and that
larger mappings reduce it, is known: C4 uses 2 MiB mappings to make remapping
cheap inside a garbage collector, and Linux moves and protects whole
page-table levels and large folios for the same reason
(`SOTA_HOSTMM_2026-10-05.md`, Section 12.1). The result here is a
measurement of that effect for revoke-before-grant transfer of GPU-visible
memory between MIG instances, where it had not been measured.

### 6.4 Portal objects on transparent huge pages of shmem

A hugetlb pool is privileged and static. shmem also has transparent huge
pages, which need no pool; their policy is `never` on this system.
`run_portal_shmem_thp.sh` runs the unmodified published pipeline binary with
the policy set to `never` and to `always`, in six paired repetitions
(`results/20261005-portal-shmem-thp-v1/`). Under `always`, at least 168 MiB
of shmem were mapped by huge pages while a run held its 128 MiB object, which
more than one process maps; under `never`, none.

| Objects | Portal round (ms) | Handoffs per round (ms) | Portal / unprotected |
|---|---:|---:|---|
| shmem, policy `never` | 7.682 | 4.789 | 2.714x [2.650, 2.780] |
| shmem, policy `always` | 4.919 | 2.139 | 1.720x [1.623, 1.824] |
| hugetlbfs (Section 6.3) | 3.125 | 0.310 | 1.090x [0.989, 1.201] |

Transparent huge pages make the Portal round 1.56 times faster
[1.51, 1.62], about half of what hugetlbfs gives, with no pool and no change
to the program. The 2.1 ms of handoff that remain were not profiled. The
policy is system-wide and affects every shmem user; confining it to the
object with the `advise` policy and `MADV_HUGEPAGE` was not tested.

## 7. Accounting: the memory cgroup as the limit of a MIG tenant

A memoryless MIG instance has no memory limit of its own: each instance
reports all DRAM as its total. Section 2 showed that three of the four
allocation planes are charged to the memory cgroup. `run_accounting_probe.sh`
tests enforcement, the cost of each plane, and a way to bring device
allocations under the limit (`results/20261005-accounting-v1/`).

**Enforcement.** A 768 MiB allocation that a GPU kernel writes, in a
transient systemd scope with `MemoryMax=512M` and no swap, on each MIG
instance:

| Allocation | 2g instance | 1g instance |
|---|---|---|
| pageable (`mmap`) | killed by the cgroup | killed by the cgroup |
| pinned (`cudaMallocHost`) | killed by the cgroup | killed by the cgroup |
| managed (`cudaMallocManaged`) | killed by the cgroup | killed by the cgroup |
| device (`cudaMalloc`) | allocated and written, 0.0 MiB charged | allocated and written, 0.0 MiB charged |
| device, with the shim, runtime linked statically | allocated, 0.0 MiB charged | allocated, 0.0 MiB charged |
| device, runtime linked dynamically | allocated, 0.2 MiB charged | allocated, 0.0 MiB charged |
| device, with the shim, runtime linked dynamically | killed by the cgroup | killed by the cgroup |

**The shim.** `libcuda_memcg_shim.so` is an `LD_PRELOAD` interposer of 47
lines that redirects `cudaMalloc` and `cuMemAlloc_v2` to the managed plane,
which is charged. It works only where interposition can reach the call. A
program that links the CUDA runtime dynamically is stopped by the limit under
the shim. A program built with `nvcc` defaults links the runtime statically
and is not affected, and neither is cuStateVec: `libcustatevec.so.1` imports
no allocation symbol (`metadata.txt`). The shim is therefore accounting for
cooperative programs of one linkage class and not a mechanism.

**Cost of the planes.** GPU write and read of 1 GiB, five repetitions on
each MIG instance:

| Plane | First GPU write (ms) | Steady GPU write (ms) | Steady GPU read (ms) |
|---|---:|---:|---:|
| device | 5.3 | 5.2 | 20.7 |
| device, runtime linked dynamically | 5.3 | 5.2 | 20.7 |
| device redirected by the shim (managed) | 271.8 | 5.0 | 20.8 |
| managed | 263.3 | 4.9 | 20.7 |
| pageable, populated by the CPU | 6.7 | 6.7 | 20.8 |
| pinned | 6.3 | 6.2 | 22.6 |

**A vendor library under the shim.** The published cuStateVec circuit suite,
unmodified, with and without the shim (22 qubits, Portal mode, five paired
repetitions):

| Circuit | Without the shim (ms) | With the shim (ms) | Validation failures |
|---|---:|---:|---:|
| bell | 4.03 | 4.18 | 0 |
| ghz8 | 14.10 | 13.88 | 0 |
| clifford16 | 28.61 | 28.91 | 0 |
| qft4 | 27.70 | 28.16 | 0 |

Reading the three parts:

- **The memory cgroup is an enforced limit for three planes on both MIG
  instances.** A tenant whose GPU data is pageable, pinned, or managed is
  bounded by `memory.max`, although its MIG instance reports all DRAM.
- **Device allocations escape it.** 768 MiB are allocated and written under
  a 512 MiB limit with nothing charged. The same bypass is reported for
  another unified-memory GPU (ROCm issue 6370); the fact that is specific to
  this platform is that a MIG instance adds no limit of its own.
- **Steady-state GPU throughput does not depend on the plane for reads**
  (20.7 to 22.6 ms/GiB). Writes to pageable memory are 28% slower than to
  device memory (6.7 against 5.2 ms/GiB). A managed allocation pays 263 ms
  per GiB once, at its first GPU write.
- **The vendor library is unchanged under the shim** because the shim does
  not reach it. The four timings differ by at most 4% in either direction.

Thus, what can be said is narrower than "the cgroup limits GPU memory". The
host limit bounds exactly the planes that the host memory manager owns. A
tenant that must be bounded has to keep its GPU data out of the device
plane, and a library that allocates device memory through its own runtime,
as cuStateVec does, stays outside. Closing that gap needs the driver or an
interception layer of the kind that HAMi and KRYPTON provide.

## 8. Copy-on-write sharing across MIG instances

The two MIG instances have separate compute and no memory of their own. If
both follow the host page tables, the host MMU can give two tenants one
physical copy of a model and a private copy of only what each tenant changes.
Section 8.1 tests whether plain `MAP_PRIVATE` semantics deliver this, and
finds one way in which they fail. Section 8.2 measures a design that cannot
fail in that way.

### 8.1 Private mappings of one model

`mig_cow_share` maps one model into a tenant in each MIG instance.

A controller writes a 4 GiB model into a `memfd` once. Two tenants are
started with `exec`, one per MIG instance, because CUDA does not support a
forked child that does not `exec`. Each tenant maps the model and a GPU
kernel in its instance reads all of it. Tenant A then overwrites
256 MiB of the model in eight 32 MiB extents from a GPU kernel, and both
tenants read the model again. The mapping is one of:

- **separate copies**: each tenant reads the file into its own anonymous
  memory;
- **shared mapping** (`MAP_SHARED`): one copy, no isolation;
- **private mapping** (`MAP_PRIVATE`) with three ways of bringing the pages
  in: the GPU faults them in, the tenant calls `MADV_POPULATE_READ`, or the
  tenant's CPU reads one byte of every page;
- for the last of these, the patch is resolved either by the kernel per page
  or by **extent privatization** of the patched extents before the kernel
  writes them.

`run_cow_share.sh` runs the six cases six times; the MIG instance that
hosts tenant A alternates and the case order rotates
(`results/20261005-cow-share-v1/`). Memory is the proportional set size of
both tenants, which divides a shared page among the processes that map it.

| Tenants map the model as | Runs | CPU step (ms) | First GPU read (ms) | Steady GPU read (ms) | Memory of both tenants before / after the patch (MiB) | Patch (ms) | Per GiB patched (ms) | A sees its patch | B still sees the original |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| separate copies | 6 | 0 | 83 | 82.6 +/- 0.1 | 8314 / 8314 | 1.9 +/- 0.5 | 8 | 6/6 | 6/6 |
| shared mapping, CPU read first (no isolation) | 6 | 242 | 89 | 88.9 +/- 0.1 | 4218 / 4218 | 658.4 +/- 4.9 | 2634 | 6/6 | 0/6 |
| private mapping, GPU faults first | 6 | 0 | 3156 | 88.8 +/- 0.1 | 8314 / 8314 | 1.9 +/- 0.5 | 7 | 6/6 | 6/6 |
| private mapping, `MADV_POPULATE_READ` first | 6 | 588 | 11172 | 89.0 +/- 0.1 | 8315 / 8315 | 1.9 +/- 0.4 | 8 | 6/6 | 6/6 |
| private mapping, CPU read first; kernel copy-on-write | 6 | 240 | 89 | 89.0 +/- 0.2 | 4218 / 4474 | 728.8 +/- 7.9 | 2915 | 6/6 | 6/6 |
| private mapping, CPU read first; extent privatization | 6 | 240 | 89 | 88.8 +/- 0.3 | 4218 / 4474 | 45.0 +/- 2.5 | 180 | 6/6 | 6/6 |

Reading the table:

- **One copy serves both instances.** With private mappings whose pages the
  CPU read first, the two tenants occupy 4,218 MiB instead of 8,314 MiB.
  Kernels in both instances read the model in 88.9 ms per pass, 8% slower
  than from a private anonymous copy on transparent huge pages (82.6 ms),
  because the file pages are 4 KiB.
- **Divergence is private and costs only the patched bytes.** After tenant
  A's kernel has overwritten 256 MiB, A reads its patch and B reads the
  original in all six runs, and memory grows by 256 MiB to 4,474 MiB. With a
  shared mapping B reads A's patch in all six runs.
- **Extent privatization makes the divergence 16 times cheaper.** The kernel
  resolves it at 2.9 s per GiB, extent privatization at 180 ms per GiB
  (728.8 against 45.0 ms for 256 MiB).
- **The first GPU access decides whether anything is shared.** If the GPU
  faults the pages in, the first read takes 3.2 s and afterwards each tenant
  holds a private copy of the whole model, 8,314 MiB as with separate
  copies. `MADV_POPULATE_READ` before the first GPU read does not help: that
  read takes 11.2 s and the copies are again private. Only when the CPU has
  read every page does the GPU read without faulting (89 ms, the same as
  its steady read) and the copy stay shared. In every variant the isolation
  holds; what is lost is the sharing.

A system-wide kernel profile of the first GPU read explains the two
failures (`results/20261005-share-fault-trace-v1/`, 1 GiB model). The driver
serves a GPU fault in a writable mapping as a write, also when the kernel
only reads:

- pages that are not mapped yet take the write path of a private file
  mapping (`uvm_ats_service_faults -> uvm_populate_pageable_vma ->
  handle_mm_fault -> copy_user_highpage`), so each page is copied into the
  tenant at its first GPU read;
- pages that `MADV_POPULATE_READ` has mapped still fault on the GPU, and the
  service then takes `handle_mm_fault -> do_wp_page -> ptep_clear_flush`: a
  copy-on-write break per page with its IOMMU invalidation (59.5% of the
  samples that are not idle are in `arm_smmu_cmdq_issue_cmdlist`). This is
  why the variant is the slowest.

After a CPU read of every page the GPU does not fault, and the profiled run
shows one copy: 1,146 MiB for both tenants against 2,170 MiB in the other
two variants.

The driver and kernel sources explain both observations
(`SOTA_HOSTMM_2026-10-05.md`, Section 12, which quotes the lines):

- In the driver's fault service for host-page-table memory, a writable
  mapping has the pages of the prefetch region added to the write mask and
  every page of the write mask serviced as a write
  (`uvm_ats_faults.c`, `if (vma->vm_flags & VM_WRITE)`). This is public
  source, and an issue thread of 2026-09-28 on another NVIDIA unified-memory
  system describes the same lines and the same loss of a private mapping.
  It is not a finding of this record.
- The GPU faults on a mapped page whose accessed flag is clear. In Linux 6.8
  the context descriptor for shared virtual addressing does not let the SMMU
  set that flag, and the kernel maps part of the pages around a file fault
  without it. `MADV_POPULATE_READ` marks the page structure accessed and not
  the page-table entry, so the entries it relies on stay without the flag. A
  CPU access sets the flag in hardware.

Section 8.2 tests the second point by clearing the flag directly.

Thus, copy-on-write sharing of GPU-visible state across MIG instances works
on this platform with ordinary `MAP_PRIVATE` semantics, under one rule that
the measurements impose: the GPU must never fault on a writable page that is
to stay shared. The CPU read that establishes this costs 240 ms for 4 GiB
with one thread.

### 8.2 A sealed base with explicit divergence

Section 8.1 leaves two problems. A tenant that holds the model's descriptor
can map it shared and writable and change the model under the other tenant.
A writable private mapping also loses its sharing whenever the GPU faults on
it. `shared_model_bench` implements a design without either problem
(`results/20261005-shared-model-v1/`, 114 runs, none failed):

- **Sealed base.** The controller seals the `memfd` against writes,
  resizing, and further seals before any tenant starts. On this kernel a
  populated hugetlbfs file refuses the write seal with `EBUSY`; the
  future-write seal is used there, which is equivalent once no writable
  mapping is left.
- **Read-only private mappings.** A tenant maps the base `MAP_PRIVATE` and
  `PROT_READ`. For a GPU fault in such a mapping the driver cannot obtain
  write access, so the fault costs time and cannot copy.
- **Explicit divergence.** Before a kernel writes an extent, the tenant
  replaces it by a private copy with one copy and one `mremap`
  (Section 4.1). A kernel that writes an extent that was not replaced fails
  with `cudaErrorIllegalAddress` and leaves the base untouched.

**Attach modes.** Two tenants, a 4 GiB model, six repetitions per row,
tenant 0 alternating between the MIG instances. A run passes if every tenant
first reads the original, tenant 0 then reads its divergence, every other
tenant still reads the original, and the base is unchanged at the end.
Attach is the time from starting the tenant process to a mapped model.

| Base | Tenants attach by | Tenants | Runs (all checks pass) | Attach (ms) | First GPU read (ms) | Steady GPU read (ms) | Memory of all tenants before / after divergence (MiB) | Divergence of 256 MiB (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shmem | private copy per tenant | 2 | 6 (6) | 365 | 83 | 82.6 +/- 0.1 | 8314 / 8314 | 2.0 +/- 0.6 |
| shmem | writable private mapping, CPU read first | 2 | 6 (6) | 34 | 89 | 88.9 +/- 0.2 | 4218 / 4474 | 43.1 +/- 3.9 |
| shmem | read-only private mapping, GPU faults first | 2 | 6 (6) | 2 | 2576 | 88.8 +/- 0.2 | 4218 / 4474 | 17.2 +/- 0.7 |
| shmem | read-only private mapping, CPU read first | 2 | 6 (6) | 34 | 89 | 88.8 +/- 0.1 | 4218 / 4474 | 44.5 +/- 3.8 |
| hugetlbfs | read-only private mapping, GPU faults first | 2 | 6 (6) | 2 | 532 | 82.6 +/- 0.0 | 4218 / 4474 | 10.0 +/- 0.3 |
| hugetlbfs | read-only private mapping, CPU read first | 2 | 6 (6) | 5 | 83 | 82.6 +/- 0.1 | 4218 / 4474 | 9.7 +/- 0.4 |

- With the CPU read first, a tenant attaches in 34 ms against 365 ms for a
  private copy, and in 5 ms on a hugetlbfs base.
- Without the CPU read the first GPU read takes 2.6 s (0.5 s on hugetlbfs),
  and, unlike the writable mapping of Section 8.1, the base stays shared:
  4,218 MiB for both tenants.
- On a hugetlbfs base the GPU reads at the speed of a private copy (82.6 ms
  per pass). On shmem it is 7.5% slower (88.8 ms).
- Divergence of 256 MiB costs 44 ms on shmem and 10 ms on hugetlbfs, and
  memory grows by the diverged bytes.

**More tenants.** Three repetitions per cell, shmem base, CPU read first.

| Tenants | Private copies: memory (MiB) | Sealed base: memory (MiB) | Ratio | Private copies: attach per tenant (ms) | Sealed base: attach per tenant (ms) |
|---:|---:|---:|---:|---:|---:|
| 2 | 8314 | 4218 | 1.97x | 365 | 34 |
| 4 | 16621 | 4333 | 3.84x | 304 | 32 |
| 8 | 33234 | 4562 | 7.28x | 281 | 32 |

A further tenant costs its process overhead, about 57 MiB, instead of a copy
of the model.

**Tenants reading at the same time.** `shared_model_concurrent` asks every
tenant to read the whole 4 GiB model in the same instant
(`results/20261005-shared-concurrent-v1/`, six repetitions, shmem base):

| Tenants | Attach by | Read alone (ms) | Read at the same time (ms) | Memory of all tenants (MiB) |
|---:|---|---:|---:|---:|
| 2, one per MIG instance | private copy | 82.6 | 82.7 +/- 0.1 | 8,314 |
| 2, one per MIG instance | sealed base | 88.9 | 90.1 +/- 0.1 | 4,218 |
| 4, two per MIG instance | private copy | 82.6 | 174.6 +/- 0.1 | 16,621 |
| 4, two per MIG instance | sealed base | 88.9 | 192.2 +/- 0.3 | 4,333 |

Reading the same physical pages from both MIG instances at once adds 1.4% to
a tenant's read. The remaining gap to private copies, 9 to 10%, is the
page-size gap that the solo reads already show and that a hugetlbfs base
removes for solo reads; concurrent reads of a hugetlbfs base were not
measured. Two tenants in one MIG instance are time-sliced and take twice as
long in either case.

**The accessed state.** One tenant, 1 GiB, five repetitions on each MIG
instance. After steady reads the tenant calls `MADV_COLD` on the mapping,
which clears the accessed state of its pages and leaves them mapped, and the
GPU reads again.

| Tenant attaches by | Runs (all checks pass) | Steady GPU read (ms) | GPU read after `MADV_COLD` (ms) | Next GPU read (ms) | Growth of the tenant's anonymous memory (MiB) |
|---|---:|---:|---:|---:|---:|
| writable private mapping, CPU read first | 10 (10) | 20.8 | 2841.1 +/- 21.2 | 20.8 | 1024.0 |
| read-only private mapping, CPU read first | 10 (10) | 20.8 | 217.8 +/- 26.8 | 20.8 | 0.0 |
| read-only private mapping, GPU faults first | 10 (10) | 20.8 | 225.0 +/- 34.3 | 20.8 | 0.0 |

The writable mapping loses everything: the GPU faults on every page, the
read takes 2.8 s, and the tenant ends with a private copy of the whole model.
This reproduces the failure of Section 8.1 by direct manipulation, which
confirms that a cleared accessed state is what makes the GPU fault. It also
shows that the CPU read of Section 8.1 is not a durable remedy: anything that
clears the accessed state undoes it. The read-only mapping pays 0.22 s once
and keeps the base shared.

**One page is enough for a whole block.** The driver upgrades its 2 MiB
prefetch region, so the damage is not limited to the pages whose flag is
clear. `af_block_probe` clears the accessed state of chosen 4 KiB pages of a
1 GiB private mapping and reads the model from the GPU
(`results/20261005-af-block-v1/`, six repetitions that alternate between the
MIG instances, every run identical in the copied amount):

| Mapping | Pages whose accessed state is cleared | GPU read afterwards (ms) | Private copies created (MiB) | Pages copied per cleared page |
|---|---:|---:|---:|---:|
| writable | none | 20.8 | 0 | |
| writable | 1 | 25.0 | 2 | 512 |
| writable | 64, one in every eighth 2 MiB block | 383.3 | 128 | 512 |
| writable | 512, one in every 2 MiB block | 2,804.2 | 1,024 | 512 |
| writable | all 262,144 | 2,838.4 | 1,024 | 1 |
| read-only | 512, one in every 2 MiB block | 268.5 | 0 | 0 |
| read-only | all 262,144 | 237.8 | 0 | 0 |

One page without the accessed flag turns its whole 2 MiB block into private
copies. Clearing the flag on one page per block, 0.2% of the pages, is
enough to turn the entire model into a private copy at the next GPU read.
A read-only mapping copies nothing in either case. No source that the three
audit passes opened reports this amplification.

**Attempts to modify the base.** Nine attempts on a shmem base and on a
hugetlbfs base, on both MIG instances (36 runs,
`shared_model_attacks.txt`):

| Attempt | Outcome | Base afterwards |
|---|---|---|
| map the base shared and writable | refused (`EPERM`) | intact |
| `mprotect` a shared read-only mapping to writable | refused (`EACCES`) | intact |
| `pwrite` | refused (`EPERM`; `EINVAL` on hugetlbfs) | intact |
| `ftruncate` to zero | refused (`EPERM`) | intact |
| punch a hole | refused (`EPERM`) | intact |
| add seals again | refused (`EPERM`) | intact |
| GPU kernel writes through a shared read-only mapping | `cudaErrorIllegalAddress` | intact |
| GPU kernel writes through a private read-only mapping | `cudaErrorIllegalAddress` | intact |
| GPU kernel writes through a private writable mapping | succeeds on a private copy | intact |

Thus, neither a tenant's CPU nor its GPU kernels can change what another
tenant reads, the memory of N tenants is one base plus what each has
diverged, and no kernel or driver event can silently turn the base into
private copies. The price is that a tenant must name the extents it will
write: an undeclared write stops the tenant instead of copying.

Limits of this section. The device has no swap, and reclaim was exercised
only through `MADV_COLD`; a model in a regular file was not tested. All
tenants run under one user, so channels such as `ptrace` are outside the
attempts. The workload is a read of the whole model, not an inference
engine.

### 8.3 The same base under time slicing and MPS

Nothing in Section 8.2 depends on MIG: the host MMU isolates processes.
`sharing_modes_probe` repeats it with all tenants time-sliced in one MIG
instance, as clients of one MPS server, and with an MPS server in each MIG
instance (`results/20261005-sharing-modes-v1/`, 174 runs, none failed).
Memory, divergence, isolation, and the accessed-flag amplification are the
same in every configuration: 4.1 to 4.5 GiB for 2 to 8 tenants of the
shared base against 8.1 to 32.5 GiB for copies. What differs is how long
tenants wait for each other and what a GPU fault of one tenant does to the
others: clients of the faulting client's MPS server do not survive (0 of
18), time-sliced processes and the clients of an MPS server in the other
MIG instance do (36 of 36). `RESEARCH_LLM_SHARE_2026-10-05.md`, Section 5,
has the tables and builds on them.

### 8.4 The same rule in an inference engine

Sections 8.1 and 8.2 use a synthetic kernel that reads a model. On
2026-10-06 the two findings behind them were repeated in llama.cpp, for the
key-value cache of a prompt prefix that one process computes and others
continue (`RESEARCH_LLM_SHARE_2026-10-05.md`, Section 6).

The amplification of Section 8.1 is what a private writable mapping does to
a shared prefix there. Eight agents that map the cache file of a
16,321-token prefix (893 MiB) writable and private, without a CPU read pass,
each end with a private copy of the whole prefix although none of them
writes it: 12.7 GiB against 5.6 GiB, and the first token after 4.9 to 5.2 s
against 1.7 s.

The remedy of Section 8.2, an immutable base mapped read-only with what a
tenant writes placed in memory that was private from the start, is built
into the engine's cache: the rows of the prefix are a read-only mapping of
the publisher's file and the rows an agent writes are a private tail. Eight
agents hold 5.7 GiB instead of 19.9 GiB with copies and write the texts of
agents that copy, and a running process hands its state to four children
with a pause of 3 ms.

## 9. Novelty boundary

`SOTA_HOSTMM_2026-10-05.md` audits more than one hundred and fifty works in
three passes, with a verification tag on every row. The second pass covers
the IOMMU literature and kernels after 6.8; the third attacks the huge-page,
sharing, GPU-fault, and fork-repair results. Every facet of this record has
prior work that takes part of it. What follows separates the part that is
claimed from the part that is cited.

### 9.1 The claim set

Four claims are made. Each is a measurement or a design in a setting in which
the three passes found no measurement or design of that kind, and none is a
new technique.

1. **A sealed base with private divergence across MIG instances.** Tenants
   in different MIG instances read one sealed physical copy of GPU-visible
   state through read-only private mappings and diverge by replacing the
   extents they will write. Measured: memory, attach time, read throughput
   alone and at the same time, divergence cost, tenant scaling, and nine
   attempts to modify the base (Section 8.2). Prior work shares read-only
   state in place through shared mappings and gives up private mappings
   because the GPU read copies them; private views that stay shared and
   diverge, a sealed base, and MIG instances are the part not found.
2. **Why a private mapping stops being shared, and how little it takes.**
   The write-intent fault service is public source and was described before
   this record. The parts not found are the condition under which sharing
   survives (every entry present with the accessed flag set), the
   insufficiency of `MADV_POPULATE_READ`, the reproduction by clearing the
   flag, and the amplification: one page without the flag costs a 2 MiB
   block, and 0.2% of the pages cost the whole model (Sections 8.1 and 8.2).
3. **The memory-management tax of binding a process to the GPU.** The
   mechanism is intended kernel behaviour and was predicted in words. The
   parts not found are any measurement for a GPU: about 9 us per single-page
   flush across four operations, no tax on range operations, saturation at
   four threads, the process-wide scope, the kernel profiles of both fault
   sides, the 2.7 s per GiB that `fork` leaves behind, and its three
   remedies measured against the kernel's own repair (Sections 3.1, 3.2,
   and 6.1).
4. **Two uses of range operations on GPU-visible memory.** Huge-page objects
   reduce the protection cost of revoke-before-grant transfer between MIG
   instances from 2.69 to 1.09 times (Section 6.3), and extent privatization
   makes a fork snapshot of GPU-written memory lose 42 to 122 times less
   than kernel copy-on-write (Section 5). Both apply known effects; the
   measurements in this setting are the part not found.

Not claimed, and cited wherever it is used: that GPU first touch is slow and
CPU pre-population removes it; copy-and-remap as a technique; region-granular
copy-on-write; cgroup accounting of GPU memory or its bypass by device
allocations; that per-page operations should be batched; read-only in-place
sharing of weights across processes; and a snapshot that beats an
application-specific stop-and-copy.

### 9.2 Facet by facet

| Facet | Verdict in the audited set | Closest prior work | What remains ours |
|---|---|---|---|
| Per-page cost of memory-management events in a process bound to a GPU by shared virtual addressing | partially taken: known in words, not measured | an IOMMU maintainer's statement of 2023 that `fork`, `mmap`, and `munmap` become slower once SVA is enabled; the 6.6 change that put the secondary-TLB notifier into the flush functions; the 6.19 arm64 patch that measures the CPU-TLB analogue after `fork`; Border Control, which found the cost negligible in simulation | the measurement on a GPU: the tax table of Section 3.2 (about 9 us per single-page flush for four operations, no tax on range operations, saturation at four threads), the kernel profiles of both fault sides, and the process-wide scope shown on memory the GPU never touched |
| GPU first touch is slow and CPU pre-population removes it | taken | Grace Hopper study; MI300A study | only the attribution on this platform: the driver's migration pass flushes each page it has just populated |
| Kernel copy-on-write gives a consistent image of memory a GPU kernel is writing, with no interception and no driver change | open as an observation and measurement; not a mechanism | CRUM (fork-based checkpoint for UVM, with a proxy and a drain copy); NVIDIA's HMM notes and Caldera say fork and GPUs do not combine | the demonstration on a host-page-table GPU, on both MIG instances, with 1,004 verified images |
| Extent privatization | partially taken; the technique is not new | RUMA and AnKer (user-space copy-and-remap); CCoW (region-granular copy-on-write in the kernel, for fault count and CPU TLB shootdowns); MSched (predict, batch, fault fallback) | the reason for it (N synchronous secondary-TLB invalidations become one), the measured ratio, the `MADV_DONTFORK` staging rule that the process-wide cost forces, and the property that a wrong prediction cannot corrupt the image |
| The cost that `fork` plus `exec` leaves behind in a GPU process, and its remedies | partially taken: the mechanism is kernel source, the remedies exist | the unmerged kernel series that reuses a whole exclusive large folio on a write fault; `MADV_POPULATE_WRITE` as the kernel's bulk repair; the RDMA practice of `MADV_DONTFORK` on registered memory | part of claim 3: the measurement (2.7 s per GiB at the next kernel) and the remedies measured side by side against the kernel's repair |
| Write-set learning from copy-on-write sentinels | not found in the audited set; small | PhoenixOS (validated speculation on kernel arguments); soft-dirty and `userfaultfd` write-protect where the kernel has them | a detector that needs no kernel facility and costs one per-page event per extent |
| cgroup accounting of GPU memory on memoryless MIG | partially taken | Linux `dmem` and its open main-memory question; ROCm issue 6370 (same bypass on an AMD APU); KRYPTON, HAMi (quota by interception) | the per-plane enforcement matrix on both MIG instances and the fact that a MIG instance adds no limit of its own; no mechanism is claimed |
| Sharing of GPU-visible state across MIG instances through the host MMU | partially taken | "The Ingestion Tax" (processes share one shared file mapping that the GPU reads in place, on three unified-memory systems); llama.cpp on Metal; C2CServe (read-mostly weights for MIG tenants, pinned); ForkKV and vLLM (copy-on-write inside one runtime) | claim 1: private views that stay shared and diverge, a sealed base, MIG instances, and the measurements of Section 8.2 |
| A GPU read copies a private mapping | partially taken | an issue thread of 2026-09-28 on another NVIDIA unified-memory system that cites the same driver lines; reports of the same symptom on Apple systems | claim 2: the accessed-flag condition, `MADV_POPULATE_READ`, the reproduction, and the 512-fold amplification |
| Huge-page objects for revoke-before-grant transfer | partially taken: a known per-entry effect | C4 (2 MiB mappings to make remapping cheap in a garbage collector); Linux page-table-level `mremap` and large-folio `mprotect`; fbufs and libmpk for the per-page cost of permission changes | claim 4: the measurement for GPU-visible memory between MIG instances (Section 6.3) |
| One owner for accounting, revocation, snapshots, and sharing | open as a combination | GMEM (central OS management of device memory, coalesced invalidations); MoonBright (host-serialized TLB coherence of a discrete GPU's own page tables, avoided by restructuring the operation) | the combination, on a platform where pageable GPU memory needs no separate manager |

### 9.3 Wording

Permitted:

> Since Linux 6.6 the architecture TLB-flush functions notify the IOMMU of
> every flush in a process that is bound for shared virtual addressing, and
> kernel developers expected this to slow `fork`, `mmap`, and `munmap`. To
> our knowledge, among the works we audited, no measurement of that cost
> exists for a GPU. On Thor a copy-on-write break costs 293 ms/GiB before a
> process initializes CUDA and 1,705 ms/GiB afterwards, on memory the GPU
> never touched as well, and a kernel profile attributes 73.5% of the samples
> to the SMMU command queue.

> On Thor, the kernel's copy-on-write governs GPU writes to private pageable
> memory. A forked child therefore holds a consistent image without API
> interception or a driver change. Prior systems obtain the same isolation by
> intercepting GPU APIs (PhoenixOS, GCR) or by a proxy and a drain copy
> (CRUM).

> Extent privatization applies user-space copy-and-remap, which RUMA and
> AnKer introduced for CPU snapshots, so that one secondary-TLB invalidation
> replaces one per page. Kernel copy-on-write remains the fallback. Thus, a
> mispredicted write set costs time but cannot corrupt the snapshot.

> To our knowledge, among the audited works, no system shares GPU-visible
> state copy-on-write across MIG instances through the host MMU. Tenants in
> separate MIG instances read one sealed physical copy of a model through
> read-only private mappings and diverge by replacing the extents they will
> write. Eight tenants of a 4 GiB model occupy 4.5 GiB instead of 32.5 GiB,
> and neither a tenant's CPU nor its GPU kernels can change what another
> tenant reads.

> A writable private mapping is not a safe way to share on this platform.
> The driver serves a GPU fault in a writable mapping as a write, as its
> source shows and as others have reported. We find that the GPU faults on
> any page whose accessed flag is clear, that one such page costs a whole
> 2 MiB block, and that clearing the flag on 0.2% of the pages turns the
> entire model into a private copy at the next GPU read.

> Placing a Portal object on 2 MiB pages reduces the measured protection
> cost of revoke-before-grant transfer from 2.69 to 1.09 times. That a
> permission change costs per page-table entry is known; we measure it for
> GPU-visible memory between MIG instances.

The following must not be claimed:

- that SVA slowing the memory manager is a discovery, that 8 us is the cost
  of one SMMU command, or that newer kernels fix or batch it; the mechanism
  is intended behaviour, the number of submissions per page in the Tegra
  kernel is not known, and only 6.8.12-tegra was measured;
- first copy-on-write for GPU memory, first GPU snapshot, first concurrent GPU
  checkpoint, or first fork for GPU processes (PhoenixOS, CRUM, GPU Snapshot,
  Caldera);
- that GPUs lack copy-on-write in general, or that they now have it; every
  sentence is scoped to pageable memory on a host-page-table GPU;
- first to characterize GPU page-fault cost, first to show that GPU first
  touch is expensive, or first to recommend CPU pre-faulting (ISPASS 2016,
  ICPP 2024, MI300A);
- a new batching principle (NPF, MSched, GMEM, MoonBright), or that a range
  operation invalidating once is a finding (it is the kernel's design);
- a novel extent-granular copy-on-write mechanism (RUMA, AnKer, CCoW);
- a snapshot that is faster than an application-specific stop-and-copy; it
  is not (Sections 5.3 and 5.4);
- first cgroup accounting of GPU memory, first to find that GPU memory
  escapes cgroups, or a mechanism that bounds device allocations (`dmem`,
  KRYPTON, HAMi, ROCm issue 6370; Section 7);
- that MIG provides no memory isolation in general; only what was measured on
  Thor profiles `2g.0gb` and `1g.0gb` with driver 595.78;
- that the host memory manager owns all GPU memory; only pageable memory;
- first to share model weights across MIG tenants (C2CServe, Flex-MIG), or
  first to share weights in place across processes on a unified-memory GPU
  ("The Ingestion Tax", llama.cpp);
- that the write-intent fault service is a finding of this record, or that
  huge pages reducing permission-change cost is a new technique;
- "transparent" without the qualifier that state must live in pageable
  memory;
- "1000x" or "160x" as a constant; report the measured values, the kernel,
  the page size, and the driver.

## 10. Limits

- **Device allocations are outside everything here.** `cudaMalloc` memory is
  not charged, not copy-on-write, and not revocable by `mprotect`. cuStateVec
  rejects pageable pointers and TensorRT allocates device memory, so the
  snapshot results use kernels that run in place on pageable memory, and the
  accounting shim does not reach either library.
- **One platform and one kernel.** The per-page cost is measured on the
  SMMUv3 SVA path of 6.8.12-tegra. The path exists in mainline source and in
  the x86 SVA drivers, and another host-page-table GPU, such as Grace Hopper
  in ATS mode, should show the same structure, but neither is measured.
- **A stop-and-copy that knows the mutable region is cheaper than a fork
  snapshot** wherever both were measured, by 2.2 to 8.8 times (Section 5).
  Against a copy of everything, the learned policy wins at 33 GiB over eight
  snapshots and loses at 8 GiB over four. Fork snapshots are for cases in
  which the checkpointer does not know the state layout, wants the image as
  a whole process, or cannot afford a second copy of everything.
- **The parent's other memory still pays.** Pages of the parent that are not
  in a registered region, such as its heap, are copied by the kernel one at a
  time when they are written while a child lives, at the per-page cost of
  Section 3.1.
- **Snapshots need a quiescent point.** `begin` must run between kernels, as
  any consistent checkpoint must.
- **Consumers are restricted.** The child of a multithreaded CUDA process may
  not use CUDA, allocate, or take locks.
- **Sharing through a writable mapping can be undone by the kernel.** The
  design of Section 8.2 avoids this by mapping the base read-only, at the
  price that a tenant must name the extents it will write; an undeclared
  write stops the tenant. Reclaim was exercised only through `MADV_COLD` on
  a device without swap, and a model in a regular file was not tested.
- **The remedies for the per-page cost are per program.** `posix_spawn`,
  `MADV_DONTFORK`, extent privatization, and read-only bases are things a
  program or a runtime must do. Nothing here changes the kernel or the
  driver, where the cost could be removed for every program.
- **Small samples.** The Portal huge-page campaigns, the sharing campaigns,
  and the tax table have six repetitions per cell; tenant scaling has three,
  and the scale check has three on one MIG instance.
- **Synthetic sparse workload.** The read-mostly case is a synthetic model
  with a small mutable state; the dense case is a real statevector
  simulation.

## 11. Artifacts

| Evidence | Directory under `thor_hostmm/results/` |
|---|---|
| cost matrix and allocation planes, both MIG instances | `20261005-hostmm-matrix-v1/` |
| kernel profile of a CPU-side copy-on-write break | `20261005-smmu-trace-v1/` |
| kernel profile of GPU-side faults | `20261005-gpu-fault-trace-v1/` |
| scope of the per-page cost, `fork` plus `exec` repair, registration | `20261005-side-probes-v1/` |
| end-to-end snapshots, synthetic and statevector | `20261005-snapshot-v1/` |
| snapshots of a 33 GiB model and a 30-qubit statevector | `20261005-scale-v1/` |
| Portal pipeline on hugetlbfs objects | `20261005-portal-hugetlb-v1/` |
| Portal authority campaigns on hugetlbfs objects | `20261005-portal-hugetlb-suite-v1/` |
| Portal pipeline on shmem transparent huge pages | `20261005-portal-shmem-thp-v1/` |
| cgroup enforcement, plane throughput, the shim | `20261005-accounting-v1/` |
| copy-on-write sharing across MIG instances | `20261005-cow-share-v1/` |
| kernel profile of the first GPU read of a shared model | `20261005-share-fault-trace-v1/` |
| sealed base: attach modes, hugetlbfs base, tenant scaling, accessed state, attempts to modify the base | `20261005-shared-model-v1/` |
| tenants reading the shared model at the same time | `20261005-shared-concurrent-v1/` |
| pages copied per page without the accessed flag | `20261005-af-block-v1/` |
| the shared base under MIG, time slicing, and MPS; fault containment | `20261005-sharing-modes-v1/` |
| tax on twelve memory-management operations; ways to start a helper program | `20261005-mm-tax-v1/` |

`thor_hostmm/verify_hostmm_artifact.sh` rebuilds every derived file from its
raw log, checks the pinned sources, and re-evaluates the invariants above
without running GPU work. `thor_hostmm/README.md` lists the reproduction
commands. No experiment reconfigured MIG or rebooted the device. Four
campaigns reserve a hugetlb pool, one sets the shmem huge-page policy, and
two take a system-wide profile; each uses `sudo` and restores the previous
setting on exit.
