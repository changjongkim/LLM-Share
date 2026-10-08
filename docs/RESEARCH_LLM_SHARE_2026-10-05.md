# LLM-Share: several serving processes on one copy of a model, on a GPU that follows the host page tables

**Evidence cutoff:** 2026-10-06
**Platform:** NVIDIA Jetson AGX Thor, kernel 6.8.12-1021-tegra, open GPU
kernel modules 595.78 (`uvm_ats_mode=1`), MIG `2g.0gb` (12 SMs) + `1g.0gb`
(6 SMs), CUDA 13.0
**Engine:** llama.cpp at commit `6f767fe96` with `llm_share/inplace_weights.patch`
(Sections 3 to 5), with `llm_share/kv_extents.patch` (Section 6), and with
`llm_share/kv_vmm.patch` for the comparison of Section 6.9
**Model:** Qwen2.5-7B-Instruct, Q4_K_M (4.36 GiB)
**Code and artifacts:** `llm_share/`, `thor_hostmm/`
**Substrate record:** `RESEARCH_HOSTMM_2026-10-05.md`
**Prior-art audit:** `SOTA_HOSTMM_2026-10-05.md`, Sections 12, 14 and 16
**Status:** sharing of weights, and sharing, forking and moving of the
key-value cache of a prompt prefix, are implemented in a real engine and
measured

## 1. Question and claim

On-device assistants are turning into groups of agents: several processes
that run the same base model, often each with its own adapter. On a device
whose CPU and GPU share one DRAM, a process that loads the model into device
memory pays for the whole model again. The engine most used on such devices
does this by design: llama.cpp turns memory mapping off on integrated CUDA
GPUs and copies the weights into device memory, and its CUDA backend cannot
take a host mapping as a buffer.

On Thor the GPU follows the host page tables, so a CUDA kernel can read a
mapped file directly. The question is whether serving processes can use this
to hold one physical copy of the model between them, what it costs in
tokens per second, and what must be true for the sharing and the isolation
to hold. The claim, with the scope that the evidence supports:

> A serving process can use the mapped model file as its weight buffer.
> Text is identical; generation is 6% slower when the file is mapped with
> 4 KiB pages and not slower with 2 MiB pages; a process starts in half the
> time or less; and every further process needs 4.4 GiB less memory for a
> 4.36 GiB model, so that eight processes hold 10 GiB instead of 41 GiB.
> The processes may sit in different MIG instances, be time-sliced in one,
> or be clients of an MPS server; the host MMU shares and isolates them in
> every case. Three properties of a GPU that follows the host page tables
> decide whether this is safe, and each is measured: a writable mapping of
> the weights turns into a private copy when one page in five hundred loses
> its accessed flag; a `fork` leaves a cost of 2.7 s per GiB at the next GPU
> write; and a GPU fault of one MPS client ends the work of every client of
> that server, while processes that are time-sliced or in the other MIG
> instance continue.

The second claim is about the state that a process computes at run time,
and it is an approach, not only a measurement. The three properties above
make the kernel's copy-on-write the wrong way to give one process the state
of another on this device. Section 6 states the rule that follows (shared
state is immutable and mapped read-only, what a process writes is private
from the start, and sharing is a range operation on mappings), builds it
into the engine for the key-value cache, and measures it:

> A process that has computed a prompt prefix publishes its key-value cache
> in 3 to 9 ms as a file that names its rows; any number of processes, in
> either MIG instance, map those rows read-only in place of computing or
> copying them and write their own rows to a private tail. Every agent
> writes the text of an agent that copies the same state. Eight agents on a
> 16,321-token prefix hold 5.7 GiB instead of 19.9 GiB, with one copy of
> the prefix between them. A running process hands its state to four
> children with a pause of 3 ms instead of 218 ms and continues unchanged.
> In the 12-SM instance agents on extents generate as fast as agents that
> copy; in the 6-SM instance a cache in host memory, shared or not, costs
> 2% of the generation speed with a 4,081-token prefix and 6% with a
> 16,321-token prefix.

The same measurements bound both claims. Separate processes do not raise
throughput on this GPU: one process that batches eight sequences generates
about three times as much as eight processes, and it shares a prefix between
its sequences without any of this. Sharing in place is for deployments whose
agents are separate programs anyway, and for putting one batching server
into each MIG instance. That configuration is the best one measured: two
servers on one published prefix serve eight agents at 102.9 tokens/s in
2.4 GiB, against 75.6 tokens/s in 2.5 GiB for one server and 105.0 tokens/s
in 5.0 GiB for two servers that copy the prefix. Inside one MIG instance
the same layout can also be built from device memory with CUDA's virtual
memory interface. Built into the same engine for comparison, it generates at
the speed of device memory, attaches about ten times slower than host
extents (325 to 438 ms against 27 to 48 ms), holds about 30% more, and
cannot reach the other instance. The first token of an agent comes
only 4 to 23% earlier with extents than with a copy, because loading the
model dominates it; the gain is memory. And the two MIG instances do not
compute the same bits, so an agent that continues a prefix from the other
instance does not always write what a recomputation there would write,
whichever way it received the prefix.

Not claimed: that reading weights in place is new (it is known on Apple
devices and was proposed for this engine); a new batching technique; the
parts of the cache mechanism taken one by one, each of which exists (a
shared immutable prefix with private continuation inside one process, a
cache whose memory follows use, a cache in a mapped file with its metadata
beside it for CPU inference, and one cache shared by processes in device
memory on one GPU; Section 7); and any result on a device other than Thor.

## 2. The substrate in five facts

Measured in `RESEARCH_HOSTMM_2026-10-05.md` on the same device; each row
names the section there.

| Fact | Measurement | Section |
|---|---|---|
| A GPU kernel reads pageable host memory at the speed of device memory | 20.8 against 20.7 ms per GiB | 7 |
| A flush of one page costs a process that uses the GPU about 9 us, anywhere in its address space; a range operation costs nothing extra | four operations, 4.3 to 9.0 times slower | 3.2 |
| The GPU faults on a mapped page whose accessed flag is clear, and the driver serves a fault in a writable mapping as a write for its whole 2 MiB prefetch block | one such page turns 512 pages into private copies | 8.2 |
| After `fork`, the next GPU write to pre-fork memory costs 2.7 s per GiB; `posix_spawn`, `MADV_DONTFORK`, or one remap pass avoid it | 2,680 against 6.6 to 6.8 ms | 6.1 |
| A sealed `memfd` mapped private and read-only is shared by any number of processes and cannot be changed by a tenant's CPU or GPU | 9 attempts, 36 runs; 8 tenants hold 4.5 instead of 32.5 GiB | 8.2 |

## 3. One process: weights read in place

### 3.1 What the engine does today

At commit `6f767fe96` the CUDA backend of llama.cpp reports
`buffer_from_host_ptr = false`, and since commit `153d324bc` (2026-08-11,
"add default load-mode auto, which avoids mmap on iGPUs") a CUDA device that
reports itself as integrated also reports `mmap_support = false`. On Thor
the default load mode therefore reads the model file into device memory.
Every process holds its own copy. A pull request of 2026-04-19 proposed
host-pointer buffers for the CUDA backend on another NVIDIA unified-memory
system; it was closed unmerged and carries no measurement
(`SOTA_HOSTMM_2026-10-05.md`, Section 14.2).

### 3.2 The change

`llm_share/inplace_weights.patch` changes one file, `ggml-cuda.cu`
(59 lines added, 6 removed). With `GGML_CUDA_HOST_PTR=1`:

- the device reports `buffer_from_host_ptr` and `mmap_support`, so the
  loader maps the file and asks the backend to wrap the mapped range;
- the backend wraps the range in a buffer that it does not own: the tensor
  data pointers point into the mapping, nothing is registered with CUDA,
  and nothing is copied;
- at load, twelve CPU threads read one byte of every page, because the GPU
  faults on a page whose accessed flag is clear (Section 2);
- a quantized tensor whose rows are not a multiple of 512 elements is
  refused. The engine's kernels read up to that boundary and the device
  path zero-fills it; a read-only mapping has no room. This model has no
  such tensor among its 198 quantized ones. A model that has would need
  padding in the file, which is not implemented.

The engine maps the file shared and read-only. The driver therefore cannot
serve a GPU fault in it as a write, and every process that maps the file
shares its page-cache pages.

### 3.3 Result

`run_engine_single.sh`: six paired repetitions that alternate between the
MIG instances and rotate the order of the modes
(`results/20261005-engine-single-v1/`). Throughput is from the engine's
benchmark (512 prompt tokens, 128 generated, three passes); load time,
memory, and text are from a 256-token greedy generation with a 4,096-token
context. "Memory outside the model file" is the drop of available memory
while the process generates; "model file mapped" is the proportional set
size of the mapping.

| Weights | Runs (failed) | Identical text | Prompt (tokens/s) | Generation (tokens/s) | Generation against device copy | Load (ms) | Memory outside the model file (MiB) | Model file mapped (MiB) |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| device copy (upstream) | 6 (0) | 6/6 | 817 | 24.12 | 1.000x [1.000, 1.000] | 942 | 5423 | 0 |
| in place, CPU reads every page at load | 6 (0) | 6/6 | 813 | 22.79 | 0.940x [0.914, 0.967] | 517 | 878 | 4460 |
| in place, no CPU read | 6 (0) | 6/6 | 812 | 22.80 | 0.940x [0.912, 0.969] | 1013 | 921 | 4460 |

- **The text is identical** in 6 of 6 runs for both in-place modes.
- **Generation is 6.0% slower**, 95% CI of the paired ratio [0.914, 0.967].
  It is 3.7% on the 12-SM instance (28.7 to 27.6 tokens/s) and 8.2% on the
  6-SM instance (19.6 to 18.0). Prompt processing is unchanged (0.5%).
- **4.4 GiB leave the process.** Memory outside the model file falls from
  5,423 to 878 MiB, and the 4,460 MiB of weights are page cache that other
  processes can map.
- **Load takes 517 ms instead of 942 ms.** Without the CPU read it takes
  1,013 ms: the GPU's faults on first use cost more than the CPU read saves.
  Generation speed afterwards is the same.

### 3.4 Page size

The 6% is the page size of the mapping, not the absence of a copy.
`run_engine_pages.sh` serves the same model from the ext4 page cache
(4 KiB pages), from a tmpfs mounted with `huge=always` (transparent 2 MiB
pages), and from hugetlbfs (2 MiB pages from a reserved pool), against the
device copy (`results/20261005-engine-pages-v1/`, six paired repetitions;
`results/20261005-engine-pages-small-v1/` repeats it with a 1.1B model):

| Weights | Runs (failed) | Identical text | Prompt (tokens/s) | Generation (tokens/s) | Generation against device copy | Load (ms) |
|---|---:|---:|---:|---:|---|---:|
| device copy (upstream) | 6 (0) | 6/6 | 817 | 24.03 | 1.000x [1.000, 1.000] | 943 |
| in place, ext4 page cache (4 KiB) | 6 (0) | 6/6 | 812 | 22.77 | 0.943x [0.918, 0.969] | 516 |
| in place, tmpfs with huge pages (2 MiB) | 6 (0) | 6/6 | 817 | 24.08 | 0.997x [0.971, 1.024] | 270 |
| in place, hugetlbfs (2 MiB) | 6 (0) | 6/6 | 817 | 24.04 | 0.996x [0.970, 1.023] | 267 |

- **With 2 MiB pages the cost is gone.** Generation is 0.997 times the
  device copy on tmpfs huge pages (95% CI [0.971, 1.025]) and 0.996 times on
  hugetlbfs ([0.970, 1.023]), against 0.943 times on 4 KiB pages
  ([0.918, 0.969]). The text is identical in 6 of 6 runs of every mode.
- **Load falls to 270 ms**, against 943 ms for the device copy and 516 ms
  for the 4 KiB mapping: there are 512 times fewer pages to read once.
- **The small model shows the same and more strongly**: 0.902 times on
  4 KiB pages ([0.894, 0.910]), 1.009 times on tmpfs huge pages
  ([0.996, 1.022]), and 1.013 times on hugetlbfs ([1.005, 1.020]), with
  identical text in 6 of 6 runs of every mode.
- tmpfs with huge pages needs no reserved pool and no change to the engine;
  the file is copied there once. Both variants hold the model in memory
  that cannot be evicted, where the page cache can.

### 3.5 Adapters

A base model is rarely served alone. `run_engine_adapters.sh` starts five
processes at the same time, three in one MIG instance and two in the other:
the base and four processes with their own LoRA adapter, which the engine
applies at run time without changing the base tensors
(`results/20261005-engine-adapters-v1/`, six repetitions). The adapters are
synthetic rank-8 factors for the query and value projections (`make_lora.py`,
10 MB each); they stand in for fine-tuned adapters and say nothing about
quality.

| Base weights | Runs (failed) | Runs with five distinct texts | Texts equal to the device-copy text | Memory outside the model file (MiB) | Model file mapped (MiB) | Total (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| device copy (upstream) | 6 (0) | 6/6 | 30/30 | 28273 | 0 | 28273 |
| in place | 6 (0) | 6/6 | 30/30 | 5653 | 4467 | 10120 |

Every process produces its own text in 6 of 6 runs, and each text equals
the text of the same adapter over a device copy in 30 of 30 comparisons.
The five processes hold 9.9 GiB instead of 27.6 GiB: one mapped base and
1.1 GiB each for context, compute buffers, and adapter.

## 4. Several agents

### 4.1 N processes under four ways of sharing the GPU's compute

`run_engine_agents.sh` starts 1, 2, 4, or 8 engine processes at the same
time, each processing a 512-token prompt and generating 128 tokens, with
weights copied to device memory or read in place from the page cache
(4 KiB pages), under the four configurations of Section 5
(`results/20261005-engine-agents-v1/`, six repetitions per cell, 192 runs).
Memory is the largest drop of available memory while the processes run plus
the proportional size of the model mapping.

| Compute shared by | Agents | Generation, device copy (tokens/s, sum) | Generation, in place (tokens/s, sum) | In place / device copy | Memory, device copy (MiB) | Memory, in place (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| one MIG instance, time-sliced | 1 | 28.5 +/- 0.1 | 27.7 +/- 0.1 | 0.971x | 5259 | 5198 |
| one MIG instance, time-sliced | 2 | 26.5 +/- 0.0 | 25.6 +/- 0.0 | 0.966x | 10462 | 5941 |
| one MIG instance, time-sliced | 4 | 26.6 +/- 0.0 | 25.4 +/- 0.0 | 0.956x | 20975 | 7366 |
| one MIG instance, time-sliced | 8 | 26.7 +/- 0.1 | 25.3 +/- 0.0 | 0.946x | 41921 | 10316 |
| one MIG instance each (alternating) | 1 | 28.5 +/- 0.0 | 27.6 +/- 0.0 | 0.969x | 5259 | 5178 |
| one MIG instance each (alternating) | 2 | 44.8 +/- 0.1 | 44.1 +/- 0.0 | 0.986x | 10430 | 5937 |
| one MIG instance each (alternating) | 4 | 41.4 +/- 0.2 | 41.0 +/- 0.0 | 0.990x | 20925 | 7283 |
| one MIG instance each (alternating) | 8 | 41.4 +/- 0.2 | 40.4 +/- 0.1 | 0.974x | 41798 | 10236 |
| one MIG instance, one MPS server | 1 | 28.6 +/- 0.1 | 27.7 +/- 0.1 | 0.967x | 5283 | 5148 |
| one MIG instance, one MPS server | 2 | 29.3 +/- 0.0 | 28.3 +/- 0.0 | 0.966x | 10456 | 5851 |
| one MIG instance, one MPS server | 4 | 29.6 +/- 0.0 | 28.6 +/- 0.0 | 0.967x | 20921 | 7313 |
| one MIG instance, one MPS server | 8 | 29.4 +/- 0.1 | 28.6 +/- 0.0 | 0.971x | 41822 | 10228 |
| two MIG instances, an MPS server in each | 1 | 28.5 +/- 0.1 | 27.6 +/- 0.1 | 0.970x | 5236 | 5174 |
| two MIG instances, an MPS server in each | 2 | 44.8 +/- 0.1 | 44.1 +/- 0.1 | 0.985x | 10452 | 5794 |
| two MIG instances, an MPS server in each | 4 | 45.7 +/- 0.1 | 45.1 +/- 0.0 | 0.986x | 20847 | 7218 |
| two MIG instances, an MPS server in each | 8 | 45.9 +/- 0.2 | 45.4 +/- 0.0 | 0.989x | 41680 | 10138 |

- **Memory.** Eight processes hold 9.9 to 10.1 GiB with the weights in
  place and 40.7 to 40.9 GiB with device copies, in every configuration:
  one mapped model and about 0.7 GiB per process.
- **Reading in place costs 1 to 5% of the total generation rate**, least
  with an MPS server in each MIG instance (0.989 times at eight agents) and
  most with time slicing (0.946 times).
- **More processes do not generate more.** One generating process already
  occupies its MIG instance. Time slicing keeps the total of one instance
  at 25 to 27 tokens/s for any number of agents, an MPS server raises it to
  29, and using both instances gives 40 to 46. The best configuration is an
  MPS server in each MIG instance: 45.4 tokens/s for eight agents in place,
  5.7 tokens/s each.

### 4.2 One process with N sequences

The alternative to N processes is one process that decodes N sequences in a
batch. It holds one copy of the weights by construction and gives the
sequences no isolation from each other. The engine's batched benchmark, six
repetitions alternating between the MIG instances
(`results/20261005-engine-batched-v1/`):

| Sequences in one process | Generation, device copy (tokens/s, sum) | Generation, in place (tokens/s, sum) | Per sequence, in place (tokens/s) |
|---:|---:|---:|---:|
| 1 | 23.7 | 22.5 | 22.5 |
| 2 | 49.5 | 45.8 | 22.9 |
| 4 | 64.1 | 62.5 | 15.6 |
| 8 | 75.2 | 75.0 | 9.4 |

- **Batching inside one process is about three times faster in total.**
  On the 12-SM instance one process generates 89.7 tokens/s for eight
  sequences, against 29.4 tokens/s for eight processes under MPS in the same
  instance; on the 6-SM instance 60.6 tokens/s. In place the batch reaches
  the same rate (89.6 and 60.4).
- Separate processes are therefore not a way to get throughput on this GPU.
  They are what a deployment has when its agents are separate programs:
  applications that each bring their own engine, agents that must not share
  an address space, or groups that must not take each other down. For those
  the in-place path removes the memory cost of being separate.

### 4.3 A batching server per MIG instance

The two results combine: batch inside a group of agents that may share a
process, put groups that must survive each other into different MIG
instances, and let all of them read one copy of the weights.
`run_engine_groups.sh` runs one batching process in each MIG instance at the
same time (`results/20261005-engine-groups-v1/`, six repetitions):

| Weights | Sequences per server | Runs (failed) | Generation, both servers (tokens/s) | 12-SM instance | 6-SM instance | Memory outside the model file (MiB) | Model file mapped (MiB) | Total (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| device copy | 4 | 6 (0) | 126.0 +/- 0.1 | 75.0 | 51.1 | 11366 | 0 | 11366 |
| in place | 4 | 6 (0) | 123.5 +/- 0.2 | 73.2 | 50.2 | 2324 | 4460 | 6784 |
| device copy | 8 | 6 (0) | 149.5 +/- 0.1 | 89.2 | 60.4 | 11362 | 0 | 11362 |
| in place | 8 | 6 (0) | 149.0 +/- 0.2 | 88.9 | 60.1 | 2324 | 4460 | 6784 |

- **Two servers generate 149 tokens/s for sixteen sequences**, 89 in the
  12-SM instance and 60 in the 6-SM instance, with the weights in place or
  copied (0.997 times). With four sequences each they generate 123 tokens/s
  in place (0.980 times).
- **In place the two servers hold 6.6 GiB instead of 11.1 GiB**: one mapped
  model and 1.1 GiB each.
- Against eight separate processes with an MPS server in each instance
  (Section 4.1: 45.4 tokens/s, 10.0 GiB), this configuration generates
  3.3 times as much, for twice as many sequences, in two thirds of the
  memory, and a fault in one server cannot reach the other (Section 5.2).

## 5. What isolates the agents

### 5.1 The way the GPU's compute is shared does not change the sharing of memory

The host MMU isolates processes, not GPU partitions. `sharing_modes_probe`
runs the sealed base of `RESEARCH_HOSTMM_2026-10-05.md` Section 8.2 with a
synthetic kernel that reads the whole 4 GiB model, for 2, 4, and 8 tenants,
under four ways of sharing the GPU
(`thor_hostmm/results/20261005-sharing-modes-v1/`, six repetitions per row,
174 runs, none failed). In every run each tenant first reads the original,
tenant 0 diverges 128 MiB, every other tenant still reads the original, and
the base is unchanged at the end.

| Compute shared by | Tenants | Read at the same time, copies (ms) | Read at the same time, shared base (ms) | Slowdown against a tenant alone, shared base | Memory, copies (MiB) | Memory, shared base (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| one MIG instance, time-sliced | 2 | 140.0 | 159.8 | 2.15x | 8314 | 4218 |
| one MIG instance, time-sliced | 4 | 281.4 | 321.9 | 4.32x | 16621 | 4333 |
| one MIG instance, time-sliced | 8 | 572.6 | 646.0 | 8.68x | 33234 | 4562 |
| one MIG instance each (alternating) | 2 | 82.7 | 90.4 | 1.01x | 8314 | 4218 |
| one MIG instance each (alternating) | 4 | 174.4 | 193.1 | 2.16x | 16621 | 4333 |
| one MIG instance each (alternating) | 8 | 351.6 | 389.0 | 4.35x | 33234 | 4563 |
| one MIG instance, one MPS server | 2 | 86.9 | 91.3 | 1.23x | 8296 | 4200 |
| one MIG instance, one MPS server | 4 | 144.5 | 148.8 | 2.00x | 16588 | 4300 |
| one MIG instance, one MPS server | 8 | 270.3 | 278.0 | 3.74x | 33170 | 4498 |
| two MIG instances, an MPS server in each | 2 | 82.7 | 90.4 | 1.01x | 8295 | 4199 |
| two MIG instances, an MPS server in each | 4 | 107.0 | 112.1 | 1.25x | 16587 | 4299 |
| two MIG instances, an MPS server in each | 8 | 178.6 | 187.1 | 2.09x | 33169 | 4497 |

- **Memory does not depend on the configuration**: 4.1 to 4.5 GiB for the
  shared base against 8.1, 16.2, and 32.5 GiB for copies.
- **Throughput does.** Time slicing inside one MIG instance multiplies a
  tenant's read time by the number of tenants. One MPS server halves that.
  An MPS server in each MIG instance halves it again: eight tenants read in
  2.1 times the time of one.
- **The shared base costs 3 to 14% in read time against private copies**,
  least where several tenants share an MPS server (3 to 5%). That is the
  4 KiB page size of the file mapping; a hugetlbfs base reads at the speed
  of a copy when a tenant is alone (`RESEARCH_HOSTMM_2026-10-05.md`,
  Section 8.2).

### 5.2 A GPU fault of one agent

The sealed design stops a kernel that writes the base with
`cudaErrorIllegalAddress`. What that fault does to the other agents depends
on how the compute is shared. Victims keep reading (40 queued reads each)
while an attacker process writes through a read-only mapping and faults
(same campaign, six repetitions):

| Configuration | Victim is | Victims | Finished the read in progress | Read again afterwards | Base intact |
|---|---|---:|---:|---:|---:|
| no MPS | other MIG instance | 6 | 6/6 | 6/6 | 6/6 |
| no MPS | same MIG instance, time-sliced | 12 | 12/12 | 12/12 | 12/12 |
| MPS server in the faulting tenant's instance | other MIG instance | 6 | 6/6 | 6/6 | 6/6 |
| MPS server in the faulting tenant's instance | client of the same MPS server | 12 | 0/12 | 0/12 | 12/12 |
| MPS server in each instance | other MIG instance, client of its own MPS server | 12 | 12/12 | 12/12 | 12/12 |
| MPS server in each instance | client of the same MPS server | 6 | 0/6 | 0/6 | 6/6 |

- Time-sliced processes do not notice the fault of another process, in the
  same MIG instance or in the other.
- Clients of the faulting client's MPS server lose the read in progress and
  cannot read again: 0 of 18.
- Clients of an MPS server in the other MIG instance are not affected:
  12 of 12.

That an MPS client's fatal fault reaches the other clients of its server is
documented by NVIDIA and measured by others
(`SOTA_HOSTMM_2026-10-05.md`, Section 14.3). The result here is which
processes survive on this device, and that the MIG boundary contains the
damage of an MPS group. The base is intact in all 54 victim records.

The accessed-flag amplification of Section 2 is the same for an MPS client:
one page per 2 MiB block without the flag turns a 1 GiB writable private
mapping into a 1 GiB private copy in 6 of 6 runs, and a read-only mapping
grows by nothing in 6 of 6.

### 5.3 Which route can share at all

Device memory can be shared between processes with CUDA IPC, which is how
one existing tool shares this engine's weights. `run_sharing_routes.sh`
tries both routes for three placements of two processes
(`results/20261005-sharing-routes-v2/`, five repetitions):

| Placement | CUDA IPC | Host page table |
|---|---|---|
| two processes in one MIG instance | works 5/5 | works 5/5 |
| one process in each MIG instance | works 0/5 (`UNSUPPORTED_at_open_cudaErrorInvalidValue`) | works 5/5 |
| two clients of one MPS server | works 5/5 | works 5/5 |

CUDA IPC shares a device allocation between two processes of one MIG
instance, with and without MPS, and cannot cross MIG instances: opening the
handle in the other instance fails with `cudaErrorInvalidValue` in 5 of 5
attempts. The host page table shares in all three placements. On this
device and driver CUDA IPC did work between two clients of one MPS server,
which NVIDIA's documentation for Tegra describes as unsupported; the probe
shares one 64 MiB allocation and does not test more. Beyond reach, the two
routes differ in kind: an IPC handle is exported by a process that must
remain alive and exposes device memory that the importer can write, while a
sealed mapping has no owner process and cannot be written by anyone.

### 5.4 The rule that follows

- Keep the base sealed and map it private and read-only; let an agent
  replace the extents it writes. A writable mapping of the base is not safe
  on this driver.
- Agents that may share a process should be batched in one (Section 4.2).
  Agents that are separate programs but may take each other down can share
  an MPS server. Agents that must survive each other's faults belong in
  different MIG instances, or outside MPS. Thor has two MIG instances, so
  this gives two fault domains.
- One batching server per MIG instance, both reading the weights in place,
  is the configuration that these rules select; Section 4.3 measures it.
- Start helper programs with `posix_spawn`.

## 6. Session state as extents: publish, attach, fork, move

Sections 3 to 5 share what never changes. After the weights are shared, what
each agent still holds for itself is its key-value cache: 56 KiB per token
for this model (28 layers, 4 key-value heads of 128 dimensions, 16-bit
values), 1.75 GiB for a 32,768-token context. Agents of one application
start from the same prefix, a system prompt and tool descriptions of several
thousand tokens, so a large part of that state is the same bytes in every
agent. This section shares it.

The engine change is `llm_share/kv_extents.patch`, applied to a second
clone, `llm_share/llama.cpp-kv`. It contains the weights change of Section 3
and adds 30 lines to the CUDA backend, so that the CPU can read and write a
buffer of host memory, 397 lines to `llama-kv-cache.cpp`, and 3 to its
header. Nothing changes in the computation graph. The mode is chosen by the
environment of the process; without it the engine behaves as upstream. The
driver of the experiments is `llm_share/kv_fork.cpp`: one program that
computes a prefix alone, publishes it as a parent, or attaches to it as a
child, and decodes the same batches in every role.

### 6.1 The rule

The substrate facts of Section 2 rule out the two obvious ways to give an
agent a prefix that another process computed. A copy costs time and memory
in proportion to the prefix, for every agent. A private writable mapping of
the other process's cache, which is what the kernel offers for this
(`MAP_PRIVATE`, or `fork`), is taxed three times on this device: a GPU read
of a page whose entry lacks the accessed flag is served as a write and
copies the 2 MiB block around it; each page the kernel copies costs a
process-wide invalidation of about 9 us; and the first write into a block
stalls the agent while the block is copied. The rule that follows:

> Shared state is immutable and mapped read-only. What a process writes goes
> to memory that was private from the start. Sharing, forking and moving are
> range operations on mappings; no page of GPU-visible memory changes owner
> through a fault.

A key-value cache fits the rule because of how it is written. A cell is
written once, when its token is decoded, and the cells of a sequence are
filled in order. With flash attention and one stream, which is what the
engine uses on this GPU, a cell is a row of every key and value tensor.
Therefore, the cells of a common prefix are a leading range of every tensor,
and everything an agent adds lies after it. That a prefix cache keeps shared
blocks unmodified and appends into private ones is how every block-level
prefix cache works inside one process (Section 7). The layout is not ours.
What the rule adds is that between processes the split has to be made in the
address space, by mappings, because the kernel's own way of making it is the
taxed one.

### 6.2 Mechanism

**The cache in host memory.** With `LLAMA_KV_HOST` set, the tensors of the
cache are placed in one host mapping, each on a 2 MiB boundary, and the CUDA
backend takes the mapping as a buffer through the same host-pointer path as
the weights. CPU reads and writes of cache tensors (state save and restore,
clearing) become memory operations after a device synchronization.

**Publish.** The mapping of a publisher is a file on a tmpfs
(`LLAMA_KV_HOST=PATH`). Saving the state of its context, which is the
engine's existing call, publishes: the patch writes the metadata of the
cells (position, sequence, token; 16 bytes per cell) and none of their rows,
makes the rows in use read-only in the publisher with one `mprotect` per
tensor, and marks their cells frozen.

**Attach.** An agent creates its context with the number of rows of the
prefix (`LLAMA_KV_PREFIX`). For every tensor the patch maps the whole pages
of those rows from the file, private and read-only, at the start of the
tensor, copies the bytes of the last, partly filled page, and leaves the
rest of the tensor as private anonymous memory. A CPU read pass over the
mapped pages sets their accessed flags, so that the GPU does not fault on
them. Loading the state file restores the metadata of the cells; no row is
read.

**Frozen cells.** The allocator of the cache never hands out a frozen cell.
An agent that drops or rewrites tokens of the prefix gets new cells in its
own tail, and a GPU write can not reach a mapped row by way of the cache.
Behind that sits the page protection: the mapping of the prefix is
read-only, and a write through it is a fault in the process that tried.

**A tail that follows use.** With `LLAMA_KV_GROW` the private part starts
without memory. Before the device writes the cells of a batch, the CPU
populates the pages of their rows, 256 rows ahead. The CPU does this because
a first touch by the GPU is the expensive path on this device
(`RESEARCH_HOSTMM_2026-10-05.md`, Section 3.1), and populating absent pages
needs no invalidation. Upstream allocates and zeroes the cache of the whole
context when the context is created. Backing a cache on demand is known for
device memory (Section 7); here it is ordinary demand paging of host memory.

**Fork.** A publisher that has saved its state keeps running: its next cells
lie after the frozen ones, in the same file, and no agent maps them.
Children attach as agents. Nothing is copied on either side, and no page is
shared copy-on-write.

**Move.** Publish, attach in another process, end the publisher. The file
outlives the process that filled it.

All of this is a property of host mappings, so it holds between processes in
different MIG instances as well.

**The alternative that the rule excludes** is in the patch for comparison
(`LLAMA_KV_COW=1`): the agent maps the whole cache file writable and
private, after the same CPU read pass over the prefix, and the kernel
copies what the GPU writes.

### 6.3 What an agent pays today

`run_engine_prefix.sh` measured, before the mechanism existed, what an agent
that starts from a long shared prefix pays with the engine's own tools and
the weights in place (`results/20261005-engine-prefix-v1/`, six repetitions;
the times are whole `llama-completion` processes that generate one token):

| Prefix (tokens) | Process that recomputes it (ms) | of which prompt evaluation (ms) | Process that restores it from the cache file (ms) | Cache file (MiB) | KiB per token |
|---:|---:|---:|---:|---:|---:|
| 1021 | 2731 | 1314 | 1448 | 55.9 | 56.0 |
| 4081 | 6798 | 5369 | 1481 | 223.2 | 56.0 |

The engine's prompt-cache file removes the recomputation and leaves the
copy: every agent holds its own 223 MiB of the same state, and the file that
carries it is as large again. The campaigns below repeat this baseline in
the same harness as the mechanism and call it "copy".

### 6.4 Agents on one published prefix

`run_engine_kvshare.sh` (`results/20261006-engine-kvshare-v1/`, six
repetitions, 252 cases, no failed agent). A publisher computes a prefix of
4,081 or 16,321 tokens in one MIG instance and exits. Then 1, 4, or 8 agent
processes start at the same time, every second one in the other MIG
instance, each with its own task appended to the prefix, and generate 64
tokens. The publisher's instance alternates between repetitions, and the
order of the modes is reversed in every second repetition.

Handing the state over:

| Prefix (tokens) | Way | Runs (failed) | Compute the prefix (ms) | Hand it over (ms) | State file (MiB) | Cache file in use (MiB) |
|---:|---|---:|---:|---:|---:|---:|
| 4081 | save the state file (upstream) | 6 (0) | 5405 | 230.8 | 223.24 | - |
| 4081 | publish, cache file with 2 MiB pages | 6 (0) | 5443 | 3.0 | 0.06 | 224 |
| 4081 | publish, cache file with 4 KiB pages | 6 (0) | 5556 | 4.3 | 0.06 | 224 |
| 16321 | save the state file (upstream) | 6 (0) | 24453 | 561.2 | 892.80 | - |
| 16321 | publish, cache file with 2 MiB pages | 6 (0) | 24646 | 9.1 | 0.25 | 896 |
| 16321 | publish, cache file with 4 KiB pages | 6 (0) | 24816 | 14.2 | 0.25 | 896 |

Publishing is 3 ms for 4,081 tokens and 9 ms for 16,321, against 231 and
561 ms for writing the state file, and its state file is 0.06 and 0.25 MiB
instead of 223 and 893 MiB. The cache file holds the rows that are in use
and nothing else.

Agents, 4,081-token prefix (context of 8,192 cells):

| Agents | Way to obtain the prefix | Texts equal to copy | Texts equal to recompute: publisher's instance, other instance | Attach (ms) | First token after process start (ms) | Generation, sum (tokens/s) | Generation against copy | Memory of the agents (MiB) | Prefix pages the agents map: resident sum / proportional sum (MiB) |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | recompute | - | 6/6, - | 28 | 6151 | 20.7 | 0.996x [0.993, 0.999] | 1145 | - |
| 1 | copy from the state file (upstream) | 6/6 | 6/6, - | 81 | 819 | 20.8 | - | 1141 | - |
| 1 | copy-on-write mapping, 2 MiB file | 6/6 | 6/6, - | 24 | 864 | 20.5 | 0.984x [0.973, 0.994] | 931 | 238 / 238 |
| 1 | copy-on-write mapping, 4 KiB file | 6/6 | 6/6, - | 27 | 1158 | 20.4 | 0.979x [0.965, 0.994] | 826 | 238 / 238 |
| 1 | extent, whole tail allocated | 6/6 | 6/6, - | 32 | 775 | 20.4 | 0.980x [0.971, 0.989] | 913 | 223 / 223 |
| 1 | extent, tail follows use | 6/6 | 6/6, - | 25 | 761 | 20.4 | 0.981x [0.973, 0.989] | 698 | 223 / 223 |
| 1 | extent, tail follows use, 4 KiB file | 6/6 | 6/6, - | 27 | 776 | 20.4 | 0.978x [0.967, 0.989] | 726 | 223 / 223 |
| 4 | recompute | - | 12/12, 12/12 | 39 | 13199 | 39.1 | 1.014x [1.001, 1.027] | 4556 | - |
| 4 | copy from the state file (upstream) | 24/24 | 12/12, 0/12 | 122 | 1274 | 38.6 | - | 4586 | - |
| 4 | copy-on-write mapping, 2 MiB file | 24/24 | 12/12, 0/12 | 36 | 1493 | 37.7 | 0.977x [0.964, 0.990] | 3235 | 952 / 616 |
| 4 | copy-on-write mapping, 4 KiB file | 24/24 | 12/12, 0/12 | 34 | 2442 | 37.3 | 0.968x [0.955, 0.980] | 3247 | 952 / 616 |
| 4 | extent, whole tail allocated | 24/24 | 12/12, 0/12 | 40 | 1209 | 37.8 | 0.979x [0.969, 0.989] | 3647 | 892 / 229 |
| 4 | extent, tail follows use | 24/24 | 12/12, 0/12 | 33 | 1217 | 37.6 | 0.974x [0.962, 0.987] | 2814 | 892 / 223 |
| 4 | extent, tail follows use, 4 KiB file | 24/24 | 12/12, 0/12 | 37 | 1189 | 37.6 | 0.974x [0.960, 0.988] | 2808 | 892 / 223 |
| 8 | recompute | - | 24/24, 24/24 | 44 | 25608 | 39.0 | 1.014x [1.007, 1.021] | 9158 | - |
| 8 | copy from the state file (upstream) | 48/48 | 24/24, 12/24 | 161 | 1797 | 38.4 | - | 9169 | - |
| 8 | copy-on-write mapping, 2 MiB file | 48/48 | 24/24, 12/24 | 34 | 2186 | 37.1 | 0.966x [0.960, 0.973] | 6545 | 1904 / 1120 |
| 8 | copy-on-write mapping, 4 KiB file | 48/48 | 24/24, 12/24 | 36 | 4130 | 36.9 | 0.959x [0.951, 0.966] | 6529 | 1904 / 1120 |
| 8 | extent, whole tail allocated | 48/48 | 24/24, 12/24 | 45 | 1706 | 37.3 | 0.971x [0.963, 0.979] | 7320 | 1785 / 228 |
| 8 | extent, tail follows use | 48/48 | 24/24, 12/24 | 41 | 1721 | 37.3 | 0.970x [0.956, 0.984] | 5640 | 1785 / 239 |
| 8 | extent, tail follows use, 4 KiB file | 48/48 | 24/24, 12/24 | 49 | 1788 | 37.0 | 0.961x [0.954, 0.969] | 5629 | 1785 / 223 |

Agents, 16,321-token prefix (context of 32,768 cells):

| Agents | Way to obtain the prefix | Texts equal to copy | Texts equal to recompute: publisher's instance, other instance | Attach (ms) | First token after process start (ms) | Generation, sum (tokens/s) | Generation against copy | Memory of the agents (MiB) | Prefix pages the agents map: resident sum / proportional sum (MiB) |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | recompute | - | 6/6, - | 59 | 25254 | 19.9 | 0.995x [0.990, 1.000] | 2556 | - |
| 1 | copy from the state file (upstream) | 6/6 | 6/6, - | 225 | 964 | 20.0 | - | 2532 | - |
| 1 | copy-on-write mapping, 2 MiB file | 6/6 | 6/6, - | 43 | 856 | 19.1 | 0.954x [0.928, 0.981] | 923 | 910 / 910 |
| 1 | copy-on-write mapping, 4 KiB file | 6/6 | 6/6, - | 52 | 1151 | 19.2 | 0.955x [0.925, 0.986] | 815 | 910 / 910 |
| 1 | extent, whole tail allocated | 6/6 | 6/6, - | 70 | 845 | 19.4 | 0.967x [0.932, 1.004] | 1617 | 892 / 892 |
| 1 | extent, tail follows use | 6/6 | 6/6, - | 43 | 791 | 19.2 | 0.955x [0.923, 0.987] | 743 | 892 / 892 |
| 1 | extent, tail follows use, 4 KiB file | 6/6 | 6/6, - | 52 | 814 | 19.3 | 0.961x [0.927, 0.996] | 744 | 892 / 892 |
| 4 | recompute | - | 12/12, 12/12 | 122 | 55866 | 37.2 | 1.062x [1.059, 1.065] | 10275 | - |
| 4 | copy from the state file (upstream) | 24/24 | 12/12, 9/12 | 396 | 1583 | 35.1 | - | 10147 | - |
| 4 | copy-on-write mapping, 2 MiB file | 24/24 | 12/12, 9/12 | 55 | 1413 | 34.6 | 0.986x [0.984, 0.988] | 3325 | 3612 / 1260 |
| 4 | copy-on-write mapping, 4 KiB file | 24/24 | 12/12, 9/12 | 67 | 2377 | 34.3 | 0.979x [0.975, 0.982] | 3326 | 3612 / 1260 |
| 4 | extent, whole tail allocated | 24/24 | 12/12, 9/12 | 93 | 1288 | 34.8 | 0.992x [0.988, 0.996] | 6441 | 3570 / 892 |
| 4 | extent, tail follows use | 24/24 | 12/12, 9/12 | 57 | 1223 | 34.8 | 0.992x [0.987, 0.997] | 2890 | 3570 / 892 |
| 4 | extent, tail follows use, 4 KiB file | 24/24 | 12/12, 9/12 | 62 | 1242 | 34.7 | 0.988x [0.983, 0.993] | 2892 | 3570 / 917 |
| 8 | recompute | - | 24/24, 24/24 | 152 | 110880 | 37.2 | 1.063x [1.054, 1.071] | 20283 | - |
| 8 | copy from the state file (upstream) | 48/48 | 24/24, 15/24 | 484 | 2205 | 35.0 | - | 20331 | - |
| 8 | copy-on-write mapping, 2 MiB file | 48/48 | 24/24, 15/24 | 58 | 1980 | 34.1 | 0.975x [0.966, 0.984] | 6705 | 7224 / 1736 |
| 8 | copy-on-write mapping, 4 KiB file | 48/48 | 24/24, 15/24 | 83 | 3971 | 33.8 | 0.965x [0.958, 0.972] | 6718 | 7224 / 1738 |
| 8 | extent, whole tail allocated | 48/48 | 24/24, 15/24 | 92 | 1774 | 34.4 | 0.984x [0.972, 0.995] | 12969 | 7140 / 1025 |
| 8 | extent, tail follows use | 48/48 | 24/24, 15/24 | 65 | 1778 | 34.4 | 0.982x [0.974, 0.990] | 5849 | 7140 / 908 |
| 8 | extent, tail follows use, 4 KiB file | 48/48 | 24/24, 15/24 | 75 | 1806 | 34.2 | 0.978x [0.969, 0.986] | 5849 | 7140 / 905 |

"Attach" is the creation of the context plus the loading of the state. The
memory column is the largest drop of available memory while the agents run;
it does not contain the cache file, which exists before they start (224 or
896 MiB, once). The generation sum of the agents that recompute is not
comparable with the other rows when several agents run: they reach
generation at different times and overlap less.

**Texts.** Every agent of every mode writes the text of the agent that
obtains the same state by copy: 78 of 78 agents per mode and prefix length.
The mechanism changes where the rows are, not what is computed from them.
Against an agent that recomputes the prefix, the texts are equal for all 42
agents per mode and prefix length that run in the publisher's instance, and
for 12 of 36 (4,081 tokens) and 24 of 36 (16,321 tokens) of the agents in
the other instance, for the copy exactly as for the extents. Section 6.6
gives the reason.

**Memory.** Eight agents on a 16,321-token prefix hold 19.9 GiB when each
copies the prefix, 12.7 GiB when each maps it and allocates its whole tail,
and 5.7 GiB when the tail follows use; the cache file adds 0.9 GiB once.
Per agent that is 2,541, 1,621 and 731 MiB. The first step, 920 MiB, is the
prefix that is no longer duplicated; the second, 890 MiB, is the half of
the context that the agent has not written. The eight mappings of the
prefix sum to 7,140 MiB of resident pages and 908 MiB of proportional
share: one copy. With the 4,081-token prefix the three numbers are 9.0, 7.1
and 5.5 GiB.

**Attach and first token.** Attaching takes 25 to 65 ms and hardly depends
on the prefix (43 ms for one agent on 16,321 tokens); copying takes 81 to
484 ms and grows with the prefix and with the number of agents that copy at
once. The first token after process start moves less, because loading the
model (about 590 ms) and creating the CUDA context dominate it: 791 against
964 ms for one agent on the long prefix, 1,778 against 2,205 ms for eight.
Attach time is not where the mechanism earns its place; memory is.

**Generation.** Agents on extents generate 0.955 to 0.992 times as fast as
agents that copy. The campaign's gate was 0.97; it holds in four of six
cells and fails in two (0.955 [0.923, 0.987] for one agent on the long
prefix, 0.970 [0.956, 0.984] for eight on the short one). The page size of
the cache file is not the reason: the 4 KiB file is within one percentage
point of the 2 MiB file in every cell. Section 6.8 decomposes the loss.

**Copy-on-write** with a CPU read pass over the prefix gives the same texts.
On the 2 MiB file it holds 105 to 233 MiB more per agent than extents (the
2 MiB blocks around the place where each tensor is written) and reaches the
first token 8 to 27% later. On the 4 KiB file eight agents reach the first
token after 4.0 to 4.1 s against 1.7 to 1.8 s, because every copied page
costs its invalidation. Section 6.7 measures what happens without the read
pass.

### 6.5 A parent that forks while it runs

`run_engine_kvfork.sh` (`results/20261006-engine-kvfork-v1/`, six
repetitions, no failed process). A parent computes a 4,081-token prefix,
hands its state to four children, two of them in the other MIG instance,
and generates 64 tokens of its own continuation while the children generate
theirs.

| Hand-over | Runs (failed processes) | Pause of the parent (ms) | State file (MiB) | Parent text equal to alone | Child texts equal to copy | Child texts equal to alone: parent's instance, other instance | Child attach (ms) | Child first token (ms) | Memory of the children (MiB) | Prefix pages the children map: resident sum / proportional sum (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| copy (state file with the rows) | 6 (0) | 218.2 | 223.24 | 6/6 | 24/24 | 12/12, 0/12 | 150 | 1373 | 4550 | - |
| extent (freeze and map) | 6 (0) | 3.0 | 0.06 | 6/6 | 24/24 | 12/12, 0/12 | 33 | 1257 | 2912 | 892 / 225 |

The parent pauses for 3 ms instead of 218 ms. It writes the text of a
process that computes the same tokens alone in six of six repetitions, in
both modes: handing its state over, and children that read it while it
continues, do not change what the parent computes. Every child writes the
text of the child that received the state by copy (24 of 24), and the text
of a process alone when it runs in the parent's instance (12 of 12). The
children hold 2.8 GiB instead of 4.4 GiB.

The generation rates of this campaign cannot be compared between the two
modes. Children on extents reach generation about 120 ms earlier and compete
with the parent for longer, so the parent's rate over its 64 tokens is lower
(10.2 against 11.3 tokens/s in the 12-SM instance) for a reason that is not
a cost. Section 6.8 measures the speed with equal overlap.

A move is the same operation followed by the exit of the first process;
Section 6.4 is that case, with the agent in the other MIG instance for every
second agent.

### 6.6 The bits of a cache across MIG instances

`run_engine_kvdet.sh` (`results/20261006-engine-kvdet-v1/`, six repetitions)
computes the cache of one prefix into a file in each MIG instance and
compares the files.

| Prefix (tokens) | Runs (failed) | Distinct caches, 12-SM instance | Distinct caches, 6-SM instance | Runs with the same bits in both instances | 16-bit values that differ (mean) | Largest difference | First tensor that differs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 6 (0) | 1 | 1 | 0/6 | 6922004 | 5.88867 | 0 |
| 1021 | 6 (0) | 1 | 1 | 0/6 | 24179266 | 13.2109 | 0 |
| 4081 | 6 (0) | 1 | 1 | 0/6 | 107376991 | 13.4688 | 0 |

Each instance computes the same bits every time. The two instances never
compute the same bits: nine of ten 16-bit values differ from the first
tensor on. The engine is deterministic within an instance and not across
the 12-SM and the 6-SM instance. Therefore, an agent that attaches to a
prefix from the other instance continues from a cache that its own instance
would not have computed, and after 64 tokens its text is sometimes different
from a recomputation there. This holds for the engine's own state file as well,
so it is a property of moving state between the instances, not of how the
state is moved. The comparison that tests the mechanism is with the copy.

### 6.7 Copy-on-write, with and without the remedies

The kernel's own way to give a process a private view of another process's
memory is a private writable mapping. `run_engine_kvcow.sh`
(`results/20261006-engine-kvcow-v1/`, six repetitions, no failed agent) has
1 and 8 agents obtain a 16,321-token prefix that way, with and without the
CPU read pass that Section 2 prescribes, over a cache file with 2 MiB and
with 4 KiB pages:

| Agents | Way to obtain the prefix | Runs (failed agents) | Texts equal to copy | Decode of the agent's own task (ms) | First token after process start (ms) | Generation against copy | Memory of the agents (MiB) | Per agent above extents (MiB) | Prefix pages the agents map: resident sum / proportional sum (MiB) |
|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | copy from the state file (upstream) | 6 (0) | 6/6 | 104 | 999 | - | 2549 | 1812 | - |
| 1 | copy-on-write, 2 MiB file, CPU read pass | 6 (0) | 6/6 | 178 | 854 | 0.961x [0.933, 0.989] | 892 | 154 | 910 / 910 |
| 1 | copy-on-write, 2 MiB file, no read pass | 6 (0) | 6/6 | 654 | 1339 | 0.953x [0.926, 0.979] | 1648 | 911 | 910 / 910 |
| 1 | copy-on-write, 4 KiB file, CPU read pass | 6 (0) | 6/6 | 473 | 1160 | 0.959x [0.927, 0.992] | 828 | 90 | 910 / 910 |
| 1 | copy-on-write, 4 KiB file, no read pass | 6 (0) | 6/6 | 701 | 1393 | 0.957x [0.921, 0.995] | 1632 | 894 | 910 / 910 |
| 1 | extent, tail follows use | 6 (0) | 6/6 | 110 | 790 | 0.967x [0.936, 0.999] | 738 | 0 | 892 / 892 |
| 8 | copy from the state file (upstream) | 6 (0) | 48/48 | 339 | 2181 | - | 20261 | 1815 | - |
| 8 | copy-on-write, 2 MiB file, CPU read pass | 6 (0) | 48/48 | 595 | 2001 | 0.974x [0.953, 0.996] | 6704 | 120 | 7224 / 1742 |
| 8 | copy-on-write, 2 MiB file, no read pass | 6 (0) | 48/48 | 3511 | 4941 | 0.967x [0.956, 0.977] | 12992 | 907 | 7224 / 7224 |
| 8 | copy-on-write, 4 KiB file, CPU read pass | 6 (0) | 48/48 | 2469 | 3923 | 0.957x [0.948, 0.966] | 6720 | 122 | 7224 / 1737 |
| 8 | copy-on-write, 4 KiB file, no read pass | 6 (0) | 48/48 | 3727 | 5161 | 0.967x [0.957, 0.976] | 12990 | 906 | 7224 / 7224 |
| 8 | extent, tail follows use | 6 (0) | 48/48 | 322 | 1735 | 0.980x [0.973, 0.988] | 5740 | 0 | 7140 / 1036 |

Without the read pass the mapping copies on read. The GPU faults on the
pages of the prefix, the driver serves the faults as writes, and every agent
ends with a private copy of the whole prefix: 894 to 911 MiB more per agent
than with extents for an 893 MiB prefix, on either page size, and the
proportional share of the mapped prefix equals its resident sum (7,224 MiB
for eight agents: eight copies). Eight agents hold 12.7 GiB against 5.6 GiB.
The copies are made while the agent decodes its own task, which takes 3.5 to
3.7 s against 0.32 s, and the first token comes after 4.9 to 5.2 s against
1.7 s. This is the amplification of Section 2 in the engine: the agent never
writes the prefix.

With the read pass the mapping copies only the 2 MiB blocks that the agent
writes, 90 to 154 MiB per agent. On the 2 MiB file eight agents then reach
the first token after 2.0 s, on the 4 KiB file after 3.9 s, because the
entries of the copied block are present and each costs its invalidation.
The texts are those of the copy in every case; the cost is memory and time,
not correctness.

Copy-on-write with every remedy applied, on the 2 MiB file, comes within
15% of extents in first-token time and within 160 MiB per agent. The difference between the
two designs is what happens when a remedy is missing: a read pass that is
left out, or a page whose accessed flag the kernel has cleared since, turns
the shared prefix into a private one without an error. Extents have no such
state. A mapping that is read-only cannot be copied by a GPU read, and a
tail that was private from the start has nothing to copy.

### 6.8 Where the generation speed goes

Section 6.4 measures 64 generated tokens and finds agents on extents 0.8 to
4.5% slower than agents that copy. `run_engine_kvspeed.sh` generates 128 or
256 tokens with one process in each MIG instance, for both prefix lengths,
and separates the kinds of host memory; in the 12-SM instance with the short
prefix it also runs two processes that are time-sliced (four result
directories `results/20261006-engine-kvspeed-*`, six repetitions each, no
failed run):

| MIG instance | Prefix (tokens) | Tokens generated | Processes | Cache | Runs (failed) | Generation, sum (tokens/s) | Against device memory |
|---|---:|---:|---:|---|---:|---:|---|
| 12-SM | 4081 | 256 | 1 | device memory, prefix computed (upstream) | 6 (0) | 25.28 | - |
| 12-SM | 4081 | 256 | 1 | private host memory, 2 MiB pages, prefix computed | 6 (0) | 25.29 | 1.000x [0.998, 1.003] |
| 12-SM | 4081 | 256 | 1 | private host memory that follows use, 4 KiB pages, prefix computed | 6 (0) | 25.01 | 0.989x [0.987, 0.991] |
| 12-SM | 4081 | 256 | 1 | device memory, prefix copied from the state file | 6 (0) | 25.27 | 1.000x [0.997, 1.003] |
| 12-SM | 4081 | 256 | 1 | mapped prefix, private tail on 2 MiB pages | 6 (0) | 25.18 | 0.996x [0.993, 0.999] |
| 12-SM | 4081 | 256 | 1 | mapped prefix, private tail that follows use | 6 (0) | 25.17 | 0.996x [0.992, 0.999] |
| 12-SM | 4081 | 256 | 2 | device memory, prefix computed (upstream) | 6 (0) | 24.31 | - |
| 12-SM | 4081 | 256 | 2 | private host memory, 2 MiB pages, prefix computed | 6 (0) | 24.43 | 1.005x [1.004, 1.006] |
| 12-SM | 4081 | 256 | 2 | private host memory that follows use, 4 KiB pages, prefix computed | 6 (0) | 24.22 | 0.996x [0.996, 0.997] |
| 12-SM | 4081 | 256 | 2 | device memory, prefix copied from the state file | 6 (0) | 24.32 | 1.000x [0.999, 1.002] |
| 12-SM | 4081 | 256 | 2 | mapped prefix, private tail on 2 MiB pages | 6 (0) | 24.38 | 1.003x [1.002, 1.003] |
| 12-SM | 4081 | 256 | 2 | mapped prefix, private tail that follows use | 6 (0) | 24.35 | 1.002x [1.001, 1.003] |
| 12-SM | 16321 | 128 | 1 | device memory, prefix computed (upstream) | 6 (0) | 23.89 | - |
| 12-SM | 16321 | 128 | 1 | private host memory, 2 MiB pages, prefix computed | 6 (0) | 23.98 | 1.004x [0.998, 1.009] |
| 12-SM | 16321 | 128 | 1 | private host memory that follows use, 4 KiB pages, prefix computed | 6 (0) | 23.91 | 1.001x [1.000, 1.002] |
| 12-SM | 16321 | 128 | 1 | device memory, prefix copied from the state file | 6 (0) | 23.94 | 1.002x [0.998, 1.006] |
| 12-SM | 16321 | 128 | 1 | mapped prefix, private tail on 2 MiB pages | 6 (0) | 23.94 | 1.002x [0.998, 1.006] |
| 12-SM | 16321 | 128 | 1 | mapped prefix, private tail that follows use | 6 (0) | 23.93 | 1.002x [0.997, 1.006] |
| 6-SM | 4081 | 256 | 1 | device memory, prefix computed (upstream) | 6 (0) | 16.26 | - |
| 6-SM | 4081 | 256 | 1 | private host memory, 2 MiB pages, prefix computed | 6 (0) | 15.90 | 0.978x [0.972, 0.984] |
| 6-SM | 4081 | 256 | 1 | private host memory that follows use, 4 KiB pages, prefix computed | 6 (0) | 15.78 | 0.970x [0.968, 0.972] |
| 6-SM | 4081 | 256 | 1 | device memory, prefix copied from the state file | 6 (0) | 16.28 | 1.001x [1.000, 1.002] |
| 6-SM | 4081 | 256 | 1 | mapped prefix, private tail on 2 MiB pages | 6 (0) | 15.90 | 0.978x [0.975, 0.981] |
| 6-SM | 4081 | 256 | 1 | mapped prefix, private tail that follows use | 6 (0) | 15.88 | 0.976x [0.974, 0.979] |
| 6-SM | 16321 | 128 | 1 | device memory, prefix computed (upstream) | 6 (0) | 15.85 | - |
| 6-SM | 16321 | 128 | 1 | private host memory, 2 MiB pages, prefix computed | 6 (0) | 14.95 | 0.943x [0.939, 0.948] |
| 6-SM | 16321 | 128 | 1 | private host memory that follows use, 4 KiB pages, prefix computed | 6 (0) | 14.91 | 0.941x [0.938, 0.943] |
| 6-SM | 16321 | 128 | 1 | device memory, prefix copied from the state file | 6 (0) | 15.89 | 1.002x [1.000, 1.005] |
| 6-SM | 16321 | 128 | 1 | mapped prefix, private tail on 2 MiB pages | 6 (0) | 14.93 | 0.942x [0.940, 0.944] |
| 6-SM | 16321 | 128 | 1 | mapped prefix, private tail that follows use | 6 (0) | 14.93 | 0.942x [0.940, 0.945] |

In the 12-SM instance the place of the cache does not matter. A cache in
private host memory generates 1.000 and 1.004 times as fast as a cache in
device memory, a cache on extents 0.996 and 1.002 times, and two
time-sliced processes are at 1.002 to 1.005. The one lasting cost there is a
short cache that lies on 4 KiB pages as a whole (0.989).

In the 6-SM instance a cache in host memory is slower: 0.978 times the
device-memory cache with the 4,081-token prefix and 0.942 to 0.943 times
with the 16,321-token prefix. A private cache with nothing shared loses the
same as a cache on extents, and at the long prefix the page size makes no
difference. The loss belongs to host memory in the small instance, not to
the mapping, and it grows with the cache that every token reads: 1.4 ms per
token for 223 MiB and 3.9 ms for 893 MiB. The copy, which is device memory,
is at 1.001 to 1.002 in both instances.

This is the loss of Section 6.4. Split by instance, its single agent keeps
0.98 to 1.00 of the copy's speed in the 12-SM instance and 0.975 and 0.93 in
the 6-SM instance, and a check with the time of every generated token
(manual, 16,321-token prefix, 12-SM instance, three runs per mode) shows the
copy and the extents at the same 41.7 ms per token from the first token on.
With several agents the rates of Section 6.4 also contain an effect that is
not a cost: agents on extents attach sooner, all reach generation together,
and overlap for longer than agents that copy.

We did not find the cause inside the GPU. The substrate measurement that a
kernel reads host memory at the speed of device memory
(`RESEARCH_HOSTMM_2026-10-05.md`, Section 7) is a sequential read of 1 GiB
by a synthetic kernel; it does not carry over to the attention kernels of
the engine in the 6-SM instance. For a deployment the consequence is a rule of
placement: agents with a long shared prefix belong in the larger instance,
or pay 2 to 6% of their generation speed for the memory they save.

### 6.9 The device-memory counterpart

Inside one GPU instance CUDA has its own way to compose a range from shared
and private memory: the virtual memory interface (`cuMemCreate`,
`cuMemExportToShareableHandle`, `cuMemMap`). The closest prior designs share
a cache between processes that way, in device memory (Section 7).

**The route.** `cuda_vmm_probe.cu` exports an allocation from one process,
and a second process maps it read-only next to an allocation of its own, and
the GPU reads across the seam (`results/20261006-vmm-routes-v1/`, six
repetitions):

| Placement | Shared and private device memory composed | Refusal | Read-only access to the shared part | Producer's data unchanged |
|---|---:|---|---|---:|
| two processes in one MIG instance | 6/6 | - | enforced | 6/6 |
| one process in each MIG instance | 0/6 | `UNSUPPORTED_at_import_CUDA_ERROR_NOT_INITIALIZED` | - | 0/0 |
| two clients of one MPS server | 6/6 | - | enforced | 6/6 |

Between two processes of one MIG instance, also as clients of one MPS
server, the composition works, at a granularity of 2 MiB, and read-only
access keeps the consumer from changing the producer's memory. Between the
two MIG instances the import is refused.

**In the engine.** `kv_vmm.patch`, in a third clone (`llama.cpp-vmm`), builds
the layout of Section 6.2 from device memory. The cache consists of 2 MiB
allocations of the virtual memory interface. A publisher makes the
allocations that a prefix fills read-only and exports them; a child maps
them read-only, copies the allocation that the prefix ends in, and maps
allocations of its own behind them as it grows. The handles are file
descriptors, so the publisher starts its children itself and they inherit
them (`kv_spawn.cpp`). This is our implementation of that design, written
for the comparison; it is not one of the systems of Section 7.
`run_engine_kvvmm.sh` (`results/20261006-engine-kvvmm-v1/`, six repetitions,
three with the parent in each MIG instance) runs a parent and four children
in one instance with the copy, with host extents, and with the device-memory
cache, all in the one engine build:

| Parent's instance | Prefix (tokens) | Hand-over | Runs | Pause of the parent (ms) | State file (MiB) | Children that ran | Child texts equal to copy | Child attach (ms) | Child first token (ms) | Child generation (tokens/s, each) | Parent generation (tokens/s) | Memory of the children (MiB) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12-SM | 4081 | copy (state file with the rows) | 3 | 216.2 | 223.24 | 12/12 | 12/12 | 192 | 1581 | 5.34 | 7.63 | 4546 |
| 12-SM | 4081 | host memory: mapped file, private tail | 3 | 3.0 | 0.06 | 12/12 | 12/12 | 27 | 1363 | 5.25 | 6.89 | 2948 |
| 12-SM | 4081 | device memory: shared allocations, private tail | 3 | 7.6 | 0.06 | 12/12 | 12/12 | 325 | 1684 | 5.41 | 7.53 | 3807 |
| 12-SM | 4081 | host memory, one child in the other instance | 3 | 2.8 | 0.06 | 3/3 | 3/3 | 25 | 701 | 15.58 | 23.88 | 809 |
| 12-SM | 4081 | device memory, one child in the other instance | 3 | 7.3 | 0.06 | 0/3 | - | - | - | - | 24.70 | 386 |
| 6-SM | 4081 | copy (state file with the rows) | 3 | 213.6 | 223.24 | 12/12 | 12/12 | 208 | 1678 | 3.28 | 3.98 | 4553 |
| 6-SM | 4081 | host memory: mapped file, private tail | 3 | 3.0 | 0.06 | 12/12 | 12/12 | 27 | 1479 | 3.16 | 3.68 | 2876 |
| 6-SM | 4081 | device memory: shared allocations, private tail | 3 | 7.2 | 0.06 | 12/12 | 12/12 | 332 | 1762 | 3.31 | 4.03 | 3727 |
| 6-SM | 4081 | host memory, one child in the other instance | 3 | 2.8 | 0.06 | 3/3 | 3/3 | 24 | 693 | 23.69 | 15.49 | 837 |
| 6-SM | 4081 | device memory, one child in the other instance | 3 | 7.4 | 0.06 | 0/3 | - | - | - | - | 16.16 | 416 |
| 12-SM | 16321 | copy (state file with the rows) | 3 | 551.1 | 892.80 | 12/12 | 12/12 | 500 | 1882 | 5.11 | 7.50 | 10148 |
| 12-SM | 16321 | host memory: mapped file, private tail | 3 | 8.2 | 0.25 | 12/12 | 12/12 | 48 | 1432 | 5.00 | 6.54 | 3078 |
| 12-SM | 16321 | device memory: shared allocations, private tail | 3 | 41.0 | 0.25 | 12/12 | 12/12 | 438 | 1835 | 5.14 | 7.48 | 3968 |
| 12-SM | 16321 | host memory, one child in the other instance | 3 | 7.9 | 0.25 | 3/3 | 0/3 | 44 | 728 | 14.48 | 22.08 | 835 |
| 12-SM | 16321 | device memory, one child in the other instance | 3 | 42.6 | 0.25 | 0/3 | - | - | - | - | 23.37 | 271 |
| 6-SM | 16321 | copy (state file with the rows) | 3 | 570.6 | 892.80 | 12/12 | 12/12 | 524 | 2000 | 3.24 | 4.01 | 10092 |
| 6-SM | 16321 | host memory: mapped file, private tail | 3 | 9.0 | 0.25 | 12/12 | 12/12 | 49 | 1553 | 2.98 | 3.44 | 3040 |
| 6-SM | 16321 | device memory: shared allocations, private tail | 3 | 39.1 | 0.25 | 12/12 | 12/12 | 434 | 1915 | 3.25 | 4.02 | 3904 |
| 6-SM | 16321 | host memory, one child in the other instance | 3 | 8.0 | 0.25 | 3/3 | 3/3 | 43 | 721 | 21.71 | 14.32 | 829 |
| 6-SM | 16321 | device memory, one child in the other instance | 3 | 39.5 | 0.25 | 0/3 | - | - | - | - | 15.66 | 305 |

The texts are the same in all three: the parent's in 3 of 3 repetitions per
cell and the children's in 12 of 12.

**Speed.** The device-memory cache generates at the speed of the copy, which
is device memory too: its children are at 1.00 to 1.01 times the copy in
both instances. The children on host extents are at 0.98 in the 12-SM
instance and at 0.96 (4,081 tokens) and 0.92 (16,321 tokens) in the 6-SM
instance. Part of that is the cost of host memory in the small instance
(Section 6.8) and part is the overlap of Section 6.5: children on host
extents reach generation 0.2 to 0.5 s before the others and compete for
longer.

**Attach and first token.** A child on host extents attaches in 27 ms and
48 ms for the two prefix lengths; a child of the device-memory cache needs
325 to 332 ms and 435 to 438 ms, about as long as the copy (192 to 524 ms).
It pays one import and one mapping for every 2 MiB allocation of every
tensor, and a device copy for the allocation that the prefix ends in. The
first token of a child comes after 1.36 to 1.55 s with host extents, 1.68
to 1.92 s with the device-memory cache, and 1.58 to 2.00 s with the copy.
The parent pauses for 3 to 9 ms, 7 to 41 ms, and 214 to 571 ms.

**Memory.** Four children hold 2.9 and 3.0 GiB on host extents, 3.7 and
3.9 GiB on the device-memory cache, and 4.4 and 9.9 GiB with the copy. The
device-memory cache shares whole 2 MiB allocations only, which is one
allocation in two per tensor for the 4,081-token prefix, and it grows by
2 MiB per tensor, 112 MiB, at a time.

**Reach.** With one child in the other MIG instance the child on host
extents runs in 12 of 12 attempts and the child of the device-memory cache
in 0 of 12: the import is refused.

Inside one instance both designs work. Device memory keeps the generation
speed of device memory. Host extents attach an order of magnitude faster,
hold about a fifth less, need a path and not a live exporter, and are the
only one of the two that reaches the other instance.

### 6.10 When agents may share a process

Section 4 found that one process that batches sequences generates about
three times as much as separate processes. The engine also shares a prefix
between the sequences of one process: the cells of the prefix get every
sequence id, and no row is copied. This is the sharing that the prefix
caches of serving engines provide, and it is the comparison that extents
have to stand. `kv_batch.cpp` is a server of that kind, and
`run_engine_kvbatch.sh` (`results/20261006-engine-kvbatch-v1/`, six
repetitions, no failed server) has it serve eight agents on the
16,321-token prefix, 64 tokens each, as one server or as one server in each
MIG instance:

| Configuration | Runs (failed servers) | Generation, all agents (tokens/s) | 12-SM server | 6-SM server | First server ready (s) | All agents ready (s) | Hand-over (ms) | State file (MiB) | Memory of all servers (MiB) | Texts equal to the copy configuration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one server in the 12-SM instance, 8 sequences | 6 (0) | 75.6 | 75.6 | - | 20.7 | 20.7 | - | - | 2578 | 36/48 |
| a server in each instance, 4 sequences each; each computes the prefix | 6 (0) | 106.2 | 61.7 | 44.5 | 20.9 | 30.6 | - | - | 5081 | 36/48 |
| a server in each instance; the second copies the state file | 6 (0) | 105.0 | 61.4 | 43.7 | 21.2 | 22.2 | 565.4 | 892.80 | 5101 | 48/48 |
| a server in each instance; the second maps the published prefix | 6 (0) | 102.9 | 60.9 | 41.9 | 20.7 | 21.4 | 8.8 | 0.25 | 2437 | 48/48 |

One server holds eight agents in 2.5 GiB and generates 75.6 tokens/s. Eight
separate processes on extents hold 5.7 GiB and generate 34 tokens/s
(Section 6.4). Whenever the agents can live in one process, sharing inside
the process is the better design, and extents do not change that.

Two servers, one in each MIG instance, generate 103 to 106 tokens/s, 1.36 to
1.41 times the single server, because they use both instances. Between two
servers the engine's own sharing ends. Each computes the prefix, and the
server in the 6-SM instance has its agents ready after 30.6 s; or the second
copies the state of the first, and the two hold 5.0 GiB.

With extents between the two servers the second maps the prefix that the
first computed. Both together hold 2.4 GiB, half of the two servers that
copy and no more than the single server, whose cache is allocated whole. All
eight agents are ready after 21.4 s, 0.7 s after the first server. The
servers generate 102.9 tokens/s, 2% less than the two that copy; the
difference is the server in the 6-SM instance (41.9 against 43.7 tokens/s),
which is the cost of host memory there (Section 6.8). Every agent writes the
text of its counterpart in the copy configuration (48 of 48).

Thus, extents are not an alternative to sharing inside a process. They
continue it across the process boundary and the MIG boundary, where it
stops. The best configuration measured for eight agents on this device is
one batching server in each instance on one published prefix.

### 6.11 Compute-sharing modes

`run_engine_kvmps.sh` (`results/20261006-engine-kvmps-v1/`, six
repetitions, 192 cases, no failed process) runs a parent and four or eight
children under time slicing, one MPS server, the two MIG instances, and one
MPS server in each instance. It repeats the 4,081-token and 16,321-token
prefixes with a copy and with extents.

| Compute shared by | Children | Prefix (tokens) | Memory, copy / extents (MiB) | Throughput, copy / extents (tokens/s) | Extents / copy | Parent pause, copy / extents (ms) | Attach, copy / extents (ms) | Extent texts equal to copy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| time slicing | 4 | 4,081 | 4607 / 2945 | 21.32 / 20.99 | 0.985 | 207.8 / 3.1 | 199.6 / 32.8 | 24/24 |
| time slicing | 8 | 4,081 | 9250 / 5925 | 23.06 / 22.74 | 0.986 | 223.6 / 3.0 | 279.3 / 31.0 | 48/48 |
| time slicing | 4 | 16,321 | 10167 / 3067 | 20.26 / 20.05 | 0.990 | 540.5 / 8.5 | 458.6 / 63.1 | 24/24 |
| time slicing | 8 | 16,321 | 20300 / 5894 | 21.83 / 21.68 | 0.993 | 539.8 / 8.7 | 576.3 / 51.1 | 48/48 |
| two MIG instances | 4 | 4,081 | 4810 / 2934 | 33.15 / 32.22 | 0.972 | 217.2 / 2.8 | 146.2 / 32.3 | 24/24 |
| two MIG instances | 8 | 4,081 | 9109 / 5795 | 36.09 / 34.71 | 0.962 | 213.2 / 2.8 | 198.6 / 33.4 | 48/48 |
| two MIG instances | 4 | 16,321 | 10188 / 3135 | 30.46 / 29.52 | 0.969 | 537.6 / 9.0 | 385.7 / 59.8 | 24/24 |
| two MIG instances | 8 | 16,321 | 20189 / 6005 | 32.97 / 32.05 | 0.972 | 557.0 / 8.7 | 477.6 / 56.4 | 48/48 |
| one MPS server | 4 | 4,081 | 4726 / 2898 | 23.81 / 23.52 | 0.988 | 212.9 / 3.0 | 121.2 / 29.6 | 24/24 |
| one MPS server | 8 | 4,081 | 9179 / 5827 | 25.66 / 25.58 | 0.997 | 212.2 / 3.0 | 145.1 / 30.4 | 48/48 |
| one MPS server | 4 | 16,321 | 10236 / 3022 | 22.83 / 22.49 | 0.985 | 542.1 / 9.2 | 407.7 / 60.6 | 24/24 |
| one MPS server | 8 | 16,321 | 20283 / 5853 | 24.45 / 24.34 | 0.996 | 528.4 / 9.2 | 514.3 / 51.5 | 48/48 |
| one MPS server per MIG instance | 4 | 4,081 | 4760 / 2936 | 36.70 / 35.31 | 0.962 | 219.1 / 2.9 | 103.8 / 27.8 | 24/24 |
| one MPS server per MIG instance | 8 | 4,081 | 9105 / 5713 | 40.42 / 38.78 | 0.959 | 214.4 / 2.9 | 133.8 / 31.8 | 48/48 |
| one MPS server per MIG instance | 4 | 16,321 | 10185 / 2993 | 33.93 / 32.39 | 0.955 | 543.7 / 7.7 | 374.1 / 50.0 | 24/24 |
| one MPS server per MIG instance | 8 | 16,321 | 20129 / 5820 | 37.03 / 35.72 | 0.965 | 531.8 / 9.2 | 454.4 / 54.3 | 48/48 |

With eight children on the 16,321-token prefix, the copy holds 19.7-19.8
GiB in every configuration, and extents hold 5.7-5.9 GiB in addition to the
0.9 GiB cache file that exists once. The throughput ratios are 0.993 under
time slicing, 0.996 under MPS, 0.972 under MIG, and 0.965 under MPS inside
the two MIG instances. The lower two ratios contain the 6-SM instance, for
which Section 6.8 measures the cost of a host-memory cache. The pause of the
parent falls from 528-557 ms to 8.7-9.2 ms, and the attach time falls from
454-576 ms to 51-56 ms. Every one of the 288 children on extents writes the
text of its copy counterpart.

The host page table does not change with the way in which processes share
the GPU. MPS changes when the processes execute and which faults they share,
and MIG changes which SMs execute a process, but the mappings of each
process still name the same host pages. The result is one state-sharing path
for every placement measured here.

### 6.12 Stress, implementation, and external-baseline campaigns

The following campaigns extend the main matrix without changing its raw
measurements.  Every cell has six repetitions.  Unless an exception is
called out below, every process or request completed, no run timed out, and
the generated token sequence matched its copy counterpart.

**Mechanisms in one engine.** `run_engine_kvsota.sh` compares five ways of
handing the same 16,321-token prefix to eight children.  `same` puts every
child in the publisher's MIG instance; `cross` alternates them across the
two instances.

| Placement | Hand-over | Children complete | Texts equal to copy | Parent pause (ms) | Attach (ms) | First token (ms) | Memory (MiB) | Throughput (tokens/s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| same | copy | 48/48 | 48/48 | 539.54 | 548.6 | 2670 | 20306 | 21.85 |
| same | demand-backed copy | 48/48 | 48/48 | 542.80 | 596.6 | 2721 | 14210 | 21.98 |
| same | shared device memory | 48/48 | 48/48 | 41.04 | 769.5 | 2864 | 7808 | 22.33 |
| same | copy-on-write | 48/48 | 48/48 | 8.12 | 50.3 | 2770 | 6892 | 21.54 |
| same | extents | 48/48 | 48/48 | 8.37 | 48.8 | 2137 | 6036 | 21.74 |
| cross | copy | 48/48 | 48/48 | 548.04 | 494.8 | 2395 | 20259 | 33.23 |
| cross | demand-backed copy | 48/48 | 48/48 | 539.12 | 559.1 | 2445 | 13732 | 33.55 |
| cross | shared device memory | 24/48 | 24/24 | 39.73 | 648.6 | 2541 | 3984 | 21.73 |
| cross | copy-on-write | 48/48 | 48/48 | 8.87 | 50.9 | 2279 | 6785 | 32.05 |
| cross | extents | 48/48 | 48/48 | 8.00 | 50.4 | 1918 | 5923 | 32.29 |

The device-memory route completes only the 24 children in the publisher's
instance when placement crosses the MIG boundary.  Copy, demand-backed
copy, copy-on-write, and extents complete all 48.  Extents use 5.8 to 5.9
GiB instead of 19.8 GiB, reduce the pause of the publisher from 540 to 548
ms to 8 ms, and attach in 49 to 50 ms.  Copy-on-write mappings (with the CPU
read of the prefix) also pause for 8 to 9 ms, attach in 50 to 51 ms, keep the
output and cross the MIG boundary; against them, extents hold less (5.9
against 6.7 GiB in one instance, 5.8 against 6.6 GiB across the two) and
reach the first token sooner (2.14 against 2.77 s, and 1.92 against 2.28 s).
(An earlier version of this paragraph gave 20.3 GiB and 5.9 to 6.0 GiB,
which were MiB divided by 1000, and called extents the only route with a
short pause, the same output and both instances, which the copy-on-write
row contradicts.)

**Trees.** `run_engine_kvtree.sh` builds two leaders and four leaves per
leader.  With a chain each leader publishes the rows it adds; the flat route
makes leaves recompute those rows.

| Tree | Runs (failed, timed out) | Root / leader publish (ms) | Leaf attach (ms) | Leaf first token (ms) | Memory (MiB) | Leaves complete | Exact texts | Minimum common tokens | Files removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| copy | 6 (0, 0) | 540.74 / 603.77 | 616.9 | 2889 | 27954 | 48 | 48/48 | 64 | - |
| one published prefix | 6 (0, 0) | 8.70 / - | 48.0 | 18880 | 9843 | 48 | 42/48 | 41 | 6/6 |
| extent chain, 2 MiB pages | 6 (0, 0) | 9.01 / 6.19 | 53.5 | 2246 | 9469 | 48 | 48/48 | 64 | 6/6 |
| extent chain, 4 KiB pages | 6 (0, 0) | 8.82 / 11.05 | 57.3 | 2349 | 9204 | 48 | 48/48 | 64 | 6/6 |

Both extent-chain page sizes produce all 48 copy-exact leaf sequences and
remove every segment file after each run.  The flat route produces 42 of 48
exact sequences; all 48 have at least 41 common generated tokens.  This is a
deterministic consequence of recomputing the group prefix on the leaf's MIG
instance, not an attach failure.

**Scale, models, and an agent trace.** `run_engine_kvscale.sh` uses 8, 16,
and 32 Qwen2.5-7B agents.  The model variants repeat the eight-agent cell
with Llama-3.1-8B and Qwen2.5-14B, and the trace variant uses a 14,282-token
agent workload.

| Workload | Agents | Mode | Runs (failed agents) | Exact texts | Attach (ms) | First token (ms) | Throughput (tokens/s) | Extents / copy | Memory (MiB) | Memory saved (MiB) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5 7B | 8 | restore | 6 (0) | 48/48 | 534.2 | 2248 | 34.79 | 1.0000 | 20343 | 0 |
| Qwen2.5 7B | 8 | extent_lazy | 6 (0) | 48/48 | 57.7 | 1713 | 34.32 | 0.9863 | 5855 | 14488 |
| Qwen2.5 7B | 16 | restore | 6 (0) | 96/96 | 715.6 | 3790 | 34.55 | 1.0000 | 40600 | 0 |
| Qwen2.5 7B | 16 | extent_lazy | 6 (0) | 96/96 | 65.9 | 3081 | 34.06 | 0.9859 | 11663 | 28937 |
| Qwen2.5 7B | 32 | restore | 6 (0) | 192/192 | 20855.4 | 45603 | 41.85 | 1.0000 | 80678 | 0 |
| Qwen2.5 7B | 32 | extent_lazy | 6 (0) | 192/192 | 100.5 | 6199 | 34.39 | 0.8600 | 23210 | 57468 |
| Llama 3.1 8B | 8 | restore | 6 (0) | 48/48 | 1101.7 | 3005 | 28.93 | 1.0000 | 38738 | 0 |
| Llama 3.1 8B | 8 | extent_lazy | 6 (0) | 48/48 | 64.2 | 1971 | 28.31 | 0.9786 | 5664 | 33074 |
| Qwen2.5 14B | 8 | restore | 6 (0) | 48/48 | 1773.8 | 4020 | 16.64 | 1.0000 | 55847 | 0 |
| Qwen2.5 14B | 8 | extent_lazy | 6 (0) | 48/48 | 78.8 | 2335 | 15.97 | 0.9597 | 6263 | 49584 |
| Qwen2.5 7B, agent trace | 8 | restore | 6 (0) | 48/48 | 442.2 | 2215 | 35.23 | 1.0000 | 20287 | 0 |
| Qwen2.5 7B, agent trace | 8 | extent_lazy | 6 (0) | 48/48 | 57.1 | 1811 | 34.86 | 0.9895 | 5910 | 14377 |

Every one of the 480 extent outputs is copy-exact.  At 32 agents, extents
save 57,468 MiB.  The aggregate throughput ratio of 0.860 is not a stable
32-agent slowdown: one copy run stalled for 253.6 s and then reported an
artificially high sum of per-process rates, producing a paired ratio of
0.455.  The other five paired ratios are 0.948, 0.975, 1.000, 0.980, and
0.981.  The verifier retains this run and checks this exact exception.

**Server integration.** `run_engine_kvserver.sh` exercises the save and
restore requests of a long-running server rather than the small process
driver.

| Mode | Agents | Runs (failed) | Requests complete | Exact texts | Minimum common tokens | Save, request / server (ms) | Restore (ms) | First token (ms) | Memory / cache file (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| copy | 4 | 6 (0) | 24/24 | 24/24 | 64 | 505.66 / 503.89 | 365.2 | 560 | 9658 / 0 |
| extent | 4 | 6 (0) | 24/24 | 6/24 | 22 | 9.05 / 7.59 | 149.6 | 378 | 2454 / 784 |
| copy | 8 | 6 (0) | 48/48 | 48/48 | 64 | 507.48 / 505.85 | 496.1 | 872 | 19302 / 0 |
| extent | 8 | 6 (0) | 48/48 | 30/48 | 22 | 8.41 / 7.12 | 245.0 | 661 | 4893 / 784 |

All requests complete and restore evaluates at most 32 prompt tokens.  The
extent route does not meet the exact-output gate: the four-agent cells match
6 of 24 sequences and the eight-agent cells match 30 of 48.  In every
repetition the mismatching task indices are exactly 1, 2, and 3, with at
least 22 common generated tokens.  The raw sequences are retained; the
result is a deterministic server-path numerical divergence rather than a
timeout or failed restore.

The cause was found on 2026-10-07 and a second campaign confirms it.  The
tokenizer joins the last character of the published prefix (a line break)
with the line breaks that begin every task: 14,281 of the 14,282 published
tokens are tokens of an agent's prompt (measured through the server's
`/tokenize` for all eight tasks; the tokens an agent evaluates, 27, 22, 24,
32, ..., are its task plus that one token).  The server discards the last
published cell and evaluates the token again.  A copy writes it back into
the same cell.  With extents the cell is frozen, so the token goes to the
first cell of the private tail and the cache has another layout than the
copy's, which changes the rounding of the attention sums; three of the
eight tasks cross a near-tie within 64 tokens.

`run_engine_kvserver2.sh` is the same campaign with one change in its
client (`kv_server_client2.py`): the publisher evaluates and saves the
prefix together with the two line breaks that begin every task, so that all
14,282 published tokens are tokens of every prompt.  The prompts of the
agents are the same text as before.  Its expectation was written into the
runner before it ran: gate V2 holds for every response.

| Mode | Agents | Runs (failed) | Requests complete | Exact texts | Minimum common tokens | Save, request / server (ms) | Restore (ms) | First token (ms) | Memory / cache file (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| copy | 4 | 6 (0) | 24/24 | 24/24 | 64 | 501.28 / 499.75 | 350.9 | 551 | 9665 / 0 |
| extent | 4 | 6 (0) | 24/24 | 24/24 | 64 | 9.13 / 7.68 | 147.1 | 370 | 2445 / 784 |
| copy | 8 | 6 (0) | 48/48 | 48/48 | 64 | 504.85 / 503.36 | 517.6 | 894 | 19286 / 0 |
| extent | 8 | 6 (0) | 48/48 | 48/48 | 64 | 9.60 / 8.26 | 265.1 | 686 | 4931 / 784 |

It holds: 24 of 24 and 48 of 48 responses equal those of the copy for all
64 tokens, and an agent server evaluates at most 31 tokens.  The other
numbers repeat the first campaign (slot save 9.6 against 505 ms, restore
265 against 518 ms, memory of eight agent servers 4.8 against 18.8 GiB).
The rule for a deployment: publish a prefix at a token boundary that the
prompts of the agents keep.

**Memory limits.** `run_engine_kvlimit.sh` first accounts for four agents,
then gives agent 0 a 435-MiB cgroup limit while the other agents continue.

| Phase | Mode | Limit (MiB) | Agent 0 complete / OOM | Agent 0 peak (MiB) | Agent 0 task tokens | Other agents complete / OOM | Other texts exact | Available-memory drop (MiB) | Charged fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| account | restore | max | 6 / 0 | 317 | 16 | 18 / 0 | 18/18 | 10130 | 0.125 |
| account | extent_lazy | max | 6 / 0 | 336 | 16 | 18 / 0 | 18/18 | 2903 | 0.462 |
| limit | restore | 435 | 6 / 0 | 320 | 4097 | 18 / 0 | 18/18 | 10108 | 0.126 |
| limit | extent_lazy | 435 | 0 / 6 | 435 | 0 | 18 / 0 | 18/18 | 3004 | 0.479 |

The accounting campaign distinguishes process charges from physical
sharing: the extent processes are charged for mapped pages even while the
system-wide available-memory drop is 2.8 GiB rather than the copy's 9.9
GiB.  Under the enforced limit, only the selected extent process is killed;
all other agents finish and preserve their account-phase text.

**Read path and device-memory attach.** The read-path probe separates raw
host-memory bandwidth from end-to-end model behavior, while the VMM probe
varies the allocation granule for an 896-MiB import.

| MIG instance | Size (MiB) | Memory | Runs (failed) | Bandwidth (GiB/s) | Against device | 95% CI | Random-page latency (ns/page) |
|---|---:|---|---:|---:|---:|---:|---:|
| 12sm | 256 | device | 6 (0) | 67.38 | 1.0000 | [1.0000, 1.0000] | 0.90 |
| 12sm | 256 | host_huge | 6 (0) | 66.53 | 0.9880 | [0.9482, 1.0279] | 0.95 |
| 12sm | 256 | host_small | 6 (0) | 67.85 | 1.0073 | [0.9824, 1.0321] | 0.98 |
| 12sm | 1024 | device | 6 (0) | 68.45 | 1.0000 | [1.0000, 1.0000] | 0.56 |
| 12sm | 1024 | host_huge | 6 (0) | 68.72 | 1.0039 | [0.9975, 1.0103] | 0.59 |
| 12sm | 1024 | host_small | 6 (0) | 68.62 | 1.0024 | [0.9961, 1.0088] | 2.87 |
| 6sm | 256 | device | 6 (0) | 45.30 | 1.0000 | [1.0000, 1.0000] | 1.11 |
| 6sm | 256 | host_huge | 6 (0) | 45.53 | 1.0055 | [0.9749, 1.0361] | 1.33 |
| 6sm | 256 | host_small | 6 (0) | 45.20 | 0.9980 | [0.9789, 1.0171] | 1.28 |
| 6sm | 1024 | device | 6 (0) | 46.07 | 1.0000 | [1.0000, 1.0000] | 1.01 |
| 6sm | 1024 | host_huge | 6 (0) | 46.08 | 1.0004 | [0.9994, 1.0013] | 0.90 |
| 6sm | 1024 | host_small | 6 (0) | 46.07 | 1.0000 | [0.9943, 1.0058] | 2.96 |

| Allocation (MiB) | Handles | Runs (failed) | Export (ms) | Attach (ms) | Attach (us/handle) | Wrong words |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 448 | 6 (0) | 113.019 | 25.030 | 55.9 | 0 |
| 4 | 224 | 6 (0) | 89.070 | 13.278 | 59.3 | 0 |
| 8 | 112 | 6 (0) | 56.140 | 7.116 | 63.5 | 0 |
| 16 | 56 | 6 (0) | 30.523 | 4.440 | 79.3 | 0 |
| 32 | 28 | 6 (0) | 30.748 | 2.894 | 103.4 | 0 |
| 64 | 14 | 6 (0) | 2.714 | 2.441 | 174.4 | 0 |
| 128 | 7 | 6 (0) | 2.117 | 1.724 | 246.2 | 0 |
| 448 | 2 | 6 (0) | 1.654 | 1.485 | 742.7 | 0 |
| 896 | 1 | 6 (0) | 1.588 | 1.452 | 1452.5 | 0 |

Sequential host reads retain 0.988--1.007 of device-memory bandwidth in all
cells.  VMM attach falls from 25.030 ms for 448 two-MiB handles to 1.452 ms
for one 896-MiB handle, showing that its attach cost is primarily per
allocation rather than per byte.

Two conclusions follow, and both correct earlier readings.  First, the read
path of a kernel that scans a buffer is the same for host and device memory
in both instances, so it does not explain the 2 to 6% that a cache in host
memory costs in the 6-SM instance (Section 6.8); hypothesis H1 of the runner
is rejected and the cause remains unidentified.  Second, the device-memory
baseline of the engine attaches a child in 438 to 770 ms (Section 6.9 and
the mechanisms table above), while importing and mapping the same 448
handles takes 25 ms in the probe.  Most of the baseline's attach time
therefore belongs to its implementation in the engine, which was not tuned,
and not to the interface: that extents attach faster than this baseline is
a statement about the two implementations.

**Protection.** `run_protect.sh` asks an importer to read and then write
shared state through the two routes.

| Route | Attempts | Read / access change | Write refused / succeeded | Write result | Exporter's state intact |
|---|---:|---:|---:|---|---:|
| host, read-only mapping | 12 | 12 reads complete | 12 refused | `cudaErrorIllegalAddress` | 12/12 |
| device VMM import | 6 | 6 access raises succeeded | 6 succeeded | allowed | 0/6 |

The host route rejects every GPU write with `cudaErrorIllegalAddress` and
leaves the shared state intact.  CUDA VMM permits every importer write and
corrupts the exporter's state in every attempt.  This is a measured
protection limitation of that comparison route.

**Ollama.** `run_ollama_agents.sh` runs the unmodified Ollama 0.16.1 server
on isolated ports and model stores.  One-server and one-server-per-MIG
placements each serve eight agents; `/api/ps` must report nonzero GPU VRAM.

| Servers | Round | Runs | Requests complete | Prompt tokens | First token (ms) | Request (ms) | Throughput (tokens/s) | Memory (MiB) | GPU servers | Reported VRAM (MiB) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one | cold | 6 | 48/48 | 16336 | 192288 | 207250 | 39.83 | 19966 | 1/1 | 33366 |
| one | warm | 6 | 48/48 | 16336 | 886 | 17162 | 33.02 | 20040 | 1/1 | 33366 |
| two | cold | 6 | 48/48 | 16336 | 121350 | 131878 | 57.86 | 25111 | 2/2 | 37548 |
| two | warm | 6 | 48/48 | 16336 | 732 | 12687 | 47.50 | 25197 | 2/2 | 37548 |

All 192 requests finish and every server reports GPU residency.  Warming
the prompt reduces first-token latency from 192.3 s to 0.886 s for one
server and from 121.4 s to 0.732 s for two.  Ollama does not expose this
cache across servers; the two-server cold round therefore computes the
prefix independently.  The campaign uses a preload compatibility shim only
to preserve the selected MIG UUID during Ollama's CUDA discovery.  It uses
separate ports and leaves the pre-existing system Ollama process alive.

### 6.13 Gates

The runners of Sections 6.4 and 6.5 state their gates in their headers, and
the headers are pinned by hash. Outcomes:

| Gate, as stated before the campaign | Outcome | Evidence |
|---|---|---|
| Every agent of every mode writes the text of the agent that recomputes the prefix | **failed as stated** | holds for 42 of 42 agents per mode and prefix length in the publisher's instance; for 12 of 36 and 24 of 36 in the other instance, for the copy as for the extents. Cause: the instances compute different bits (Section 6.6). Against the copy: 78 of 78 in every mode |
| The agents of an extent mode hold one copy of the prefix between them | passed | proportional share of the mapped prefix 223 to 239 MiB for a 223 MiB prefix and 892 to 1,025 MiB for an 893 MiB prefix, for 1 to 8 agents; at least 0.8 prefix saved per agent in all 18 cells |
| Extents with a tail that follows use keep 97% of the generation speed of the copy | **failed in two of six cells** | 0.955 and 0.970; the other four 0.974 to 0.992. Cause: a cache in host memory costs nothing in the 12-SM instance and 2 to 6% in the 6-SM instance, shared or not (Section 6.8) |
| Attaching does not take longer than copying | passed | 25 to 65 ms against 81 to 484 ms |
| Publishing does not take longer than saving the state | passed | 3.0 and 9.1 ms against 231 and 561 ms |
| A parent that forks, and each of its children, write the text of a process that computes the same tokens alone | parent passed; children **failed in the other instance** | parent 6 of 6 in both modes; children 12 of 12 in the parent's instance and 0 of 12 in the other one, in both modes; against the child that copies, 24 of 24 |
| Handing over and attaching are not slower with extents than with copies | passed | 3 against 218 ms; 33 against 150 ms |
| The children hold one copy of the prefix between them | passed | proportional share 225 MiB for a 223 MiB prefix; 1.6 GiB less memory for four children |

The verifier checks the relations that hold, and for the two failed gates
it checks the outcome that was measured: the number of cells below 97%, and
equality with recomputation only in the publisher's instance.

`verify_stator_campaigns.sh` independently rebuilds the summaries for all
13 campaigns in Sections 6.11--6.12, verifies their pinned source hashes,
and enforces the completion, memory, latency, output, cleanup, and isolation
relations.  It reports four explicit measured exceptions rather than
hiding them: the flat-tree output count, the one 32-agent timing outlier,
and the two server-integration exact-output cells.

### 6.14 Improvement campaigns of 2026-10-07

The comparisons above keep the weights in place on both sides and compare
ways to hand a prefix over. The campaigns of this section add the
unmodified engine as a whole, a tuned device-memory baseline, the steps of
the mechanism left out one at a time, faults, deeper trees and a pipeline on
a public workload. Their runners state their gates in their headers, and
`verify_stator_campaigns.sh` rebuilds every summary and evaluates the gates
again.

**Whole stack (`run_engine_kvstack.sh`, `20261007-engine-kvstack-v1`).**
Four stacks by what an agent shares: `unmodified` is the engine at the same
commit with no patch of this repository applied (`llama.cpp-stock`), in
which every agent copies the weights to device memory and restores the
prefix from the state file; `prefix mapped` and `weights in place` apply
one of the two mechanisms (the first reads the model file without mapping
it, `kv_fork_nommap.cpp`, so that the weights are copied although the
device accepts host memory); `both` is STATOR. The agent workload (14,282
tokens), a context of 16,384 tokens, every second agent in the 6-SM
instance. The page cache is dropped and the model file read again before
every repetition. Stacks that copy are not run at counts that would leave
less than 20 GiB available.

| Stack | Agents | Failed | Memory (MiB) | 95% CI | Per agent (MiB) | Weights load (ms) | Attach (ms) | First token (ms) | Throughput (tokens/s) | Speed vs unmodified | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unmodified | 1 | 0 | 6162 | 6 | 6162 | 1147 | 151.3 | 1470 | 21.10 | 1.000 | 1.000 |
| prefix mapped | 1 | 0 | 5277 | 16 | 5277 | 1154 | 32.5 | 1362 | 20.68 | 0.975 | 0.974 |
| weights in place | 1 | 0 | 1655 | 12 | 1655 | 634 | 152.7 | 963 | 19.91 | 0.940 | 0.939 |
| both (STATOR) | 1 | 0 | 751 | 31 | 751 | 628 | 32.7 | 836 | 19.44 | 0.913 | 0.914 |
| unmodified | 2 | 0 | 12249 | 23 | 6124 | 1291 | 189.4 | 1688 | 38.12 | 1.000 | 1.000 |
| prefix mapped | 2 | 0 | 10481 | 31 | 5241 | 1298 | 32.9 | 1550 | 37.03 | 0.971 | 0.972 |
| weights in place | 2 | 0 | 3244 | 24 | 1622 | 706 | 244.5 | 1161 | 36.99 | 0.970 | 0.972 |
| both (STATOR) | 2 | 0 | 1470 | 20 | 735 | 711 | 37.2 | 962 | 36.02 | 0.945 | 0.945 |
| unmodified | 4 | 0 | 24488 | 43 | 6122 | 1616 | 189.5 | 2170 | 36.59 | 1.000 | 1.000 |
| prefix mapped | 4 | 0 | 20829 | 144 | 5207 | 1617 | 34.4 | 2017 | 35.46 | 0.969 | 0.967 |
| weights in place | 4 | 0 | 6411 | 32 | 1603 | 867 | 211.1 | 1468 | 34.93 | 0.954 | 0.954 |
| both (STATOR) | 4 | 0 | 2831 | 38 | 708 | 862 | 41.3 | 1301 | 34.65 | 0.947 | 0.947 |
| unmodified | 8 | 0 | 48981 | 96 | 6123 | 2429 | 275.2 | 3385 | 37.25 | 1.000 | 1.000 |
| prefix mapped | 8 | 0 | 41796 | 167 | 5224 | 2432 | 38.6 | 3146 | 35.73 | 0.959 | 0.964 |
| weights in place | 8 | 0 | 12888 | 30 | 1611 | 1134 | 307.9 | 2172 | 34.71 | 0.932 | 0.935 |
| both (STATOR) | 8 | 0 | 5759 | 25 | 720 | 1081 | 43.7 | 1834 | 34.48 | 0.926 | 0.931 |
| unmodified | 12 | 0 | 73594 | 20 | 6133 | 3384 | 347.1 | 4745 | 37.32 | 1.000 | 1.000 |
| prefix mapped | 12 | 0 | 62894 | 39 | 5241 | 3616 | 42.0 | 4624 | 36.52 | 0.979 | 0.977 |
| weights in place | 12 | 0 | 19370 | 32 | 1614 | 1376 | 366.7 | 2776 | 34.85 | 0.934 | 0.936 |
| both (STATOR) | 12 | 0 | 8662 | 42 | 722 | 1388 | 51.4 | 2490 | 34.48 | 0.924 | 0.927 |
| weights in place | 16 | 0 | 25803 | 26 | 1613 | 1791 | 444.5 | 3558 | 35.05 | - | - |
| both (STATOR) | 16 | 0 | 11534 | 36 | 721 | 1782 | 55.8 | 3192 | 34.33 | - | - |
| weights in place | 32 | 0 | 51460 | 41 | 1608 | 3669 | 896.6 | 7171 | 35.98 | - | - |
| both (STATOR) | 32 | 0 | 22869 | 78 | 715 | 3560 | 74.2 | 6228 | 34.80 | - | - |
| both (STATOR) | 64 | 0 | 45252 | 179 | 707 | 7659 | 200.2 | 13096 | 35.75 | - | - |

The memory of every stack is linear in the number of agents; the line is
fitted per repetition:

| Stack | Counts | Largest count | Memory per added agent (MiB) | 95% CI | Ratio to unmodified | Shared files (MiB) | Available (MiB) | Agents that fit by the line |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unmodified | 5 | 12 | 6130.2 | 4.1 | 1.0000 | 0 | 120032 | 19 |
| prefix mapped | 5 | 12 | 5237.5 | 3.7 | 0.8544 | 784 | 119998 | 22 |
| weights in place | 7 | 32 | 1607.9 | 1.3 | 0.2623 | 4466 | 120093 | 71 |
| both (STATOR) | 8 | 64 | 707.0 | 2.6 | 0.1153 | 5250 | 120094 | 163 |

`Agents that fit by the line` extends the line to the memory that was
available before the cases (the model file is subtracted for the stacks
that read it in place, since its pages count as available although they
are in use). It is an extrapolation above the largest count that was run.
At 8 agents the saving of both mechanisms together (43,222 MiB) differs
from the sum of the two separate savings (43,278 MiB) by 0.1%. Within a
repetition every agent index writes one text in all stacks and counts
(32 of 32 indices, 268 texts per repetition).

Gate K5 (summed generation speed of `both` at least 0.92 of `unmodified`
at every common count) is not met at one agent: 0.913 (median 0.914). The
single agent runs in the 12-SM instance in the odd repetitions and in the
6-SM instance in the even ones: `prefix mapped` 1.00 / 0.947, `weights in
place` 0.958 / 0.923, `both` 0.955 / 0.873. The weights are read from the 4
KiB pages of the ext4 page cache in this campaign.

The GPU driver keeps device memory that processes have freed as
reclaimable kernel memory: `KReclaimable` remains at 68 GiB between cases
while `MemFree` is 41 to 48 GiB and `MemAvailable` 117 GiB. This is why the
drop of `MemAvailable` is the measure, and it is the likely cause of the
stall of the first repetition of the 32-agent copy cell in Section 6.12
(demand above the pool and the free pages has to wait for the reclaim of
page cache).

According to the MIG profile table the two instances have 12 and 6 SMs;
`cudaDevAttrMultiProcessorCount` reports 12 and 8.

**Larger counts (`20261008-engine-kvstack-capacity-v2` and
`20261008-engine-kvstack-capacity-136-v1`, six repetitions).** The stack
that shares both with 128 and 136 agents, split evenly between the two MIG
instances:

| Stack | Agents | Failed | Memory (MiB) | 95% CI | Per agent (MiB) | Weights load (ms) | Attach (ms) | First token (ms) | Throughput (tokens/s) | Speed vs unmodified | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| both (STATOR) | 136 | 0 | 95433 | 461 | 702 | 18927 | 691.3 | 31037 | 38.48 | - | - |

No agent fails among the 1,584 agent processes. The agents use 89,874 MiB
and 95,433 MiB, within 1% of the line of the campaign above, and generate
36.31 and 38.48 tokens/s in total. The 136-agent run leaves at least
22.7 GiB available.

**The whole stack with Qwen2.5-14B
(`20261008-engine-kvstack-qwen14b-v1` and `-none8-v1`).** Counts 1 and 4
first establish a line of 12,486 MiB per unmodified agent. Its conservative
estimate for eight agents plus the 16 GiB reserve fits in the 118.0 GiB
available, so count 8 is then run separately. Eight agents use 99,717 MiB
(97.4 GiB) unmodified and 6,309 MiB on STATOR; adding its one 8,572 MiB
model file and 2,688 MiB cache file gives 17.2 GiB for the whole stack.
Every corresponding agent writes one text across the two stacks. K1--K4
hold. K5 does not: the paired throughput ratio is 0.8955 (95% confidence
interval 0.8294--0.9668) at one agent and 0.8788 (0.7573--1.0198) at eight,
below its 0.92 floor; the medians are 0.8980 and 0.9234.

**The same with the model file on 2 MiB pages
(`20261007-engine-kvstack-huge-v1`).** The model file is copied to a tmpfs
mounted with `huge=always` and every stack reads it from there; the runner
is unchanged (`MODEL=`, counts 1 and 8, and 32 for `both`).

| Stack | Agents | Failed | Memory (MiB) | 95% CI | Per agent (MiB) | Weights load (ms) | Attach (ms) | First token (ms) | Throughput (tokens/s) | Speed vs unmodified | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unmodified | 1 | 0 | 6147 | 31 | 6147 | 1003 | 187.1 | 1362 | 21.10 | 1.000 | 1.000 |
| weights in place | 1 | 0 | 1642 | 15 | 1642 | 355 | 179.8 | 709 | 21.13 | 0.997 | 0.994 |
| both (STATOR) | 1 | 0 | 715 | 33 | 715 | 350 | 34.7 | 562 | 20.49 | 0.959 | 0.969 |
| unmodified | 8 | 0 | 48852 | 165 | 6106 | 2313 | 328.2 | 3319 | 37.02 | 1.000 | 1.000 |
| weights in place | 8 | 0 | 12833 | 34 | 1604 | 897 | 314.5 | 1911 | 37.24 | 1.006 | 1.017 |
| both (STATOR) | 8 | 0 | 5693 | 35 | 712 | 903 | 38.6 | 1645 | 36.78 | 0.994 | 1.007 |
| both (STATOR) | 32 | 0 | 22576 | 41 | 706 | 3177 | 85.0 | 5839 | 36.87 | - | - |

At 8 agents `weights in place` generates at 1.006 (0.980 to 1.033) and
`both` at 0.994 (0.969 to 1.020) of `unmodified`; K5 holds in both cells.
The single agent: `weights in place` 1.02 in the 12-SM instance and 0.974
in the 6-SM instance, `both` 1.02 and 0.90. The loss of the campaign above
that comes from the weights is therefore one of 4 KiB pages. The count of
agents that fit is not taken from this campaign: the tmpfs copy of the
model is not available memory, and the summary subtracts the model file
once more.

**Mechanisms with a tuned device-memory baseline
(`run_engine_kvmech.sh`, `20261007-engine-kvmech-v1`).** The comparison of
Section 6.12 again, in an engine that adds two steps of tuning to the
device-memory path (`llama.cpp-tuned`, `kv_tuned.patch`): `device_tuned`
sets the access of the allocations of a tensor in one call, copies the
allocations that the prefix ends in with one wait for the device, zeroes
with one wait and gives a child's own memory one allocation per tensor and
step (`LLAMA_KV_VMM_BATCH=1`); `device_merged` also makes the publisher copy
the rows it publishes into one allocation per tensor, so that a child
imports one handle per tensor (`=2`). Cells: 8 children in the parent's
instance and across instances on the 16,321-token prefix, 4 and 1 children,
and 8 children on the 4,081-token prefix.

| Placement | Children | Prefix (tokens) | Mode | Children complete | Texts equal to copy | Parent pause (ms) | Attach (ms) | 95% CI | First token (ms) | Memory (MiB) | 95% CI | Throughput (tokens/s) | Speed vs copy | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| same | 8 | 16321 | copy | 48/48 | 48/48 | 553.01 | 645.4 | 78.7 | 2748 | 20309 | 30 | 21.54 | 1.0000 | 1.0000 |
| same | 8 | 16321 | demand | 48/48 | 48/48 | 549.38 | 792.7 | 63.1 | 2936 | 14227 | 76 | 21.74 | 1.0093 | 1.0095 |
| same | 8 | 16321 | device | 48/48 | 48/48 | 39.48 | 811.4 | 51.4 | 2906 | 7803 | 11 | 21.85 | 1.0145 | 1.0160 |
| same | 8 | 16321 | device_tuned | 48/48 | 48/48 | 36.60 | 473.3 | 98.6 | 2548 | 7804 | 19 | 21.65 | 1.0054 | 1.0078 |
| same | 8 | 16321 | device_merged | 48/48 | 48/48 | 114.63 | 194.6 | 30.6 | 2285 | 7741 | 10 | 21.31 | 0.9895 | 0.9902 |
| same | 8 | 16321 | cow | 48/48 | 48/48 | 8.77 | 57.7 | 9.2 | 2780 | 6906 | 48 | 21.14 | 0.9818 | 0.9819 |
| same | 8 | 16321 | extent | 48/48 | 48/48 | 8.61 | 54.0 | 6.1 | 2123 | 6036 | 29 | 21.33 | 0.9904 | 0.9911 |
| cross | 8 | 16321 | copy | 48/48 | 48/48 | 547.42 | 527.7 | 60.9 | 2372 | 20252 | 31 | 32.64 | 1.0000 | 1.0000 |
| cross | 8 | 16321 | demand | 48/48 | 48/48 | 551.92 | 598.7 | 112.5 | 2463 | 13740 | 64 | 32.82 | 1.0055 | 1.0052 |
| cross | 8 | 16321 | device | 24/48 | 24/24 | 39.97 | 638.0 | 38.8 | 2506 | 3962 | 15 | 21.11 | 0.6466 | 0.6459 |
| cross | 8 | 16321 | device_tuned | 24/48 | 24/24 | 41.19 | 456.7 | 88.8 | 2318 | 3977 | 16 | 20.87 | 0.6393 | 0.6395 |
| cross | 8 | 16321 | device_merged | 24/48 | 24/24 | 116.05 | 260.3 | 34.9 | 2149 | 3946 | 34 | 20.52 | 0.6286 | 0.6299 |
| cross | 8 | 16321 | cow | 48/48 | 48/48 | 8.61 | 51.7 | 6.7 | 2243 | 6774 | 18 | 31.33 | 0.9597 | 0.9608 |
| cross | 8 | 16321 | extent | 48/48 | 48/48 | 8.68 | 56.7 | 6.4 | 1897 | 5918 | 16 | 31.57 | 0.9672 | 0.9722 |
| same | 4 | 16321 | copy | 24/24 | 24/24 | 551.69 | 491.2 | 54.2 | 1905 | 10185 | 62 | 20.04 | 1.0000 | 1.0000 |
| same | 4 | 16321 | demand | 24/24 | 24/24 | 548.53 | 435.5 | 89.6 | 1845 | 7179 | 9 | 20.01 | 0.9987 | 0.9949 |
| same | 4 | 16321 | device | 24/24 | 24/24 | 40.16 | 424.6 | 15.3 | 1846 | 3970 | 12 | 20.12 | 1.0039 | 1.0019 |
| same | 4 | 16321 | device_tuned | 24/24 | 24/24 | 37.78 | 165.0 | 26.8 | 1575 | 3980 | 17 | 19.69 | 0.9826 | 0.9803 |
| same | 4 | 16321 | device_merged | 24/24 | 24/24 | 117.23 | 106.9 | 31.3 | 1518 | 3939 | 20 | 19.58 | 0.9771 | 0.9750 |
| same | 4 | 16321 | cow | 24/24 | 24/24 | 8.33 | 49.9 | 4.9 | 1817 | 3514 | 30 | 19.57 | 0.9766 | 0.9746 |
| same | 4 | 16321 | extent | 24/24 | 24/24 | 8.50 | 49.1 | 3.7 | 1438 | 3087 | 20 | 19.66 | 0.9811 | 0.9797 |
| same | 1 | 16321 | copy | 6/6 | 6/6 | 546.59 | 333.0 | 4.2 | 1188 | 2538 | 4 | 14.02 | 1.0000 | 1.0000 |
| same | 1 | 16321 | demand | 6/6 | 6/6 | 547.90 | 315.7 | 4.8 | 1162 | 1867 | 17 | 13.94 | 0.9943 | 0.9958 |
| same | 1 | 16321 | device | 6/6 | 6/6 | 41.91 | 219.3 | 14.4 | 1075 | 1077 | 12 | 13.69 | 0.9763 | 0.9765 |
| same | 1 | 16321 | device_tuned | 6/6 | 6/6 | 39.02 | 78.8 | 1.1 | 892 | 1072 | 8 | 13.17 | 0.9392 | 0.9396 |
| same | 1 | 16321 | device_merged | 6/6 | 6/6 | 112.80 | 59.7 | 1.2 | 864 | 1071 | 10 | 13.10 | 0.9344 | 0.9348 |
| same | 1 | 16321 | cow | 6/6 | 6/6 | 8.73 | 42.3 | 0.7 | 986 | 954 | 8 | 13.14 | 0.9370 | 0.9398 |
| same | 1 | 16321 | extent | 6/6 | 6/6 | 8.82 | 43.5 | 0.5 | 850 | 850 | 13 | 13.11 | 0.9346 | 0.9347 |
| same | 8 | 4081 | copy | 48/48 | 48/48 | 210.45 | 263.4 | 19.2 | 2294 | 9157 | 44 | 22.60 | 1.0000 | 1.0000 |
| same | 8 | 4081 | demand | 48/48 | 48/48 | 221.45 | 292.2 | 25.0 | 2328 | 8448 | 9 | 22.59 | 0.9994 | 0.9994 |
| same | 8 | 4081 | device | 48/48 | 48/48 | 6.81 | 450.6 | 17.6 | 2444 | 7556 | 27 | 22.67 | 1.0029 | 1.0031 |
| same | 8 | 4081 | device_tuned | 48/48 | 48/48 | 6.61 | 137.9 | 32.4 | 2141 | 7557 | 17 | 22.42 | 0.9920 | 0.9914 |
| same | 8 | 4081 | device_merged | 48/48 | 48/48 | 24.52 | 167.4 | 54.0 | 2193 | 7574 | 46 | 22.47 | 0.9940 | 0.9950 |
| same | 8 | 4081 | cow | 48/48 | 48/48 | 2.95 | 31.5 | 4.6 | 2883 | 6709 | 9 | 22.33 | 0.9878 | 0.9883 |
| same | 8 | 4081 | extent | 48/48 | 48/48 | 3.08 | 29.7 | 2.4 | 2011 | 5757 | 46 | 22.33 | 0.9881 | 0.9874 |

With 8 children the attach time of the device path falls from 811 to 473
to 195 ms. The first step removes one access setting and one wait per 2 MiB
allocation. What remains is the import of 392 handles, which the driver
serializes across the children: one child alone attaches in 79 ms. One
allocation per tensor leaves 56 handles; the copy that makes it raises the
pause of the parent from 37 to 115 ms. An allocation cannot be extended or
joined with another, and 2 MiB is the smallest one, so a child still copies
the allocation that the prefix ends in for each of the 56 tensors, which is
why the children hold 7.6 GiB against 5.9 GiB on extents. Across instances
the import is refused at every step of the tuning (24 of 48 children
complete). All 1,146 children that complete write the text of copy.

Gates that are not met, as the verifier reports them: B3 in three cells,
where `device_tuned` holds 1 to 10 MiB more than `device` (7,804 against
7,803 MiB with 8 children); B4 in two cells, where a child attaches faster
on the copy-on-write mapping than on extents (51.7 against 56.7 ms across
instances, 42.3 against 43.5 ms with one child). The copy-on-write mapping
and the extents do the same work at attach; they differ in memory (6.7
against 5.9 GiB) and in the first token (2.78 against 2.12 s).

**Putting a model file on 2 MiB pages (`run_weights_publish.sh`,
`20261007-weights-publish-v1`).** Reading the weights in place costs nothing
when the model file is on 2 MiB pages, which a tmpfs mounted with
`huge=always` provides. Three ways to put a file from storage there:
`cp` (buffered), `dd iflag=direct` (direct reads through a buffer of the
program, one reader), and `weights_publish.c`, which maps the target shared
and reads the source into the mapping with O_DIRECT, eight readers on
chunks of 32 MiB. The page cache is dropped before every case. The third
file is four copies of the 14B model, to show a size that no model of this
repository has.

| File | Size (GiB) | Method | Runs (failed) | Time (s) | 95% CI | GiB/s | Speed-up vs cp | Page cache left (MiB) | Share of the file | Largest drop of MemFree (MiB) | Identical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct-Q4_K_M.gguf | 4.4 | cp | 6 (0) | 2.208 | 0.013 | 1.98 | 1.00 | 4468 | 1.001 | 8764 | 6/6 |
| Qwen2.5-7B-Instruct-Q4_K_M.gguf | 4.4 | dd | 6 (0) | 1.469 | 0.034 | 2.97 | 1.50 | 2 | 0.000 | 4091 | 6/6 |
| Qwen2.5-7B-Instruct-Q4_K_M.gguf | 4.4 | publish | 6 (0) | 1.154 | 0.009 | 3.78 | 1.91 | 4 | 0.001 | 4008 | 6/6 |
| Qwen2.5-14B-Instruct-Q4_K_M.gguf | 8.4 | cp | 6 (0) | 4.048 | 0.034 | 2.07 | 1.00 | 8580 | 1.001 | 16137 | 6/6 |
| Qwen2.5-14B-Instruct-Q4_K_M.gguf | 8.4 | dd | 6 (0) | 2.844 | 0.087 | 2.95 | 1.42 | 10 | 0.001 | 8234 | 6/6 |
| Qwen2.5-14B-Instruct-Q4_K_M.gguf | 8.4 | publish | 6 (0) | 1.804 | 0.015 | 4.64 | 2.24 | 5 | 0.001 | 7762 | 6/6 |
| synthetic-4x-Qwen2.5-14B-Instruct-Q4_K_M.gguf | 33.5 | cp | 6 (0) | 16.710 | 3.655 | 2.06 | 1.00 | 33207 | 0.968 | 65561 | 6/6 |
| synthetic-4x-Qwen2.5-14B-Instruct-Q4_K_M.gguf | 33.5 | dd | 6 (0) | 11.230 | 0.379 | 2.98 | 1.47 | 10 | 0.000 | 34056 | 6/6 |
| synthetic-4x-Qwen2.5-14B-Instruct-Q4_K_M.gguf | 33.5 | publish | 6 (0) | 6.949 | 0.049 | 4.82 | 2.37 | 10 | 0.000 | 34126 | 6/6 |

`cp` leaves the file in the page cache as well (the whole file for the two
models, 97% of the synthetic one), and the largest drop of `MemFree` while
loading is twice the size of the file. With direct reads nothing is left in
the page cache and the drop is the size of the file. `weights_publish` is
1.9 to 2.4 times as fast as `cp` and 1.3 to 1.6 times as fast as `dd`. All
54 targets equal their sources. The steady state does not depend on how the
file was put there.

**vLLM (`run_ext_vllm.sh`, `20261007-ext-vllm-v1`).** vLLM 0.20.0 (the
wheel of the Jetson AI Lab index, torch 2.10.0) with the 4-bit AWQ weights
of the same model, one server in the 12-SM instance, eight concurrent
requests on the agent workload, twice. It does not start on this platform
as installed. Needed: the modules xgrammar, compressed-tensors and triton;
the device by index (vLLM parses `CUDA_VISIBLE_DEVICES` as integers; index
0 is the 12-SM instance, and the 6-SM instance cannot be selected; Section
6.15 lifts this with a launcher);
`TRITON_CUDART_PATH` for the CUDA headers, because a kernel of its block
table is a Triton kernel; `-O0`, which switches the Inductor compilation
off. `--kv-cache-memory-bytes` reserves 8 GiB.

| Round | Servers ready | Agents complete | Start-up (ms) | Memory when ready (MiB) | First token (ms) | Prompt tokens (cached) | Throughput (tokens/s) | Peak memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | 6/6 | 48/48 | 23291 | 15519 | 9834 | 14307 (12488) | 156.10 | 15618 |
| warm | 6/6 | 48/48 | 23291 | 15519 | 676 | 14307 (14298) | 153.57 | 15618 |

In the first round seven of the eight requests take the prefix from the
cache blocks that the first one computes (12,488 of 14,307 prompt tokens on
average). One server with batching generates 4.5 times the tokens per
second of eight separate processes on the engine. The texts are not
compared with those of the engine: the weights and their quantization
differ.

**The steps of the mechanism, one at a time (`MODE_SET=ablation`,
`20261007-engine-kvablate-v1`).** A parent and eight children, every second
child in the 6-SM instance, the 16,321-token prefix. `no_read` leaves out the
CPU read of the mapped rows (`LLAMA_KV_TOUCH=0`), `no_populate` leaves out
the memory that a child puts behind its own rows before the device writes
them (`LLAMA_KV_POPULATE=0`), `writable` and `writable_no_read` are the
copy-on-write mapping with and without the CPU read, `grow_*` change the
step in which a child's memory follows use, and `small_pages` puts the file
on a tmpfs with 4 KiB pages.

| Placement | Children | Prefix (tokens) | Mode | Children complete | Texts equal to copy | Parent pause (ms) | Attach (ms) | 95% CI | First token (ms) | Memory (MiB) | 95% CI | Throughput (tokens/s) | Speed vs copy | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cross | 8 | 16321 | copy | 48/48 | 48/48 | 536.35 | 529.8 | 78.6 | 2447 | 20241 | 36 | 32.94 | 1.0000 | 1.0000 |
| cross | 8 | 16321 | extent | 48/48 | 48/48 | 8.96 | 50.6 | 3.6 | 1827 | 5913 | 28 | 31.74 | 0.9636 | 0.9656 |
| cross | 8 | 16321 | no_read | 48/48 | 48/48 | 8.99 | 45.5 | 1.2 | 2427 | 5933 | 18 | 31.78 | 0.9648 | 0.9695 |
| cross | 8 | 16321 | no_populate | 48/48 | 48/48 | 8.51 | 49.9 | 2.0 | 1982 | 5929 | 18 | 31.20 | 0.9472 | 0.9513 |
| cross | 8 | 16321 | writable | 48/48 | 48/48 | 8.88 | 49.1 | 3.2 | 2212 | 6799 | 12 | 31.51 | 0.9567 | 0.9597 |
| cross | 8 | 16321 | writable_no_read | 48/48 | 48/48 | 9.07 | 47.7 | 4.7 | 5174 | 13074 | 23 | 32.01 | 0.9719 | 0.9760 |
| cross | 8 | 16321 | grow_1024 | 48/48 | 48/48 | 8.47 | 48.5 | 2.0 | 1840 | 6079 | 23 | 31.71 | 0.9626 | 0.9656 |
| cross | 8 | 16321 | grow_4096 | 48/48 | 48/48 | 8.69 | 49.6 | 2.6 | 1845 | 6880 | 40 | 31.74 | 0.9634 | 0.9668 |
| cross | 8 | 16321 | grow_all | 48/48 | 48/48 | 9.14 | 86.6 | 5.4 | 1908 | 12932 | 22 | 31.68 | 0.9617 | 0.9652 |
| cross | 8 | 16321 | small_pages | 48/48 | 48/48 | 13.15 | 63.6 | 3.6 | 1885 | 5841 | 15 | 31.63 | 0.9603 | 0.9639 |

Every case completes and every child writes the text of copy (480 of 480),
also in the modes that leave a step out, so the gates A1 and A2 hold and the
counts that were to be reported are zero failures. A3 (memory does not fall
with a larger step) and A4 (attach is faster on 2 MiB pages than on 4 KiB
pages) hold. Without the CPU read a child attaches 5 ms earlier and
produces its first token 0.60 s later. Without populate-ahead the first
token is 0.16 s later and the summed generation speed is 0.947 of copy
against 0.964.

**Where the loss of speed with host memory comes from (`MODE_SET=locality`,
`20261007-engine-kvlocal-12sm-v1` and `-6sm-v1`).** Section 6.8 found that a
cache in host memory costs generation speed in the 6-SM instance and not in
the 12-SM instance, and did not find the cause. This campaign separates
four candidates with a parent and four children in one instance, a
16,321-token prefix and 128 generated tokens: host memory against device
memory (`host_copy_huge`, a private cache in 2 MiB pages that receives a
copy of the rows, against `copy`), the page size of private memory
(`host_copy`, 4 KiB pages), sharing (`extent` against `host_copy`), the
page size of an agent's own rows (`grow_all`) and the page size of the
shared rows (`small_pages`). The ratios are the summed generation speed of
the children against `copy` in the same repetition.

12-SM instance:

| Placement | Children | Prefix (tokens) | Mode | Children complete | Texts equal to copy | Parent pause (ms) | Attach (ms) | 95% CI | First token (ms) | Memory (MiB) | 95% CI | Throughput (tokens/s) | Speed vs copy | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| same | 4 | 16321 | copy | 24/24 | 24/24 | 560.45 | 463.6 | 58.0 | 1837 | 10147 | 21 | 18.98 | 1.0000 | 1.0000 |
| same | 4 | 16321 | host_copy_huge | 24/24 | 24/24 | 514.82 | 384.8 | 22.2 | 1744 | 10094 | 28 | 19.25 | 1.0143 | 1.0142 |
| same | 4 | 16321 | host_copy | 24/24 | 24/24 | 515.05 | 513.4 | 30.8 | 1873 | 6581 | 55 | 19.35 | 1.0192 | 1.0192 |
| same | 4 | 16321 | extent | 24/24 | 24/24 | 8.70 | 46.1 | 1.0 | 1392 | 3093 | 22 | 18.93 | 0.9971 | 0.9989 |
| same | 4 | 16321 | grow_all | 24/24 | 24/24 | 8.68 | 76.0 | 4.4 | 1430 | 6517 | 24 | 18.96 | 0.9991 | 1.0006 |
| same | 4 | 16321 | small_pages | 24/24 | 24/24 | 12.22 | 55.8 | 2.8 | 1426 | 2952 | 25 | 18.98 | 0.9999 | 1.0018 |

6-SM instance:

| Placement | Children | Prefix (tokens) | Mode | Children complete | Texts equal to copy | Parent pause (ms) | Attach (ms) | 95% CI | First token (ms) | Memory (MiB) | 95% CI | Throughput (tokens/s) | Speed vs copy | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| same | 4 | 16321 | copy | 24/24 | 24/24 | 549.33 | 440.6 | 24.1 | 1898 | 10166 | 103 | 12.40 | 1.0000 | 1.0000 |
| same | 4 | 16321 | host_copy_huge | 24/24 | 24/24 | 518.83 | 362.1 | 33.0 | 1828 | 10044 | 18 | 11.65 | 0.9401 | 0.9401 |
| same | 4 | 16321 | host_copy | 24/24 | 24/24 | 516.85 | 483.7 | 23.4 | 1971 | 6591 | 145 | 11.69 | 0.9426 | 0.9430 |
| same | 4 | 16321 | extent | 24/24 | 24/24 | 8.59 | 44.4 | 0.9 | 1501 | 3080 | 168 | 11.55 | 0.9316 | 0.9319 |
| same | 4 | 16321 | grow_all | 24/24 | 24/24 | 9.06 | 73.5 | 3.0 | 1534 | 6454 | 14 | 11.55 | 0.9319 | 0.9325 |
| same | 4 | 16321 | small_pages | 24/24 | 24/24 | 12.76 | 54.2 | 2.8 | 1560 | 3324 | 1064 | 11.56 | 0.9325 | 0.9325 |

In the 12-SM instance no form of host memory costs speed: the private
caches are at 1.014 and 1.019 of the device cache, the three forms of
extents at 0.997 to 1.000 with intervals that contain 1. In the 6-SM
instance every form costs: a private cache in host memory generates at
0.940 (2 MiB pages) and 0.943 (4 KiB pages) of the device cache, and the
three forms of extents at 0.932 (0.9316 to 0.9325). So the loss is a property of
host memory in that instance (6.0%); sharing adds 0.9 points; the page
size of the private memory, of an agent's own rows and of the shared rows
changes it by 0.3 points or less. The intervals are narrow (half-widths of
0.001 to 0.002 in the 6-SM instance). All 288 children write the text of
the child that received a copy, and both gates hold in both campaigns.

**Scattered reads and writes by the kind of memory (`run_tlb_probe.sh`,
`20261007-tlb-probe-v1`).** The read-path probe of Section 6.8 scans a
buffer and found host memory as fast as device memory in both instances.
This probe adds accesses that jump: it reads consecutive words (scan),
reads words at pseudo-random strides of 4 and 64 KiB (gather) and writes at
the same strides (scatter) over 1 GiB, in device memory and in pageable
host memory with 2 MiB and with 4 KiB pages (private anonymous memory that
the CPU has written), in each instance. All threads work on the buffer at
the same time, each on its share (scan) or on all of it. The rates are
relative to device memory in the same instance and repetition; the CUDA
runtime reports the 6-SM instance as 8 SMs.

| SMs (runtime) | Buffer (MiB) | Stride (KiB) | Memory | Runs | Scattered reads (Mwords/s) | vs device | low | high | Scattered writes (Mwords/s) | vs device | low | high |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 1024 | 4 | device | 6 | 3414.8 | 1.000 | 1.000 | 1.000 | 1495.2 | 1.000 | 1.000 | 1.000 |
| 12 | 1024 | 4 | host_huge | 6 | 3252.2 | 0.952 | 0.911 | 0.994 | 1388.9 | 0.929 | 0.927 | 0.930 |
| 12 | 1024 | 4 | host_small | 6 | 37.1 | 0.011 | 0.011 | 0.011 | 37.2 | 0.025 | 0.025 | 0.025 |
| 12 | 1024 | 64 | device | 6 | 3387.3 | 1.000 | 1.000 | 1.000 | 1489.2 | 1.000 | 1.000 | 1.000 |
| 12 | 1024 | 64 | host_huge | 6 | 3289.1 | 0.971 | 0.961 | 0.981 | 1387.2 | 0.931 | 0.931 | 0.932 |
| 12 | 1024 | 64 | host_small | 6 | 41.9 | 0.012 | 0.012 | 0.013 | 41.3 | 0.028 | 0.028 | 0.028 |
| 8 | 1024 | 4 | device | 6 | 1920.4 | 1.000 | 1.000 | 1.000 | 1473.4 | 1.000 | 1.000 | 1.000 |
| 8 | 1024 | 4 | host_huge | 6 | 1571.3 | 0.818 | 0.815 | 0.822 | 1376.1 | 0.934 | 0.931 | 0.937 |
| 8 | 1024 | 4 | host_small | 6 | 33.3 | 0.017 | 0.017 | 0.017 | 33.3 | 0.023 | 0.022 | 0.023 |
| 8 | 1024 | 64 | device | 6 | 1818.9 | 1.000 | 1.000 | 1.000 | 1472.7 | 1.000 | 1.000 | 1.000 |
| 8 | 1024 | 64 | host_huge | 6 | 1565.2 | 0.861 | 0.857 | 0.864 | 1375.4 | 0.934 | 0.931 | 0.937 |
| 8 | 1024 | 64 | host_small | 6 | 43.7 | 0.024 | 0.024 | 0.024 | 38.2 | 0.026 | 0.026 | 0.026 |

With 2 MiB pages, host memory is scanned as fast as device memory or faster
(1.21 to 1.27 in the 12-SM instance, 0.97 to 0.99 in the 6-SM instance) and
written at scattered addresses at 0.93 of device memory in both instances.
Scattered reads separate the instances: 0.95 to 0.97 of device memory in
the 12-SM instance and 0.82 to 0.86 in the 6-SM instance. H1 holds for
reads and not for writes. With 4 KiB pages every rate is 0.01 to 0.03 of
device memory in both instances, so H2 holds as well: a buffer of 262,144
pages that all threads use at once is far slower than the same buffer in
512 pages.

The engine does not show the second effect. With the shared prefix on 4 KiB
pages (`small_pages`) and with a private cache on 4 KiB pages
(`host_copy`) it generates as fast as with 2 MiB pages in both instances
(the locality campaign above); its kernels use the tensors of one layer at
a time and not the whole cache at once. The probe therefore gives one
difference between the two instances that has the direction of the loss,
scattered reads of host memory, and it explains neither the size of the
loss in the engine nor why the page size does not matter there. What the
6-SM instance does differently when it reads host memory at addresses that
jump was not examined below the CUDA interface.

**A pipeline of agents on the tools of a public benchmark
(`run_engine_kvpipe.sh`, `20261007-engine-kvpipe-huge-v1`).** A planner
computes a prefix of a system prompt and the schemas of 90 tools of BFCL v4
(11,788 tokens) and publishes it. Two leaders attach, add a plan and publish
again; four workers per leader attach to what their leader published and
work through three turns of requests and tool results. The processes
alternate between the two MIG instances, and the model file is on a tmpfs
with 2 MiB pages for all three stacks. The power rails of the board are
sampled with `tegrastats` while a case runs. Memory is the largest drop of
MemAvailable from before the planner starts, so the cache files of `chain`
are in it.

| Stack | Groups x workers x turns | Runs (failed) | Prefix (tokens) | Completion (s) | 95% CI | vs stock | Memory (MiB) | Files (MiB) | Worker attach (ms) | Worker first token (ms) | Energy, input rail (J) | prefix / leaders / workers (J) | vs stock | GPU rail (J) | Worker texts equal to stock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock | 2x4x3 | 6 (0) | 11788 | 80.3 | 0.2 | 1.000 | 67384 | 0 | 376.3 | 6305 | 5906 | 833 / 394 / 4679 | 1.000 | 2545 | 48/48 |
| copy | 2x4x3 | 6 (0) | 11788 | 77.9 | 0.1 | 0.970 | 17635 | 0 | 380.0 | 2312 | 5834 | 832 / 373 / 4630 | 0.988 | 2580 | 48/48 |
| chain | 2x4x3 | 6 (0) | 11788 | 77.9 | 0.1 | 0.971 | 8577 | 896 | 36.3 | 1926 | 5880 | 859 / 359 / 4662 | 0.996 | 2602 | 48/48 |

The unmodified engine holds 65.8 GiB for the pipeline, the stack with the
weights in place and state copies 17.2 GiB, and extent chains 8.4 GiB. The
planner publishes in 6.7 ms against 422 ms, a leader in 4.6 ms against
448 ms, and a worker attaches in 36 ms against 376 ms; the first token of
a worker comes after 1.93 s against 6.31 s. The workers generate at 0.936
of their summed speed on the unmodified engine (half of them run in the
6-SM instance), and the pipeline still completes in 0.970 of its time
(77.9 s against 80.3 s), because the hand-overs are shorter. The energy of
the input rail is 5,880 J against 5,906 J, a paired ratio of 0.996. All 48
workers and 12 leaders write the text of the unmodified engine. The gates
P1 to P5 hold.

**What a fault of one sharer does to the others (`run_engine_kvfault.sh`,
`20261007-engine-kvfault-v1`).** A publisher publishes the 14,282-token
prefix of the agent workload as extents and exits; eight agents map it and generate 128 tokens.
Eight seconds into the generation either nothing happens (`none`), a
further process maps the prefix file read-only as an agent does and
launches a GPU kernel that writes into it (`write`), or agent 0 is killed
(`kill`). Four ways of sharing the GPU: time slicing in the 12-SM instance,
one MPS server in that instance, the two MIG instances with time slicing,
and one MPS server in each instance. "Others" are the seven agents besides
agent 0, 42 over six repetitions.

| GPU sharing | Fault | Runs | Agents complete | Others complete | Others with the text of the run without a fault | Writes refused | Error of the write | File intact (runs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| timeslice | none | 6 | 48/48 | 42/42 | 42/42 | 0/0 | none | 6/6 |
| timeslice | write | 6 | 48/48 | 42/42 | 42/42 | 6/6 | cudaErrorIllegalAddress | 6/6 |
| timeslice | kill | 6 | 42/48 | 42/42 | 42/42 | 0/0 | none | 6/6 |
| mps | none | 6 | 48/48 | 42/42 | 42/42 | 0/0 | none | 6/6 |
| mps | write | 6 | 0/48 | 0/42 | 0/42 | 6/6 | cudaErrorIllegalAddress | 6/6 |
| mps | kill | 6 | 0/48 | 0/42 | 0/42 | 0/0 | none | 6/6 |
| mig | none | 6 | 48/48 | 42/42 | 42/42 | 0/0 | none | 6/6 |
| mig | write | 6 | 48/48 | 42/42 | 42/42 | 6/6 | cudaErrorIllegalAddress | 6/6 |
| mig | kill | 6 | 42/48 | 42/42 | 42/42 | 0/0 | none | 6/6 |
| mig_mps | none | 6 | 48/48 | 42/42 | 42/42 | 0/0 | none | 6/6 |
| mig_mps | write | 6 | 24/48 | 24/42 | 24/42 | 6/6 | cudaErrorIllegalAddress | 6/6 |
| mig_mps | kill | 6 | 30/48 | 30/42 | 30/42 | 0/0 | none | 6/6 |

The GPU write is refused in all 24 cases with a fault (`cudaErrorIllegalAddress`
in the writing process), and the prefix file has the same contents after
every case. Under time slicing and under MIG all 42 other agents complete
after the write and after the kill, with the text of the case without a
fault: sharing a prefix through host mappings does not join the agents
into one fault domain. Under one MPS server no agent completes after either
event (0 of 42); the server shares the fate of its clients, as Section 5
found for processes that share nothing. With one MPS server per instance,
the agents of the instance without the fault complete after the write (24
of 42) and 30 of 42 after the kill. The gates F1 to F3 hold; the counts
under MPS were to be reported and are not gated.

**Deeper and wider trees, and when the memory of a subtree comes back
(`run_engine_kvdeep.sh`, `20261007-engine-kvdeep-v1` and `-wide-v1`).** The
tree of Section 6.12 has three levels and 11 processes. Here a root
publishes the 16,321-token prefix; every inner agent attaches to what its
parent published, appends the context of its subtree and publishes again;
a leaf attaches, appends its task and generates. `2 2 4` is a tree of four
levels with 23 processes and 16 leaves, `1 16` one leader with 16 leaves.
The agents of a level alternate between the two MIG instances. The agents
below the first child of the root generate 64 tokens and the others three
times as many, so that the first subtree leaves while the others run. With
chains the files are removed from their directory as soon as every leaf
has attached, and the pages that the tmpfs holds are sampled every 0.2 s.

Four levels:

| Hand-over | Fan-outs | Processes (leaves) | Runs (failed) | Inner attach / publish (ms) | Leaf attach (ms) | Leaf first token (ms) | Memory (MiB) | 95% CI | Files: all alive / first subtree left / end (MiB) | Leaf texts equal to copy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| copy | 2x2x4 | 23 (16) | 6 (0) | 288.5 / 611.10 | 1008.2 | 4982 | 58289 | 52 | 0 / 0 / 0 | 96/96 |
| chain | 2x2x4 | 23 (16) | 6 (0) | 49.5 / 7.80 | 61.2 | 4050 | 19166 | 23 | 2352 / 1680 / 0 | 96/96 |

One leader with 16 leaves:

| Hand-over | Fan-outs | Processes (leaves) | Runs (failed) | Inner attach / publish (ms) | Leaf attach (ms) | Leaf first token (ms) | Memory (MiB) | 95% CI | Files: all alive / first subtree left / end (MiB) | Leaf texts equal to copy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| copy | 1x16 | 18 (16) | 6 (0) | 216.3 / 563.36 | 708.2 | 4030 | 45658 | 20 | 0 / -1 / 0 | 96/96 |
| chain | 1x16 | 18 (16) | 6 (0) | 45.2 / 6.36 | 57.8 | 3287 | 14258 | 18 | 1232 / -1 / 0 | 96/96 |

The tree of four levels holds 18.7 GiB with chains against 56.9 GiB with
copies, the wide tree 13.9 GiB against 44.6 GiB. An inner agent publishes
in 7.8 ms against 611 ms and attaches in 50 ms against 289 ms; a leaf
attaches to the segments of its three ancestors in 61 ms against 1,008 ms
and produces its first token after 4.05 s against 4.98 s. The leaves
generate at the same summed speed (30.2 against 30.5 tokens/s). All 192
leaves write the text of the leaf that received copies.

Memory return in the tree of four levels: the tmpfs holds 2,352 MiB while
every process runs, 1,680 MiB after the first subtree has left, and nothing
after the last process. The segments of a subtree are released when its
last sharer leaves, although the files have no name any more; the segments
of the ancestors that the other subtree still maps remain. The gates D1 to
D5 hold in both campaigns.

**Shorter and longer prefixes (`run_engine_kvscale.sh` with `PARAGRAPHS=20`
and `600`, `20261007-engine-kvscale-p20-v1` and `-p600-v1`).** Section 6.12
measures eight agents on the 16,321-token prefix. The same case with a
prefix of 1,021 tokens (context 4,096) and of 30,601 tokens (context
32,768):

| Prefix (tokens) | Agents | Mode | Runs (failed agents) | Exact texts | Attach (ms) | First token (ms) | Throughput (tokens/s) | Extents / copy | Memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1021 | 8 | restore | 6 (0) | 48/48 | 92.9 | 1753 | 39.37 | 1.0000 | 7336 |
| 1021 | 8 | extent_lazy | 6 (0) | 48/48 | 28.3 | 1661 | 39.20 | 0.9958 | 5582 |

| Prefix (tokens) | Agents | Mode | Runs (failed agents) | Exact texts | Attach (ms) | First token (ms) | Throughput (tokens/s) | Extents / copy | Memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30601 | 8 | restore | 6 (0) | 48/48 | 737.5 | 2460 | 31.41 | 1.0000 | 20478 |
| 30601 | 8 | extent_lazy | 6 (0) | 48/48 | 64.4 | 1780 | 30.29 | 0.9644 | 5856 |

On the short prefix the agents hold 5.45 GiB with extents against 7.16 GiB
with copies, attach in 28 ms against 93 ms and generate at 0.996 of the
speed of the copy (interval 0.986 to 1.006). On the long prefix they hold
5.72 GiB against 20.00 GiB, attach in 64 ms against 738 ms, produce the
first token after 1.78 s against 2.46 s and generate at 0.964 (0.955 to
0.974). The cache file adds 56 MiB and 1,674 MiB once. The memory of the
copies follows the context and not the prefix: an agent that restores a
state reserves the cache of its whole context in device memory, so the
copies of the 30,601-token prefix hold what those of the 16,321-token
prefix hold in the same context (19.9 GiB). With extents the agents hold
5.5 to 5.7 GiB at every length, because the prefix is mapped and the tail
follows use. The loss of speed grows with the prefix (0.4%, 1.4% at
16,321 tokens, 3.6%). Half of the agents run in the 6-SM instance, where
host memory costs speed, and a longer prefix means more rows read from
it; the campaign does not separate the two instances. All 96 agents write the text of the
copy.

**The 32-agent cell again (`20261007-engine-kvscale-32-v2`).** In the first
measurement of 32 agents (Section 6.12) the agents of the copy waited up to
240 s for their memory in one repetition, which spread their generation over
time; the paired ratio of that repetition was 0.455 and the gate S4 failed
on the mean (0.860, median 0.978). The cell was measured again with the
page cache dropped and the model file read again before the campaign.

| Prefix (tokens) | Agents | Mode | Runs (failed agents) | Exact texts | Attach (ms) | First token (ms) | Throughput (tokens/s) | Extents / copy | Memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16321 | 32 | restore | 6 (0) | 192/192 | 1233.6 | 7641 | 35.36 | 1.0000 | 81034 |
| 16321 | 32 | extent_lazy | 6 (0) | 192/192 | 88.6 | 6295 | 34.09 | 0.9641 | 23237 |

No repetition stalls: the first tokens arrive after 7.6 s with copies and
6.3 s with extents. The 32 agents hold 79.1 GiB with copies and 22.7 GiB
with extents, attach in 89 ms against 1,234 ms, and generate at 0.964 of
the summed speed of the copies (interval 0.952 to 0.976). All 192 agents
write the text of the copy. The first measurement stays in the record with
its failed gate; the figure of the paper takes this cell from the rerun.

**A long generation (`N_GEN=2048`, `20261007-engine-kvscale-gen2k-v1`).**
Every campaign above generates 64 to 192 tokens per agent. Here two agents,
one in each MIG instance, generate 2,048 tokens on the 4,081-token prefix
(context 8,192), so that the private tail grows to half the size of the
prefix.

| Prefix (tokens) | Agents | Mode | Runs (failed agents) | Exact texts | Attach (ms) | First token (ms) | Throughput (tokens/s) | Extents / copy | Memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4081 | 2 | restore | 6 (0) | 12/12 | 91.0 | 997 | 38.97 | 1.0000 | 2500 |
| 4081 | 2 | extent_lazy | 6 (0) | 12/12 | 25.7 | 956 | 37.83 | 0.9707 | 1944 |

Both agents write the text of the copy over all 2,048 tokens in the six
repetitions (12 of 12). They generate at 0.971 of the summed speed of the
copies (interval 0.969 to 0.972) and hold 1.90 GiB against 2.44 GiB. The
tail grows in steps as it is used and does not change the text or the
speed over the length of the generation.

**An existing way to share weights (`run_ext_weightshare.sh`,
`20261007-ext-weightshare-v1`).** cuda-llm-weight-share is a preloaded
library that exports the allocation of the weights of the first process
through CUDA IPC and maps it in the processes that follow; it runs with the
unmodified engine. A master computes the prefix of the agent workload,
saves the engine's state file and remains alive; eight agents restore the
prefix and generate. The three stacks differ only in the weights:
`stock`, `ipc` (the library) and `inplace` (this work). Memory is the
largest drop of MemAvailable from before the master starts; the model file
is in the page cache in every stack, and its 4.36 GiB are added to
`inplace`, which keeps it mapped.

| Placement | Stack | Runs | Agents complete | In the other instance | Library: workers / fallbacks | Weights load (ms) | First token (ms) | Throughput (tokens/s) | Memory (MiB) | 95% CI | Texts equal to stock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| same | stock | 6 | 48/48 | 0/0 | 0 / 0 | 2616 | 3824 | 24.33 | 55206 | 164 | 48/48 |
| same | ipc | 6 | 48/48 | 0/0 | 48 / 0 | 2020 | 3240 | 22.90 | 21585 | 71 | 48/48 |
| same | inplace | 6 | 48/48 | 0/0 | 0 / 0 | 1138 | 2383 | 22.85 | 14614 | 38 | 48/48 |
| cross | stock | 6 | 48/48 | 24/24 | 0 / 0 | 2540 | 3527 | 37.88 | 55162 | 31 | 48/48 |
| cross | ipc | 6 | 24/48 | 0/24 | 24 / 24 | 2374 | 3210 | 22.31 | 38267 | 46 | 24/24 |
| cross | inplace | 6 | 48/48 | 24/24 | 0 / 0 | 1119 | 2161 | 34.87 | 14523 | 77 | 48/48 |

In one MIG instance the library works: all 48 agents map the weights of
the master, and the stack holds 21.1 GiB against 53.9 GiB unmodified;
with the weights in place it holds 14.3 GiB plus the model file, 18.6 GiB.
The weights are ready after 2.0 s with the library and 1.1 s in place
(2.6 s unmodified). With every second agent in the other MIG instance the
24 agents of that instance do not complete with the library (CUDA IPC does
not cross the instances, Section 5.3), and the 24 of the master's instance
hold 37.4 GiB; in place all 48 complete in 18.5 GiB with the file. Every
agent that completes writes the text of the unmodified stack. W1 and W2
hold; W3 is the count above and is not gated.

### 6.15 vLLM servers on extents

The servers of the sections above belong to the engine that this work
changes. This section applies the design to vLLM 0.20.0, a batching server
that shares the blocks of a prefix between the sequences of one process and
nothing between processes. With one server in each MIG instance, the
weights and the prefix exist once per server.

**The plugin (`vllm_stator/`, 460 lines of Python).** No file of vLLM is
edited. The package registers under `vllm.general_plugins` and wraps three
places, each only when its environment variable is set.

- Weights. The first server lets vLLM load and convert the model (AWQ to
  the layout of its Marlin kernels), then writes the 283 parameters and
  buffers of at least 64 KiB to one file on a tmpfs with 2 MiB pages, each
  at a 4 KiB offset. The file is mapped and the device copies each tensor
  into the mapping, so 5.19 GiB are written in 0.15 s (a copy through
  `tensor.cpu()` took 21 s). Every server then maps the file read-only and
  shared, wraps the mapping in CUDA tensors (`__cuda_array_interface__`,
  no registration, no copy) and assigns them to the parameters. After the
  attach torch reports 17 MiB allocated on the device, down from 5,343 MiB,
  and vLLM logs "Model loading took 0.0 GiB memory".
- Prefix. vLLM is asked for the cache layout in which a block is one
  contiguous range over all layers (a connector with
  `prefer_cross_layer_blocks`); a block of 16 tokens is then 917,504 bytes.
  The one allocation of that layout is replaced: in the first server by a
  shared mapping of a file, in a later server by private anonymous memory in
  which the range of the published blocks is a read-only shared mapping of
  that file. When the first request of the first server finishes, the
  connector records the hashes of its full blocks, keeps a reference on
  them and write-protects their range. A later server registers these
  hashes in its block pool with a reference of its own, so vLLM's prefix
  caching finds them and never hands them out. The connector moves no data;
  its load and save functions are empty.
- Device names. vLLM parses `CUDA_VISIBLE_DEVICES` as integers while it
  imports its layers, which is before plugins load, and the engine is a
  spawned child. `python -m stator_vllm.serve` accepts a name first and then
  runs vLLM's command line; a spawned child imports the main module of its
  parent before it reads its arguments, so the engine inherits the change.
  Every configuration below starts this way, the baselines included.

Both servers need the same `PYTHONHASHSEED`, since vLLM seeds its block
hashes from it. A server attaches when it starts: the prefix has to be
published before a later server allocates its cache.

**Campaign (`run_vllm_share.sh`, `20261007-vllm-share-v1`).** Qwen2.5-7B-
Instruct from its 4-bit AWQ weights, a cache of 2 GiB per server, `-O0`,
`--gpu-memory-utilization 0.4` (vLLM otherwise refuses to start a second
server on unified memory). Server 1 starts in the 12-SM instance and
answers the 14,282-token prefix of the agent workload alone; server 2
starts in the 6-SM instance. Round `first`: four agents on server 2, which
has not computed the prefix. Round `both`: four other agents on each
server at the same time. Rounds `alone1`, `alone2`: one agent on a server
with nothing else running. Six configurations, six repetitions, the order
of the configurations turning with the repetition:

- `vllm`: two unmodified servers;
- `cpu`: vLLM's own offloading of cache blocks to host memory, 2 GiB per
  server (`--kv-offloading-backend native`, which needs
  `--disable-hybrid-kv-cache-manager`); the buffer belongs to one server;
- `lmcache`: LMCache 0.5.5 through vLLM's `LMCacheConnectorV1`, with a
  store process in host memory that both servers reach (`lm://`), chunks of
  256 tokens, no compression, no cache of its own in a server and a staging
  pool of 1 GiB per server. The multiprocess mode of LMCache was not used:
  its server opens the cache of a vLLM worker through CUDA IPC, which the
  two MIG instances do not accept from each other (Section 5.3);
- `stator_weights`, `stator_kv`, `stator`: the plugin with the weights
  only, the prefix only, and both.

| Mode | Servers | Cases (failed) | Requests complete | Memory (MiB) | 95% CI | vs vllm (MiB) | Peak (MiB) | Shared files (MiB) | Mapped by a later server: weights, prefix (MiB) | First token, second server (ms) | 95% CI | Smallest cached share | Throughput (tokens/s) | 95% CI | vs vllm | median | 12-SM, 6-SM (tokens/s) | Start-up, first and later server (ms) | Weights publish, attach (ms) | Prefix attach (ms) | LMCache store, retrieve (ms) | Texts equal to vllm: first, both, alone1, alone2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vllm | 2 | 6 (0) | 84/84 | 21272 | 33 | 0 | 21706 | 0 | 0, 0 | 14108 | 48 | 0.0000 | 145.69 | 0.85 | 1.0000 | 1.0000 | 85.40, 60.29 | 24223, 26023 | 0, 0 | 0 | 0, 0 | 24/24, 48/48, 6/6, 6/6 |
| cpu | 2 | 6 (0) | 84/84 | 25281 | 435 | 4009 | 25367 | 0 | 0, 0 | 14188 | 19 | 0.0000 | 145.74 | 1.33 | 1.0003 | 0.9957 | 85.83, 59.91 | 24782, 26100 | 0, 0 | 0 | 0, 0 | 22/24, 48/48, 6/6, 6/6 |
| lmcache | 2 | 6 (0) | 84/84 | 26617 | 252 | 5345 | 26670 | 0 | 0, 0 | 1809 | 99 | 0.9841 | 145.26 | 0.14 | 0.9970 | 0.9965 | 85.68, 59.58 | 25541, 26726 | 0, 0 | 0 | 407, 924 | 19/24, 48/48, 6/6, 6/6 |
| stator_weights | 2 | 6 (0) | 84/84 | 15690 | 216 | -5582 | 15705 | 5732 | 5321, 0 | 14220 | 14 | 0.0000 | 137.35 | 0.58 | 0.9427 | 0.9449 | 86.09, 51.26 | 24230, 18013 | 162, 1405 | 0 | 0, 0 | 22/24, 48/48, 6/6, 6/6 |
| stator_kv | 2 | 6 (0) | 84/84 | 20552 | 106 | -720 | 21802 | 2050 | 0, 780 | 661 | 59 | 0.9971 | 141.59 | 0.96 | 0.9718 | 0.9721 | 84.91, 56.67 | 24036, 25036 | 0, 0 | 23 | 0, 0 | 24/24, 48/48, 6/6, 6/6 |
| stator | 2 | 6 (0) | 84/84 | 14840 | 116 | -6432 | 14863 | 7782 | 5321, 780 | 645 | 57 | 0.9971 | 138.20 | 0.67 | 0.9485 | 0.9480 | 89.37, 48.82 | 24296, 18124 | 162, 1414 | 23 | 0, 0 | 22/24, 48/48, 6/6, 6/6 |

Memory is the drop of MemAvailable from before server 1 starts to the end
of the case, shared files included; the peak is the largest drop during
the case, the start of the servers included.

*Memory.* Two unmodified servers hold 20.71 GiB. vLLM's offloading adds its
buffers (24.82 GiB) and LMCache its staging pools, its process and a third
copy of the prefix in the store (26.01 GiB). With the plugin the servers
hold 14.46 GiB, 6.25 GiB less: the second server maps 5.19 GiB of weights
and 0.76 GiB of prefix blocks that exist once. The weights alone account
for 5.49 GiB and the prefix alone for 0.71 GiB. The peak is not lower
(20.99 GiB against 21.16 GiB): vLLM's loader runs unchanged, so a starting
server holds a device copy of the weights until the plugin replaces it.

*First token of the second server.* It computes the prefix itself in `vllm`,
`cpu` and `stator_weights` (14.1 to 14.2 s). With LMCache it copies 14,080
tokens from the store in 0.90 s and answers after 1.79 s; storing them had
cost the first server 0.41 s inside its request. With the plugin the
server maps the 892 blocks in 23 ms when it starts and answers after
0.65 s; publishing copies nothing.

*Throughput.* The summed generation speed of the eight agents of round
`both` is 138.4 tokens/s with the plugin against 145.7 tokens/s, a paired
ratio of 0.950 (median 0.951). The loss is in the 6-SM instance: its
server generates at 0.820 of its speed in `vllm`, the server of the 12-SM
instance at 1.041. With the weights alone the 6-SM server is at 0.860,
with the prefix alone at 0.950. This is the cost of host memory in that
instance that Section 6.8 measures for the engine, larger here because the weights
are read from host memory as well.

*Texts.* In `alone1` and `alone2` every configuration writes the text of
`vllm` in all six repetitions, in both instances; in `alone2` the prefix of
the plugin and of LMCache comes from the other instance. With concurrent
agents the texts differ in a few cases in every configuration that is not
`vllm` itself, the ones that share nothing included (`cpu`: 67 of 72
equal; `lmcache`: 69; the three plugin modes: 66 to 67), because vLLM
batches the sequences of a server and the batches differ between runs.

All gates hold: V1 (every server starts, every request completes), V2
(every agent of `stator` in `first` is served at least 99.7% of its prompt
from the cache; the first agent of `vllm` none), V3 (first token 0.65 s
against 14.13 s), V4 (6.25 GiB saved against 5.95 GiB mapped), V5 (0.950)
and V6 (`alone1` equal in every mode).

**Four servers (`SERVERS=4`, `20261007-vllm-share-4srv-v1`, two
repetitions).** Servers 3 and 4 start after server 2, in the 12-SM and the
6-SM instance, and attach like server 2.

| Mode | Servers | Cases (failed) | Requests complete | Memory (MiB) | 95% CI | vs vllm (MiB) | Peak (MiB) | Shared files (MiB) | Mapped by a later server: weights, prefix (MiB) | First token, second server (ms) | 95% CI | Smallest cached share | Throughput (tokens/s) | 95% CI | vs vllm | median | 12-SM, 6-SM (tokens/s) | Start-up, first and later server (ms) | Weights publish, attach (ms) | Prefix attach (ms) | LMCache store, retrieve (ms) | Texts equal to vllm: first, both, alone1, alone2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vllm | 4 | 2 (0) | 44/44 | 42620 | 267 | 0 | 42647 | 0 | 0, 0 | 14067 | 395 | 0.0000 | 237.19 | 4.20 | 1.0000 | 1.0000 | 131.74, 105.44 | 23985, 25016 | 0, 0 | 0 | 0, 0 | 8/8, 32/32, 2/2, 2/2 |
| lmcache | 4 | 2 (0) | 44/44 | 51551 | 165 | 8931 | 51574 | 0 | 0, 0 | 1739 | 40 | 0.9841 | 288.64 | 0.24 | 1.2169 | 1.2169 | 165.88, 122.76 | 25515, 26321 | 0, 0 | 0 | 415, 841 | 7/8, 31/32, 2/2, 2/2 |
| stator | 4 | 2 (0) | 44/44 | 23269 | 305 | -19351 | 29843 | 7370 | 5317, 780 | 658 | 758 | 0.9971 | 123.05 | 0.47 | 0.5188 | 0.5188 | 77.10, 45.95 | 23946, 25542 | 152, 97 | 23 | 0, 0 | 6/8, 29/32, 2/2, 2/2 |

Four unmodified servers hold 41.62 GiB, four with LMCache 50.34 GiB and
four with the plugin 22.72 GiB. From two to four servers the memory grows
by 10.5 GiB per server unmodified and by 4.1 GiB per server with the
plugin (the cache reservation of 2 GiB less the prefix, and the process).
The largest drop during the case is 29.14 GiB with the plugin against
41.65 GiB: the device copy of a starting server exists once at a time. The
first token of server 2 and the texts of the two rounds with one agent are
as with two servers.

The throughput column of this campaign does not compare the
configurations, and the gate V5 fails on it (0.519). Round `both` sends
four agents to every server, but only servers 1 and 2 have the prefix in
`vllm` and `lmcache`: servers 3 and 4 compute it (first tokens after 13.8
and 20.2 s) or fetch it from the store (4.9 and 6.8 s) during the round,
and their agents generate after those of servers 1 and 2 have finished.
The sum of the rates of these two configurations therefore adds rates of
different times (237.2 and 288.6 tokens/s) and is not a throughput. Only
with the plugin do all 16 agents generate at the same time, at
123.1 tokens/s. The like-for-like part of the round is the rate of an
agent on servers 1 and 2 while the other server of its instance is busy:
9.3 tokens/s (12-SM) and 6.7 tokens/s (6-SM) in `vllm`, 9.6 and 5.6
tokens/s with the plugin. The next campaign repeats the case with every
server holding the prefix before the round.

**Four servers that all hold the prefix (`run_vllm_share2.sh`,
`20261007-vllm-share-4srv-v2`, two repetitions).** The runner is
`run_vllm_share.sh` with one addition: after round `first`, servers 3 and 4
each answer one agent alone, so that every server holds the prefix when
round `both` starts.

| Mode | Servers | Cases (failed) | Requests complete | Memory (MiB) | 95% CI | vs vllm (MiB) | Peak (MiB) | Shared files (MiB) | Mapped by a later server: weights, prefix (MiB) | First token, second server (ms) | 95% CI | Smallest cached share | Throughput (tokens/s) | 95% CI | vs vllm | median | 12-SM, 6-SM (tokens/s) | Start-up, first and later server (ms) | Weights publish, attach (ms) | Prefix attach (ms) | LMCache store, retrieve (ms) | Texts equal to vllm: first, both, alone1, alone2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vllm | 4 | 6 (0) | 144/144 | 42549 | 50 | 0 | 42567 | 0 | 0, 0 | 14028 | 31 | 0.0000 | 131.56 | 0.22 | 1.0000 | 1.0000 | 76.40, 55.17 | 23972, 24996 | 0, 0 | 0 | 0, 0 | 24/24, 96/96, 6/6, 6/6 |
| lmcache | 4 | 6 (0) | 144/144 | 51613 | 35 | 9064 | 51628 | 0 | 0, 0 | 1726 | 68 | 0.9841 | 131.22 | 0.29 | 0.9974 | 0.9974 | 76.31, 54.91 | 25322, 26618 | 0, 0 | 0 | 402, 840 | 20/24, 79/96, 6/6, 6/6 |
| stator | 4 | 6 (0) | 144/144 | 23273 | 27 | -19276 | 29841 | 7370 | 5317, 780 | 595 | 45 | 0.9971 | 122.08 | 0.30 | 0.9279 | 0.9277 | 76.50, 45.57 | 24544, 25000 | 148, 97 | 23 | 0, 0 | 22/24, 85/96, 6/6, 6/6 |

The 16 agents of every configuration now generate at the same time: 131.5
tokens/s unmodified, 131.0 tokens/s with LMCache and 121.8 tokens/s with
the plugin, a paired ratio of 0.926. The two servers of the 12-SM instance
are at 1.000 of their speed in `vllm` and the two of the 6-SM instance at
0.823, as with two servers (1.041 and 0.820). Memory and first token repeat
the first run: 41.68, 50.35 and 22.74 GiB; 14.08, 1.86 and 0.66 s; the
largest drop with the plugin is 29.28 GiB. All gates hold in this
campaign, V5 included.

## 7. Novelty boundary

`SOTA_HOSTMM_2026-10-05.md` Section 14 is the audit for this concept, its
Section 16 the audit for the cache mechanism of Section 6, and its Section
12 covers the substrate. The sources named as closest were opened directly
and exist. The verdicts, with what was measured here:

| Claim | Verdict in the audited set | Closest prior work | What remains ours |
|---|---|---|---|
| Share: N processes on one copy of the weights, read in place by the GPU | partially taken | "The Ingestion Tax" (arXiv 2608.12114): shared read-only mappings read in place, up to four processes, measured on Apple hardware; a closed, unmeasured pull request for host-pointer CUDA buffers in this engine; a tool that shares this engine's CUDA weights through CUDA IPC | the measured in-place weight path in a CUDA engine on an NVIDIA unified-memory device (Section 3), its page-size dependence (no cost on 2 MiB pages), sharing across MIG instances, which CUDA IPC cannot do here (Section 5.3), the measurement up to eight processes and with adapters, and the sealed read-only base with divergence |
| A writable private mapping of weights becomes a private copy | partially taken | an issue thread of 2026-09-28 that states the driver's write-intent fault service with the source lines | the accessed-flag trigger and the amplification: one page per 2 MiB block, 0.2% of the pages, costs the whole model; the same under MPS. It also corrects a sentence of "The Ingestion Tax" that Linux keeps private file mappings file-backed, for writable mappings on this driver |
| The cost that `fork` leaves in a GPU process | open in the audited set, with thin search coverage | driver source comments on per-fault TLB invalidation | the measurement (2.7 s per GiB) and three remedies measured against the kernel's own repair |
| Blast radius of a GPU fault under MPS, MIG, and time slicing | the propagation fact is taken; the recommendation is partially taken | NVIDIA's MPS documentation; a 2026 characterization of MPS fault propagation; one MPS server per MIG instance is a documented workflow | who survives on this device (Section 5.2) together with the memory sharing that holds in every configuration, and the throughput of each configuration |
| One copy of a computed prefix cache shared by separate processes by mapping, each with private continuation | partially taken | Omni-Flow (arXiv 2606.31093): role processes bind an alias to one physical copy of a cache pool and keep changing state in private writable pages, in device memory on one GPU; an open proposal for SGLang that exports a cache slab through CUDA IPC to replicas under MPS | the same through the host page tables: the rows of the prefix are pages of a file that every agent maps read-only into its own cache tensors, measured in an engine up to eight agents (Section 6.4) |
| The same across GPU partitions | open in the audited set | every mapping-based design found rests on CUDA IPC, which NVIDIA documents as unsupported across GPU instances; everything that crosses such a boundary copies into each engine (LMCache, vLLM and SGLang host pools) | half of the agents of Section 6.4 run in the other MIG instance; the device-memory route, built into the same engine for comparison, stops at the instance boundary (Section 6.9) |
| An immutable shared extent with a private tail, chosen to avoid copy-on-write | the layout is taken; the reason is partially taken | block-level prefix caches (vLLM, SGLang), ForkKV, Omni-Flow; issue #81 for the write-intent fault service; Markuze et al. (ASPLOS 2016) for "an invalidation can cost more than a copy" in the DMA setting | the measured cost of a single-page copy-on-write break in a process bound to a GPU by shared virtual addressing (substrate record), the amplification of a GPU read, and the engine-level comparison with a copy-on-write mapping (Sections 6.4 and 6.7) |
| State files that name rows instead of carrying them | partially taken | llama.cpp pull request #21792 (open): cache tensors in a `MAP_SHARED` file with a metadata sidecar, resumed by another process, "CPU only ... GPU device memory cannot be mmap'd" | the same for a cache that the GPU reads in place, with several agents attached at once to a frozen range (publish in 3 to 9 ms) |
| A cache whose memory follows the tokens in use | taken | vAttention (ASPLOS 2025), kvcached | nothing; here it is the kernel's demand paging of host memory, and it is reported as a consequence, not as a contribution |
| Fork or move of a running serving context between processes without a copy | open in the audited set for GPU-visible state; taken for CPU inference as a sequential hand-off | request-level fork inside one process (SGLang, Parrot, ForkKV); Execution-State Capsules fork by copy on this device; llama.cpp pull request #21792 | a parent that publishes with a pause of 3 ms and keeps generating while four children in both MIG instances continue from its rows (Section 6.5) |
| Move of a generic GPU-visible object with revocation | open at the mechanism level | zero-copy hand-off inside one CUDA context on this device class | GPU Portals (`RESEARCH_GPU_PORTALS_2026-10-04.md`); not connected to the engine |

Permitted wording:

> Reading model weights in place from a mapped file is known on Apple
> devices and was proposed for llama.cpp's CUDA backend. To our knowledge,
> among the works we audited, no measurement of it exists for a CUDA engine
> on an NVIDIA unified-memory device. On Thor it generates identical text
> 6% slower on 4 KiB pages, and each further serving process needs 4.4 GiB
> less memory for a 4.36 GiB model.

> The host page table shares the weights between processes in every
> placement we tried: across MIG instances, time-sliced in one instance, and
> as clients of an MPS server. CUDA IPC did not cross MIG instances.

> Separate serving processes do not raise throughput on this GPU; one
> batching process generates about three times as much as eight processes.
> One batching server per MIG instance, both reading one copy of the
> weights in place, generates 149 tokens/s for sixteen sequences in 6.6 GiB.

> On this driver a writable private mapping of the weights is not a safe
> way to share them: one page per 2 MiB block without the accessed flag
> turns the whole model into a private copy at the next GPU read. The
> write-intent fault service that causes this is public source and was
> reported before us.

> Sharing a prefix cache between processes without a copy exists in device
> memory on one GPU, and a cache in a mapped file exists for CPU inference.
> To our knowledge, among the works we audited, none maps the pages of a
> computed prefix into the cache tensors of several GPU-serving processes
> through the host page tables, none shares a cache across GPU partitions
> without a copy, and none forks a running serving process without one. On
> Thor eight agents on a 16,321-token prefix hold 5.7 GiB instead of
> 19.9 GiB and write the texts of agents that copy the prefix.

> We do not use the kernel's copy-on-write for GPU-visible state. The
> layout that replaces it, an immutable prefix and a private tail, is how
> prefix caches work inside one process; we make it between processes with
> mappings because, on this device, a page that changes owner through a
> fault costs an invalidation for the whole process and a GPU read can be
> served as a write.

Must not be claimed: first in-place or zero-copy weight sharing; first
multi-adapter serving; that separate processes give more throughput; that
CUDA IPC fails under MPS on this device (it worked in the probe); a new finding that MPS propagates faults; that the
results hold on DGX Spark or Grace Hopper (the driver path is shared, MIG
is not, and nothing was measured there); first cross-process sharing of a
key-value cache, without the qualification "through the host page tables";
the immutable-prefix layout, demand-backed caches or metadata-only state
files as ideas; that extents make an agent start much faster (the first
token moves by 4 to 23%); that extents cost no generation speed on this
device (nothing in the 12-SM instance, 2 to 6% in the 6-SM instance); that a prefix moved between the MIG instances reproduces a
recomputation there (the instances differ in their bits).

## 8. Limits

- **One device, one engine, one 7B model** (and a 1.1B model for the
  page-size result). Nothing was measured on DGX Spark, Grace Hopper, or an
  Apple device.
- **Serving several agents from one process is faster in total.** Section 4
  measures it. Separate processes are for agents that need their own
  address space, their own engine version, or containment of each other's
  faults.
- **The engine path shares a file, not a sealed object.** The engine maps
  the model file shared and read-only, which is enough against the
  write-intent fault service, and a process cannot write the weights
  through its mapping. Whoever can write the file can still change the
  model for every agent. The sealed `memfd` of
  `RESEARCH_HOSTMM_2026-10-05.md` Section 8.2 removes that, and was measured
  with a synthetic kernel, not with the engine.
- **Adapters are synthetic and applied at run time.** No adapter was merged
  into the base, so the divergence of extents was exercised only by the
  synthetic kernel.
- **Row padding.** A model with a quantized tensor whose rows are not a
  multiple of 512 elements is refused by the in-place path.
- **Page cache can be evicted.** Under memory pressure the kernel may drop
  pages of the mapped model; the next GPU access then faults and reads them
  back. The mapping is read-only, so this costs time and never a copy. It
  was not measured.
- **Memory is measured from outside the process**, as the drop of available
  memory plus the proportional size of the model mapping, because device
  allocations are not charged to a process on this platform.
- **Fault containment is specific to this device and driver.** Thor has two
  MIG instances. Driver 595.78 predates the partial error isolation between
  MPS partitions that NVIDIA documents for a later release.
- **The cache mechanism was driven by a small program, not by a server.**
  `kv_fork.cpp` decodes fixed batches and takes the most probable token.
  The engine's own `llama-completion` ran on the path in a manual check
  (identical text), not in a campaign, and `llama-server` did not run on it.
- **One level of fork.** A publisher cannot load a state and an agent cannot
  publish, so a child cannot hand its own rows on. Chains of forks need a
  file per agent.
- **Rows must be cells.** The path refuses a cache with a transposed value
  tensor (no flash attention) or with several streams, because a prefix is
  then not a leading range of each tensor.
- **A frozen cell is lost to its agent.** An agent that drops tokens of the
  prefix, for example when its context shifts, does not get their cells
  back.
- **Agents trust the publisher.** An agent cannot change the prefix: its
  mapping is read-only and its cache never hands out a mapped cell. Whoever
  can write the cache file can change the prefix for every agent, and
  nothing here addresses timing channels between processes that share a
  cache. Sharing across the two MIG instances is for agents of one tenant
  that are separated for performance or fault containment, not for mutually
  distrusting tenants.
- **The cache file is memory.** It lives on a tmpfs and cannot be evicted;
  the device has no swap.
- **State moved between the MIG instances is not what the other instance
  would compute** (Section 6.6).
- **The device-memory counterpart is our own baseline** (Section 6.9): one
  2 MiB allocation and one handle per step of every tensor, not tuned, and
  not the code of any of the systems it stands for. A version with larger
  allocations would attach faster and share less.
- **The cost of a host-memory cache in the 6-SM instance is measured, not
  explained** (Section 6.8).
- **The mapping of a host-memory cache is not released** when its context is
  freed. The experiments create one context per process.
- **Six repetitions per cell**; five for the CUDA IPC route check.

## 9. Artifacts

| Evidence | Directory |
|---|---|
| one process, device copy against in place | `llm_share/results/20261005-engine-single-v1/` |
| page size of the mapped model, 7B and 1.1B | `llm_share/results/20261005-engine-pages-v1/`, `20261005-engine-pages-small-v1/` |
| five processes, four adapters, one base | `llm_share/results/20261005-engine-adapters-v1/` |
| N processes under MIG, time slicing, MPS | `llm_share/results/20261005-engine-agents-v1/` |
| one process with N sequences | `llm_share/results/20261005-engine-batched-v1/` |
| one batching server per MIG instance | `llm_share/results/20261005-engine-groups-v1/` |
| a shared prompt prefix: recompute against the prompt-cache file | `llm_share/results/20261005-engine-prefix-v1/` |
| CUDA IPC against the host page table | `llm_share/results/20261005-sharing-routes-v2/` |
| agents on one published prefix: recompute, copy, copy-on-write, extents | `llm_share/results/20261006-engine-kvshare-v1/` |
| a parent that forks its state while it keeps running | `llm_share/results/20261006-engine-kvfork-v1/` |
| the bits of a prefix cache across repetitions and MIG instances | `llm_share/results/20261006-engine-kvdet-v1/` |
| copy-on-write with and without the CPU read pass | `llm_share/results/20261006-engine-kvcow-v1/` |
| generation speed with the cache in device and in host memory, both instances and prefix lengths | `llm_share/results/20261006-engine-kvspeed-v1/`, `-long-v1/`, `-6sm-v1/`, `-6sm-long-v1/` |
| shared and private device memory composed with CUDA's virtual memory interface | `llm_share/results/20261006-vmm-routes-v1/` |
| eight agents in one batching server, and in one server per MIG instance with and without extents | `llm_share/results/20261006-engine-kvbatch-v1/` |
| host extents against a device-memory cache and the copy, a parent and four children in one instance | `llm_share/results/20261006-engine-kvvmm-v1/` |
| extents under time slicing, MPS, MIG, and MPS within MIG | `llm_share/results/20261006-engine-kvmps-v1/` |
| copy, demand-backed copy, shared device memory, copy-on-write, and extents in one engine | `llm_share/results/20261006-engine-kvsota-v1/` |
| two-level agent trees with flat and chained prefixes | `llm_share/results/20261006-engine-kvtree-v1/` |
| 8, 16, and 32 agents; Llama-3.1-8B, Qwen2.5-14B, and an agent trace | `llm_share/results/20261006-engine-kvscale-v1/`, `-llama8b-v1/`, `-qwen14b-v1/`, `-agent-v1/` |
| long-running server save and restore path | `llm_share/results/20261006-engine-kvserver-v1/` |
| process accounting and an enforced cgroup memory limit | `llm_share/results/20261006-engine-kvlimit-v1/` |
| raw host- and device-memory read path | `llm_share/results/20261006-read-path-v1/` |
| shared device-memory attach cost by allocation granule | `llm_share/results/20261006-vmm-attach-v1/` |
| write protection of host mappings and device VMM imports | `llm_share/results/20261006-protect-v1/` |
| one and two unmodified Ollama servers, cold and warm prefixes | `llm_share/results/20261006-ollama-agents-v1/` |
| sealed base under MIG, time slicing, MPS; fault containment | `thor_hostmm/results/20261005-sharing-modes-v1/` |
| sealed base: modes, scaling, attempts to modify it | `thor_hostmm/results/20261005-shared-model-v1/` |
| pages copied per page without the accessed flag | `thor_hostmm/results/20261005-af-block-v1/` |
| tax on memory-management operations; ways to start a helper | `thor_hostmm/results/20261005-mm-tax-v1/` |

`llm_share/verify_llm_share_artifact.sh`,
`llm_share/verify_stator_campaigns.sh`, and
`thor_hostmm/verify_hostmm_artifact.sh` rebuild every summary from its raw
log, check the pinned sources, and re-evaluate the relations stated above
without running the GPU. The engine patches are
`llm_share/inplace_weights.patch`, `llm_share/kv_extents.patch` and, for the
comparison of Section 6.9, `llm_share/kv_vmm.patch`, each against llama.cpp
commit `6f767fe96` in a clone of its own; each contains the one before it. No experiment reconfigured MIG or rebooted the device. MPS servers, the hugetlb
pool, and the temporary mounts were started with the campaigns and
removed by them. Result directories carry the tag of the day the series
began; `metadata.txt` in each holds the time of the run.
