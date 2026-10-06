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
stay alive and exposes device memory that the importer can write, while a
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
private, after the same CPU read pass over the prefix, and lets the kernel
copy what the GPU writes.

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
(`results/20261006-engine-kvcow-v1/`, six repetitions, no failed agent) lets
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
and a second process maps it read-only next to an allocation of its own and
lets the GPU read across the seam (`results/20261006-vmm-routes-v1/`, six
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
repetitions, no failed server) lets it serve eight agents on the
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

### 6.11 Gates

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
| sealed base under MIG, time slicing, MPS; fault containment | `thor_hostmm/results/20261005-sharing-modes-v1/` |
| sealed base: modes, scaling, attempts to modify it | `thor_hostmm/results/20261005-shared-model-v1/` |
| pages copied per page without the accessed flag | `thor_hostmm/results/20261005-af-block-v1/` |
| tax on memory-management operations; ways to start a helper | `thor_hostmm/results/20261005-mm-tax-v1/` |

`llm_share/verify_llm_share_artifact.sh` and
`thor_hostmm/verify_hostmm_artifact.sh` rebuild every summary from its raw
log, check the pinned sources, and re-evaluate the relations stated above
without running the GPU. The engine patches are
`llm_share/inplace_weights.patch`, `llm_share/kv_extents.patch` and, for the
comparison of Section 6.9, `llm_share/kv_vmm.patch`, each against llama.cpp
commit `6f767fe96` in a clone of its own; each contains the one before it. No experiment reconfigured MIG or rebooted the device. MPS servers, the hugetlb
pool, and the temporary mounts were started with the campaigns and
removed by them. Result directories carry the tag of the day the series
began; `metadata.txt` in each holds the time of the run.
