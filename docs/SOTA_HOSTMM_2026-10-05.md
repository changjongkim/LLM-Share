# Host-MM GPU memory on Thor: prior-art audit (2026-10-05)

Provenance: produced by a delegated search pass on 2026-10-05 that used primary
sources only. Every row carries a verification tag. Tags marked V (doc) are text
extracted by a fetch tool and must be re-opened in a browser before a sentence
that depends on them is submitted. The measured platform facts quoted in the
brief are those of `RESEARCH_HOSTMM_2026-10-05.md` at the time of the audit;
two were refined afterwards and are corrected in Section 10 below.


## 0. Method, verification legend, and limits

Only primary sources were used. Search-engine summaries were used to find sources, never as evidence.

| Tag | Meaning |
|---|---|
| **V (pdf)** | The PDF was downloaded from the listed primary URL, converted to text, and the cited passage was read. Highest assurance. |
| **V (src)** | Source file downloaded and read. Linux files are tag `v6.8` from the `gregkh/linux` GitHub mirror, because `git.kernel.org` served a bot challenge and `torvalds/linux` raw access was rate-limited on this date. |
| **V (doc)** | Official documentation, mailing-list archive, LWN article, or official repository page, opened through a fetch tool that returns model-extracted text. Quotes were requested verbatim but must be re-read in a browser before they are quoted in a paper. |
| **V (abs)** | Abstract or landing page only. "No" in such a row means "absent from the abstract". |
| **M** | Bibliographic metadata only (title, authors, venue). Content not opened. |
| **NV** | Not verified. Content comes from an index snippet or from a citation inside another paper. Must be opened before citing. |

Limits of this pass:

- ACM DL, IEEE Xplore, Wiley, MDPI, and lkml.org refused the fetch. Works reachable only there are M or NV.
- All unit conversions to ms/GiB or us/page are mine and are marked "(conv.)". They assume 4 KiB pages unless the source states otherwise.
- The device measurements listed in the task (2026-10-05) are treated as given and were not re-run.
- "Not found" means "not found by the queries in Section 6", not "does not exist".

Device numbers used for comparison (given): fork 25 ms/GiB (4 KiB) and 1.6 ms/GiB (THP); steady GPU write 5 ms/GiB; first GPU write after fork 3,100 ms/GiB (about 12 us per 4 KiB page); first GPU touch of fresh memory 4,900 ms/GiB (about 19 us/page, conv.); CPU-side CoW pre-break 2,000 to 2,900 ms/GiB; range `mprotect` 13 ns/page.

## 1. Verdicts

| Item | Verdict | Single closest work | Most dangerous prior work |
|---|---|---|---|
| Facet A: accounting (cgroup) | **PARTIALLY TAKEN** | Linux `dmem` cgroup controller, whose author defers main memory because of double accounting | Upstream Linux work in 2026 (dmem reclaim, memcg accounting of dma-buf heaps) plus NVIDIA's own driver accounting; KRYPTON (ATC 2025) on the academic side |
| Facet B: snapshot by fork CoW with extent privatization | **PARTIALLY TAKEN** | CRUM (fork-based CoW checkpoint for CUDA UVM, 2018) | PhoenixOS (SOSP 2025): concurrent GPU checkpoint with "soft copy-on-write" |
| Facet C: CoW sharing across MIG tenants | **OPEN in the verified set**, but it is a known OS technique on a new platform and it is not yet built | C2CServe (MIG instances read CPU-resident weights) | ForkKV and vLLM (CoW of GPU inference state inside one runtime); Tetris (mmap-based tensor deduplication across instances) |
| Characterization: about 1000x per-page versus range cost on a host-page-table GPU | **PARTIALLY TAKEN** | Grace Hopper system-memory study (ICPP 2024): same mechanism, GPU first touch is slow, CPU-side population removes it | MI300A study (2025) and ISPASS 2016 for per-fault numbers; MSched (347x) and NPF (ASPLOS 2017) for the per-page versus batched cliff |
| Umbrella claim: host MM as single owner of device memory | **PARTIALLY TAKEN** | GMEM (Rice, arXiv 2023) and Linux HMM | GMEM |

Details and justifications are in Sections 3, 4, and 8.

## 2. Tables by group

Column key: **Mechanism** = how CoW, dirty tracking, accounting, or revocation is obtained. **UMA?** = works on a unified-memory GPU. **Intercept/driver?** = needs API interception or a driver change.

### Group 1. Transparent GPU checkpoint, restore, snapshot; CoW or dirty tracking for GPU memory

| Work (authors, venue, year, URL, tag) | What it does | Mechanism | UMA? | Intercept/driver? | Comparable cost numbers | Exact overlap with the thesis | What remains different |
|---|---|---|---|---|---|---|---|
| **PhoenixOS**. Wei, Huang, et al., SOSP 2025. [arXiv 2405.12079](https://arxiv.org/abs/2405.12079). **V (pdf)** | Concurrent OS-level checkpoint and restore of GPU processes | Write set is speculated from kernel-launch arguments and validated by PTX binary instrumentation. "Soft CoW": before a kernel runs, a buffer that is not yet checkpointed is copied to a spare GPU buffer (up to 2 GB reserved). Granularity is a whole buffer. A mis-speculation discards the checkpoint and retries stop-the-world | Not evaluated. Testbed is 8x A800 80 GB, CUDA 11.3 | Yes: intercepts all GPU APIs and instruments kernels | Stop-the-world checkpoint and restore each stall more than 2.1 s; Llama2-13B migration downtime 9.8 s to 2.3 s; start in 622 ms | Concurrent snapshot of GPU state with CoW semantics; predicting the GPU write set before the write; a fallback path when prediction fails | Its premise is that GPUs lack "OS-mediated data paths (e.g., copy-on-write)" and dirty bits. On Thor pageable memory this premise does not hold: kernel CoW is the native mechanism. PhoenixOS's fallback aborts the checkpoint; in the thesis kernel CoW is the fallback and the snapshot stays correct. PhoenixOS handles `cudaMalloc` memory, which the thesis cannot |
| **GCR**. Zeng, Ren, Shu, Lu, FAST 2026. [USENIX PDF](https://www.usenix.org/system/files/fast26-zeng.pdf). **V (pdf)** | Fast and lightweight GPU checkpoint and restore with incremental checkpoints | Driver-integrated C/R for control state, interception of GPU memory (de)allocation for data buffers. Dirty buffers are found by "shadow execution" of kernels on the CPU using dirty templates. States that "current GPU hardware lacks dirty bits" | Not evaluated. Testbed is 2x A100-40GB | Yes: selective interception plus `cuda-checkpoint` | Checkpoint latency reduced 72.1% versus cuda-checkpoint and 63.6% versus PhOS; incremental checkpoint cuts size 86.6% and latency 43.8% | Incremental (dirty-only) GPU checkpointing; identifying what the GPU wrote | Software emulation of dirty tracking at buffer level for device memory. No OS page-table mechanism, no fork |
| **CRIUgpu**. Stoyanov, Spišaková, Ramos, Gurfinkel, Vagin, Reber, Armour, Bruno, [arXiv 2502.16631](https://arxiv.org/abs/2502.16631), 2025 (venue NV). **V (pdf)** | Transparent unified CPU-GPU snapshots through CRIU plugins | Calls `cuda-checkpoint` (lock, checkpoint, restore, unlock) and AMD KFD ioctls. Stop-the-world. No CoW, no dirty tracking | Not evaluated. H100, A100, V100, A6000, MI210 | No interception. Needs driver checkpoint support | GPU state to host memory: 4.9 s (GPT-2 Small) and 28 s (GPT-2 XL); lock 240 ms and 500 ms; restore 2.5 s and 11 s | Transparent, interception-free GPU snapshot | Pause-and-copy of device memory. The pause scales with state size. The thesis pauses for one fork (25 ms/GiB or 1.6 ms/GiB) |
| **NVIDIA cuda-checkpoint**. [README](https://github.com/NVIDIA/cuda-checkpoint). **V (doc, raw README read)** | Suspend and resume CUDA state of a process for CRIU | Locks CUDA APIs, drains work, "device memory is copied to the host, into allocations managed by the CUDA driver", releases GPU resources | Not stated. Driver 595 adds "Support for ARM CPUs" (aarch64 binary). Tegra is not mentioned | Driver feature (550+) | None in README | Vendor path for transparent GPU checkpoint | Stop-the-world copy. README: "does not support UVM memory or IPC memory created with `cuMemExportToShareableHandle()`". Whether it runs on Thor is unknown |
| **CRUM**. Garg, Mohan, Sullivan, Cooperman, [arXiv 1808.00117](https://arxiv.org/abs/1808.00117), 2018 (venue NV). **V (pdf)** | Checkpoint-restart for CUDA UVM applications (hybrid CUDA/MPI) | A proxy process owns the device and UVM regions. Shadow pages in the application are synchronized with a segfault handler and page protection bits. After draining GPU data to the host, the application **forks a child that writes the image under kernel CoW** while the parent resumes | Discrete GPUs with UVM. Not UMA | Proxy and API forwarding | Forked checkpoint 4.1 s versus 45 s naive for a 32 to 33 GB image; 6% average runtime overhead | **Fork-based CoW snapshot of GPU-shared memory that overlaps checkpoint I/O with GPU computation.** Also uses page protection for CPU/GPU synchronization | States "UVM memory is incompatible with shared memory and fork on Linux", so it needs the proxy and a drain copy. On Thor the GPU writes into the forked pages directly and the kernel CoW governs those writes. CRUM does not measure per-page CoW cost and has no extent mechanism |
| **CRAC**. Jain, Cooperman, [arXiv 2008.10596](https://arxiv.org/abs/2008.10596) (SC 2020 per author copy; venue M). **V (pdf, partial)** | Checkpoint-restart for CUDA with streams and UVM | Split-process design; log and replay of allocations. Notes that CRUM used shadow memory | Discrete | Yes | Not extracted | Transparent checkpoint with UVM | No fork CoW of GPU-written memory |
| **GPU Snapshot**. Lee, Sullivan, Hari, Tsai, Keckler, Erez, ICS 2019. [author PDF](https://lph.ece.utexas.edu/merez/uploads/MattanErez/ics19_gpusnapshot.pdf). **V (pdf)** | Logical snapshot with asynchronous transfer and incremental checkpoints for GPU-dense HPC nodes | New hardware: memory zone monitor and zone table, 64 KB zones, duplicate-on-write hardware. Simulator only | Not applicable (proposed hardware) | Hardware change | 4 to 40x node-level checkpoint overhead reduction (simulated) | Logical snapshot at an instant with later duplication of regions the GPU is about to overwrite; **zone (extent) granularity** | Argues that the GPU virtual memory system "cannot be used effectively" for CoW and that "just-in-time duplication like CoW has very high latency" because of exception handling and TLB shootdowns. This is the 2019 qualitative version of the measured 3,100 ms/GiB. The thesis reaches the same goal with no hardware change |
| **gCROP**. Yang, Du, Song, Xia, SoCC 2024. [author PDF](https://ipads.se.sjtu.edu.cn/_media/publications/yang-socc24.pdf). **V (pdf)** | On-demand, parallel restore of GPU apps for serverless start | On-demand restore through GPU page faults (AMD XNACK); deduplication across checkpoint images | Discrete AMD MI50 | Driver and runtime changes | "<100ms startup latency" for large GPU apps such as GPT-2-Large (abstract) | Uses GPU page faults as an OS mechanism | States "GPUs lack the OS support for fork(), and GPU memory usually does not support the copy-on-write feature". Restore only, no concurrent checkpoint |
| **Singularity**. Shukla et al., [arXiv 2202.07848](https://arxiv.org/abs/2202.07848), 2022. **V (pdf)** | Preemptible, migratable DL jobs | Device proxy with `LD_PRELOAD` interception; per-buffer content checksums deduplicate uploads | Discrete | Yes | Not extracted | Transparent GPU checkpoint; dedup of GPU buffers | Interception, stop-the-world |
| **Caldera**. Appuswamy, Karpathiotakis, Porobic, Ailamaki, CIDR 2017. [PDF](http://cidrdb.org/cidr2017/papers/p21-appuswamy-cidr17.pdf). **V (pdf)** | HTAP engine: CPU OLTP writers and GPU OLAP readers on shared memory | Application-level shadow-copy CoW per data page with epochs. GPU kernels read host memory through UVA | Discrete (Maxwell) | Application design | Not comparable | **CoW snapshot shared between CPU writers and GPU kernels without copying to the device** | States fork-based snapshotting "is not applicable with GPGPUs because CUDA memory allocations cannot be shared across process boundaries". CoW is done by the application, the GPU only reads. In the thesis the kernel does CoW and the GPU is the writer |
| **gMig**. Ma, Zheng, Dong, Li, Qi, He, Guan, VEE 2018. [VEE page](https://conf.researchr.org/details/vee-2018/vee-2018-Research-Papers/9/gMig-Efficient-GPU-Live-Migration-Optimized-by-Software-Dirty-Page-for-Full-Virtuali). **V (abs)** | GPU live migration for full virtualization | "Software Dirty Page": hashing detects modified GPU pages because commodity GPUs lack dirty tracking; one-shot pre-copy | GPU model not stated in the abstract (NV: believed to be Intel integrated graphics under mediated pass-through) | Hypervisor mediation | Downtime 302 ms (Windows), 119 ms (Linux); 80% fewer pages sent in downtime | Dirty tracking of GPU-written memory at page granularity | Hash scanning, not page-table write protection. Full text not opened |
| **Modal GPU memory snapshots**. Capelo, Weld, [blog](https://modal.com/blog/gpu-mem-snapshots), 2025-07-30. **V (doc)** | Serverless cold start from a GPU snapshot | CUDA checkpoint API inside gVisor checkpoint/restore: copy GPU memory to host, release GPU | Discrete | Driver API | 20 s to 2 s (Parakeet), 45 s to 5 s (vLLM, Qwen2.5-0.5B) | Snapshot-based GPU start | No CoW, no fork |
| **Anchor**, SOSP 2026 (title on [accepted list](https://sigops.org/s/conferences/sosp/2026/accepted.html), **M**; content **NV**) | Per index snippet: a daemon owns GPU memory so a restarted worker remaps it through CUDA IPC | Not verified | Not verified | Not verified | Snippet only | Decouples GPU memory ownership from the failing process | Must be opened. If accurate, it is daemon ownership through CUDA IPC, not host-MM ownership |
| Cricket (Eiling et al., CCPE 2022, [DOI](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.6474)); NVCR (Nukada et al., IEEE, 2011, [Xplore](https://ieeexplore.ieee.org/document/6008825/)); CheCUDA (Takizawa et al., 2009); Checkpoint/Restart for CUDA Kernels (Eiling, SC-W 2023, [DOI](https://dl.acm.org/doi/10.1145/3624062.3624254)). **M** | API-forwarding or proxy-based CUDA checkpointing | Not verified (NV) | NV | NV: believed interception or proxy | None | Transparent CUDA checkpoint | Publisher pages blocked. Nothing in titles or index text suggests fork CoW of GPU-written memory |

### Group 2. GPUs that share host page tables or fault through the host; device page-fault and invalidation cost

| Work (authors, venue, year, URL, tag) | What it does | Mechanism | UMA? | Intercept/driver? | Comparable cost numbers | Exact overlap with the thesis | What remains different |
|---|---|---|---|---|---|---|---|
| **Grace Hopper system-memory study**. Schieffer, Wahlgren, Ren, Faj, Peng, ICPP 2024. [arXiv 2407.07850](https://arxiv.org/abs/2407.07850). **V (pdf)** | Compares system-allocated (`malloc`), managed, and explicit-copy memory on GH200 | Describes the single system page table used by the GPU through the SMMU/ATS. GPU first touch raises a fault that "the CPU then handles ... and populates the system page table" | Coherent superchip with two memories; system page table shared | None | GPU-side initialization is "significantly longer" for system memory; 64 KB pages cut 33-qubit initialization by 5x and total runtime by 2.9x; `cudaHostRegister` pre-population costs about 300 ms in Rodinia; de-allocation 4.6x to 38x faster with 64 KB pages. No per-page figure | **GPU first touch of host-page-table memory is slow; CPU-side first touch avoids it; larger pages reduce it.** These three observations are already published | No fork, CoW, `mprotect`, MMU notifier, cgroup, or MIG (all zero occurrences in the text). No us/page or ms/GiB number. No write-fault (CoW) path |
| **MI300A unified physical memory study**. Wahlgren, Schieffer, Shi, León, Pearce, Gokhale, Peng, [arXiv 2508.12743](https://arxiv.org/abs/2508.12743), 2025 (venue NV). **V (pdf)** | Characterizes allocation, page faults, TLB, and bandwidth on AMD MI300A APUs | GPU has its own page table kept in sync with the system page table by Linux HMM; XNACK replays faults | **Yes**: unified physical memory APU | None | Single-page fault latency: CPU 9 us, GPU minor 16 us, GPU major 18 us (tail 20 and 22 us). Throughput: GPU major 1.1 M pages/s (about 238 ms/GiB, conv.), GPU minor 9.0 M pages/s (about 29 ms/GiB, conv.), 12 CPU cores 3.7 M pages/s. CPU pre-fault plus GPU minor fault is up to 2.2x faster than GPU major | **Published per-fault GPU cost on a unified-memory GPU that mirrors host page tables, plus the recommendation to pre-fault on the CPU** | HMM mirror, not shared page tables: a GPU "minor" fault still costs 16 us after CPU population, whereas on Thor the post-population GPU touch costs 5 ms/GiB. No fork or CoW, no `mprotect`, no notifier attribution, no cgroup |
| **Vesely, Basu, Oskin, Loh, Bhattacharjee**, "Observations and Opportunities in Architecting Shared Virtual Memory for Heterogeneous Systems", ISPASS 2016. [author PDF](https://www.csa.iisc.ac.in/~arkapravab/papers/ispass16.pdf). **V (pdf)** | Real-system study of shared virtual memory between CPU and integrated GPU (AMD A10-7850K, Linux 4.0) | GPU uses the process page tables through the IOMMU (ATS, PPR) | **Yes**: integrated GPU sharing the CPU page table | None | GPU page fault 5 to 140 us depending on concurrency; CPU fault about 1.7 us; "GPU page-faults are 3-80x slower". GPU TLB shootdown 4.2 us single entry and 4.4 us full flush, with "no significant difference" between them; shootdowns are serialized | **First published characterization of fault and invalidation cost on a host-page-table integrated GPU. The observation that one-entry and whole-TLB invalidation cost the same is the ingredient of "batch into ranges"** | 2016 hardware and OpenCL. No fork, CoW, or `mprotect` experiment. No batching mechanism proposed for CoW. No tenant or cgroup aspect |
| **Zheng, Nellans, Zulfiqar, Stephenson, Keckler**, "Towards High Performance Paged Memory for GPUs", HPCA 2016. [author PDF](https://www.cs.utexas.edu/~skeckler/pubs/HPCA_2016_Paged_Memory.pdf). **V (pdf)** | Hardware proposals for replayable far-faults | GPU far-fault through the host driver | Discrete | Hardware | Far-fault cost 20 us (Maxwell) to 50 us (Kaveri) | Per-fault cost in tens of us is long established | Migration faults over PCIe |
| **MSched**. Shen, Chen, Chen, Chen, [arXiv 2512.24637](https://arxiv.org/abs/2512.24637), 2025-2026. **V (pdf)** | OS-level GPU multitasking by proactive memory scheduling | Predicts each kernel's working set from launch arguments and populates it in one batch before the task runs. Demand paging "retains ... as a fallback"; false negatives "are handled transparently via standard page faults" | Discrete RTX 5080 (UVM) | Intercepts kernel launch; extends the UVM driver | One GPU page fault 31.79 us, of which 1.35 us is data transfer; per-fault migration 0.12 GB/s versus batched 41.7 GB/s, "a 347x improvement" | **Same control structure as extent privatization: predict the set, act in one batch, keep the fault path as the correctness fallback. Same class of per-page versus batch cliff** | Target is migration into device memory, not CoW breaking or snapshot. No host-page-table GPU. Requires interception |
| **NPF**. Lesokhin, Eran, Raindel, Shapiro, Grimberg, Liss, Ben-Yehuda, Amit, Tsafrir, "Page Fault Support for Network Controllers", ASPLOS 2017. [author PDF](https://www.cs.technion.ac.il/~dan/papers/npf-asplos-2017.pdf). **V (pdf)** | On-demand paging for RDMA and Ethernet NICs | Device faults resolved by the driver; invalidation through a Linux MMU notifier | Not a GPU | Driver and firmware | Minor NPF 220 us for a 4 KB message, 350 us for 4 MB. With the ATS/PRI rule of one page per request and no batching, a cold 4 MB message "would have been prohibitive (more than 220 milliseconds)". Invalidations "are cheaper than NPFs" | **The per-page device fault versus batched update cliff (about 630x by these numbers) and an MMU-notifier invalidation cost breakdown were published in 2017** | NIC, not GPU. No CoW or fork experiment, although fork-with-CoW is named as a case the design supports |
| **NVIDIA HMM documentation**: blog "Simplifying GPU Application Development with HMM" (Hubbard et al., 2023-08-22, [link](https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management)) and [CUDA 12.2 release notes](https://docs.nvidia.com/cuda/archive/12.2.0/cuda-toolkit-release-notes/index.html). **V (doc)** | Makes system-allocated memory GPU-accessible on x86 through Linux HMM | Software coherence through HMM mirroring | Discrete (PCIe) | Driver | "Page migrations are handled in chunks of 4 KB-page size". "HMM is not yet fully optimized" | Vendor states the fork limitation: "The `fork()` system call is not fully supported yet when attempting to share GPU-accessible memory between parent and child processes"; blog: "fork(2) without a following exec(3) is not fully supported" | This is the HMM (software) path on x86. The Thor result that fork CoW governs GPU writes on the hardware-coherent path is not documented by NVIDIA in anything opened here |
| **CUDA Programming Guide 2.6** ([link](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)) and **CUDA for Tegra app note** ([link](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html)). **V (doc)** | Define the platform facts | `PageableMemoryAccessUsesHostPageTables`: "1 is hardware, 0 is software". "Full Coherency is supported on Tegra devices starting with Thor SoC". Pageable memory with `pageableMemoryAccess` = 1 "is also directly accessible on the GPU and also cached in iGPU's L2 cache" | Yes (Thor) | None | None | The two platform facts are vendor-documented | Neither document mentions fork, copy-on-write, `mprotect`, page-fault cost, or cgroups |
| **Linux HMM** ([docs](https://docs.kernel.org/mm/hmm.html)) and **DRM GPU SVM** ([docs](https://docs.kernel.org/gpu/rfc/gpusvm.html)). **V (doc)** | Kernel framework for mirroring a process address space into a device | Driver registers an `mmu_interval_notifier`; "During the ops->invalidate() callback the device driver must perform the update action to the range". GPU SVM: notifiers "512M or larger"; ranges are not split, so on partial unmap "the driver is expected to invalidate and destroy the entire range" | Framework | Driver | None | Revocation and CoW visibility to a device through notifiers is the designed kernel contract | No cost data, no fork or CoW discussion in the pages opened |
| **Linux v6.8 `mm/`** (`memory.c`, `mprotect.c`, `mremap.c`, `huge_memory.c`, [mirror](https://github.com/gregkh/linux/tree/v6.8/mm)). **V (src)** | Shows where secondary-MMU notifications are issued | `copy_page_range` (fork): one `MMU_NOTIFY_PROTECTION_PAGE` range for the whole CoW VMA. `wp_page_copy` (CoW fault): one notifier range of `PAGE_SIZE` per fault. `change_pmd_range` (`mprotect`): one `MMU_NOTIFY_PROTECTION_VMA` range. `move_page_tables` (`mremap`): one `MMU_NOTIFY_UNMAP` range. `do_huge_pmd_wp_page`: a shared THP falls back to `__split_huge_pmd` and `VM_FAULT_FALLBACK` | n/a | n/a | n/a | **The per-page versus per-range asymmetry is the structure of mainline Linux, and THP does not batch CoW.** The measured cliff and the failure of THP to help CoW pre-break follow from this code | The kernel gives the structure, not the cost of a notifier into a GPU driver. Newer kernels were not checked |
| **GMEM**. Zhu, Cox, Rixner, [arXiv 2310.12554](https://arxiv.org/abs/2310.12554), 2023 (preprint; PDF header reads "ASPLOS'23 Submission"). **V (pdf)** | "OS support for centralized memory management of both CPU and devices" (FreeBSD 13) | Drivers attach to a process address space and "let the OS take charge of their memory management". Coalesces TLB/IOTLB invalidations for batched unmap. Case studies: IOMMU driver, Intel integrated GPU, simulated faultable integrated GPU that shares the CPU page table | Yes (integrated GPU case, partly simulated) | Kernel refactoring | 54% higher network receive throughput with 32% less CPU for the IOMMU driver | **Closest statement of the umbrella claim: the OS memory manager owns device memory; invalidations are coalesced** | No cgroup accounting, no fork snapshot of device-written memory, no CoW sharing across tenants, no MIG. FreeBSD. GPU part is simulated or a refactor |
| **Cooper, Scogland, Ge**, "Shared Virtual Memory: Its Design and Performance Implications for Diverse Applications", ICS 2024. [arXiv 2405.06811](https://arxiv.org/abs/2405.06811). **V (pdf, partial)** | Studies AMD SVM/HMM on discrete MI-series GPUs | HMM ranges; one fault can service a range of up to 256K pages | Discrete | None | Not extracted | Range-granular fault service in a production GPU driver | Discrete; migration |
| **Fusco et al.**, "Understanding Data Movement in Tightly Coupled Heterogeneous Systems", [arXiv 2408.11556](https://arxiv.org/abs/2408.11556), 2024. **V (pdf, partial)** | Data-movement study on GH200 | Describes ATS and allocation classes | Coherent superchip | None | Bandwidth only | Background on ATS memory classes | No fault, fork, or CoW measurement |
| **gpu_ext**. Zheng et al., [arXiv 2512.12615](https://arxiv.org/abs/2512.12615), 2025. **V (pdf, partial)** | eBPF policy hooks in the NVIDIA UVM driver | Programmable prefetch and eviction for UVM regions | Discrete | Driver extension | Not extracted | OS-level policy over GPU memory | UVM placement policy; no cgroup, fork, or CoW (zero occurrences) |
| **Psistakis et al.**, "Optimized Page Fault Handling During RDMA", IEEE TPDS 2022. [author PDF](https://psistakis.cs.illinois.edu/files/publications/psistakis-tpds22.pdf). **V (pdf, partial)** | Device page-fault handling for RDMA | Touch-ahead and retransmission schemes | Not a GPU | Driver, hardware | Not extracted | Device fault cost motivates touch-ahead | NIC |
| Apple Silicon unified memory, Arm Mali, Qualcomm Adreno | n/a | n/a | n/a | n/a | n/a | n/a | **NV**. No primary source was opened that states whether these GPUs dereference pageable process memory through host page tables or how they behave under fork. See Section 6 |

### Group 3. fork and CoW latency on CPUs; batched, extent-granular, or user-space CoW

| Work (authors, venue, year, URL, tag) | What it does | Mechanism | UMA? | Intercept/driver? | Comparable cost numbers | Exact overlap with the thesis | What remains different |
|---|---|---|---|---|---|---|---|
| **On-demand-fork**. Zhao, Gong, Fonseca, EuroSys 2021. [author PDF](https://sishuaigong.github.io/pdf/eurosys21-odf.pdf). **V (pdf)** | Microsecond fork for large processes | Shares last-level page tables at fork and copies them on demand | CPU only | Kernel patch | Fork of 1 GB: 6.5 ms average with 4 kB pages, 0.17 ms with 2 MB pages; ODF is 65x faster than fork. Notes that huge pages make a CoW fault up to 512x longer | Fork cost is page-table copying; huge pages cut it by about 50x. This bounds the thesis's 25 ms/GiB and 1.6 ms/GiB numbers as ordinary | No device. CoW data copy is still per page (or per huge page) |
| **Async-fork**. Pang, Deng, et al., PVLDB 16(5), 2023. [arXiv 2301.05861](https://arxiv.org/abs/2301.05861). **V (pdf, abstract and introduction)** | Removes query latency spikes from fork-based snapshots (Redis) | Offloads page-table copying to the child; "proactive synchronization" when the parent modifies a PTE during the copy | CPU only | Kernel patch | Tail latency of snapshot queries reduced 81.76% (8 GB) and 99.84% (64 GB) | Fork-based snapshot latency as a systems problem; proactive copying of page-table state before the fault | Proactive copy concerns page tables, not data extents. The PTE-table granularity detail was not located in the text read (NV) |
| **A fork() in the road**. Baumann, Appavoo, Krieger, Roscoe, HotOS 2019. [PDF](https://www.microsoft.com/en-us/research/uploads/prod/2019/04/fork-hotos19.pdf). **V (pdf)** | Argues against fork | n/a | n/a | n/a | n/a | States that a process using "OpenCL with a GPU, cannot safely fork since the OS cannot duplicate the process state on the NIC/GPU" | A reviewer will quote this. The thesis must say exactly what is forked (anonymous pageable memory) and that the child does not use CUDA |
| **RUMA**. Schuhknecht, Dittrich, Sharma, PVLDB 9(10), 2016. [PDF](http://www.vldb.org/pvldb/vol9/p768-schuhknecht.pdf). **V (pdf)** | Rewiring of virtual-to-physical mappings from user space | Memory is backed by a main-memory file; "rewired COW" copies a page into a pooled page and remaps it with `mmap`, instead of kernel "implicit COW" | CPU only | None | Per 2 MB page: implicit CoW 594 us, rewired CoW 442 us (25.5% less); update throughput +12% to +96% | **User-space CoW by copy-then-remap that outperforms kernel CoW, for snapshots.** This is the technique of extent privatization | Motivation is page-allocation cost, not secondary-MMU invalidation. Page-granular. No device |
| **AnKer**. Sharma, Schuhknecht, Dittrich, SIGMOD 2018. [author PDF](https://bigdata.uni-saarland.de/publications/AnKer_SIGMOD2018.pdf). **V (pdf)** | Fine-granular, high-frequency snapshots for MVCC | Compares fork, rewiring with `mprotect` and "manual copy-on-write", and a new `vm_snapshot` system call | CPU only | Kernel patch (4.8) | Snapshot creation table (not converted) | **Partial, range-scoped snapshots created without a full fork; `mprotect` plus manual CoW** | CPU database columns; no device, no extent privatization ahead of a predicted write set |
| **bpf_fault**. Zussman, Dey, Zengin, Fang, Hildenbrand, Cidon, SOSP 2026. [draft PDF](https://github.com/bpf-fault/bpf-fault/blob/main/bpf_fault_draft.pdf). **V (pdf, draft)** | Custom page-fault handlers in eBPF (Linux 6.17) | Handles missing, write-protect, and minor faults in-kernel; integrated with Firecracker and QEMU live snapshots | CPU only | Kernel patch | `userfaultfd` missing read fault 10.2 us versus 0.75 us baseline | Custom write-protect fault policy for live snapshots without fork is current top-venue work | No device or secondary MMU. Disables THP at fault time. Per-fault, not extent-ahead |
| **userfaultfd write-protect**. Corbet, LWN, 2019-05-02. [link](https://lwn.net/Articles/787308/). **V (doc)** | Kernel feature for user-space write tracking | Write-protect faults delivered to a monitor; "could eliminate the need to fork to get a stable set of pages" | CPU | Mainline | None | Fork-free live snapshot by write protection is mainline | Per-page events; article notes THP must be disabled |
| **"Patching until the COWs come home"**. Babka, LWN, 2021-03-22. [link](https://lwn.net/Articles/849638/). **V (doc)** | History of CoW versus pinned pages | Kernel breaks CoW when a pinned (GUP) reference is taken on a CoW-shared page | CPU, pinning devices | Mainline | None | Pinned memory (`cudaHostRegister`, `cudaMallocHost`) and fork CoW interact through special kernel rules | The thesis uses unpinned pageable memory. Any claim must exclude pinned buffers or test them |
| **CCoW**. "CCoW: Optimizing Copy-on-Write Considering the Spatial Locality in Workloads", Electronics 11(3):461, 2022. [MDPI](https://www.mdpi.com/2079-9292/11/3/461). **M** (HTTP 403); content **NV** | Per index text: copies several neighbouring pages in one CoW fault ("region") | NV | CPU | Kernel patch (NV) | NV | **If the index text is accurate, region-granular proactive CoW already exists for CPUs** | Must be opened before any "extent-granular CoW" sentence is written |
| **TrEnv**. Huang et al., SOSP 2024. [author PDF](https://madsys.cs.tsinghua.edu.cn/publication/trenv-transparently-share-serverless-execution-environments-across-different-functions-and-nodes/SOSP24-huang.pdf). **V (pdf, partial)** | Shares execution environments across serverless functions | CoW "mm-templates" on CXL or RDMA memory | CPU only (no occurrence of "GPU") | Kernel patch | Not extracted | CoW templates for warm start | No GPU |
| HyPer fork snapshots (Kemper, Neumann, ICDE 2011). **M** (cited by RUMA and AnKer) | Fork-based OLAP snapshots | Kernel CoW | CPU | None | NV | Origin of fork-as-snapshot in data systems | Not opened |

### Group 4. Accounting and isolation of GPU memory, with emphasis on unified memory

| Work (authors, venue, year, URL, tag) | What it does | Mechanism | UMA? | Intercept/driver? | Comparable cost numbers | Exact overlap with the thesis | What remains different |
|---|---|---|---|---|---|---|---|
| **Linux `dmem` cgroup controller**. [cgroup-v2 docs](https://www.kernel.org/doc/Documentation/admin-guide/cgroup-v2.rst) (**V (doc)**, text read); Lankhorst, LKML 2024-12-17, [archive](https://lists.openwall.net/linux-kernel/2024/12/17/1367) (**V (doc)**) | Per-cgroup `dmem.max/min/low/current/peak` for device memory regions | Driver registers regions (example: `drm/0000:03:00.0/vram0`) and charges allocations | Designed for VRAM regions. Author: "Main memory will be a followup, but requires some discussions on hwo to be prevent double accounting" | Driver must charge | None | **cgroup accounting and limiting of GPU memory is mainline.** The unified-memory case is acknowledged as open by its author | No answer for a GPU whose memory is system DRAM. Separate from memcg |
| **dmem follow-ups**: pinned device memory (Lankhorst, 2025-08-19, [LWN](https://lwn.net/Articles/1034421/)); reclaim (Hellström, 2026-05-11, [LWN](https://lwn.net/Articles/1072437/)). **V (doc)** | Extend dmem semantics | Reclaim callback when `max` is lowered; drivers xe and amdgpu | VRAM regions | Driver | None | Revocation (reclaim) under a cgroup limit is being upstreamed for device memory | Not memcg, not UMA |
| **GPU cgroup controller proposal**. Valsaraju (Google), 2022-01-14, [LWN](https://lwn.net/Articles/881554/); **memcg tracking of exported dma-bufs**. Mercier, 2023-01-09, [LWN](https://lwn.net/Articles/919548/). **V (doc)** | Attribute dma-buf and GPU allocations to cgroups on Android | Charge at export; transfer charge across processes | Yes: Android SoCs are unified memory | Kernel patches | None | The unified-memory accounting problem has been posted to the kernel lists since 2022 | Prototype "does not include resource limit enforcements". dma-buf path, not CUDA |
| **dma-buf system heap memcg accounting** (Chanudet, LKML, 2026-01) and **memcg dma-buf per-cgroup accounting via pidfd** (Esteve, RFC, 2026-05). **NV** (lkml.org refused; index text only) | Per index text: `__GFP_ACCOUNT` for system-heap allocations so they no longer "escape" memcg limits; charge to a client cgroup by pidfd | NV | Yes (embedded) | Kernel patches | NV | If accurate, the same hole (device-path allocations of system DRAM escaping `memory.max`) is being closed upstream in 2026 for dma-buf heaps | Must be opened on lore.kernel.org |
| **ROCm issue 6370**, "Charge amdgpu GTT allocations to the memory cgroup (memcg) for OOM/Kubernetes accounting on RDNA3.5 APUs", opened 2026-06-19, open. [link](https://github.com/ROCm/legacy-rocm-build/issues/6370). **V (doc)** | Bug report on AMD APUs | GTT is "system RAM mapped into GPU virtual address spaces"; "the Linux memory cgroup (memcg) subsystem does not account for them" | **Yes** | n/a | Reporter: about 50.6 GB of GTT against 37.0 GB of cgroup working sets | **Same observation as the Thor measurement, on another vendor's unified-memory GPU, three and a half months earlier** | A user report without a fix or a design. Not MIG |
| **NVIDIA open GPU kernel modules**: `video_mem.c` at commit 61dcc93 ([link](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/61dcc93722ecb418bb5f2e00923f05b4b8051dd1/src/nvidia/src/kernel/mem_mgr/video_mem.c)); `uvm_linux.h` ([link](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/kernel-open/nvidia-uvm/uvm_linux.h)). **V (doc)** | Vendor driver accounting hooks | `if (!IS_MIG_ENABLED(pGpu)) { ... memacctTryCharge(...) }`. UVM defines `NV_UVM_GFP_FLAGS_ACCOUNT` as `NV_UVM_GFP_FLAGS` OR-ed with `__GFP_ACCOUNT`, and `uvm_memcg_context_start/end` | Code is generic; behaviour on Tegra not verified | Driver | None | NVIDIA already charges device allocations (skipped under MIG) and can charge UVM system memory to memcg | Whether these paths are active on Thor with driver 595.78 is not verified. The Thor measurement says `cudaMalloc` is not charged. Managed memory on Thor is unmeasured |
| **NVIDIA MIG documentation for Thor**. [supported profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html), [Jetson guide r39.2](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/MiG.html). **V (doc)** | Defines Thor profiles | "The Thor iGPU uses unified system memory shared with the CPU. There is no dedicated video memory, so all profiles report 0 GB." | Yes | n/a | None | The memoryless property is vendor-documented | No statement on memory isolation, limits, cgroups, or out-of-memory behaviour between instances |
| **KRYPTON**. Zhang et al., USENIX ATC 2025. [PDF](https://www.usenix.org/system/files/atc25-zhang-shulai.pdf). **V (pdf)** | Kernel-space interception for GPU sharing with compatibility and isolation | Intercepts `ioctl` and command-buffer writes; write-protects command buffers with `do_mprotect_pkey`; enforces a per-process GPU memory quota | No: A100 and RTX 4090 | Kernel module | Not comparable | Kernel-enforced GPU memory quota; `mprotect` used as a GPU-control primitive | Quota on dedicated VRAM, enforced by interception. No aliasing with memcg |
| **TGS**. Wu et al., NSDI 2023. [PDF](https://www.usenix.org/system/files/nsdi23-wu.pdf). **V (pdf)** | Transparent GPU sharing for containers | Redirects allocations to CUDA unified memory at the OS layer for oversubscription | No: A100 | Interception | Not comparable | OS-layer GPU memory management for containers | Discrete VRAM versus host RAM |
| **HAMi-core**. [README](https://github.com/Project-HAMi/HAMi-core). **V (doc)** | Per-container GPU memory and compute limits | "hijacking the API calls between CUDA-Runtime (libcudart.so) and CUDA-Driver (libcuda.so)"; `LD_PRELOAD` of `libvgpu.so`; `CUDA_DEVICE_MEMORY_LIMIT` | Not stated | API interception | None | GPU memory quota by interception | Cooperative; unrelated to memcg |
| GaiaGPU (Gu et al., ISPA 2018). **M** (content NV) | Container GPU sharing | Per index text: intercepts CUDA memory and compute APIs | NV | Interception | NV | Quota by interception | Not opened |
| **Nixie**. Xu et al., OSDI 2026. [PDF](https://www.usenix.org/system/files/osdi26-xu-yechen.pdf). **V (pdf, partial)** | Temporal multiplexing of consumer GPUs with memory coordination | Coordinates GPU memory allocation; contrasts with UVM page-fault-driven swapping | No: RTX 5090 | States no application or driver change | Not extracted | System-wide GPU memory arbitration | VRAM to host swapping |
| **Prism**. Yu et al., OSDI 2026. [PDF](https://www.usenix.org/system/files/osdi26-yu-shan.pdf). **V (pdf, partial)** | Multi-LLM serving by "GPU memory ballooning" | A balloon driver (`kvcached`) decouples virtual and physical GPU memory inside serving engines | Discrete | Engine integration | Not extracted | Reclaiming GPU memory across tenants | Application-level balloon, not OS memory manager |
| **Sereno**. Xin, Shi, Dong, Mi, OSDI 2026, "Inference in the Shadows: Taming Memory Bandwidth Contention in Mobile LLM Inference with Sereno". [PDF](https://www.usenix.org/system/files/osdi26-xin.pdf). **V (pdf)** | Protects foreground QoS from background on-device LLM inference on mobile UMA SoCs | Yield points from speculative decoding; reacts to memory **bandwidth** contention | Yes (mobile) | Framework | Not comparable | A 2026 OSDI paper on unified-memory contention exists | It concerns bandwidth, not capacity, accounting, cgroups, fork, or CoW (none of these terms appear) |
| **gpuoom**. [repository](https://github.com/tom-doerr/gpuoom). **V (doc)** | Early OOM killer for unified-memory NVIDIA machines (GB10, Jetson, Grace) | Samples per-process GPU memory | Yes | None | None | README: "On a unified-memory machine a CUDA allocation is ordinary system memory that the driver pins. It appears in no `Rss*` counter of the process that owns it" | A community tool. Shows the accounting hole is known to practitioners |
| **Performance isolation on edge GPUs**. Martín, Flich, Hernández, [arXiv 2601.07600](https://arxiv.org/abs/2601.07600), 2026. **V (abs)** | MPS, MIG, Green Contexts on A100 and Jetson Orin | Temporal isolation measurements | Orin (no MIG there) | None | Not comparable | Edge GPU isolation study | Timing only; no memory capacity isolation |
| MxGPU (Dreimann et al., JSA 2026, [DOI](https://doi.org/10.1016/j.sysarc.2025.103669)). **M** (from the project's earlier audit; not reopened) | Integrated Intel GPU multiplexing on Genode | Explicit shared regions per cell | Yes | Custom OS | NV | Memory management for an integrated GPU exists | Not Linux, not cgroup |

### Group 5. CoW or deduplicated sharing of model or GPU state across tenants or processes

| Work (authors, venue, year, URL, tag) | What it does | Mechanism | UMA? | Intercept/driver? | Comparable cost numbers | Exact overlap with the thesis | What remains different |
|---|---|---|---|---|---|---|---|
| **C2CServe**. Luo, Sadiq, Yang, et al., [arXiv 2605.19481](https://arxiv.org/abs/2605.19481), 2026. **V (pdf)** | Serverless LLM serving on MIG with weights left in CPU memory | Weights sit in pinned CPU memory (`cudaHostAlloc` with `cudaHostAllocMapped`) and are streamed over NVLink-C2C by GPU kernels in MIG instances | Coherent superchip (GH200), not memoryless MIG | Custom kernels | Not comparable | **MIG tenants compute directly on host-resident model state** | Read-mostly, pinned, no CoW, no private divergence, no cgroup statement. It does not say that one physical copy is mapped by several tenants at once |
| **Tetris**. Li, Zhao, Yang, Zhan, Li, USENIX ATC 2022. [PDF](https://www.usenix.org/system/files/atc22-li-jie.pdf). **V (pdf)** | Memory-efficient serverless inference | User-space page deduplication into a shared tensor store; instances `mmap` tensors with reference counts | CPU inference mainly; notes frameworks keep a CPU copy for GPU inference | Framework change | Not comparable | **Deduplicated sharing of model tensors across isolated instances through `mmap`** | Read-only sharing of CPU memory; no CoW divergence managed for GPU writers; no MIG |
| **vLLM / PagedAttention**. Kwon et al., SOSP 2023. [arXiv 2309.06180](https://arxiv.org/abs/2309.06180). **V (pdf)** | Paged KV cache | Reference-counted physical blocks with block-granular "copy-on-write", explicitly likened to OS virtual memory | n/a (allocator inside one process) | Runtime design | Memory savings (not converted) | **CoW of GPU inference state at block (extent) granularity** | Inside one address space and one runtime, done by bookkeeping, not by the MMU |
| **ForkKV**. Wang, Ren, Gui, et al., [arXiv 2604.06370](https://arxiv.org/abs/2604.06370), 2026. **V (pdf, abstract and introduction)** | Multi-LoRA agent serving | "fork with copy-on-write (CoW)": a shared base KV cache "analogous to the parent process's memory pages" and per-agent residual caches | n/a | Runtime design | Not extracted | **Explicitly frames sharing of GPU state among agents as OS fork with CoW** | An analogy implemented in a serving runtime. No OS fork, no MMU, no hardware tenant boundary |
| **SAGE**. Zhao, Cui, Chen, Zhang, et al., [arXiv 2404.14691](https://arxiv.org/abs/2404.14691), 2024. **V (pdf)** | Fast setup for GPU serverless functions | A unified memory daemon shares read-only GPU memory and context across invocations of the same function | Discrete | API remoting | Function duration reduced 11.3x (abstract) or 13.3x (introduction); the two passages differ | Read-only GPU data shared across invocations | One daemon-owned context; no isolation boundary; no CoW |
| **Tangram**. Zhu, Shen, Shao, [arXiv 2512.01357](https://arxiv.org/abs/2512.01357), 2025. **V (pdf, abstract)** | Serverless LLM loading by GPU memory reuse | Unified GPU memory pool with tensor-level parameter sharing across models | Discrete | Runtime | Not extracted | Tensor sharing across models on the GPU | Application pool |
| **gCROP** and **Singularity** (see Group 1). **V (pdf)** | Deduplicate GPU checkpoint content | Image comparison; content checksums | Discrete | Yes | n/a | Dedup of GPU state | Storage or upload dedup, not live sharing |
| **Flex-MIG**. Kim, Yeom, Kim, [arXiv 2511.09143](https://arxiv.org/abs/2511.09143), 2025. **V (pdf, partial)** | One job across several MIG instances | Host shared-memory collectives | Discrete MIG | Runtime | Not extracted | Cross-MIG host shared memory | One trust domain, no CoW |
| **Prism** (Group 4), **StreamBox** (Wu et al., ATC 2024, [PDF](https://www.usenix.org/system/files/atc24-wu-hao.pdf), V (pdf, partial)), **Torpor** (Yu et al., [arXiv 2306.03622](https://arxiv.org/abs/2306.03622), V (pdf, partial)), **ServerlessLLM** ([arXiv 2401.14351](https://arxiv.org/abs/2401.14351), V (pdf, partial)), **BlitzScale** (Zhang et al., OSDI 2025, [PDF](https://www.usenix.org/system/files/osdi25-zhang-dingyan.pdf), V (pdf, partial)) | Model swapping, shared GPU runtime, fast loading, live autoscaling | Models held in host memory and swapped or loaded quickly; shared runtime | Discrete | Runtime, API redirection | Not extracted | Host memory as the home of model state | No OS-level CoW; no cross-tenant CoW |
| Medusa (ASPLOS 2025, [repository](https://github.com/MachineLearningSystem/25ASPLOS-Medusa)). **M** | Per title: state materialization for serverless LLM start | NV | NV | NV | NV | Snapshot-style warm start | Not opened |
| KSM-style deduplication of GPU memory | n/a | n/a | n/a | n/a | n/a | n/a | **Not found.** No primary source located for kernel same-page merging over GPU-accessed memory. NV, not proof of absence |

### Group 6. Application-side anchors

| Work (authors, venue, year, URL, tag) | What it does | Mechanism | UMA? | Intercept/driver? | Comparable cost numbers | Exact overlap with the thesis | What remains different |
|---|---|---|---|---|---|---|---|
| **CheckFreq**. Mohan, Phanishayee, Chidambaram, FAST 2021. [PDF](https://www.usenix.org/system/files/fast21-mohan.pdf). **V (pdf)** | Frequent DNN checkpointing | Two phases: `snapshot()` makes "a consistent in-memory copy of all the learnable model state"; `persist()` writes it asynchronously | Discrete | Framework | Not extracted | The snapshot-then-persist split, with the stall confined to the snapshot | The snapshot is a full copy. The thesis replaces the copy by fork CoW |
| **DataStates-LLM**. Maurya, Underwood, Rafique, Cappello, Nicolae, HPDC 2024. [arXiv 2406.10707](https://arxiv.org/abs/2406.10707). **V (pdf, abstract and introduction)** | Lazy asynchronous LLM checkpointing | Exploits that parameters "remain immutable" during most of an iteration to make lazy device-to-host copies | Discrete | Framework | Not extracted | Uses application knowledge of when state is immutable to avoid CoW | Application-level; no OS mechanism |
| Just-In-Time Checkpointing (Gupta et al., EuroSys 2024, [DOI](https://dl.acm.org/doi/abs/10.1145/3627703.3650085)). **M** | Per abstract index: checkpoint at failure time | NV | NV | NV | NV | Crash recovery for GPU jobs | Not opened |
| **Caldera** (Group 1). **V (pdf)** | Snapshot isolation between CPU writers and GPU readers | Application CoW | Discrete | Application | n/a | Snapshot isolation for GPU-visible state | See Group 1 |
| **Smart Black Box**. Yao, Atkins, [arXiv 1903.01450](https://arxiv.org/abs/1903.01450), 2019. **V (abs)** | Value-driven high-bandwidth automotive event data recorder | Buffers and compresses sensor data by value | n/a | n/a | n/a | Black-box recording for autonomous vehicles is an established application | Records sensor streams, not accelerator memory state |
| Forensic or black-box recording of on-device GPU inference state for robots | n/a | n/a | n/a | n/a | n/a | n/a | **Not found** in primary sources by the queries in Section 6. Index results were patents and sensor-data recorders. NV |

### Group 7. Jetson Thor, memoryless MIG, Sysmem Full Coherency (2025 to 2026)

| Work (authors, venue, year, URL, tag) | What it does | Mechanism | UMA? | Intercept/driver? | Comparable cost numbers | Exact overlap with the thesis | What remains different |
|---|---|---|---|---|---|---|---|
| **CUDA for Tegra app note**, **CUDA Programming Guide**, **NVIDIA MIG docs** (Groups 2 and 4). **V (doc)** | Vendor definitions | Full Coherency from Thor; `0 GB` profiles; at most one compute and one graphics instance | Yes | n/a | None | Both platform facts | No fork, CoW, cgroup, or cost statement |
| **Echo** (v1 title "DejaVu"). Zhu, Zhao, Li, Yoon, [arXiv 2609.05635](https://arxiv.org/abs/2609.05635), v2 2026-09-30. **V (pdf)** | Merges host and device buffers to remove copies on unified-memory SoCs | Proves and profiles when buffers can alias; evaluated on Orin Nano, AGX Orin, and **AGX Thor** | Yes | Interposer | Up to 7.05x on copy-dominated workloads and up to 1.40x on closed-source end-to-end applications (abstract) | A systems preprint that evaluates on Thor and exploits unified memory already exists | Copy elimination inside one application. No fork, CoW, cgroup, `mprotect`, or MIG (zero occurrences) |
| **Hydra**. "Phase-Aware Workload Characterization of LLM Inference across Edge SoC Generations", [arXiv 2608.25053](https://arxiv.org/abs/2608.25053), 2026. **V (pdf, partial)** | LLM inference characterization on Xavier, Orin, and Thor | Tracing | Yes | None | Not extracted | Thor is already a characterization target | No memory-management mechanism study |
| NVIDIA forum thread, "MIG on AGX Thor - MIG 1g instances enumerate as CUDA devices but hang on the first kernel", 2026-10-01. [link](https://forums.developer.nvidia.com/t/mig-on-agx-thor-mig-1g-instances-enumerate-as-cuda-devices-but-hang-on-the-first-kernel/384828). **V (doc)** | User report on JetPack 7.2 | Staff directed the user to profiles 78 and 83 | Yes | n/a | None | Confirms the two supported coexisting profiles | No memory statement |
| **Linux AGX**. Shirvani, Liu, SOSP 2026 (title on [accepted list](https://sigops.org/s/conferences/sosp/2026/accepted.html), **M**; content **NV**) | Per index snippet: GPU-aware weights for Linux fair scheduling on embedded robots | NV | NV | NV | NV | A SOSP 2026 paper on Linux as the manager of the GPU for physical AI | CPU scheduling, not memory. Must be opened to learn the platform |
| Any 2025-2026 paper on memoryless MIG, "Sysmem Full Coherency", or `pageableMemoryAccessUsesHostPageTables` | n/a | n/a | n/a | n/a | n/a | n/a | **Not found.** Queries returned only NVIDIA documentation and third-party explainers |

## 3. Claims that are already taken

Each line is a sub-claim of the thesis that a verified source already makes.

1. **A GPU can dereference ordinary pageable host memory through the host page tables.** NVIDIA CUDA Programming Guide and CUDA for Tegra app note; Grace Hopper study (ICPP 2024); ISPASS 2016 for an integrated GPU.
2. **The first GPU touch of such memory is slow because the CPU services the fault, and CPU-side population removes the cost.** Grace Hopper study (ICPP 2024); MI300A study (recommends CPU pre-faulting).
3. **Larger pages reduce the GPU first-touch penalty.** Grace Hopper study: 5x with 64 KB pages.
4. **A GPU or device page fault costs tens of microseconds.** ISPASS 2016 (5 to 140 us), HPCA 2016 (20 to 50 us), MI300A (16 to 18 us), MSched (31.79 us), NPF (220 us for a NIC).
5. **Per-page device faults are orders of magnitude slower than one batched operation.** NPF, ASPLOS 2017 (more than 220 ms without batching versus about 350 us for 4 MB); MSched (347x).
6. **Invalidating one translation in a GPU costs about the same as invalidating a whole range.** ISPASS 2016 (4.2 us versus 4.4 us).
7. **Fork notifies secondary MMUs once per VMA, a CoW fault once per page, `mprotect` and `mremap` once per range; a write to a shared THP is split and handled per 4 KiB page.** Linux v6.8 source.
8. **Predict the GPU's access or write set from kernel-launch arguments, act before the access, and keep a fault or retry path for mispredictions.** MSched (proactive population with demand paging as fallback); PhoenixOS (validated speculation).
9. **Concurrent GPU checkpoint with copy-on-write semantics.** PhoenixOS (soft CoW by interception); GPU Snapshot (duplicate-on-write hardware, zone granularity).
10. **A fork-based CoW snapshot can overlap checkpoint writing with GPU computation.** CRUM (2018).
11. **CoW through the GPU's virtual memory is slow because of exception handling and TLB shootdowns.** GPU Snapshot (ICS 2019), stated qualitatively.
12. **Fork latency is page-table copying, huge pages reduce it by about 50x, and fork-based snapshots cause latency spikes that can be engineered away.** On-demand-fork; Async-fork.
13. **User-space CoW by copy-then-remap can replace and outperform kernel CoW; partial snapshots can be made with `mprotect` and manual CoW.** RUMA (2016); AnKer (2018). Fork-free write-protect snapshots: `userfaultfd` write-protect; bpf_fault (SOSP 2026).
14. **cgroup accounting and limiting of GPU memory.** Linux `dmem`; GPU cgroup proposal; memcg tracking of dma-bufs. **Quota by interception.** HAMi-core, KRYPTON, TGS.
15. **On unified-memory GPUs, device-path allocations of system DRAM escape memcg.** ROCm issue 6370 (AMD APU); the dmem author's note on main memory and double accounting; gpuoom README (NVIDIA unified memory).
16. **Memory protection as a GPU control primitive.** KRYPTON (`do_mprotect_pkey` on command buffers); CRUM (page protection and a fault handler for shadow pages); Linux HMM (the `invalidate` callback contract). The project's own GPU Portals prototype already demonstrates `mprotect` revoke-before-grant on coherent pages, so revocation is not a new claim of this direction.
17. **MIG tenants can compute on host-resident model state.** C2CServe. **Model tensors can be deduplicated across isolated instances by `mmap`.** Tetris. **GPU inference state can be shared copy-on-write at block granularity.** vLLM; ForkKV (which uses the words "fork with copy-on-write").
18. **The OS memory manager should own device memory centrally, and invalidations should be coalesced.** GMEM (preprint); Linux HMM as the mainline framework.
19. **Fork and GPUs do not mix in general.** HotOS 2019; NVIDIA HMM release notes; Caldera; gCROP; CRUM. These are claims against the thesis that must be answered, not claims the thesis makes.

## 4. Claims that appear open in the verified set

**O1. Kernel fork CoW, with no API interception and no driver change, yields a consistent snapshot of memory that a GPU kernel is concurrently writing.**
Closest prior work: CRUM (fork-based CoW checkpoint for UVM, but only after a drain copy and with a proxy, because "UVM memory is incompatible with shared memory and fork"); NVIDIA's HMM notes (fork "not fully supported"); Caldera (fork snapshot "not applicable with GPGPUs").
Honest statement: no verified source demonstrates this on any GPU. It is, however, the behaviour the Linux MMU-notifier contract prescribes for any device that follows host page tables, so the contribution is an observation and a measurement on a new platform, not a mechanism. It applies only to private anonymous pageable memory; `cudaMalloc`, pinned, and managed memory are outside it.

**O2. Extent privatization: copy the extents the GPU is about to write on CPU threads and install each with one `mremap(MREMAP_FIXED)`, with kernel CoW as the correctness fallback.**
Closest prior work: RUMA and AnKer (user-space CoW by copy-then-remap); MSched (predict, batch, fault fallback); GPU Snapshot (zone-granular duplicate-on-write in hardware); CCoW (region-granular CoW in the kernel, content not verified).
Honest statement: this is a known OS technique applied on a new platform. The part not found in the verified set is the reason for it: the cost being avoided is the per-page secondary-MMU invalidation into a GPU driver, not page allocation (RUMA) or data migration (MSched). The safety property that a wrong prediction costs time and cannot corrupt the snapshot is stronger than PhoenixOS's abort-and-retry, and that comparison is fair to make.

**O3. Measured cost of fork, CoW, first touch, and `mprotect` on a GPU that uses host page tables, including the finding that CPU-side CoW pre-break is also slow.**
Closest prior work: Grace Hopper study (first touch only, no per-page number); MI300A (per-fault latency through HMM); ISPASS 2016 (fault and shootdown on an integrated GPU).
Honest statement: the write-fault and fork paths are unmeasured in every source opened, and no source attributes a cost to MMU-notifier invalidation of a GPU. The magnitude of a GPU fault (12 to 19 us) is within the published range, and the existence of a per-page versus batch cliff is published for other devices. The attribution to notifier invalidation is at present an inference ("appears to be") and needs a direct trace before it is stated as a result.

**O4. Using memcg as the capacity-isolation mechanism for memoryless MIG tenants by keeping tenant GPU data in pageable memory, together with the measurement that `cudaMalloc` bypasses `memory.max` in both instances.**
Closest prior work: Linux `dmem` and its open main-memory question; ROCm issue 6370; the kernel dma-buf memcg patches.
Honest statement: that anonymous memory is charged to memcg is ordinary Linux behaviour, and the bypass by device-path allocations is known on other unified-memory platforms. What is not in the verified set is the MIG-specific fact (each instance reports all of DRAM and the driver skips its own accounting when MIG is on) and a design in which the host limit is the only limit. As long as `cudaMalloc` remains outside, the claim is about what can be accounted, not about an enforced bound.

**O5. OS-level CoW sharing of mutable or read-mostly state across MIG tenants (MMU-enforced private divergence on shared physical pages).**
Closest prior work: C2CServe (MIG tenants on host-resident weights, read-mostly, pinned); Tetris (`mmap` dedup); ForkKV and vLLM (CoW in a runtime).
Honest statement: not found as an OS mechanism across hardware GPU partitions. It is `MAP_PRIVATE` semantics on a new platform, and it is planned, not measured. Two risks from prior art apply: CUDA is documented as not fully supporting fork without exec, so tenants must be separate exec'd processes over a shared mapping, and the measured per-page CoW cost makes the facet depend on O2.

**O6. The combination: one owner (host MM) for accounting, revocation, snapshot, and CoW sharing of GPU-visible memory on a memoryless MIG GPU, enabled by extent batching.**
Closest prior work: GMEM (centralized OS management of device memory with coalesced invalidations).
Honest statement: the combination is not in the verified set, and the platform (no device memory to manage separately) is what makes it possible. Each ingredient has a named precedent in Section 3, so the claim must be the combination and the platform, never an ingredient.

## 5. Permitted wording and forbidden wording

"First" may be used only with "to our knowledge", only for the combination or for a measurement on this class of platform, and only scoped to the works audited in Section 2.

### Permitted

> To our knowledge, among the GPU checkpointing systems and unified-memory studies we audited, this is the first measurement of fork, copy-on-write, and range-protection costs for memory that a GPU accesses through the host page tables.

> On Thor, the kernel's copy-on-write governs GPU writes to private pageable memory. A forked child therefore holds a consistent image without API interception or a driver change. Prior systems obtain the same isolation by intercepting GPU APIs (PhoenixOS, GCR) or by a proxy and a drain copy (CRUM).

> A per-page event that reaches the GPU driver costs about 12 us, whereas a range operation costs 13 ns per page. Similar gaps are reported for NIC page faults (NPF) and for GPU demand paging (MSched); we observe the gap for copy-on-write and first touch on a GPU that shares the host page tables.

> Extent privatization applies user-space copy-and-remap, which RUMA and AnKer introduced for CPU snapshots, to avoid per-page invalidation of the GPU mapping. Kernel copy-on-write remains the fallback. Thus, a mispredicted write set costs time but cannot corrupt the snapshot.

> Device allocations (`cudaMalloc`) are not charged to the memory cgroup in either MIG instance, as has been reported for other unified-memory GPUs. Pageable memory is charged as usual, so the host limit bounds exactly the memory that the host memory manager owns.

> To our knowledge, among the audited works, no system shares GPU-visible state copy-on-write across MIG instances through the host MMU. (Use only after the mechanism is implemented and measured.)

### Required acknowledgements next to any of the above

- GPU first-touch cost and CPU pre-population on host-page-table GPUs: Grace Hopper study (ICPP 2024) and MI300A study.
- Integrated-GPU fault and shootdown costs: Vesely et al. (ISPASS 2016).
- Fork-based CoW checkpointing with GPUs: CRUM. Concurrent GPU checkpoint with soft CoW: PhoenixOS.
- cgroup control of device memory: Linux `dmem`; the unified-memory accounting gap: the dmem discussion and ROCm issue 6370.
- OS ownership of device memory: GMEM.
- The limitation: device allocations and the vendor libraries that require them are outside the mechanism.

### Forbidden

- "First copy-on-write for GPU memory", "first GPU snapshot", "first concurrent GPU checkpoint", "first fork for GPU processes". PhoenixOS, CRUM, GPU Snapshot, and Caldera exist.
- "GPUs lack copy-on-write" as a general statement, and its opposite "GPUs now support copy-on-write". Scope every sentence to host-pageable memory on a host-page-table GPU.
- "First to characterize GPU page-fault cost", "first to show GPU first touch is expensive", "first to recommend CPU pre-faulting". ISPASS 2016, ICPP 2024, MI300A.
- "We discover that per-page operations must be batched" or "a new batching principle". NPF, MSched, GMEM, ISPASS 2016.
- "Novel extent-granular copy-on-write" or "new CoW mechanism". RUMA, AnKer; CCoW unverified.
- "First cgroup accounting of GPU memory", "first GPU memory isolation", "first to find that GPU memory escapes cgroups". `dmem`, KRYPTON, HAMi, ROCm issue 6370.
- "MIG provides no memory isolation" as a general statement. Say only what was measured on Thor profiles `2g.0gb` and `1g.0gb` with driver 595.78.
- "The OS memory manager owns all GPU memory" or "single owner of GPU memory" without the qualifier "for pageable memory". `cudaMalloc` allocations are outside, and cuStateVec and TensorRT need them.
- "First to share model weights across MIG tenants" or "first zero-copy sharing across MIG". C2CServe, Flex-MIG.
- "Transparent" without qualification. The application must place its state in pageable memory.
- "1000x" as a headline constant. Report the two measured values and the ratio, and state the kernel version, page size, and driver.
- Any causal statement that the cost "is" MMU-notifier invalidation until a kernel trace shows it.

## 6. Threats I could not rule out

Works that could not be opened:

- **CCoW** (Electronics 2022). MDPI returned HTTP 403. If it does region-granular CoW in the fault handler, it is the closest kernel precedent for O2.
- **Anchor** and **Linux AGX** (SOSP 2026). Only the accepted-list titles were confirmed; ACM DL was blocked. Anchor may overlap the crash-recovery anchor of Group 6. Linux AGX may target Jetson AGX hardware.
- **MorphX** ("Efficient GPU Multitasking with Morphable Kernels") and "Beyond Utilization: Energy-Conscious GPU Sharing for Inference Serving" (SOSP 2026). Titles only.
- **gMig** full text. Whether its page-level dirty tracking runs on an integrated GPU with host-shared memory is not confirmed.
- **Cricket, NVCR, CheCUDA, Checkpoint/Restart for CUDA Kernels, Just-In-Time Checkpointing, Medusa, GaiaGPU, MxGPU, HyPer.** Metadata only.
- **Kernel patches of 2026**: dma-buf system-heap memcg accounting (Chanudet), memcg dma-buf accounting by pidfd (Esteve), dmem for CMA heaps. lkml.org refused the fetch. They may already define how unified-memory device allocations are charged.
- **NVIDIA drivers after 595.78.** The project's earlier notes cite 615.71.09; a `memacct` path exists and is skipped under MIG at the commit opened. A newer Jetson release could charge device memory on Tegra, which would change Facet A.
- **Linux kernels newer than 6.8.** Batching of CoW for large anonymous folios, or batched notifier calls, may have been merged or proposed. Not checked. If present, the per-page cost is a property of the 6.8 Tegra kernel, not of the platform.
- **A Linux GMEM RFC** (2023). Believed to exist; not opened.
- **Apple Silicon (Metal), Arm Mali, Qualcomm Adreno.** No primary source opened on whether these GPUs follow host page tables for pageable memory or on fork behaviour.
- **Intel integrated GPUs with OpenCL or Level Zero shared virtual memory.** GMEM mentions the capability; no fork or CoW measurement was searched for specifically.
- **KVM and secondary-MMU literature.** MMU-notifier cost under fork, CoW, and KSM is studied for virtual machines; no paper with a per-page number was located in this pass.
- **Venue lists.** Only OSDI 2026 and SOSP 2026 lists were scanned, through a fetch tool. USENIX ATC 2026 (page returned 404), EuroSys 2026, ASPLOS 2026, MobiSys, SenSys, RTSS, RTAS, and EMSOFT 2026 were not scanned. Real-time and embedded venues are the likeliest home for Jetson Thor MIG work.
- **NVIDIA GTC talks and white papers on Thor MIG.** Not searched beyond documentation.

Queries tried (web search):

- PhoenixOS SOSP 2025 concurrent OS-level GPU checkpoint restore speculation copy-on-write
- CRIUgpu transparent checkpointing GPU-accelerated workloads cuda-checkpoint
- Cricket, CheCUDA, NVCR transparent checkpoint-restart CUDA
- "GPU snapshot: checkpoint offloading for GPU-dense systems"
- gMig GPU live migration software dirty page full virtualization
- fork GPU process copy-on-write GPU memory serverless warm start zygote CUDA context
- fork() snapshot GPU-accessible memory copy-on-write integrated GPU / unified memory / shared virtual memory
- "Harnessing Integrated CPU-GPU System Memory for HPC: a first look into Grace Hopper" page fault first touch
- GPU page fault cost HMM mmu notifier invalidation overhead ATS "system-allocated memory"
- NVIDIA "Heterogeneous Memory Management" system allocated memory fork copy-on-write limitations
- "Page Fault Support for Network Controllers" on-demand paging
- MMU notifier invalidation overhead KVM secondary MMU copy-on-write fork cost
- On-Demand-Fork EuroSys 2021; Async-fork VLDB 2023
- RUMA rewired user-space memory access; AnKer vm_snapshot
- userfaultfd write-protect live snapshot without fork
- copy-on-write huge page proactive copy neighbouring pages CCoW
- Linux dmem cgroup controller; dma-buf memcg accounting "gpu cgroup controller"; "memcg: dma-buf per-cgroup"
- Jetson GPU memory container memory limit cgroup unified memory OOM
- HAMi GPU memory limit interception; GaiaGPU vCUDA
- nvidia-uvm cgroup accounting
- SERENO OSDI 2026
- copy-on-write sharing model weights across GPU processes MPS tenants deduplication
- share base model weights LoRA copy-on-write GPU tensors KSM GPU memory
- "copy-on-write" GPU memory across MIG instances / MPS clients unified memory
- Tetris tensor sharing; SAGE GPU serverless read-only memory sharing; TrEnv; Medusa; BlitzScale
- vLLM PagedAttention copy-on-write
- Modal GPU memory snapshots
- black box recorder robot AI state forensic recording DNN state
- "Just-In-Time Checkpointing"
- Jetson Thor "Sysmem Full Coherency" / "memoryless MIG" / "0gb"; "pageableMemoryAccessUsesHostPageTables"; "Jetson Thor" OR "AGX Thor" arXiv 2026
- gpu_ext eBPF GPU driver UVM
- SOSP 2026: Anchor; Linux AGX; bpf_fault; morphable kernels

## 7. What the audit changes for the direction

1. **The characterization cannot carry the paper alone.** GPU first-touch cost, CPU pre-population, per-fault cost, and the per-page versus batch cliff each have a citation. The new data are the fork, write-fault, and `mprotect` paths on a host-page-table GPU, and the notifier attribution once traced.
2. **PhoenixOS is the comparison a reviewer will demand for Facet B.** The answer is not "we also do CoW". The answer is that on this platform the property PhoenixOS says GPUs lack is present for pageable memory, so interception, instrumentation, and abort-on-misprediction are unnecessary there, and that PhoenixOS still covers device memory, which this design does not.
3. **CRUM must be cited for fork-based GPU checkpointing.** It is the direct ancestor, and its stated obstacle (UVM versus fork) is the one the platform removes.
4. **Facet A is the weakest novelty claim and the strongest motivation.** The hole is known on AMD APUs and Android. Position it as a consequence (pageable memory is charged, device memory is not, MIG does not help) and not as a discovery.
5. **Facet C is open and unbuilt.** It should be claimed only after measurement, and it inherits the per-page CoW cost, so it depends on extent privatization.
6. **Scope every sentence to pageable memory.** The device-allocation limitation is the first objection of any reviewer who knows cuStateVec or TensorRT, and several audited works (PhoenixOS, GCR, KRYPTON) do cover device memory.
7. **Check a newer kernel before claiming the cost is intrinsic.** The per-page notifier and the THP split are properties of Linux 6.8 code paths.

## 8. Novelty verdicts in detail

**Accounting (cgroup): PARTIALLY TAKEN.** Closest: Linux `dmem` controller. cgroup control of GPU memory is mainline, the escape of device-path allocations from memcg on unified-memory GPUs is reported (ROCm issue 6370) and acknowledged as unresolved by the dmem author, and NVIDIA's driver contains accounting hooks that are skipped under MIG. Open: the memoryless-MIG measurement and a design in which the memcg limit is the only memory limit. This is a known OS behaviour on a new platform.

**Snapshot through fork CoW with extent privatization: PARTIALLY TAKEN.** Closest in mechanism: CRUM. Most dangerous: PhoenixOS. CoW-based concurrent GPU checkpointing, fork-based GPU checkpointing, user-space copy-and-remap CoW, and predict-then-batch with a fault fallback are each published. Open: kernel CoW as the native, interception-free snapshot mechanism for GPU-written pageable memory, and the use of extent remapping to avoid secondary-MMU invalidation, with a correctness fallback that cannot corrupt the image.

**CoW sharing across MIG tenants: OPEN in the verified set.** Closest: C2CServe. No audited system uses the host MMU to give MIG instances private copy-on-write views of shared pages. The technique is `MAP_PRIVATE` or fork semantics, CoW sharing of GPU state exists inside runtimes (vLLM, ForkKV), and the facet is planned, not measured. Treat as a combination claim that must be earned by an implementation.

**Characterization (about 1000x per-page versus range cost on a host-page-table GPU): PARTIALLY TAKEN.** Closest: Grace Hopper system-memory study (ICPP 2024). The phenomenon class is established by ISPASS 2016, NPF, MSched, and MI300A, and the kernel source explains the asymmetry. Open: the specific paths (fork, CoW write fault, CPU-side pre-break, `mprotect`) and numbers on a GPU that shares host page tables, and the attribution.

## 9. Primary sources opened

Papers (PDF read): [PhoenixOS](https://arxiv.org/abs/2405.12079); [GCR](https://www.usenix.org/system/files/fast26-zeng.pdf); [CRIUgpu](https://arxiv.org/abs/2502.16631); [CRUM](https://arxiv.org/abs/1808.00117); [CRAC](https://arxiv.org/abs/2008.10596); [GPU Snapshot](https://lph.ece.utexas.edu/merez/uploads/MattanErez/ics19_gpusnapshot.pdf); [gCROP](https://ipads.se.sjtu.edu.cn/_media/publications/yang-socc24.pdf); [Singularity](https://arxiv.org/abs/2202.07848); [Caldera](http://cidrdb.org/cidr2017/papers/p21-appuswamy-cidr17.pdf); [Grace Hopper study](https://arxiv.org/abs/2407.07850); [MI300A study](https://arxiv.org/abs/2508.12743); [Vesely et al. ISPASS 2016](https://www.csa.iisc.ac.in/~arkapravab/papers/ispass16.pdf); [Zheng et al. HPCA 2016](https://www.cs.utexas.edu/~skeckler/pubs/HPCA_2016_Paged_Memory.pdf); [MSched](https://arxiv.org/abs/2512.24637); [NPF](https://www.cs.technion.ac.il/~dan/papers/npf-asplos-2017.pdf); [GMEM](https://arxiv.org/abs/2310.12554); [SVM ICS 2024](https://arxiv.org/abs/2405.06811); [Fusco et al.](https://arxiv.org/abs/2408.11556); [gpu_ext](https://arxiv.org/abs/2512.12615); [Psistakis et al.](https://psistakis.cs.illinois.edu/files/publications/psistakis-tpds22.pdf); [On-demand-fork](https://sishuaigong.github.io/pdf/eurosys21-odf.pdf); [Async-fork](https://arxiv.org/abs/2301.05861); [A fork() in the road](https://www.microsoft.com/en-us/research/uploads/prod/2019/04/fork-hotos19.pdf); [RUMA](http://www.vldb.org/pvldb/vol9/p768-schuhknecht.pdf); [AnKer](https://bigdata.uni-saarland.de/publications/AnKer_SIGMOD2018.pdf); [bpf_fault draft](https://github.com/bpf-fault/bpf-fault/blob/main/bpf_fault_draft.pdf); [TrEnv](https://madsys.cs.tsinghua.edu.cn/publication/trenv-transparently-share-serverless-execution-environments-across-different-functions-and-nodes/SOSP24-huang.pdf); [KRYPTON](https://www.usenix.org/system/files/atc25-zhang-shulai.pdf); [TGS](https://www.usenix.org/system/files/nsdi23-wu.pdf); [Nixie](https://www.usenix.org/system/files/osdi26-xu-yechen.pdf); [Prism](https://www.usenix.org/system/files/osdi26-yu-shan.pdf); [Sereno](https://www.usenix.org/system/files/osdi26-xin.pdf); [C2CServe](https://arxiv.org/abs/2605.19481); [Tetris](https://www.usenix.org/system/files/atc22-li-jie.pdf); [vLLM](https://arxiv.org/abs/2309.06180); [ForkKV](https://arxiv.org/abs/2604.06370); [SAGE](https://arxiv.org/abs/2404.14691); [Tangram](https://arxiv.org/abs/2512.01357); [Flex-MIG](https://arxiv.org/abs/2511.09143); [StreamBox](https://www.usenix.org/system/files/atc24-wu-hao.pdf); [Torpor](https://arxiv.org/abs/2306.03622); [ServerlessLLM](https://arxiv.org/abs/2401.14351); [BlitzScale](https://www.usenix.org/system/files/osdi25-zhang-dingyan.pdf); [CheckFreq](https://www.usenix.org/system/files/fast21-mohan.pdf); [DataStates-LLM](https://arxiv.org/abs/2406.10707); [Smart Black Box](https://arxiv.org/abs/1903.01450); [Echo](https://arxiv.org/abs/2609.05635); [Hydra](https://arxiv.org/abs/2608.25053); [edge GPU isolation](https://arxiv.org/abs/2601.07600).

Documentation, source, and archives: [CUDA Programming Guide 2.6](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html); [CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html); [NVIDIA HMM blog](https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management); [CUDA 12.2 release notes](https://docs.nvidia.com/cuda/archive/12.2.0/cuda-toolkit-release-notes/index.html); [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint); [MIG supported profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html); [Jetson Thor MIG guide](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/MiG.html); [open-gpu-kernel-modules `video_mem.c`](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/61dcc93722ecb418bb5f2e00923f05b4b8051dd1/src/nvidia/src/kernel/mem_mgr/video_mem.c); [`uvm_linux.h`](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/kernel-open/nvidia-uvm/uvm_linux.h); [Linux cgroup-v2 documentation](https://www.kernel.org/doc/Documentation/admin-guide/cgroup-v2.rst); [Linux HMM](https://docs.kernel.org/mm/hmm.html); [DRM GPU SVM](https://docs.kernel.org/gpu/rfc/gpusvm.html); [Linux v6.8 mm sources](https://github.com/gregkh/linux/tree/v6.8/mm); [dmem thread](https://lists.openwall.net/linux-kernel/2024/12/17/1367); LWN [881554](https://lwn.net/Articles/881554/), [919548](https://lwn.net/Articles/919548/), [1034421](https://lwn.net/Articles/1034421/), [1072437](https://lwn.net/Articles/1072437/), [787308](https://lwn.net/Articles/787308/), [849638](https://lwn.net/Articles/849638/); [ROCm issue 6370](https://github.com/ROCm/legacy-rocm-build/issues/6370); [HAMi-core](https://github.com/Project-HAMi/HAMi-core); [gpuoom](https://github.com/tom-doerr/gpuoom); [Modal blog](https://modal.com/blog/gpu-mem-snapshots); [VEE 2018 gMig page](https://conf.researchr.org/details/vee-2018/vee-2018-Research-Papers/9/gMig-Efficient-GPU-Live-Migration-Optimized-by-Software-Dirty-Page-for-Full-Virtuali); [SOSP 2026 accepted papers](https://sigops.org/s/conferences/sosp/2026/accepted.html); [OSDI 2026 sessions](https://www.usenix.org/conference/osdi26/technical-sessions); [NVIDIA forum thread](https://forums.developer.nvidia.com/t/mig-on-agx-thor-mig-1g-instances-enumerate-as-cuda-devices-but-hang-on-the-first-kernel/384828).

## 10. What changed after the audit was written

The audit was commissioned before four results existed. They change three of
its statements and none of its verdicts' direction.

1. **The cause of the per-page cost is now traced.** Section 5 forbade any
   causal statement "until a kernel trace shows it". A `perf` profile of a
   copy-on-write break in a CUDA process attributes 73.5% of all samples to
   `arm_smmu_cmdq_issue_cmdlist`, reached through
   `do_wp_page -> ptep_clear_flush -> __mmu_notifier_arch_invalidate_secondary_tlbs -> arm_smmu_mm_arch_invalidate_secondary_tlbs`
   (`thor_hostmm/results/20261005-smmu-trace-v1/`). The same break before CUDA
   is initialized shows no IOMMU symbol. The cost is therefore IOMMU
   secondary-TLB invalidation under SMMUv3 shared virtual addressing, not an
   "MMU-notifier invalidation into the GPU driver" as the brief guessed: the
   UVM module runs in ATS mode with HMM disabled, and the invalidation is
   issued by the IOMMU layer. Claim O3 may now state the attribution as a
   measured result for this kernel and driver.
2. **The cost is process-wide.** A region that no GPU kernel ever touched pays
   it once the process has initialized CUDA (1,724 to 2,163 ms/GiB against
   159 to 349 ms/GiB before initialization). No audited source reports this.
   It widens O3 from "memory a GPU uses" to "every page of a process bound to
   the GPU", and it is the reason a pre-populated staging area must be
   excluded from `fork`.
3. **Two mechanisms were added that the brief did not describe.** Write-set
   learning uses the kernel's copy-on-write as its only detector: one page per
   privatized extent is left shared and read back through
   `/proc/self/pagemap`, because this kernel has neither `userfaultfd` nor
   soft-dirty bits. No audited source uses copy-on-write sentinels this way;
   the closest are PhoenixOS's validated speculation and the kernel facilities
   that are absent here. The contribution is small and is claimed only as an
   engineering consequence of the missing facilities.
4. **O5 is no longer only planned.** Copy-on-write sharing of one model
   across the two MIG instances is implemented with separately executed
   tenants over a `MAP_PRIVATE` mapping, as the audit required, and measured
   in `RESEARCH_HOSTMM_2026-10-05.md` Section 8 (36 runs: 4,218 MiB for two
   tenants of a 4 GiB model against 8,314 MiB for separate copies; private
   divergence in 6 of 6 runs). The measurement adds a condition that no
   audited source states: the driver serves a GPU fault in a writable mapping
   as a write, so sharing survives only if the CPU has read every page before
   the GPU does. The permitted sentence for O5 may be used with the scope
   "among the audited works" and with that condition.

5. **GPU-side faults are traced as well**
   (`thor_hostmm/results/20261005-gpu-fault-trace-v1/`, system-wide profile).
   One kernel thread, the UVM bottom half, serves them. Of the samples that
   are not idle, 63.2% (first GPU touch) and 51.8% (first GPU write after
   `fork`) are in `arm_smmu_cmdq_issue_cmdlist`. The write after `fork`
   reaches it through `uvm_ats_service_faults -> handle_mm_fault ->
   do_wp_page -> ptep_clear_flush`. The first touch reaches it through
   `uvm_migrate_pageable -> migrate_vma_setup -> migrate_vma_collect_pmd ->
   ptep_clear_flush`. This answers the source-level observation of
   Section 11 that `do_anonymous_page()` does not flush: the fault itself
   does not, and the driver's migration pass over the pages it has just
   populated does, once per page.
6. **The IOMMU literature and newer kernels are audited in Section 11.** That
   pass was run because items 1 and 2 moved the cause from the GPU driver to
   the IOMMU layer, which Sections 1 to 9 had not searched. Its results:
   - the mechanism is intended kernel behaviour since v6.6 and an IOMMU
     maintainer said in 2023 that it makes `fork`, `mmap`, and `munmap`
     slower once SVA is enabled. No opened source gives a number, a profile,
     or a GPU case. Items 1 and 2 are therefore measurements of a known
     mechanism, and "no audited source reports this" in item 2 must be read
     as "no audited source measures this";
   - CCoW is opened: region-granular copy-on-write in the kernel fault
     handler, motivated by fault count and CPU TLB shootdowns, without any
     IOMMU or secondary-MMU content;
   - MoonBright (OSDI 2026) is a new neighbour in framing, on the GPU's own
     page tables of discrete GPUs;
   - the per-page path is present in mainline 7.3-rc6 source, and the
     magnitude on any kernel other than 6.8.12-tegra is not measured.

Wording that Section 11 adds to Section 5:

> Since Linux 6.6 the architecture TLB-flush functions notify the IOMMU of
> every flush in a process that is bound for shared virtual addressing, and
> the kernel developers expected this to slow `fork`, `mmap`, and `munmap`.
> We measure the effect for a GPU: a copy-on-write break costs 293 ms/GiB
> before the process initializes CUDA and 1,705 ms/GiB afterwards, on memory
> the GPU never touched as well, and a kernel profile attributes 73.5% of the
> samples to the SMMU command queue.

Required next to it: the 2023 linux-iommu thread (Gunthorpe), the Popple
series that placed the notifier in the flush functions, the v6.19 arm64 patch
that measures the CPU-TLB analogue after `fork` (Huang Ying), and Border
Control (MICRO 2015), which concluded in simulation that the cost is
negligible at far lower event rates. Forbidden in addition to Section 5:
"we discover that SVA slows the memory manager", "8 us is the cost of one
SMMU invalidation" (the number of synchronous submissions per page in the
Tegra kernel is not known), and "fixed in newer kernels".

Unchanged: the technique of copy-then-remap is RUMA's and AnKer's; GPU first
touch and CPU pre-population are the Grace Hopper and MI300A studies'; cgroup
control of device memory is `dmem`'s. Anchor, MorphX, and Linux AGX were read
as abstracts only, and the 2026 dma-buf memcg patches were not opened; they
remain threats.

## 11. Addendum: IOMMU shared virtual addressing, newer kernels, and remaining threats

Provenance: produced by a second delegated search pass on 2026-10-05, after
the kernel profile of Section 10 item 1 existed. The brief asked for work
that takes the claims. The text is the pass's own; its "findings 1 to 3" are
the per-page cost, its process-wide scope, and extent privatization.

Date of this pass: 2026-10-05. Scope: the gap left by Sections 1 to 9, which were written before the cause was known to be SMMUv3 SVA secondary-TLB invalidation. Findings are numbered as in the request: (1) per-page MM events cost about 8 us extra in an SVA-bound mm, (2) the cost is process-wide, (3) range operations invalidate once, so extent privatization (copy, then one `mremap`) replaces per-page kernel CoW.

Method. Papers were downloaded as PDF and searched with `pdftotext`. Kernel source was read from `torvalds/linux` at tags v6.8, v6.9, v6.10, v6.11, v6.12, v6.14, v6.17, v6.19, v7.0, v7.1, v7.2, v7.3-rc1 and at master (v7.3-rc6, commit `a90ee4305c4a`), plus the headers installed on the Thor (`/usr/src/linux-headers-6.8.12-1021-tegra-ubuntu24.04_aarch64`). Mailing-list messages were read as raw messages from lore.kernel.org, through the patchwork.kernel.org API, or from LWN. Search engines were used only to locate sources.

Limits. The Thor has kernel headers only. The `arm-smmu-v3` driver source of 6.8.12-tegra was not available, so every driver statement about "6.8" refers to upstream v6.8. The Tegra tree is not upstream v6.8: its `asm/tlbflush.h` already contains the v6.14 range-notifier fix, and its config sets `CONFIG_TEGRA241_CMDQV=y` (upstream since v6.12). The public NVIDIA module tree has no tag 595.78; tags 595.71.05 and 595.80 were read instead.

Tags: **V (pdf)** PDF downloaded and passage read; **V (src)** source file or commit read (version stated); **V (doc)** official doc, mailing-list message, or LWN page opened; **V (abs)** abstract only; **M** metadata only; **NV** not verified.

### Group A. IOMMU SVA and PASID secondary-TLB invalidation cost

| Work (venue, year) | What it measures or does | Overlap with findings 1-3 | Tag | URL |
|---|---|---|---|---|
| J. Gunthorpe, replies in "Cache Invalidation Solution for Nested IOMMU" (linux-iommu, 2023-04-05, 04-06, 04-11) | No measurement. States that the SVA invalidation "sits in a hot path of the mm, so it broadly impacts certain workloads once SVA is enabled", asks for a test with "a more real application that is actually using the MM (eg alloc/free memory, fork, etc)" with SVA on, and says "MM activities like mmap and munmap ... become slower once SVA is enabled". Context is nested (guest) SVA. No participant posted a number for these MM activities. | Findings 1 and 2 are stated qualitatively by a maintainer three years before our measurement. No per-page number, no CoW profile, no GPU. | **V (doc)** | [04-05](https://lore.kernel.org/linux-iommu/ZC1iQf8HdVNyG80+@nvidia.com/), [04-06](https://lore.kernel.org/linux-iommu/ZC6vlI7onlUgBidX@nvidia.com/), [04-11](https://lore.kernel.org/linux-iommu/ZDVLL6qcTxfIMT5g@nvidia.com/) |
| A. Popple (NVIDIA), "Invalidate secondary IOMMU TLB on permission upgrade", RFC to v4 (linux-mm, 2023-06 to 07; in v6.6) | Moves the secondary-TLB notifier call into the arch TLB flush functions. This is what makes arm64 `flush_tlb_page()` call the SMMU for every page. Cover: "still allowing efficient range based invalidations based on the existing TLB batching code". Gunthorpe on the RFC: "Batching is going to be important". No measurement. | Origin of the call chain in finding 1. The contrast between per-page and range calls (finding 3, first half) is the stated design. | **V (doc)** | [v3 cover on LWN](https://lwn.net/Articles/938727/), [RFC cover and reply](https://patchwork.kernel.org/project/linux-mm/cover/cover.063f3dc2100ae7cbe3a6527689589646ea787216.1687259597.git-series.apopple@nvidia.com/) |
| N. Chen (NVIDIA), commit `d5afb4b47e13` "iommu/arm-smmu-v3: Fix soft lockup triggered by arm_smmu_mm_invalidate_range" (v6.6) | Reports a 26 s soft lockup with `arm_smmu_cmdq_issue_cmdlist` on top of the stack, reached from `munmap` in an SVA process. Adds `CMDQ_MAX_TLBI_OPS`: at or above 512 pages without range-invalidation support, one ASID-wide TLBI replaces per-page commands. | Same hot function as finding 1, on the unmap path. Explains why a multi-MiB range operation costs one command (finding 3, first half). Not fork or CoW. | **V (src)** | [commit](https://github.com/torvalds/linux/commit/d5afb4b47e13) |
| P. Jaroszynski (NVIDIA), commit `f7edb07ad7c6` "Fix mmu notifiers for range-based invalidates" (v6.14, stable) | arm64 `__flush_tlb_range()` passed an empty range to the notifier since v6.6, so every range flush became invalidate-all, which "can certainly result in suboptimal perf". | Bears on finding 3: the fixed macro (`__flush_start`) is present in the Thor header, so range flushes on the Thor carry the real range. | **V (src)** | [commit](https://github.com/torvalds/linux/commit/f7edb07ad7c6) |
| J. Pan (Microsoft), "SMMU v3 CMDQ fix and improvement" v5 (linux-iommu, 2025-12) | CMDQ lock fairness and timeout fixes. Cover: problems "become more pronounced when multiple CPUs submit to a single queue, a common scenario under SVA when shared buffers (used by both CPU and device) are being unmapped". No number in the cover. | Confirms contention on the same command queue under SVA unmap. Not per-page CoW. | **V (doc)** | [cover](https://lore.kernel.org/linux-iommu/20251208212857.13101-1-jacob.pan@linux.microsoft.com/) |
| T. Zhang (Intel), "Batch IOTLB/dev-IOTLB invalidation" v3 (linux-iommu, 2024-08; VT-d) | "IOTLB and dev-IOTLB invalidation operations are performance-critical." Microbenchmark with a DSA device in SVA: batching the two commands cuts `qi_submit_sync()` by "roughly more than 800 cycles". | The only number found for an SVA invalidation call. x86, per call, no fork or CoW, no absolute per-page cost. | **V (doc)** | [cover](https://lore.kernel.org/linux-iommu/20240815065221.50328-1-tina.zhang@intel.com/) |
| Huang Ying (Alibaba), "arm64, tlbflush: don't TLBI broadcast if page reused in write fault" v6, commit `cb1fa2e99955` (v6.19) | A fork()/exec() workload on arm64 spends 50.5% of cycles in TLB flush functions during write-protect faults after fork; a local flush reduces this to 0.3% and improves usemem by 40.6%. The patch keeps the secondary-TLB notifier call in the local flush. | Per-page flush cost after fork on arm64 is published, for the CPU TLB only. No SVA, no SMMU. A reviewer may cite it as the arm64 precedent for finding 1. | **V (doc)** | [patch 2/2](https://lore.kernel.org/linux-mm/20251114085403.101552-3-ying.huang@linux.alibaba.com/), [commit](https://github.com/torvalds/linux/commit/cb1fa2e99955) |
| J.-P. Brucker, "Shared Virtual Addressing for the IOMMU" v1 cover (2018) | Design of SVA. "Although we don't have any performance measurement at the moment, SVA will likely be slower than classical DMA since it relies on page faults". | None measured. The expected cost named is device page faults, not CPU-side invalidation. | **V (doc)** | [LWN copy](https://lwn.net/Articles/747230/) |
| Linux `Documentation/arch/x86/sva.rst` (master) | SVA bind registers an MMU notifier "to keep the device TLB in sync"; on fork(2) or exec(2) the PASID is removed from the process. | Describes the mechanism. No cost. | **V (doc)** | [file](https://github.com/torvalds/linux/blob/master/Documentation/arch/x86/sva.rst) |
| Linux x86 and IOMMU drivers (master): `arch/x86/mm/tlb.c`, `drivers/iommu/intel/svm.c`, `drivers/iommu/amd/pasid.c` | `flush_tlb_page()` is `flush_tlb_mm_range(PAGE_SIZE)`, which ends in `mmu_notifier_arch_invalidate_secondary_tlbs()`. The Intel and AMD SVA drivers implement that callback (`intel_arch_invalidate_secondary_tlbs()` -> `cache_tag_flush_range()`; `sva_arch_invalidate_secondary_tlbs()` -> `amd_iommu_dev_flush_pasid_pages()`). | The per-page pattern of finding 1 is not specific to Arm. It exists in source on x86. No measurement located. | **V (src)** | [tlb.c](https://github.com/torvalds/linux/blob/master/arch/x86/mm/tlb.c) |
| Linux `arch/powerpc/mm/book3s64/radix_tlb.c` (master) | When `mm->context.copros > 0` (a coprocessor or nest MMU is attached to the mm), flushes add a broadcast `tlbie` ("coprocessors must use tlbie") and, on POWER9, are escalated. | Precedent for finding 2 at the level of design: attaching an accelerator to an mm changes the TLB-flush cost of the whole mm. | **V (src)** | [file](https://github.com/torvalds/linux/blob/master/arch/powerpc/mm/book3s64/radix_tlb.c) |
| NVIDIA open-gpu-kernel-modules, `kernel-open/nvidia-uvm/uvm_ats_sva.{c,h}` (tags 595.80 and 595.71.05) | On kernels without `arch_invalidate_secondary_tlbs`, UVM itself writes per-page `TLBI_EL2_VA` commands plus `CMD_SYNC` to a CMDQ-V queue and spins until consumed ("Bug 4130089"). On kernels that have the callback the workaround is compiled out and UVM relies on the kernel path. | Confirms that on 6.8 the SMMU invalidation for ATS is the kernel notifier path of finding 1. No cost stated. | **V (src)** | [uvm_ats_sva.c](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/595.80/kernel-open/nvidia-uvm/uvm_ats_sva.c) |
| Kuper et al., "A Quantitative Analysis and Guidelines of Data Streaming Accelerator in Modern Intel Xeon Scalable Processors" (ASPLOS 2024) | DSA throughput and latency on Sapphire Rapids, Linux 5.15. SVM and PASID are described; a page fault stalls a processing engine. | None. No fork, no CoW, no invalidation cost (0 occurrences of "fork" or "copy-on-write"). | **V (pdf)** | [arXiv 2305.02480](https://arxiv.org/abs/2305.02480) |
| Wang et al., "To PRI or Not To PRI, That's the question" (OSDI 2025) | I/O page faults for passthrough devices in VMs; average fault handling about 700 us; proposes VIO. | None. Device-side fault latency, not CPU-side MM cost. | **V (pdf)** | [PDF](https://www.usenix.org/system/files/osdi25-wang-yun.pdf) |
| Koenig, Zelioli, Benini, "Evaluating IOMMU-Based Shared Virtual Addressing for RISC-V Embedded Heterogeneous SoCs" (arXiv, 2025) | IOTLB miss and translation overhead for accelerator offload on an FPGA-emulated RISC-V SoC. | None. No fork, CoW, or invalidation measurement. Closest academic study of SVA on an embedded SoC. | **V (pdf)** | [arXiv 2502.17398](https://arxiv.org/abs/2502.17398) |
| Psistakis, "Handling of Memory Page Faults during Virtual-Address RDMA" (M.Sc. thesis 2019, arXiv 2025) | Arm SMMU fault handling for an FPGA DMA engine on Zynq UltraScale+. | None. CoW and fork appear as background only. | **V (pdf)** | [arXiv 2511.21018](https://arxiv.org/abs/2511.21018) |
| Olson, Power, Hill, Wood, "Border Control: Sandboxing Accelerators" (MICRO 2015) | Simulation. Section 5.2.4: on a permission downgrade the accelerator must "invalidate its TLB entries, and the ATS must flush its caches"; overhead about 0.02% at 10 to 200 downgrades per second, small up to 1000 per second. States that copy-on-write "incurs no extra overhead". | Describes the mechanism class and concludes it is negligible, at event rates two to four orders of magnitude below per-page CoW of a multi-GiB region (our measurement implies on the order of 10^5 page events per second). Opposite conclusion to finding 1, different regime, simulated. | **V (pdf)** | [PDF](https://research.cs.wisc.edu/multifacet/papers/micro15_border_control.pdf) |
| Haria, Hill, Swift, "Devirtualizing Memory in Heterogeneous Systems" (ASPLOS 2018) | Identity mapping for accelerators. CoW and fork make the OS revert to paging; the authors recommend against CoW for identity-mapped data. | None on cost. Notes an interaction between fork/CoW and accelerator address translation. | **V (pdf)** | [PDF](https://research.cs.wisc.edu/multifacet/papers/asplos18_dvm.pdf) |
| Malka et al., "rIOMMU" (ASPLOS 2015) | Table 1: IOTLB invalidation costs 2,127 to 2,135 cycles per `unmap` in strict mode (VT-d, kernel DMA API). | A published per-invalidation number, for the DMA API on x86. Not SVA, not process MM. | **V (pdf)** | [PDF](https://www.cs.technion.ac.il/~dan/papers/riommu-asplos-2015.pdf) |
| Peleg et al., "Utilizing the IOMMU Scalably" (USENIX ATC 2015) | Lock contention of IOVA allocation and batched IOTLB invalidation in the DMA API. | None. | **V (pdf)** | [PDF](https://www.usenix.org/system/files/conference/atc15/atc15-paper-peleg.pdf) |
| Markuze et al., "DAMN" (ASPLOS 2018) | Cost of the invalidation-queue lock and of waiting for IOTLB invalidation in `dma_unmap`. | None. | **V (pdf)** | [PDF](https://www.cs.technion.ac.il/~dan/papers/damn-asplos-2018.pdf) |
| Amit et al., "vIOMMU" (USENIX ATC 2011) | "IOTLB invalidation is a lengthy process that on bare metal takes over 40% of the overall unmapping process." | None. DMA unmap. | **V (pdf)** | [PDF](https://www.usenix.org/legacy/event/atc11/tech/final_files/Amit.pdf) |
| Rubin et al., "Fast & Safe IO Memory Protection" (SOSP 2024) | IOTLB miss and invalidation cost for 100 Gbps NICs in the DMA API. | None. No SVA, no fork. | **V (pdf)** | [PDF](https://www.cs.cornell.edu/~ragarwal/pubs/fands.pdf) |
| Korolija, Roscoe, Alonso, "Do OS abstractions make sense on FPGAs?" (Coyote, OSDI 2020) | Shared virtual memory for FPGAs with software-loaded FPGA TLBs. | None. No fork or CoW measured. | **V (pdf)** | [PDF](https://www.usenix.org/system/files/osdi20-korolija.pdf) |
| "NP-RDMA: Using Commodity RDMA without Pinning Memory" (arXiv, 2023) | Uses MMU notifiers to update IOMMU or SMMU mappings when pages are swapped. | None. Not SVA, no fork or CoW cost. | **V (pdf)** | [arXiv 2310.11062](https://arxiv.org/abs/2310.11062) |
| Danduri, Machiry, "Speed Kills: Exploring Confused Deputy Attacks Through Edge AI Accelerators" (arXiv, 2026) | Security study of edge accelerators including Jetson AGX Orin; IOMMU overhead in gem5. | None. | **V (pdf)** | [arXiv 2605.17707](https://arxiv.org/abs/2605.17707) |
| Li et al., "Automatic BLAS Offloading on Unified Memory Architecture" and follow-up (arXiv 2404.13195, 2501.00279) | First-touch data movement on Grace Hopper. | None. No SMMU, ATS, fork, or CoW content. | **V (pdf)** | [2404.13195](https://arxiv.org/abs/2404.13195), [2501.00279](https://arxiv.org/abs/2501.00279) |

Result for group A. No opened source reports a per-page or per-GiB cost of fork, CoW, first touch, or THP split in a process bound to SVA, on any architecture. The effect is acknowledged in words by an IOMMU maintainer (April 2023) and is the stated reason for range batching in the v6.6 notifier rework.

### Group B. Kernel changes after v6.8

| Work (venue, year) | What it measures or does | Overlap with findings 1-3 | Tag | URL |
|---|---|---|---|---|
| Baseline: upstream v6.8 and the Thor headers | CoW copy: `wp_page_copy()` -> `ptep_clear_flush()` -> `flush_tlb_page()` -> `__flush_tlb_page_nosync()` -> notifier with a one-page range (`asm/tlbflush.h`, identical in the Thor header). CoW reuse: `wp_page_reuse()` -> `ptep_set_access_flags()` -> `flush_tlb_page()` (`arch/arm64/mm/fault.c`). Driver: `arm_smmu_mm_arch_invalidate_secondary_tlbs()` -> `arm_smmu_tlb_inv_range_asid()` -> `arm_smmu_cmdq_batch_submit()` (TLBI plus `CMD_SYNC`), then `arm_smmu_atc_inv_domain()` (second submit when ATS masters exist). | Source of finding 1. | **V (src)** v6.8 | [arm-smmu-v3-sva.c](https://github.com/torvalds/linux/blob/v6.8/drivers/iommu/arm/arm-smmu-v3/arm-smmu-v3-sva.c) |
| "mm: batch-copy PTE ranges during fork" (v6.9; `wrprotect_ptes()` in `copy_present_ptes()`) | Batches write-protection of large folios in the parent at fork. | Fork side only. Fork already ends in one `flush_tlb_mm(oldmm)`. Does not touch the fault path. | **V (src)** v6.9 | [memory.c](https://github.com/torvalds/linux/blob/v6.9/mm/memory.c) |
| J. Gunthorpe, commit `d38c28dbefee` "Put the SVA mmu notifier in the smmu_domain" (v6.11) | Restructures the SVA domain. The callback body keeps one `arm_smmu_tlb_inv_range_asid()` and one `arm_smmu_atc_inv_domain()` per call. Message: "since ARM_SMMU_FEAT_BTM is never enabled, remove the parts of the BTM support". | Per-call cost unchanged. Broadcast TLB maintenance, which would remove the TLBI commands, is not used. | **V (src)** v6.11, v6.17 | [commit](https://github.com/torvalds/linux/commit/d38c28dbefee) |
| `tegra241-cmdqv` (v6.12; `CONFIG_TEGRA241_CMDQV=y` on the Thor) | `tegra241_cmdqv_get_cmdq()` picks a per-VINTF queue by `raw_smp_processor_id() % num_lvcmdqs_per_vintf` to spread lock contention. | Changes which queue is used, not the number of synchronous submissions per page. | **V (src)** master | [tegra241-cmdqv.c](https://github.com/torvalds/linux/blob/master/drivers/iommu/arm/arm-smmu-v3/tegra241-cmdqv.c) |
| D. Jain, commit `c320dbb7c80d` "arm64/mm: Elide TLB flush in certain pte protection transitions" (v6.19) | `pte_needs_flush()` for arm64; `mprotect` from `PROT_NONE` to read-write no longer flushes (3.2 us to 2.85 us per call). | Makes some range operations cheaper still (finding 3). | **V (src)** | [commit](https://github.com/torvalds/linux/commit/c320dbb7c80d) |
| Huang Ying, commit `cb1fa2e99955` (v6.19) | Reuse faults flush the CPU TLB locally. `local_flush_tlb_page()` still calls the secondary-TLB notifier; only the spurious-fault fix-up skips it. | The reuse half of finding 1 persists. | **V (src)** v6.19 | [commit](https://github.com/torvalds/linux/commit/cb1fa2e99955) |
| N. Chen and J. Gunthorpe, `arm_smmu_invs` series, commit `4202fddd01c7` (v7.1) | `arm_smmu_domain_inv_range()` walks a per-domain invalidation array. `arm_smmu_invs_end_batch()` forces a submit between TLBI and ATS entries ("ATS must be after a sync of the S1/S2 invalidations"). | Still one synchronous TLBI submit and one synchronous ATC submit per notifier call. No coalescing across calls. | **V (src)** v7.1, master | [commit](https://github.com/torvalds/linux/commit/4202fddd01c7) |
| R. Roberts, commit `0477fc56960d` "arm64: mm: More flags for __flush_tlb_range()" (v7.1) | Adds `TLBF_NONOTIFY`. Its only users are `flush_tlb_fix_spurious_fault()` and the PMD variant. | `flush_tlb_page()` and the write-enable flush in `__ptep_set_access_flags_anysz()` still notify. | **V (src)** master | [commit](https://github.com/torvalds/linux/commit/0477fc56960d) |
| A. Mhetre (NVIDIA), "iommu/arm-smmu-v3: Tegra264 invalidation workaround" v9, commits `95ed2da20283`, `06b15ddcfbc0` (v7.3-rc1; erratum T264-SMMU-3) | "a TLB entry can survive an invalidation that races with concurrent traffic". Every CFGI or TLBI sequence with `CMD_SYNC` is issued twice on `nvidia,tegra264-smmu`; ATC_INV is not doubled. | On mainline with the mainline device tree, the Thor's SoC pays two synchronous TLBI submissions per notifier call. Whether 6.8.12-tegra carries an equivalent is unknown (the installed DT nodes are plain `arm,smmu-v3`; driver source not available). | **V (doc)**, **V (src)** master | [cover](https://lore.kernel.org/linux-iommu/20260726081904.1408859-1-amhetre@nvidia.com/), [commit](https://github.com/torvalds/linux/commit/06b15ddcfbc0) |
| Mainline v7.3-rc6 `mm/memory.c`, `mm/huge_memory.c` | `wp_page_copy()` copies one page and calls `ptep_clear_flush()` once per fault. `do_huge_pmd_wp_page()` splits a shared anonymous PMD and falls back to PTE faults. No batched CoW copy exists. | Findings 1 (CoW break, THP split plus per-4K copy) persist in source. | **V (src)** `a90ee4305c4a` | [memory.c](https://github.com/torvalds/linux/blob/master/mm/memory.c) |
| J. Gunthorpe, "Organize the SMMUv3 invalidation flow so iommupt can use it" v8 (posted 2026-10-02, not merged) | Reworks TLBI encoding. For SVA: "SVA invalidation has no idea what the MM did"; the callback still issues one `arm_smmu_domain_tlbi()` per call, with better TTL hints for sub-PMD ranges. | No coalescing planned in this series. | **V (doc)** | [cover](https://lore.kernel.org/all/0-v8-1eaaed5e0433+3d3d2d-smmu_tlbi_jgg@nvidia.com/) |
| Yuan-Hao Hsu, "mm/memory: reuse the whole exclusive large folio on a write fault" v1 and v2 (linux-mm, 2026-09-18 and 09-19, not merged) | After fork and child exit, makes all PTEs of an exclusive large folio writable in one fault. x86 numbers: 420 ns per reuse fault today, 1,500 ns per 4K CoW copy fault; 256 MiB of 64K mTHP goes from 65,601 to 4,161 faults. "Small folios, PMD-mapped THPs, unsharing faults and the copy path are not changed." | If merged, it would cut the reuse-path count for large folios by the folio order. It does not address the copy path, 4K pages, or SVA. | **V (doc)** | [v2 cover](https://lore.kernel.org/linux-mm/20260919073134.639-1-aa9736195201@gmail.com/) |
| Barry Song, "mm: entirely reuse the whole anon mTHP in do_wp_page" RFC (2024-08, not merged) | Same idea, motivated by deferred-split mTHP on phones. | As above. | **V (doc)** | [RFC](https://lore.kernel.org/r/20240831092339.66085-1-21cnbao@gmail.com) |
| LKML discussion of the cost of `arch_invalidate_secondary_tlbs` on fork or CoW for SVA-bound processes | Not found beyond the Gunthorpe replies in group A. No stack trace with `arm_smmu_cmdq_issue_cmdlist` under `do_wp_page`, `wp_page_copy`, or `ptep_clear_flush` exists in the lore archive search. | n/a | **V (doc)** (search result pages) | see Queries tried |

### Group C. CCoW

| Work (venue, year) | What it measures or does | Overlap with findings 1-3 | Tag | URL |
|---|---|---|---|---|
| Ha and Kim (Ajou University), "CCoW: Optimizing Copy-on-Write Considering the Spatial Locality in Workloads", Electronics 11(3):461, 2022 | "Coverage-based copy-on-write". The address space is divided into fixed regions (32 KB to 2 MB evaluated, 2 MB used). Per-region counters record how many pages were CoW-faulted in the previous fork epoch; when the coverage exceeds a threshold, one write fault precopies the whole region in the page fault handler. Dirty bits track use of precopied regions. Linux 5.7.7, about 400 lines, Xeon Gold 5215, Redis with YCSB; up to 10% throughput gain with extra memory. Motivation: number of page faults, user-kernel mode switches, and CPU TLB shootdowns. | Region-granular CoW in the kernel fault handler exists (finding 3, mechanism class). It is not motivated by secondary-MMU or IOMMU cost: the PDF contains no occurrence of IOMMU, notifier, GPU, or accelerator. The paper does not describe how TLB flushes are issued for a precopied region, so its behaviour on an SVA-bound mm cannot be inferred. It is in-kernel and predictive; ours is user-space copy plus one `mremap`. | **V (pdf)** | [DOI](https://doi.org/10.3390/electronics11030461), [PDF](https://mdpi-res.com/d_attachment/electronics/electronics-11-00461/article_deploy/electronics-11-00461-v2.pdf) |

The row for CCoW in Section 2, Group 3 can be changed from "M, content NV" to **V (pdf)** with the description above. The index text was accurate.

### Group D. 2025 to 2026 venues

| Work (venue, year) | What it measures or does | Overlap with findings 1-3 | Tag | URL |
|---|---|---|---|---|
| Ma et al. (Tsinghua), "Anchor: Mitigating GPU Shallow Disruptions with Decoupled Memory" (SOSP 2026) | A daemon owns GPU memory; a restarted worker remaps it through the GPU's IPC memory mechanism. Recovery time for LLM jobs in clusters falls by 60.7% (inference) and 26.6% (training). | None. Datacenter GPUs, device memory, crash recovery. No fork, CoW, SVA, or host page tables in the abstract. Related only to the umbrella theme "who owns GPU memory". | **V (abs)** (ACM DL returned 403; first paragraph from the OpenAlex record of the DOI, full abstract from a third-party index of SOSP'26 abstracts) | [DOI](https://doi.org/10.1145/3830418.3843892), [accepted list](https://sigops.org/s/conferences/sosp/2026/accepted.html), [abstract index](https://pchaigno.github.io/academic/2026/08/03/sosp-2026-papers.html) |
| Ren et al. (Tsinghua), "Efficient GPU Multitasking with Morphable Kernels" (MorphX, SOSP 2026) | Kernels yield or reclaim compute at run time; background throughput 2.24x at equal foreground latency. | None. Compute scheduling, no memory management. | **V (abs)** (same sources; artifact README read) | [DOI](https://doi.org/10.1145/3830418.3843904), [artifact](https://github.com/thustorage/MorphX) |
| Shirvani and Liu (UC Riverside), "Linux AGX: An Adaptive GPU eXtension to Linux Fair Scheduling for Physical AI and Robotic Systems" (SOSP 2026) | Adjusts EEVDF or CFS weights so that CPU steps that gate GPU launches run sooner; makespan falls 6.0 to 20.0% and GPU utilization rises 7.6 to 22.1% on three NVIDIA platforms. | None on findings 1 to 3. CPU scheduling only. Same application domain (single-GPU embedded NVIDIA platforms), so it is a likely neighbour in related work. The abstract does not name the platforms. | **V (abs)** (same sources) | [DOI](https://doi.org/10.1145/3830418.3843893) |
| Zhang et al. (ICT CAS), "MoonBright: A GPU Memory Allocator with Device-Side Page Table Materialization and Deferred TLB Coherence" (OSDI 2026) | On discrete NVIDIA and AMD GPUs, page-table construction and TLB shootdowns are "serialized through the host control path"; the vendor flush path takes 26.8 us on an A100. Builds page tables on the GPU and avoids shootdowns by always allocating fresh virtual addresses. | Partial, in framing only: a host-serialized, tens-of-microseconds TLB coherence cost per mapping operation, avoided by restructuring the operation. Different object: the GPU's own page tables and TLB, not the host page tables or the SMMU. No OS fork, CoW, IOMMU, or SVA (0 occurrences of IOMMU, SMMU, notifier, copy-on-write; "fork" appears only for beam-search state in the application). | **V (pdf)** | [PDF](https://www.usenix.org/system/files/osdi26-zhang-yangyu.pdf) |
| Holmes et al. (MIT), "Rethinking Process Snapshots for Near-Warm Serverless Cold Starts" (Spice, OSDI 2026) | SHELF snapshot format and a kernel `spliceVMA` that overlays sparse file ranges onto one VMA to restore a process without per-page work. | Partial with finding 3 as a pattern (replace per-page faults by a VMA-level operation). CPU only, restore not CoW, no secondary MMU. | **V (pdf)** | [PDF](https://www.usenix.org/system/files/osdi26-holmes.pdf) |
| Yu et al., "Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning" (OSDI 2026) | Memory ballooning across co-served LLMs on datacenter GPUs. | None. | **V (abs)** | [program](https://www.usenix.org/conference/osdi26/technical-sessions) |
| Su, "Execution-State Capsules" (arXiv, 2026) | Application-level snapshot, "fork", and rollback of an LLM session by copying a fixed buffer set, on aarch64 unified-memory devices. | None. The fork is a buffer copy in the runtime, not an OS fork. | **V (pdf)** | [arXiv 2606.20537](https://arxiv.org/abs/2606.20537) |
| Kang et al., "ZeroSwap: Minimizing Swap Overhead for Real-Time Multi-DNN Inference via SSD-based GPU Memory Extension" (RTAS 2026) | Title only. | Unknown. Swap for GPU memory on embedded platforms; fork, CoW, or SVA not indicated by the title. | **M** | [program](https://2026.rtas.org/program/) |
| Zhuo et al., "LAIKA: Machine Learning-Assisted In-Kernel APU Acceleration" (ASPLOS 2026) | Title only. | Unknown. | **M** | [program](https://www.asplos-conference.org/asplos2026/program/) |
| SOSP 2026 accepted list, OSDI 2026 program, EuroSys 2026 accepted papers, ASPLOS 2026 program, RTSS 2025 program, RTAS 2026 program, ESWEEK 2026 program (EMSOFT, CASES, CODES+ISSS) | Keyword scan of titles (and of abstracts where the page carries them, OSDI only) for: Jetson, Thor, MIG, unified memory, IOMMU, SMMU, SVA, shared virtual, fork, copy-on-write, CoW, page fault, page table, TLB, checkpoint, snapshot. | No title on Jetson Thor, MIG on unified memory, SVA with a GPU, or fork/CoW with an accelerator. Hits are the rows above, bpf_fault and Nixie (already in Sections 2 and 9), and one ESWEEK workshop talk ("Design, Implementation and Evaluation of a RISC-V IOMMU on gem5"). | **M** | [SOSP](https://sigops.org/s/conferences/sosp/2026/accepted.html), [OSDI](https://www.usenix.org/conference/osdi26/technical-sessions), [EuroSys](https://2026.eurosys.org/papers.html), [ASPLOS](https://www.asplos-conference.org/asplos2026/program/), [RTSS](https://2025.rtss.org/program/index.html), [RTAS](https://2026.rtas.org/program/), [ESWEEK](http://esweek.org/wp-content/uploads/2026/10/ESWEEK_2026_program_11.pdf) |
| USENIX ATC 2026 | No program found. `usenix.org/conference/atc26` serves the ATC '25 page, which links "this announcement about USENIX ATC". The announcement returned HTTP 403 and was not read. | n/a | **NV** | [page](https://www.usenix.org/conference/atc26) |

### Group E. KVM secondary MMU and other host-page-table GPUs

| Work (venue, year) | What it measures or does | Overlap with findings 1-3 | Tag | URL |
|---|---|---|---|---|
| I. Eidus, commit `828502d30073` "ksm: add mmu_notifier set_pte_at_notify()" (2009) | Adds `change_pte()` so that a CoW of a KSM page updates the KVM shadow entry directly "instead of flushing the shadow page table entry and then getting vmexit". No number. | The per-page secondary-MMU cost of CoW was recognised for KVM in 2009 and a dedicated callback was added. Qualitative precedent for finding 1. | **V (src)** | [commit](https://github.com/torvalds/linux/commit/828502d30073) |
| P. Bonzini, commit `997308f9ae72` "mmu_notifier: remove the .change_pte() callback" (v6.10) | States the callback "had no actual functionality" for over ten years because `invalidate_range_start()` had already zapped the secondary PTE. | The KVM-side optimisation for per-page CoW no longer exists. No number. | **V (src)** | [commit](https://github.com/torvalds/linux/commit/997308f9ae72) |
| KVM or secondary-MMU paper with a per-page number for notifier invalidation under fork, CoW, or KSM | Not found. | n/a | **NV** | see Queries tried |
| Linux `drivers/gpu/drm/drm_gpusvm.c` (master; Intel Xe GPU SVM, upstream since v6.15) | Mirrors CPU ranges into GPU page tables through HMM and interval notifiers; ranges are created on GPU fault and removed on a notifier UNMAP event. | Different design: the GPU has its own page tables and does not walk the host tables, so `arch_invalidate_secondary_tlbs` is not involved. The file does not mention fork or copy-on-write. No measurement. | **V (src)** | [file](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/drm_gpusvm.c) |
| Arm Mali, Apple AGX, Intel Level Zero or OpenCL USM shared: fork or CoW cost | Not opened. | n/a | **NV** | none |

### Verdict changes

**Finding 1 (per-page MM events cost about 8 us extra in an SVA-bound mm): PARTIALLY TAKEN.**
Closest work: J. Gunthorpe's replies in "Cache Invalidation Solution for Nested IOMMU" (linux-iommu, April 2023), which state that the SVA invalidation sits in a hot path of the mm and that fork, alloc/free, mmap and munmap become slower once SVA is enabled.
What remains claimable: the mechanism is existing, intended kernel behaviour since v6.6 and must be described as such; the measurement is ours. No opened source gives a per-page or per-GiB cost for CoW, fault, or THP split under SVA, a profile that attributes it to `arm_smmu_cmdq_issue_cmdlist`, or any such number for a GPU. Three citations are required next to the claim: the 2023 thread, the Popple series that placed the notifier in `flush_tlb_page()`, and Huang Ying's v6.19 patch for the CPU-TLB analogue on arm64. Border Control (MICRO 2015) reaches the opposite conclusion in simulation at far lower event rates and should be cited as the contrasting prior belief.

**Finding 2 (the cost is process-wide): PARTIALLY TAKEN.**
Closest work: the same 2023 thread ("it broadly impacts certain workloads once SVA is enabled").
What remains claimable: a measured demonstration on a GPU platform that memory the device never touched pays the cost. It is not a discovery about the kernel: the notifier is registered per mm, the callback receives no information about which ranges the device has used, and PowerPC has done the equivalent for coprocessor-attached mms for years (`mm->context.copros`).

**Finding 3 (range operations invalidate once; extent privatization replaces per-page CoW): PARTIALLY TAKEN, unchanged.**
Closest work in this pass: CCoW (Electronics 2022), now verified, for region-granular CoW; RUMA and AnKer from Section 2 remain closest for user-space copy-then-remap.
What remains claimable: the reason and the number. No opened work performs extent-granular privatization in order to turn N synchronous secondary-TLB invalidations into one, and none reports the ratio on an SVA-bound mm. That range operations cost one invalidation is kernel design (Popple cover letter; `CMDQ_MAX_TLBI_OPS` in commit `d5afb4b47e13`), not a finding. "Novel CoW mechanism" stays forbidden.

**New entry for Section 6 (threats).** MoonBright (OSDI 2026) is the closest 2026 systems paper in framing: host-serialized TLB coherence at tens of microseconds per mapping operation on a GPU, removed by restructuring the operation. It concerns the GPU's own page tables on discrete GPUs. It should be cited and distinguished.

### Kernel-version statement

Sentence we may write:

> The per-page invalidation path is not specific to Linux 6.8. In mainline Linux 7.3-rc6, a copy-on-write fault still handles one page (`wp_page_copy()`), its `ptep_clear_flush()` still reaches `mmu_notifier_arch_invalidate_secondary_tlbs()` through the arm64 `flush_tlb_page()`, and the SMMUv3 SVA callback still submits one synchronous TLBI batch, and one synchronous ATC batch when ATS is in use, per call. We measured only 6.8.12-tegra; the magnitude on a newer kernel is not measured.

Evidence, all read at commit `a90ee4305c4a` (v7.3-rc6):

- `mm/memory.c`: `wp_page_copy()` calls `ptep_clear_flush(vma, vmf->address, vmf->pte)` once per fault. `wp_page_reuse()` calls `ptep_set_access_flags(..., 1)`. `do_wp_page()` has no batched copy.
- `mm/huge_memory.c`: `do_huge_pmd_wp_page()` ends in `__split_huge_pmd()` and `VM_FAULT_FALLBACK` for a shared anonymous THP.
- `arch/arm64/include/asm/tlbflush.h`: `flush_tlb_page()` -> `__flush_tlb_page()` -> `__do_flush_tlb_range()`, which calls the notifier unless `TLBF_NONOTIFY` is set. `arch/arm64/include/asm/pgtable.h`: only `flush_tlb_fix_spurious_fault()` and its PMD variant pass `TLBF_NONOTIFY`. `arch/arm64/mm/fault.c`: `__ptep_set_access_flags_anysz()` flushes with `TLBF_NOWALKCACHE | TLBF_NOBROADCAST`, so the notifier is called.
- `drivers/iommu/arm/arm-smmu-v3/arm-smmu-v3-sva.c`: `arm_smmu_mm_arch_invalidate_secondary_tlbs()` -> `arm_smmu_domain_inv_range()`. `arm-smmu-v3.c`: `__arm_smmu_domain_inv_range()` submits at each `arm_smmu_invs_end_batch()` boundary, and `arm_smmu_cmdq_batch_submit()` always passes `sync = true`. `ARM_SMMU_FEAT_BTM` is only ever cleared.
- Intermediate tags v6.9, v6.10, v6.11, v6.17 were read for the SVA callback; v6.19 and v7.1 for the arm64 flush changes. None coalesces invalidations across faults.

Changes that alter the magnitude, in either direction:

- v6.19 (`cb1fa2e99955`) removes the CPU TLBI broadcast from reuse faults and leaves the notifier call in place. Only the CPU-side part of a reuse fault becomes cheaper.
- v7.3-rc1 (`06b15ddcfbc0`, erratum T264-SMMU-3) doubles every TLBI plus `CMD_SYNC` on Tegra264 when the SMMU node is `nvidia,tegra264-smmu`. The mainline `tegra264.dtsi` uses that compatible. A mainline kernel on this SoC would therefore issue three synchronous submissions per page when ATS is in use (two TLBI, one ATC) where upstream v6.8 issues two. We must not write "8 us is the cost of one SMMU invalidation" without knowing whether 6.8.12-tegra already applies an equivalent workaround.
- Not merged: the Hsu series (September 2026) would reduce the number of reuse faults for exclusive large folios; the copy path is explicitly unchanged. Gunthorpe's TLBI rework (v8, October 2026) keeps one invalidation per notifier call.

Forbidden wording: "fixed in newer kernels", "a 6.8 artefact", "newer kernels batch these invalidations". None is supported.

Two source-level observations that affect how finding 1 is worded:

- First touch. `do_anonymous_page()` contains no `flush_tlb_*` or `ptep_clear_flush()` call in v6.8 or in mainline, and arm64 defined `flush_tlb_fix_spurious_fault()` as empty before v6.19 (also in the Thor header). A write to a never-mapped anonymous page therefore does not reach the SMMU notifier according to the source. A first write to a page that was previously read (zero page) goes through `wp_page_copy()` and does. The first-touch surcharge should be attributed with the same `perf` method before it is listed under the same mechanism.
- Range size. Without SMMU range-invalidation support, a range of 512 pages or more becomes one ASID-wide TLBI; a smaller range becomes up to 511 per-page TLBI commands inside one notifier call. "Issues the invalidation once" is exact for multi-MiB extents and should be worded as "one notifier call" in general.

### Not opened

- Tegra 6.8.12 `drivers/iommu/arm/arm-smmu-v3/*.c` (not installed on the device). NVIDIA module tag 595.78 (not published; 595.71.05 and 595.80 read).
- ACM DL full texts of Anchor, MorphX, and Linux AGX (HTTP 403). No arXiv version found by title.
- USENIX ATC announcement (HTTP 403).
- Amit, Ben-Yehuda, Yassour, "IOMMU: Strategies for Mitigating the IOTLB Bottleneck" (WIOSCA 2010): two author URLs returned 404, HAL returned HTML.
- Farshin et al., "Characterizing IOTLB Wall for Multi-100-Gbps Linux-based Networking" (PeerJ CS 2023): HTTP 403.
- Optimus (ASPLOS 2020), Coyote v2, Vogel et al. (IEEE TC 2018, configurable IOMMU for FPGA SVM), HATRIC (ISCA 2017): not attempted in this pass.
- ZeroSwap (RTAS 2026) and LAIKA (ASPLOS 2026): titles only.
- Linaro, Huawei, or LPC talk slides on UACCE and SVA performance: not searched as primary documents. The only UACCE material read is the 2023 thread, where Z. Gao reports that guest SVA with huge pages matches host throughput for DMA bandwidth.
- EuroSys 2026, ASPLOS 2026, RTSS 2025, RTAS 2026, ESWEEK 2026: titles scanned, no abstracts or papers opened.

### Queries tried

lore.kernel.org (public-inbox search, lists `all`, `linux-iommu`, `linux-mm`, `linux-arm-kernel`):

- `"arm_smmu_mm_arch_invalidate_secondary_tlbs" AND (perf OR slow OR overhead OR latency OR bottleneck)`
- `arch_invalidate_secondary_tlbs AND (overhead OR slow OR performance) AND (fork OR cow OR "copy-on-write")`
- `"secondary TLB" AND (batch OR coalesce OR defer) AND (SVA OR smmu) AND (fork OR cow OR "page fault")`
- `SVA AND (fork OR "copy-on-write" OR cow) AND (slow OR overhead OR latency OR performance) AND invalidat`
- `"mmu_notifier_arch_invalidate_secondary_tlbs" AND (batch OR overhead OR performance OR expensive)`
- `"arm_smmu_cmdq_issue_cmdlist" AND ("do_wp_page" OR "wp_page_copy" OR "ptep_clear_flush" OR "handle_mm_fault" OR "copy_page_range" OR "dup_mmap")` (0 results)
- `"arm_smmu_cmdq_issue_cmdlist" AND (SVA OR "arm_smmu_mm") AND (perf OR slow OR lockup OR contention)` (0 results)
- `(uacce OR "hisi_zip" OR hisilicon) AND SVA AND (invalidat OR tlb) AND (performance OR slow OR overhead)`
- `s:"CMDQ fix and improvement"`, `s:"Batch IOTLB/dev-IOTLB invalidation"`, `s:tegra264 AND (repeat OR erratum OR TLBI)`
- `s:(cow OR "copy-on-write" OR "write fault" OR wp) AND s:(batch OR batching OR "large folio" OR mTHP OR multiple)` since 2023-06
- `dfn:mm/memory.c AND s:(cow OR wp OR "write fault") AND s:(batch OR "large folio" OR folios)` since 2024

patchwork.kernel.org API: covers and patches matching `secondary TLB`, `arch_invalidate_secondary_tlbs`, `SVA`, `SMMU v3 CMDQ`, `Hook up ATC invalidation to mm ops`.

arXiv API: `all:"shared virtual addressing"`; `abs:IOMMU AND abs:"shared virtual" AND (abs:accelerator OR abs:GPU)`; `abs:"IOTLB" AND (abs:invalidation OR abs:"page fault") AND abs:accelerator`; `abs:"mmu notifier"`; `(abs:IOMMU OR abs:SMMU) AND (abs:"TLB invalidation" OR abs:"IOTLB invalidation" OR abs:shootdown)`; `abs:"host page table" AND abs:GPU`; `(abs:"copy-on-write" OR abs:"fork()") AND abs:GPU AND cat:cs.OS`; `(abs:"TLB shootdown" OR abs:"TLB invalidation") AND (abs:"copy-on-write" OR abs:fork) AND cat:cs.OS`; `abs:"Jetson AGX Thor" OR abs:"Jetson Thor"`; `abs:Jetson AND (abs:"copy-on-write" OR abs:fork OR abs:"page table" OR abs:IOMMU OR abs:SMMU OR abs:"shared virtual")`; `abs:"Grace Hopper" AND (abs:"system-allocated" OR abs:"address translation" OR abs:"page table" OR abs:ATS)`; `"Address Translation Services" AND GPU`; titles `"Shallow Disruptions"`, `"Morphable Kernels"`, `"Linux AGX"`, `ZeroSwap`, `LAIKA`.

OpenAlex: "shared virtual addressing IOMMU PASID accelerator overhead"; "IOTLB invalidation shared virtual memory accelerator page fault fork"; "mmu notifier invalidation overhead copy-on-write secondary MMU"; "SMMU shared virtual address accelerator Kunpeng UACCE performance"; "KVM MMU notifier invalidation cost copy-on-write KSM fork guest secondary page table"; "extended page table invalidation overhead host copy-on-write KSM unsharing latency virtual machine"; "integrated GPU shared virtual memory fork copy-on-write OpenCL unified shared memory overhead"; "Apple silicon unified memory GPU page table fork copy-on-write".

Semantic Scholar API: three queries, all returned HTTP 429.

Web search (discovery only): "shared virtual addressing SVA IOMMU overhead fork copy-on-write mmu notifier secondary TLB invalidation cost measurement paper"; "arch_invalidate_secondary_tlbs performance overhead SVA per-page invalidation fork"; "arm_smmu_cmdq_issue_cmdlist SVA performance slow page fault munmap fork"; the three SOSP 2026 titles.

## 12. Addendum: huge-page transfer, sealed copy-on-write sharing, GPU read faults, and post-fork repair

Provenance: produced by a third delegated search pass on 2026-10-05, after
the huge-page, sharing, and fork-repair results existed. The brief asked for
work that takes four claims, C1 to C4. The text is the pass's own. Rows
tagged V (doc) were read through a fetch tool and must be re-opened in a
browser before a sentence that depends on them is submitted; this applies in
particular to the 2026 issue threads and to the arXiv paper that are the
closest works for C2 and C3.

Audit date: 2026-10-05. Scope: claims C1 to C4 only. Works listed as already audited in the request were not reopened.

Method. Every row was opened as a primary source: a PDF downloaded and converted with `pdftotext`, a source file or commit fetched from `raw.githubusercontent.com` or the GitHub commit API, or a documentation, mailing-list, LWN, or issue page fetched directly. Search results were used only to locate URLs. Line numbers refer to the files as fetched at the stated tag or commit.

Tags: **V (pdf)** PDF downloaded and passage read; **V (src)** source file or commit read; **V (doc)** official doc, mailing-list, LWN, or issue page opened through a fetch tool; **V (abs)** abstract only; **M** metadata only; **NV** not verified.

Driver version note. The installed module reports `version: 595.78` (`modinfo nvidia_uvm`). The public repository has no tag 595.78. The nearest public tags are 595.71.05 and 595.80; all UVM files quoted below are byte-identical between these two tags (`cmp`). The installed `nvidia-uvm.ko` contains the symbols `uvm_ats_service_faults`, `uvm_perf_prefetch_compute_ats`, and `uvm_populate_pageable_vma`. The Jetson build of the source was not available, so the match to the running binary is by version bracket and symbol names, not by source identity.

Kernel version note. Linux source was read at the upstream tag v6.8. The running kernel is 6.8.12-tegra with vendor patches (for example `CONFIG_TEGRA241_CMDQV=y`, which is not in upstream v6.8). The vendor kernel tree (nv-tegra.nvidia.com) could not be reached, so statements about v6.8 are statements about upstream v6.8.

### 12.1 C1: huge-page backing and the cost of revoke-before-grant transfer

| Work (venue, year) | What it does or states | Overlap with the claim | Tag | URL |
|---|---|---|---|---|
| Druschel and Peterson, fbufs (SOSP 1993) | Cross-domain buffer transfer by page remapping plus shared mappings. Non-volatile fbufs need "two physical page table updates per page: one to remove write permission from the originator when the fbuf is transferred, and one to return write permissions to the originator after the fbuf was freed". Table 1 gives incremental per-page costs: 3 (cached, volatile), 21 (volatile), 29 (cached), 37 (plain), 144 (Mach COW), 316 (copy), in microseconds. | Changes permissions per transfer and quantifies the per-page cost of revocation. It removes the cost by dropping revocation ("volatile") and by caching mappings. One page size only. No large pages. No GPU memory. | V (pdf) | https://www.cs.princeton.edu/courses/archive/fall11/cos518/papers/fbufs.pdf |
| Chu, Zero-Copy TCP in Solaris (USENIX ATC 1996) | Transmit side sets a COW protection on the user buffer per send; receive side flips pages. "COW faults are expensive, so is setting up a COW protection on a user buffer, and tearing it down later." "it takes four cross-calls to remap two 8K buffers, averaging 15 µs". | Per-transfer protection change with a measured per-operation cost. Page size appears only in a remark on cache colours ("With a MMU page size of 4KB, there are 64 colors"). No large-page experiment. No GPU memory. | V (pdf) | https://www.usenix.org/legacy/publications/library/proceedings/sd96/full_papers/chu.ps |
| Pai, Druschel, Zwaenepoel, IO-Lite (OSDI 1999) | Immutable buffers shared read-only across domains. "IO-Lite's worst case cross-domain transfer overhead is that of page remapping". | Grants read access per transfer, avoids revocation by immutability. No page-size study. No GPU memory. | V (pdf) | https://www.usenix.org/legacy/events/osdi99/full_papers/pai/pai.pdf |
| Appel and Li, Virtual Memory Primitives for User Programs (ASPLOS 1991; TR CS-TR-276-90 read) | "When small pages are used, it is particularly important to trap and change page protections quickly, since this overhead is independent of page size while the actual computation (typically) takes time proportional to page size." | States the premise behind C1: the protection-change overhead is per page and independent of page size. The paper argues for smaller pages. In-process use, no transfer between domains. | V (pdf) | https://www.cs.princeton.edu/~appel/papers/vmpup.pdf |
| Tene, Iyengar, Wolf, C4 (ISMM 2011) | Lists limits of stock Linux remapping: "Each page remap includes an implicit TLB invalidate operation", "Only small (4KB on X86-64) page mappings can be remapped". Their kernel subsystem "supports explicit and mixed mapping and remapping of large (2MB on X86-64) pages" and protection changes "without requiring TLB invalidation". Table 1: sustainable remap rate 3.04 GB/s (Linux) against 6.50 TB/s (modified) with one active thread. | Closest verified work. Uses 2 MiB mappings and batched invalidation specifically to cut the cost of remap and protection changes. In-process garbage collection on x86. No cross-domain transfer, no accelerator memory, no shmem or hugetlbfs comparison. | V (pdf) | https://www.azul.com/files/c4_paper_acm.pdf |
| Du et al., XPC (ISCA 2019) | "Adopting page remapping for ownership transfer can mitigate the above security problem, but the remapping operation still requires kernel's involvement. Meanwhile, remapping may also lead to costly TLB shootdown." Relay segments are translated by a register "instead of page tables", have one owner at a time, and need "no TLB shootdown". | Same goal as revoke-before-grant (exclusive ownership moves, no copy). Solves the cost with a hardware segment register, not with larger pages. No GPU memory. | V (pdf) | https://ipads.se.sjtu.edu.cn/_media/publications/xpc-isca19.pdf |
| Stamler et al., zIO (OSDI 2022) | Section 3.4 "Huge pages": tracking "requires fine-grained page protection"; zIO's requests "force the OS to break huge pages into base page mappings"; with hugetlbfs "fine-grained page protection is disallowed and zIO can only track buffers at huge page granularity". | Uses page protection per buffer. Treats huge pages as an obstacle to protection granularity, not as a way to reduce protection cost. | V (pdf) | https://www.usenix.org/system/files/osdi22-stamler.pdf |
| Park et al., libmpk (USENIX ATC 2019) | "the overhead of mprotect() increases in proportion to the number of pages" (Figure 3). `mpk_mprotect()` is 1.73x faster for one page and 3.77x for 1,000 pages. Page counts are in 4 KB pages; no larger page size is evaluated. | Quantifies page-count dependence of `mprotect`. The remedy is protection keys. No larger-page experiment. | V (pdf) | https://www.usenix.org/system/files/atc19-park-soyeon.pdf |
| Vahldiek-Oberwagner et al., ERIM (USENIX Security 2019) | MPK domain switches. `mprotect` appears only for executable-page interception. The word "huge" does not occur in the text. | No per-transfer permission change on data pages, no page-size analysis. | V (pdf) | https://www.usenix.org/system/files/sec19-vahldiek-oberwagner_0.pdf |
| Hedayati et al., Hodor (USENIX ATC 2019) | "we use huge pages to reduce the first-level (identity-function) page tables from four levels to two, eliminating half the extra cost of a VMX TLB fill". Mentions DPDK processes that "share DPDK huge-pages" when mutually trusting. | Huge pages reduce TLB fill cost under VMFUNC, not permission-change cost. | V (pdf) | https://www.usenix.org/system/files/atc19-hedayati-hodor.pdf |
| Litton et al., Light-Weight Contexts (OSDI 2016) | Context switch between address-space views. The strings "huge", "mprotect", "remap" do not occur in the text. | No transfer by permission change, no page-size analysis. | V (pdf) | https://www.usenix.org/system/files/conference/osdi16/osdi16-litton.pdf |
| Zhang et al., Demikernel (SOSP 2021) | Huge pages appear only as the DPDK/SPDK DMA pool ("we allocate 2 GB of 2 MB huge pages, as required by DPDK"). The strings "mprotect" and "remap" do not occur. | Zero-copy without per-transfer permission change. | V (pdf) | https://irenezhang.net/papers/demikernel-sosp21.pdf |
| Sartakov et al., CAP-VMs (OSDI 2022) | Components "must either copy data or modify page tables, both of which are expensive operations"; CAP-VMs share through CHERI capabilities without page-table changes. | Avoids the page-table cost by capabilities. The word "huge" does not occur. | V (pdf) | https://www.usenix.org/system/files/osdi22-sartakov.pdf |
| Linux commit 2c91bd4a4e2e, "mm: speed up mremap by 20x on large regions" (2019) | "The bottleneck is move_page_tables, which is copying each pte at a time". Moving at PMD level: "On a 1GB mremap, the mremap completion times drops from 3.4-3.6 milliseconds to 144-160 microseconds." | Kernel statement and measurement that remap cost follows page-table granularity. Not about transfer between domains. | V (src) | https://github.com/torvalds/linux/commit/2c91bd4a4e2e |
| Linux series "Optimize mprotect() for large folios", commits b9bf6c2872c5 and cac1db8c3aad (Dev Jain, 2025-07-18) | `mprotect` read-only then read-write 40 times over 1 GiB on arm64 (Apple M3): before 2.1 s (PTE-mapped THP), 2 s (64K mTHP), 1 s (4K); after 0.65 s, 0.7 s, 1.1 s. | Measures `mprotect` cost against folio size on arm64. CPU only. | V (src) | https://github.com/torvalds/linux/commit/b9bf6c2872c5 |
| Linux commit 64fe24a3e05e, "mm/mprotect: try avoiding write faults for exclusive anonymous pages when changing protection" (2022) | Benchmark `mprotect(PROT_READ)` then `mprotect(PROT_READ|PROT_WRITE)` then write over 1 GiB, 40 loops: 6.398 s stock, 3.780 s patched. "This commit doesn't add the same handling for PMDs". | Shows that the cost of a revoke/grant cycle is dominated by the per-page write faults that follow the grant. | V (src) | https://github.com/torvalds/linux/commit/64fe24a3e05e |
| Linux v6.8 `mm/hugetlb.c`, `hugetlb_change_protection()` lines 6856 to 6905 | One `mmu_notifier_range` for the call, then `for (; address < end; address += psize)` over huge-page entries. | Source-level reason for the hugetlbfs result: one entry per 2 MiB. | V (src) | https://github.com/torvalds/linux/blob/v6.8/mm/hugetlb.c#L6856 |
| NVIDIA open-gpu-kernel-modules 595.80, `uvm_ats_faults.c` lines 553 to 566 | "The GPU will re-fetch an entry on access if the PTE is invalid and the page size is not 4K, but if the page size is 4K no re-fetch will happen ... so use the hammer of always invalidating the GPU's TLB on each fault." | The driver adds a GPU TLB invalidation to every serviced ATS fault on 4 KiB-base-page kernels. This is about the kernel base page size, not about huge mappings. | V (src) | https://github.com/NVIDIA/open-gpu-kernel-modules/blob/595.80/kernel-open/nvidia-uvm/uvm_ats_faults.c |
| Corbet, "Zero-copy TCP receive" (LWN, 2018) | "It has long been conventional wisdom in the kernel community that zero-copy schemes dependent on memory-mapping tricks will struggle to outperform implementations that simply copy the data. There is quite a bit of overhead involved in setting up and tearing down these mappings." | Per-transfer mapping cost, per-page granularity. Huge pages are not mentioned. | V (doc) | https://lwn.net/Articles/752188/ |
| vmsplice(2), `SPLICE_F_GIFT` | "The user pages are a gift to the kernel. The application may not modify this memory ever". Data "must also be properly page aligned". | Ownership transfer by convention, no permission enforcement. | V (doc) | https://man7.org/linux/man-pages/man2/vmsplice.2.html |
| DPDK Programmer's Guide, Multi-process Support | Processes share hugepage memory with identical mappings. Secondary processes "MUST have equivalent permissions and trust level." | Hugepage shared memory without per-transfer permission change. | V (doc) | https://doc.dpdk.org/guides/prog_guide/multi_proc_support.html |
| AOSP, BufferQueue and Gralloc | "Buffer contents are never copied by BufferQueue ... buffers are always passed by a handle." No statement on page permissions or huge pages. | GPU-visible buffer handoff by handle, no revocation. Read through a fetch tool summary. | V (doc) | https://source.android.com/docs/core/graphics/arch-bq-gralloc |
| CUDA 12.2 release notes (2023) | HMM limitation: "HugeTLBfs pages are not yet supported on HMM (this is an uncommon scenario)." | Vendor position on huge pages for GPU-visible system memory on the x86 HMM path. No statement for the ATS path. | V (doc) | https://docs.nvidia.com/cuda/archive/12.2.0/cuda-toolkit-release-notes/index.html |

Finding for C1. Every transfer system in the verified set that changes permissions per transfer reports a per-page cost (fbufs, Solaris, libmpk) and none of them uses larger pages to reduce it. zIO states the opposite relation (huge pages conflict with its protection granularity). The use of 2 MiB mappings to cut remap and protection cost is established outside IPC: C4 does it for garbage collection, and Linux has measured PMD-level `mremap` (2019) and large-folio `mprotect` batching (2025). No verified work measures the page-size dependence of permission transfer for GPU-visible or SVA-bound memory.

### 12.2 C2: copy-on-write sharing of GPU-visible state across GPU partitions, with a sealed base

| Work (venue, year) | What it does or states | Overlap with the claim | Tag | URL |
|---|---|---|---|---|
| Si, Lin, Li, Zhang, "The Ingestion Tax: Adopting File-Backed Weights in Tensor Frameworks" (arXiv 2608.12114v2, 2026-08-30) | Maps each tensor file `MAP_SHARED`, wraps the pages as a no-copy GPU buffer, imports through DLPack. "N processes decode from one mapped copy where resident loading creates N copies". Platforms: Apple M5 Max (Metal), AMD APU (Vulkan), NVIDIA GH200 ("File-backed mappings and pinned Grace allocations fall in the direct-read subset of the link-bound class"). "A private mapping is converted to anonymous pages when Metal wires it ... a 60 GB private mapping produced 58 GB of anonymous memory". "a GPU store to the readonly Metal mapping is silently discarded, the CUDA path faults". | Closest verified work. GPU reads file pages in place and N processes share one physical copy through the host page cache, including on an NVIDIA coherent-memory system. Sharing is read-only and `MAP_SHARED`. No private divergence by copy-on-write, no hardware GPU partitions, no memfd seals. It reports that a private mapping loses sharing on Apple. | V (pdf) | https://arxiv.org/pdf/2608.12114 |
| llama.cpp, commit 2ed93db (2026-10-05) | `src/llama-mmap.cpp` lines 485 and 495: `int flags = MAP_SHARED;` and `mmap(NULL, file->size(), PROT_READ, flags, fd, 0)`. `ggml/src/ggml-metal/ggml-metal-device.m` line 2212: `newBufferWithBytesNoCopy:ptr length:size_aligned options:MTLResourceStorageModeShared`. `ggml/src/ggml-cuda/ggml-cuda.cu` lines 144 to 166 allocate with `cudaMalloc` (or `cudaMallocManaged` under `GGML_CUDA_ENABLE_UNIFIED_MEMORY`), line 783 copies with `cudaMemcpyAsync(... cudaMemcpyHostToDevice ...)`, line 5225 sets `buffer_from_host_ptr = false`. | On Metal the GPU reads the read-only shared file mapping in place, so processes share weights through the page cache. On CUDA, including Jetson, weights are copied into CUDA allocations. No copy-on-write, no partitions, no seals. | V (src) | https://github.com/ggml-org/llama.cpp/tree/2ed93db472c267e1ef4e4570df74287d96b1a135 |
| llama.cpp Discussion 21223, "Share readonly GPU model weights across multiple llama-* processes" (2026-03-31) | Maintainer: "The mmap weight sharing works on Apple Silicon with GPU thanks to the unified memory" and "the CUDA backend would make a copy into device buffers." | Confirms the state of practice: in-place sharing on Apple, none on CUDA. Read through a fetch tool summary. | V (doc) | https://github.com/ggml-org/llama.cpp/discussions/21223 |
| pontostroy/cuda-llm-weight-share (README) | `LD_PRELOAD` library that intercepts `cudaMalloc()` and shares the weight allocation between processes with `cudaIpcGetMemHandle()` and `cudaIpcOpenMemHandle()`. | Shares weights across CUDA processes at the runtime level. No MMU enforcement, no copy-on-write, no immutability enforcement. | V (doc) | https://github.com/pontostroy/cuda-llm-weight-share |
| geistlib issues 541 and 526 (2026-09-30, 2026-10-01) | "`gguf_open` maps the model with `mmap(PROT_READ, MAP_PRIVATE)` ... When a command buffer first makes such a wrapper resident, Metal wires its pages for write. On a private mapping that is copy-on-write: every page of every GPU-bound weight becomes a private anonymous copy." Measured 2918 MiB copied on one model; zero with `MAP_SHARED`. | Reports for Apple the failure mode that C3 reports for NVIDIA: a private mapping loses sharing when the GPU reads it. The fix there is to give up the private mapping. | V (doc) | https://github.com/geisten/geistlib/issues/541 |
| huggingface/candle PR 3785 (opened 2026-07-26) | Adds `new_buffer_with_bytes_no_copy` so that an mmap'ed GGUF is viewed as a Metal buffer. | In-place GPU read of mmap'ed weights on Apple. No cross-process or copy-on-write statement. | V (doc) | https://github.com/huggingface/candle/pull/3785 |
| ml-explore/mlx Discussion 615 (2024 to 2025) | Maintainer (2024-02-03): "We don't mmap the model" and "we can't mmap memory in a way that makes it available for the GPU as well." | Negative data point for MLX at that date. Read through a fetch tool summary. | V (doc) | https://github.com/ml-explore/mlx/discussions/615 |
| Dakkak et al., TrIMS (arXiv 1811.09732, 2018) | "layer weights are constant and can be shared across processes"; "TrIMS leverages the CUDA runtime's cudaIpc* to share GPU" memory between framework processes. | Weight sharing across GPU processes by CUDA IPC with a model manager. Runtime-enforced, no copy-on-write, discrete GPUs. | V (pdf) | https://arxiv.org/pdf/1811.09732 |
| safetensors README | "On CPU, if the file is already in cache, then it can truly be zero-copy, whereas on GPU there is not such disk cache, so a copy is always required". | States that GPU loading always copies. | V (doc) | https://github.com/huggingface/safetensors/blob/main/README.md |
| Triton Inference Server, shared-memory extension | System and CUDA shared memory are for input and output tensors; CUDA shared memory uses a `cudaIPC` handle. "On Jetson, only system shared memory is supported". | Not a weight-sharing mechanism. | V (doc) | https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_shared_memory.md |
| TensorFlow Lite v2.16.1, `tensorflow/lite/mmap_allocation.cc` line 103 | `mmap(nullptr, length + offset_in_buffer_, PROT_READ, MAP_SHARED, ...)`. | Model flatbuffer is a read-only shared mapping, shareable across processes on the CPU side. GPU delegate behaviour not opened. | V (src) | https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/lite/mmap_allocation.cc#L103 |
| CUDA 12.2 release notes and NVIDIA HMM blog (2023) | HMM lets GPU kernels read `mmap`'ed files in place. Limitations: "The fork() system call is not fully supported yet when attempting to share GPU-accessible memory between parent and child processes"; blog: "fork(2) without a following exec(3) is not fully supported." | Vendor documents in-place GPU access to file mappings and states that fork-based sharing of GPU-accessible memory is not fully supported. No statement on copy-on-write across processes. | V (doc) | https://docs.nvidia.com/cuda/archive/12.2.0/cuda-toolkit-release-notes/index.html ; https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management/ |
| CUDA for Tegra application note (current) | On Thor, `cudaDeviceProp::pageableMemoryAccess` is 1 and system allocations "created via mmap() or malloc() ... can be accessed directly on the GPU without calling any CUDA APIs to register memory". CUDA IPC on Thor starts with CUDA 13.0. The strings "fork" and "copy-on-write" do not occur. | Documents the platform capability that C2 relies on. No statement on sharing between processes, MIG instances, or private mappings. | V (doc) | https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html |
| Jetson Linux Developer Guide r39.2.1, Multi-Instance GPU | MIG "partitions GPU resources so that multiple workloads can run concurrently with hardware-level isolation". Profiles are listed with 0.00 memory. | No statement on memory shared between instances or on how host memory is isolated. | V (doc) | https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/MiG.html |
| Linux v6.8 `drivers/dma-buf/udmabuf.c` lines 194 to 195 and 245 to 251 | `#define SEALS_WANTED (F_SEAL_SHRINK)` and `#define SEALS_DENIED (F_SEAL_WRITE)`; creation fails unless the memfd has the first and lacks the second. | Kernel precedent for memfd seals as the contract for memory handed to GPU drivers. The seal policy is the opposite of a write-sealed base: udmabuf refuses write-sealed memfds. | V (src) | https://github.com/torvalds/linux/blob/v6.8/drivers/dma-buf/udmabuf.c#L194 |
| fcntl `F_ADD_SEALS` man page | `F_SEAL_WRITE`: "trying to create new shared, writable memory-mappings via mmap(2) will also fail with EPERM"; adding the seal "fails with EBUSY if any writable, shared mapping exists". | Defines what the planned seal enforces. Private writable mappings are not excluded by this text. | V (doc) | https://man7.org/linux/man-pages/man2/F_GET_SEALS.2const.html |
| Linux commit 5535be309971, "mm/gup: fix FOLL_FORCE COW security issue and remove FOLL_COW" (2022) | Describes CVE-2022-2590: a CoW-path bug let unprivileged user space "modify tmpfs/shmem file content ... and to bypass memfd write sealing". | Shows that write sealing of shmem has been bypassed through a copy-on-write path before. Relevant to a threat model that relies on seals plus private mappings. | V (src) | https://github.com/torvalds/linux/commit/5535be309971 |
| MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark issue 81 (2026-09-28) | "`torch.from_file(shared=False)` is not an alternative: a private writable mapping copies every page the GPU touches into anonymous memory." Platform: GB10, kernel 6.17.0-1031-nvidia, driver 580.173.02. | States for an NVIDIA unified-memory system that private writable mappings lose sharing when the GPU touches them. See C3. | V (doc) | https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/issues/81 |
| Linux v6.8 `Documentation/arch/x86/sva.rst` lines 128 to 129 | "On fork(2) or exec(2) the PASID is removed from the process as it no longer has the same address space that it had when the device was opened." | Kernel documentation of SVA binding across fork (x86). Relevant to fork-based sharing designs. | V (src) | https://github.com/torvalds/linux/blob/v6.8/Documentation/arch/x86/sva.rst |

Finding for C2. Read-only sharing of GPU-read weights across processes through the host page cache, with the GPU reading the mapped pages in place, exists (llama.cpp on Metal; The Ingestion Tax on Apple, an AMD APU, and GH200). Runtime-level sharing by CUDA IPC exists (TrIMS, cuda-llm-weight-share). Three independent sources report that a private mapping loses sharing when the GPU reads it (geistlib 541 and The Ingestion Tax for Apple, MiaAI-Lab 81 for NVIDIA GB10), and all three respond by abandoning the private mapping. No verified work keeps a private, writable mapping shared under GPU reads, lets GPU writes diverge by copy-on-write, does so across hardware GPU partitions, or seals the base.

### 12.3 C3: GPU read faults served with write intent

| Work (venue, year) | What it does or states | Overlap with the claim | Tag | URL |
|---|---|---|---|---|
| MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark issue 81 (2026-09-28) | "the NVIDIA UVM driver services every GPU fault in a writable mapping as a write fault. In driver 580.173.02, `uvm_ats_faults.c` lines 496 to 497 add the faulted pages to the write mask when the mapping allows writes, and lines 632 to 633 service them as writes." Remedy: `mprotect(PROT_READ)`, after which "UVM then services the GPU's faults as read faults." Also: "a private writable mapping copies every page the GPU touches into anonymous memory." | Takes the first half of C3 in public, one week before this audit, on DGX Spark (GB10), kernel 6.17, for a shared file mapping. It does not state the condition under which sharing survives (CPU read of every page first), does not discuss `MADV_POPULATE_READ`, the access flag, or MIG. The cited line numbers match tag 580.65.06, which was read here. | V (doc) | https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/issues/81 |
| NVIDIA open-gpu-kernel-modules, `uvm_ats_faults.c` at 595.80 (identical at 595.71.05; same logic at 550.54.14, 565.57.01, 570.86.16, 580.65.06, 590.48.01, 610.57.04, 615.71.09) | See the next subsection. | The source that the claim describes. | V (src) | https://github.com/NVIDIA/open-gpu-kernel-modules/blob/595.80/kernel-open/nvidia-uvm/uvm_ats_faults.c |
| NVIDIA open-gpu-kernel-modules, `uvm_populate_pageable.c` at 595.80 lines 141 to 148 | Driver comment: "Kernel v6.6 introduced a bug in set_pte_range() around the handling of the ARM AF bit. Instead of setting the AF bit, the bit is incorrectly being cleared in set_pte_range() during first-touch fault handling. Calling handle_mm_fault() again takes a different code path which correctly sets the AF bit." | Vendor statement in source that first-touch faults on v6.6 and later leave the access flag clear on arm64 and that the GPU path needs it set. | V (src) | https://github.com/NVIDIA/open-gpu-kernel-modules/blob/595.80/kernel-open/nvidia-uvm/uvm_populate_pageable.c |
| Linux commit 4cd7ba16a0af, "mm: fix old/young bit handling in the faulting path" (Ram Tummala, NVIDIA, 2024-07-09) | "The polarity of prefault calculation is incorrect. This leads to prefault being incorrectly set for the faulting address. The following check will incorrectly mark the PTE old rather than young." Fixes 3bd786f76de2. | Upstream fix for the polarity that upstream v6.8 still has. Whether the Jetson 6.8.12 kernel carries it was not verified. | V (src) | https://github.com/torvalds/linux/commit/4cd7ba16a0af |
| Brucker, "[PATCH v3 04/12] iommu/arm-smmu-v3: Add support for Hardware Translation Table Update" (linux-arm-kernel, 2021-04-13, reposted by Keqian Zhu) | "If the SMMU supports it and the kernel was built with HTTU support, enable hardware update of access and dirty flags. This is essential for shared page tables, to reduce the number of access faults on the fault queue." The diff adds `CTXDESC_CD_0_TCR_HA | CTXDESC_CD_0_TCR_HD` to `arm_smmu_alloc_shared_cd()`. | States that SVA devices take access faults on shared page tables unless HA is enabled. This form was not merged; upstream v6.8 has no HTTU code. | V (doc) | https://lists.infradead.org/pipermail/linux-arm-kernel/2021-April/651829.html |
| Linux commit 74fa4c177ad0, "iommu/arm-smmu-v3-sva: Enable Hardware Access and Hardware Dirty bits" (Nicolin Chen, NVIDIA, 2026-05-03; ancestor of v7.2, not of v7.1) and its LKML thread | "SMMU has the same HTTU feature, yet it is not enabled in the SVA CD. As a result, SMMU will not clear the PTE_RDONLY bit while sharing the CPU page table, resulting in unnecessary stalls. Thus, enable CTXDESC_CD_0_TCR_HA and CTXDESC_CD_0_TCR_HD in the SVA CD." | First upstream kernel in which SVA context descriptors enable hardware access and dirty updates. On such a kernel with an HTTU-capable SMMU, a GPU read of an old PTE should not fault (inference from the commit text, not measured). | V (src), V (doc) | https://github.com/torvalds/linux/commit/74fa4c177ad0 ; https://lkml.iu.edu/2605.1/01004.html |
| geistlib issue 541 (2026-10-01); The Ingestion Tax (arXiv 2608.12114) | Apple Metal "wires its pages for write. On a private mapping that is copy-on-write". | Same effect on a different vendor stack, with a different mechanism. | V (doc), V (pdf) | see 12.2 |
| CUDA for Tegra application note | Describes pageable memory access on Thor. No statement on fault access type, private mappings, or the access flag. | No overlap. | V (doc) | https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html |
| NVIDIA Developer Forums, "Very slow mmap on DGX Spark that affects model loading" (2025-11) | NVIDIA staff attribute slow `mmap` loading to lazy page faults and suggest read-ahead tuning. Read through a fetch tool summary. | Nothing on write intent or copy-on-write. | V (doc) | https://forums.developer.nvidia.com/t/very-slow-mmap-on-dgx-spark-that-affects-model-loading-questions-to-nvidia/349886 |
| Jetson 6.8.12-tegra kernel source (`mm/memory.c`, `arm-smmu-v3`) | Not reachable (nv-tegra.nvidia.com timed out). | Needed to confirm that the running kernel matches upstream v6.8 in `set_pte_range()` and in the SVA context descriptor. | NV | https://nv-tegra.nvidia.com/ |

### What the driver and kernel source say

All UVM quotes are from tag 595.80 of github.com/NVIDIA/open-gpu-kernel-modules, directory `kernel-open/nvidia-uvm/`. All Linux quotes are from upstream tag v6.8 unless another tag is named.

1. Read and write faults are recorded in separate masks. `uvm_gpu_replayable_faults.c`, lines 1841 to 1846:

```c
        if ((access_type == UVM_FAULT_ACCESS_TYPE_READ) ||
            uvm_fault_access_type_mask_test(current_entry->access_type_mask, UVM_FAULT_ACCESS_TYPE_READ))
            uvm_page_mask_set(read_fault_mask, page_index);

        if (access_type >= UVM_FAULT_ACCESS_TYPE_WRITE)
            uvm_page_mask_set(write_fault_mask, page_index);
```

2. The prefetch step adds the prefetch mask to the write mask when the VMA is writable. `uvm_ats_faults.c`, lines 467 to 468 and 490 to 498:

```c
    if (!uvm_perf_prefetch_enabled(gpu_va_space->va_space))
        return status;
```

```c
    if (service_type == UVM_ATS_SERVICE_TYPE_FAULTS) {
        uvm_page_mask_t *read_fault_mask = &ats_context->faults.read_fault_mask;
        uvm_page_mask_t *write_fault_mask = &ats_context->faults.write_fault_mask;

        uvm_page_mask_or(read_fault_mask, read_fault_mask, prefetch_mask);

        if (vma->vm_flags & VM_WRITE)
            uvm_page_mask_or(write_fault_mask, write_fault_mask, prefetch_mask);
    }
```

3. The prefetch mask always contains the faulted pages. `uvm_perf_prefetch.c` line 227 seeds the tree with `uvm_page_mask_or(&bitmap_tree->pages, resident_mask, faulted_pages);`. The traversal starts at the leaf level (line 77, `iter->level_idx = bitmap_tree->level_count - 1;`), where a node covers one page (`uvm_perf_utils.h` line 166, `1 << (((tree)->level_count - 1) - (iter)->level_idx)`). Lines 118 to 119 select a region when `counter * 100 > subregion_pages * g_uvm_perf_prefetch_threshold`, and the default threshold is 51 (line 42), so the one-page leaf of a faulted page always qualifies. Line 297 fills `out_prefetch_mask` with the selected region. When every page of the 2 MiB VA block is already resident, every level qualifies and the region is the whole block within the VMA (derived here from lines 102 to 119 and 221 to 229, not stated in a comment). Prefetch is on by default (line 39, `static unsigned uvm_perf_prefetch_enable = 1;`), and `/sys/module/nvidia_uvm/parameters/uvm_perf_prefetch_enable` reads 1 on this machine.

4. Every page in the write mask of a writable VMA is serviced as a write. `uvm_ats_faults.c`, lines 633 to 640:

```c
    for_each_va_block_subregion_in_mask(subregion, write_fault_mask, region) {
        uvm_fault_access_type_t access_type;
        uvm_page_mask_t *serviced_mask;

        if (vma->vm_flags & VM_WRITE) {
            access_type = UVM_FAULT_ACCESS_TYPE_WRITE;
            serviced_mask = faults_serviced_mask;
        }
```

5. Write service becomes a write fault in the host MM. `uvm_ats_faults.c` line 59, `bool write = (access_type >= UVM_FAULT_ACCESS_TYPE_WRITE);`, and line 89, `uvm_migrate_args.populate_permissions = (write ? UVM_POPULATE_PERMISSIONS_WRITE : UVM_POPULATE_PERMISSIONS_ANY);`. `uvm_populate_pageable.c`, lines 60 to 65 and 137 to 151:

```c
    unsigned int fault_flags = is_write ? FAULT_FLAG_WRITE : 0;

    fault_flags |= (FAULT_FLAG_REMOTE);

    for (i = 0; i < num_pages; i++) {
        ret = UVM_HANDLE_MM_FAULT(vma, start + (i * PAGE_SIZE), fault_flags);
```

```c
    status = handle_fault(vma, start, num_pages, is_write);
    if (status != NV_OK)
        goto out;

    // Kernel v6.6 introduced a bug in set_pte_range() around the handling of
    // the ARM AF bit. Instead of setting the AF bit, the bit is incorrectly
    // being cleared in set_pte_range() during first-touch fault handling.
    // Calling handle_mm_fault() again takes a different code path which
    // correctly sets the AF bit.
    status = handle_fault(vma, start, num_pages, is_write);
    if (status != NV_OK)
        goto out;

    if (should_use_gup(vma, flags)) {
        long ret = NV_GET_USER_PAGES_REMOTE(mm, start, num_pages, is_write ? FOLL_WRITE : 0, NULL, NULL);
```

6. No source comment or public NVIDIA document was found that describes this upgrade as intended for private mappings or that mentions copy-on-write. The only explanatory comments near the code concern stale read-only entries in the SMMU and GPU TLBs (`uvm_ats_faults.c` lines 545 to 566).

7. Upstream v6.8 installs old PTEs on first-touch file faults on arm64. `mm/memory.c`, lines 4524 and 4530 to 4533:

```c
	bool prefault = in_range(vmf->address, addr, nr * PAGE_SIZE);
```

```c
	if (prefault && arch_wants_old_prefaulted_pte())
		entry = pte_mkold(entry);
	else
		entry = pte_sw_mkyoung(entry);
```

`arch/arm64/include/asm/pgtable.h`, lines 1106 to 1110: "Experimentally, it's cheap to set the access flag in hardware and we benefit from prefaulting mappings as 'old' to start with." followed by `#define arch_wants_old_prefaulted_pte	cpu_has_hw_af`. The running kernel has `CONFIG_ARM64_HW_AFDBM=y`. In v6.8 the polarity is inverted, so the PTE of the faulting address is the one made old. v6.12 `mm/memory.c` line 5019 reads `bool prefault = !in_range(vmf->address, addr, nr * PAGE_SIZE);`, which makes the fault-around neighbours old instead. With either polarity a read fault with fault-around (`shmem` sets `.map_pages = filemap_map_pages`, `mm/shmem.c` lines 4568 and 4577) leaves part of the populated range with the access flag clear.

8. `MADV_POPULATE_READ` does not set the access flag on PTEs that are already present. `mm/gup.c` line 1726 sets `gup_flags = FOLL_TOUCH | FOLL_HWPOISON | FOLL_UNLOCKABLE;`, and lines 670 to 680 say:

```c
	if (flags & FOLL_TOUCH) {
		if ((flags & FOLL_WRITE) &&
		    !pte_dirty(pte) && !PageDirty(page))
			set_page_dirty(page);
		/*
		 * pte_mkyoung() would be more correct here, but atomic care
		 * is needed to avoid losing the dirty bit: it is easier to use
		 * mark_page_accessed().
		 */
		mark_page_accessed(page);
	}
```

9. Upstream v6.8 does not enable hardware access-flag update for SVA. `drivers/iommu/arm/arm-smmu-v3/arm-smmu-v3-sva.c`, lines 147 to 151, builds the shared context descriptor as:

```c
	tcr = FIELD_PREP(CTXDESC_CD_0_TCR_T0SZ, 64ULL - vabits_actual) |
	      FIELD_PREP(CTXDESC_CD_0_TCR_IRGN0, ARM_LPAE_TCR_RGN_WBWA) |
	      FIELD_PREP(CTXDESC_CD_0_TCR_ORGN0, ARM_LPAE_TCR_RGN_WBWA) |
	      FIELD_PREP(CTXDESC_CD_0_TCR_SH0, ARM_LPAE_TCR_SH_IS) |
	      CTXDESC_CD_0_TCR_EPD1 | CTXDESC_CD_0_AA64;
```

The strings `HTTU`, `FEAT_HA`, `FEAT_HD`, `TCR_HA`, and `TCR_HD` do not occur in `arm-smmu-v3.h`, `arm-smmu-v3.c`, or `arm-smmu-v3-sva.c` at v6.8. At v6.12 and v6.17 `CTXDESC_CD_0_TCR_HA` exists but is set only for stage-1 paging domains with dirty tracking (`arm-smmu-v3.c` line 1385 at v6.12), not in `arm_smmu_make_sva_cd()`. Upstream master (2026-10-05) sets it for SVA at `arm-smmu-v3-sva.c` lines 107 to 115, added by commit 74fa4c177ad0.

10. The effect of an old PTE on an ATS translation request (no permission returned, so the GPU faults) was not read in the SMMUv3 specification. It is inferred from Brucker's statement that HTTU is needed "to reduce the number of access faults on the fault queue" and from the NVIDIA comment in item 5.

Conclusion. The measured behaviour is explained by the public source for both parts of the claim: with prefetch enabled (the default), any GPU fault on a `VM_WRITE` VMA is serviced through `handle_mm_fault(FAULT_FLAG_WRITE)`, for the faulted page and for the whole prefetch region, which is the full 2 MiB VA block when the block is populated; and on upstream v6.8 arm64 a populated but never CPU-touched file PTE can have the access flag clear, `MADV_POPULATE_READ` does not set it, and the SVA context descriptor does not let the SMMU set it, so the GPU still faults. Two points remain unverified: the exact 595.78 driver source and the 6.8.12-tegra kernel source, so the explanation holds for the bracketing public driver tags and for upstream v6.8.

Source-derived predictions that were not measured here and that the paper should either test or avoid stating:
- Loading `nvidia-uvm` with `uvm_perf_prefetch_enable=0` skips the upgrade (the early return at `uvm_ats_faults.c` lines 467 to 468), so GPU read faults on writable VMAs would be serviced as reads.
- Removing `VM_WRITE` from the mapping (`mprotect(PROT_READ)`) makes the driver service reads as reads, as issue 81 reports for GB10.
- A single old PTE in a populated 2 MiB block is enough to break copy-on-write for the whole block, because the prefetch region covers the block. Any later clearing of access flags by reclaim aging would have the same effect, so sharing that relies on "the CPU has read every page" may not be stable under memory pressure.
- On a kernel that contains 74fa4c177ad0 (v7.2 or later) with an SMMU that reports HTTU, old PTEs would no longer cause GPU read faults, and `MADV_POPULATE_READ` would then be sufficient.

### 12.4 C4: latent per-page cost after fork in an SVA-bound process, and its repair

| Work (venue, year) | What it does or states | Overlap with the claim | Tag | URL |
|---|---|---|---|---|
| Linux v6.8 `mm/memory.c` lines 972 to 979, `include/linux/rmap.h` (`ClearPageAnonExclusive` in the dup path, for example line 406), `mm/memory.c` lines 3503 to 3512 | Fork: "If it's a COW mapping, write protect it both in the parent and the child" (`ptep_set_wrprotect(src_mm, addr, src_pte)`), and the exclusive flag is cleared. Write fault: `if (folio && folio_test_anon(folio) && (PageAnonExclusive(vmf->page) || wp_can_reuse_anon_folio(folio, vma)))` then `SetPageAnonExclusive` and `wp_page_reuse`. | Source of the latent state: after the child is gone, each parent page needs one write fault to become writable again. Nothing in the exit or exec path restores write permission. | V (src) | https://github.com/torvalds/linux/blob/v6.8/mm/memory.c#L972 |
| Linux v6.8 `arch/arm64/mm/fault.c` lines 241 to 243 and `arch/arm64/include/asm/tlbflush.h` lines 272 to 283 | `/* Invalidate a stale read-only entry */ if (dirty) flush_tlb_page(vma, address);` and `__flush_tlb_page_nosync()` calls `mmu_notifier_arch_invalidate_secondary_tlbs(mm, uaddr & PAGE_MASK, (uaddr & PAGE_MASK) + PAGE_SIZE);`. | Source of the per-page IOMMU invalidation on each reuse fault on arm64 with an SVA-bound mm. | V (src) | https://github.com/torvalds/linux/blob/v6.8/arch/arm64/mm/fault.c#L212 |
| Linux v6.8 `mm/mprotect.c` lines 62 to 71, and commit 64fe24a3e05e (2022) | "Writable MAP_PRIVATE mapping: We can only special-case on exclusive anonymous pages" and `return page && PageAnon(page) && PageAnonExclusive(page);`. | The kernel's bulk path that maps pages writable without faults applies only to pages already marked exclusive. After fork the flag is clear, so an `mprotect` round trip does not repair the state. | V (src) | https://github.com/torvalds/linux/blob/v6.8/mm/mprotect.c#L42 |
| madvise(2), `MADV_POPULATE_WRITE` (Linux 5.14) | "Populate (prefault) page tables writable, faulting in all pages in the range just as if manually writing to each each page ... One example use case is preallocating memory, breaking any CoW (Copy on Write)." | Existing kernel interface for bulk repair in one call. By the v6.8 source it runs one `handle_mm_fault` per page (`mm/gup.c` line 1726 onward), so the per-page TLB and secondary-TLB invalidation remains and only the GPU fault round trip is removed. Its cost on this platform was not measured and is the baseline the repair must be compared with. | V (doc), V (src) | https://man7.org/linux/man-pages/man2/madvise.2.html |
| madvise(2), `MADV_DONTFORK`, and Linux v6.8 `kernel/fork.c` line 670 | "Do not make the pages in this range available to the child after a fork(2). This is useful to prevent copy-on-write semantics from changing the physical location of a page if the parent writes to it after a fork(2). (Such page relocations cause problems for hardware that DMAs into the page.)" Fork skips `VM_DONTCOPY` VMAs. | Same avoidance mechanism, documented for a different reason (page relocation under DMA), not for the cost of later write faults. | V (doc), V (src) | https://man7.org/linux/man-pages/man2/madvise.2.html |
| madvise(2), `MADV_WIPEONFORK`, and Linux v6.8 `kernel/fork.c` lines 705 to 707 and 744 to 745 | "VM_WIPEONFORK gets a clean slate in the child." and `if (!(tmp->vm_flags & VM_WIPEONFORK)) retval = copy_page_range(tmp, mpnt);`. | A second existing way to keep fork from write-protecting the parent's pages (private anonymous memory only). Documented for secrets, not for cost. | V (doc), V (src) | https://github.com/torvalds/linux/blob/v6.8/kernel/fork.c#L705 |
| rdma-core, ibv_fork_init(3) and ibv_is_fork_initialized(3) | "ibv_fork_init() works on Linux kernels supporting the MADV_DONTFORK flag". "Calling ibv_fork_init() will reduce performance due to an extra system call for every memory registration". `IBV_FORK_UNNEEDED` "indicates that the kernel copies DMA pages on fork". | Established practice of `MADV_DONTFORK` on device-visible memory. Motivation is data corruption after fork. No statement on post-fork fault cost. | V (doc) | https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_fork_init.3.md |
| Linux commit 2c91bd4a4e2e, "mm: speed up mremap by 20x on large regions" (2019) | 1 GiB `mremap` in 144 to 160 microseconds with PMD-level moves. | Explains why a repair that ends in `mremap` is cheap in its remap step. | V (src) | https://github.com/torvalds/linux/commit/2c91bd4a4e2e |
| CUDA 12.2 release notes; NVIDIA HMM blog (2023) | "The fork() system call is not fully supported yet when attempting to share GPU-accessible memory between parent and child processes." "fork(2) without a following exec(3) is not fully supported." | Vendor guidance on fork with GPU-accessible system memory (x86 HMM). No cost statement, nothing on the parent after the child exits. | V (doc) | https://docs.nvidia.com/cuda/archive/12.2.0/cuda-toolkit-release-notes/index.html |
| PyTorch, Multiprocessing best practices | "The CUDA runtime has the limitation described in [poison fork] when using the `fork` start method; either the `spawn` or `forkserver` start method are required to use CUDA in subprocesses." | Guidance against fork in CUDA processes for correctness. No cost statement. | V (doc) | https://github.com/pytorch/pytorch/blob/main/docs/source/notes/multiprocessing.md |
| Baumann, Appavoo, Krieger, Roscoe, "A fork() in the road" (HotOS 2019) | "a process using, say, DPDK with a kernel-bypass NIC, or OpenCL with a GPU, cannot safely fork since the OS cannot duplicate the process state on the NIC/GPU." Lists `MADV_DONTFORK/DOFORK/WIPEONFORK` as fork special cases. Measures fork time against dirty memory. | General argument and fork-time measurement. No measurement of the parent's write faults after the child is gone, no SVA. | V (pdf) | https://www.microsoft.com/en-us/research/uploads/prod/2019/04/fork-hotos19.pdf |
| Linux v6.8 `Documentation/arch/x86/sva.rst` lines 128 to 129 | "On fork(2) or exec(2) the PASID is removed from the process". | Kernel documentation on SVA and fork. Says nothing on the parent's write-protected pages. | V (src) | https://github.com/torvalds/linux/blob/v6.8/Documentation/arch/x86/sva.rst |
| Linux commit 74fa4c177ad0 (v7.2) | Enables HD in SVA context descriptors so that the SMMU clears `PTE_RDONLY` on writable-clean PTEs without a stall. | Does not help post-fork pages: fork removes write permission from the PTE, so the fault is still required (inference from commit text and item 1 of this table). | V (src) | https://github.com/torvalds/linux/commit/74fa4c177ad0 |
| NVIDIA Developer Forums, "CUDA and fork()" (2007) | Users discuss GPU state shared across fork. No NVIDIA statement in the thread as fetched. Read through a fetch tool summary. | No guidance, no cost. | V (doc) | https://forums.developer.nvidia.com/t/cuda-and-fork/2131 |
| Corbet, "Patching until the COWs come home (part 1)" (LWN, 2021) | Discusses correctness of page reuse after fork with `vmsplice` and GUP references. | About correctness of the reuse decision, not its cost. | V (doc) | https://lwn.net/Articles/849638/ |

Finding for C4. The mechanism is fully visible in upstream source: fork write-protects the parent, the exclusive flag is cleared, and each page needs one reuse fault that on arm64 flushes one page from the CPU TLB and from secondary TLBs. `MADV_DONTFORK` on device-visible memory is standard RDMA practice for a different reason. `MADV_POPULATE_WRITE` is an existing one-call repair. No verified source measures this cost in a GPU or SVA-bound process, and none proposes a user-space remap repair. The arm64 and large-folio reuse-fault patches named in the request as already known address the same faults on the CPU side and were not reopened.

### Verdicts

**C1: PARTIALLY TAKEN.**
- Closest work: Tene, Iyengar, Wolf, C4 (ISMM 2011), which uses 2 MiB mappings and batched invalidation to cut remap and protection cost, for in-process garbage collection.
- Claimable: for revoke-before-grant transfer of SVA-visible memory between GPU tenants, placing the object on 2 MiB hugetlbfs pages reduces the measured protection overhead from 2.69x to 1.09x (1.72x with shmem THP), which is a measurement of a known per-PTE effect in a setting where it had not been measured, not a new technique.
- Citations that must stand next to the claim: fbufs (per-page revocation cost), Chu 1996, Appel and Li 1991 (overhead independent of page size), XPC (remap-based ownership transfer and its TLB cost), zIO section 3.4 (huge pages against fine-grained protection), libmpk (`mprotect` cost grows with page count), C4, Linux commits 2c91bd4a4e2e and b9bf6c2872c5.

**C2: PARTIALLY TAKEN.**
- Closest work: Si et al., "The Ingestion Tax" (arXiv 2608.12114, 2026), where N processes share one `MAP_SHARED` file mapping that the GPU reads in place, on Apple, an AMD APU, and GH200.
- Claimable: private, writable mappings of one sealed memfd that stay physically shared under GPU reads and diverge by copy-on-write under GPU writes, across two MIG instances, enforced by the host MMU; read-only in-place sharing across processes is prior art and must be stated as such.
- Citations that must stand next to the claim: The Ingestion Tax, llama.cpp (source and Discussion 21223), geistlib issue 541, MiaAI-Lab issue 81, TrIMS, CUDA 12.2 release notes (fork limitation), CUDA for Tegra application note, `udmabuf.c` (seals with GPU-visible memfd), commit 5535be309971 (seal bypass through a CoW path).

**C3: PARTIALLY TAKEN.**
- Closest work: MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark issue 81 (2026-09-28), which states that UVM services every GPU fault in a writable mapping as a write fault, cites the same lines, and notes that a private writable mapping is copied.
- Claimable: the condition under which sharing survives (every PTE present with the access flag set before the GPU touches it), the insufficiency of `MADV_POPULATE_READ`, the source-level explanation that joins the UVM prefetch upgrade with arm64 old PTEs and the missing HA bit in the SVA context descriptor, and the confirmation on Thor with MIG; the write-intent servicing itself is public source and was publicly described first by issue 81.
- Citations that must stand next to the claim: issue 81, `uvm_ats_faults.c` and `uvm_populate_pageable.c` at a named tag, Linux commit 4cd7ba16a0af, Brucker's 2021 HTTU patch, Linux commit 74fa4c177ad0, geistlib issue 541 or The Ingestion Tax for the Apple analogue.

**C4: PARTIALLY TAKEN.**
- Closest work: madvise(2) `MADV_POPULATE_WRITE` (Linux 5.14) together with the upstream reuse-fault path, as the existing bulk repair; `ibv_fork_init` for the `MADV_DONTFORK` avoidance.
- Claimable: the measurement that the post-fork reuse faults cost 2.7 s/GiB when taken as GPU write faults in an SVA-bound process, and a user-space repair that is cheaper than the kernel's per-page path, provided `MADV_POPULATE_WRITE` is measured as a baseline; the avoidance by `MADV_DONTFORK` is existing practice.
- Citations that must stand next to the claim: madvise(2) (`MADV_POPULATE_WRITE`, `MADV_DONTFORK`, `MADV_WIPEONFORK`), ibv_fork_init(3), Linux commit 64fe24a3e05e, the arm64 and large-folio reuse-fault patches already known to the authors, CUDA 12.2 release notes, PyTorch multiprocessing notes, "A fork() in the road".

### Not opened

- Brustoloni and Steenkiste, "Effects of Buffering Semantics on I/O Performance" (OSDI 1996). No reachable copy (USENIX legacy URL returns 404). It may contain per-page cost models on platforms with different page sizes. NV.
- He et al., Copier, "How to Copy Memory? Coordinated Asynchronous Copy as a First-Class OS Service" (SOSP 2025). The preprint URL returned "Not Found". NV.
- Sartakov et al., CubicleOS (ASPLOS 2021). Two candidate URLs failed. NV.
- Liedtke, "On micro-kernel construction" (SOSP 1995), seL4 manual, and any L4 or seL4 measurement of map or grant with large frames. NV.
- Memif (ASPLOS 2016), Shadowfax. NV.
- Fuchsia VMO reference. The page was fetched but only the statement that VMO size is rounded to the page size was extracted; no usable statement on transfer cost. Treat as NV for the claim.
- Android Zygote documentation. The page was fetched and contains no statement on preloading or copy-on-write sharing. Android NNAPI reference (`ANeuralNetworksMemory_createFromFd`) could not be fetched. NV.
- TensorFlow Lite GPU delegate (whether weights are copied to GPU objects). Only the CPU-side `mmap` flags were read. NV for the GPU side.
- Ray object store (Plasma) documentation. Fetch failed. NV.
- DeepPlan (EuroSys 2023, direct host access). ACM and author URLs failed. NV.
- ServerlessT2I (arXiv 2607.26566) was downloaded and searched; it has no cross-process MMU sharing and was left out of the tables.
- Arm SMMUv3 specification text on ATS translation requests to entries with AF clear. NV; item 10 above is an inference.
- Jetson 6.8.12-tegra kernel source and the Jetson build of nvidia-uvm 595.78. NV.
- lore.kernel.org threads (blocked by a proof-of-work page). Commit messages were read through the GitHub API instead; Brucker's patch was read on lists.infradead.org.
- Huang Ying's arm64 patch and the large-folio reuse series, listed as already known in the request.
- NVIDIA/cuopt issue 1995 (DGX Spark SIGBUS in forked workers) was opened; it concerns a crash on any GPU memory touch and has no fork-cost content.

### Queries tried

Web searches:
- `nvidia-uvm ATS read fault serviced as write fault writable VMA copy-on-write "uvm_ats_faults.c" OR "uvm_ats_service_faults"`
- `NVIDIA forum CUDA Jetson Thor OR "DGX Spark" system memory mmap MAP_PRIVATE GPU read copies pages "copy-on-write" UVM ATS write fault read-only mprotect`
- `"iommu/arm-smmu-v3: Add support for Hardware Translation Table Update" Brucker "essential for shared page tables" access faults`
- `"arm-smmu-v3-sva" "Enable Hardware Access and Hardware Dirty bits" Nicolin Chen patch stalls SVA PTE_RDONLY`
- `huge pages reduce mprotect cost zero-copy IPC page remapping ownership transfer "huge pages" "page flipping" OR "remapping" cross-domain transfer paper`
- `zero-copy IPC "huge pages" remap ownership transfer between processes mprotect cost "2MB" "TLB shootdown" page flipping huge page paper microkernel OR unikernel`
- `hugetlb OR "huge pages" "userfaultfd write-protect" OR mprotect cost per page "512x" fewer PTE updates zero-copy buffer handoff GPU "dma-buf" OR "shared memory" protection domain huge page permission change latency measurement`
- `Druschel Peterson "Fbufs: a high-bandwidth cross-domain transfer facility" pdf`
- `Brustoloni Steenkiste "Effects of buffering semantics on I/O performance" OSDI 1996 pdf`
- `"Copier" SOSP 2025 memory copy OS service zero-copy "page remapping" huge pages paper`
- `llama.cpp mmap model weights shared across processes Metal "newBufferWithBytesNoCopy" unified memory zero-copy GPU`
- `copy-on-write model weights shared across GPU processes "unified memory" OR "shared virtual addressing" page tables inference serving deduplication zero-copy paper 2025 2026`
- `memfd sealed model weights share across processes inference "F_SEAL_WRITE" OR "memfd_create" GPU zero-copy mmap weights "MAP_PRIVATE"`
- `NVIDIA MIG instances share model weights across MIG "memoryless" OR "0gb" Jetson Thor DGX Spark MIG shared system memory tenants`
- `fork child exited parent pages remain write-protected "wp fault" reuse cost after fork "MADV_DONTFORK" GPU OR RDMA slowdown measurement`
- `CUDA process fork slow GPU page faults after fork parent write-protected pages Jetson OR "DGX Spark" OR "Grace Hopper" ATS "fork" slowdown system memory`
- `linux mm patch after fork child exit parent anonymous pages remain write-protected "PageAnonExclusive" restore writable without fault OR "avoid write faults" exclusive after child exits lkml`

GitHub commit searches (repo torvalds/linux): `"fix old/young bit handling in the faulting path"`, `"try avoiding write faults for exclusive anonymous pages"`, `"speed up mremap by 20x on large regions"`, `"Optimize mprotect" "large folios"`, and the commit history of `drivers/iommu/arm/arm-smmu-v3/arm-smmu-v3-sva.c`.

Source greps (all with results reported above): in the UVM tree, `write_fault_mask`, `read_fault_mask`, `faults_serviced_mask`, `VM_WRITE`, `FAULT_FLAG`, `FOLL_WRITE`, `UVM_POPULATE_PERMISSIONS`, `skip_mapped`, `handle_mm_fault`, `AF bit`; in Linux v6.5, v6.6, v6.8, v6.12, v6.17, and master, `prefault`, `arch_wants_old_prefaulted_pte`, `HTTU`, `FEAT_HA`, `TCR_HA`, `FOLL_TOUCH`, `can_change_pte_writable`, `ClearPageAnonExclusive`, `wp_can_reuse_anon_folio`, `mmu_notifier_arch_invalidate_secondary_tlbs`, `VM_DONTCOPY`, `VM_WIPEONFORK`, `SEALS_WANTED`.

Not found, with the queries above: any paper, document, or thread that uses huge pages to reduce the cost of permission transfer for GPU-visible memory; any work that shares GPU-read state across MIG instances through the host MMU; any NVIDIA document that describes read faults being serviced with write intent; any measurement of post-fork write-fault cost in a GPU process on Jetson, DGX Spark, or Grace Hopper.

## 13. What Section 12 changes, and what was measured in response

1. **All four claims are partially taken, and each keeps a part that the
   three passes did not find.** No claim may be worded as a technique. The
   research record states the claim set accordingly
   (`RESEARCH_HOSTMM_2026-10-05.md`, Section 9).
2. **C3, a prediction from source, was tested and holds.** Section 12 derives
   from the driver source that the fault service upgrades its 2 MiB prefetch
   block to write access in a writable mapping, so that one page whose
   accessed flag is clear should cost a whole block. Measured
   (`thor_hostmm/results/20261005-af-block-v1/`, 42 runs): one cleared page
   turns exactly 512 pages into private copies in every run; one cleared
   page per 2 MiB block, 0.2% of the pages, turns the whole 1 GiB model into
   a private copy; a read-only mapping grows by nothing. No source opened in
   the three passes reports this amplification.
3. **C2 and C3 together.** The sources that Section 12 found for read-only
   in-place sharing abandon private mappings when they observe that the GPU
   read copies them. The research record explains the copy (write-intent
   service plus a cleared accessed flag, each visible in source), reproduces
   it by direct manipulation, and measures a design that keeps private,
   divergable views of one sealed base shared across MIG instances
   (`thor_hostmm/results/20261005-shared-model-v1/`,
   `20261005-shared-concurrent-v1/`). The write-intent service itself is
   public source and was described first by the issue thread of
   2026-09-28; it is cited, not claimed.
4. **C4, the baseline that Section 12 demands, is measured.**
   `MADV_POPULATE_WRITE` after the child has gone costs 2,675 ms/GiB with
   one thread and 1,714 ms/GiB with twelve in a CUDA process
   (`thor_hostmm/results/20261005-mm-tax-v1/`, operation `reuse_fault`),
   against 20.6 ms/GiB for the privatization pass. `MADV_DONTFORK` is
   existing practice from RDMA and is reported as such.
5. **C1.** The record words the huge-page result as a measurement of a
   known per-entry effect in a setting where it had not been measured, with
   the citations that Section 12 lists.

## 14. Addendum: on-device LLM serving concept (share, fork, move)

Provenance: produced by a fourth delegated search pass on 2026-10-05, after
the concept was set to on-device LLM serving and a first engine result
existed. The brief asked for work that takes six claims: S (share weights in
place across processes), F (fork session state), M (move state between
processes), and the pitfalls P1 to P3. The text is the pass's own. The four
sources that an earlier pass had named as closest were opened directly and
exist. Rows tagged M rest on fetch-tool summaries.

Audit date: 2026-10-05. Every row tagged V was opened in this pass (PDF text extracted locally, GitHub REST API JSON, raw source file at a named commit, or the official page through a fetch tool). Search snippets were used only to locate sources. Where a fetch tool returned a model-written summary instead of the page text, the row is tagged M and says so.

Tags: **V (pdf)** = paper PDF opened and read. **V (src)** = source file opened at a named commit or tag. **V (doc)** = official documentation, issue, PR or discussion opened directly. **V (abs)** = abstract page only. **M** = from memory or from a tool-written summary, not checked against the primary text. **NV** = not verified.

Claim labels: S = share one physical copy of base weights across N processes with the GPU reading in place (sealed memfd, MAP_PRIVATE|PROT_READ, extent divergence by copy+mremap). F = fork KV or session state copy-on-write across processes at extent granularity. M = zero-copy move of a context or KV buffer between processes with revocation. P1 = writable private mapping silently becomes a full private copy (GPU fault served as write, 2 MiB prefetch block upgrade, accessed-flag trigger). P2 = post-fork GPU write cost of 2.7 s/GiB. P3 = MPS fault kills co-clients, time slicing and other MIG instances survive, so MPS groups inside MIG instances bound the blast radius.

### 14.1 Group A: the two works named by the previous pass, plus the two cited threads

| Work (venue, year) | What it does or states (with a quote where decisive) | Overlap with S / F / M / P1 / P2 / P3 | Tag | URL |
|---|---|---|---|---|
| Si, Lin, Li, Zhang, "The Ingestion Tax: Adopting File-Backed Weights in Tensor Frameworks" (arXiv 2608.12114, cs.OS, v1 2026-08-12, v2 2026-08-30) | **EXISTS**, title as given. Platforms: Apple M5 Max 128 GB (Metal, PyTorch MPS and MLX through DLPack), a held-out 8 GB Apple M3, AMD Ryzen 7 9700X APU (llama.cpp Vulkan), NVIDIA GH200 (probes and rebinding only), RTX 5070 Ti and a rented A10/A100/H100 fleet as discrete controls. No Jetson, Thor, DGX Spark, GB10 or Tegra (zero occurrences of each word). Mechanism (Algorithm 1): "𝑝 ← mmap(𝑝𝑎𝑡ℎ, 𝑛, prot_read, map_shared)" then "NewBufferWithBytesNoCopy(𝑝, 𝑛)", exported as a read-only DLPack capsule. The GPU reads in place: "the weights remain clean, shared, evictable file pages: 𝑁 processes decode from one mapped copy where resident loading creates 𝑁 copies (at capacity, 5.5 vs. 0.08 tok/s)". Table 8 measures N = 1, 2, 4 concurrent decoders of Qwen2.5-7B and N = 2 of Qwen2.5-32B on the Apple machine with vm_stat deltas. llama.cpp integration on the APU: "We connect the loader's existing buffer_from_host_ptr hook to VK_EXT_external_memory_host, advertised only on unified-memory devices; the loader and the inference loop are unchanged", 2.82 to 3.42 tok/s, peak working set 8.13 to 4.20 GiB, 4 paired processes. GH200: "a stock transformers model rebound to DLPack views of Grace-resident file pages runs a 145 GB fp16 Qwen2.5-72B at 2.48 tok/s", and the limitation "GH200 is evaluated with placement probes and storage rebinding rather than a production loader". Private mappings: "the mapping must remain shared. A private mapping is converted to anonymous pages when Metal wires it, recreating the copy at model scale: a 60 GB private mapping produced 58 GB of anonymous memory, 10× slower wiring, and a 3× throughput loss. This behavior was observed on macOS; Linux and Windows keep private file mappings file-backed." Mutation: "a GPU store to the read-only Metal mapping is silently discarded, the CUDA path faults". On discrete CUDA: "cudaHostRegister rejects a file-backed mapping under every tested flag ... The unregistered file mapping can nevertheless be read in place by a kernel at bus rate". Scope limit stated by the authors: the ratios "do not extend to batched GEMM, prefill, or concurrent serving". Absent from the text (zero hits): seal, memfd, LoRA, adapter, fork, MIG, tenant, isolation, time slicing, accessed flag, ATS, SMMU. MPS appears only as the PyTorch Metal backend. KV cache appears once, as a citation of SuperInfer. | **S: taken in its basic form** (file-backed weights mapped once, GPU reads in place, N processes share one copy, measured on Apple; llama.cpp host-pointer import on a Vulkan iGPU; in-place read on GH200 through the NVIDIA coherent path). Not taken: CUDA backend of llama.cpp, Tegra or GB10, MAP_PRIVATE with seals, per-process divergence of extents, adapters, N = 8. **P1: partly adjacent.** It reports the private-mapping copy on macOS and states the opposite for Linux ("Linux and Windows keep private file mappings file-backed"), which our P1 measurement contradicts for writable private mappings under nvidia-uvm ATS. No mention of write-fault servicing, the 2 MiB block, or the accessed flag. **F, M, P2, P3: no overlap.** | V (pdf) | https://arxiv.org/abs/2608.12114 |
| GitHub issue #81, MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark, "v0.30 lane: the PLE table is written back to the NVMe while serving (writable shared mapping + UVM write faults); mprotect it read-only" (opened 2026-09-28 by antoniocuegervas, open, 0 comments) | **EXISTS**, date as given. Platform: one GB10 (DGX Spark), kernel 6.17.0-1031-nvidia, driver 580.173.02, vLLM recipe. The mapping is `torch.from_file(path, shared=True, ...)`, a writable shared mapping of a lookup table that the GPU reads in place. Decisive sentences: "the NVIDIA UVM driver services every GPU fault in a writable mapping as a write fault. In driver 580.173.02, `uvm_ats_faults.c` lines 496 to 497 add the faulted pages to the write mask when the mapping allows writes, and lines 632 to 633 service them as writes. On ext4 that runs `page_mkwrite`: the page is marked dirty and the file's times are updated, so the kernel later writes the same bytes back." Fix: `mprotect(..., PROT_READ)`, after which "UVM then services the GPU's faults as read faults." On private mappings: "`torch.from_file(shared=False)` is not an alternative: a private writable mapping copies every page the GPU touches into anonymous memory." Measured: about 1.3 GiB written back in 7 minutes, zero dirty pages in 115 samples after the fix, prefill speed unchanged. The issue body states it was written by an AI assistant on behalf of the account owner. | **P1: the root cause is taken** (GPU fault in a writable VMA is serviced as a write, and a private writable mapping is copied page by page as the GPU touches it, stated for GB10 with file and line numbers). Not stated there: the 2 MiB prefetch-block upgrade, the accessed-flag trigger, the 0.2% of pages costing the whole model, or any number for the private case (the private-mapping sentence is an assertion with no measurement). **S: adjacent** (a single process reads a file mapping in place on GB10 inside vLLM; no cross-process sharing is discussed). **F, M, P2, P3: no overlap.** | V (doc) | https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/issues/81 |
| llama.cpp Discussion #21223, "Idea: Share readonly GPU model weights across multiple llama-* processes" (phoyd, 2026-03-31; 2 comments, 6 replies) | **EXISTS.** A user request, not an implementation. The author wants several `llama-server` processes of one model, including "the same model with a new LoRA, without disturbing the work of the other users", and notes "`--mmap` already allows the memory to be shared between instances. But there is currently no equivalent mechanism for the GPU". Asked "Does mmap weight sharing work on other unified memory archs? i.e. AI Max 395 and DGX Spark?", the maintainer (ggerganov, 2026-04-02) answers: "I don't think so. The Metal implementation basically "looks" directly at the memory mapped buffers via MTLResourceStorageModeShared, while the CUDA backend would make a copy into device buffers." He adds that one `llama_model` with several `llama_context` is supported at the API level inside one process. A later reply (pontostroy, 2026-06-04) links a CUDA IPC workaround, listed in 14.2. | **S: the need is stated publicly, including the LoRA variant and DGX Spark, and the maintainer states that the CUDA backend does not do it.** This is supporting evidence for the gap, and it also shows the idea itself is not new. **F, M, P1, P2, P3: no overlap.** | V (doc) | https://github.com/ggml-org/llama.cpp/discussions/21223 |
| geisten/geistlib issue #541, "metal: the GGUF is mapped MAP_PRIVATE — wiring copies every GPU-bound weight page (2.9 GiB on gemma4-e2b)" (geisten, 2026-10-01, closed) | **EXISTS** at github.com/geisten/geistlib/issues/541. Platform: Apple M1 Max, Metal. Not NVIDIA. Statement: "`gguf_open` maps the model with `mmap(PROT_READ, MAP_PRIVATE)` ... every weight on Metal is a `newBufferWithBytesNoCopy` wrapper over that mapping. When a command buffer first makes such a wrapper resident, Metal wires its pages for write. On a **private** mapping that is copy-on-write: **every page of every GPU-bound weight becomes a private anonymous copy.**" Evidence table: "a 16 KiB GPU blit out of a 256 MiB NoCopy wrapper (cold file): 16 384 / 16 384 pages copied" under MAP_PRIVATE and 0 under MAP_SHARED. Scope: "Keep `MAP_PRIVATE` elsewhere. Linux and the Pi do no device wiring of the mapping (Vulkan copies weights into device buffers)". | **P1: the same class of pitfall on a different platform and driver** (Metal wiring for write on a read-only private mapping copies the whole wrapper, so a 16 KiB read copies 256 MiB). The amplification shape is close to ours, the mechanism is different (wiring, not fault servicing plus prefetch), and the mapping there is PROT_READ, which on our platform is the safe case. **S: opposite design choice** (it concludes MAP_SHARED is required on Metal; our hardened form uses MAP_PRIVATE|PROT_READ on nvidia-uvm). **F, M, P2, P3: no overlap.** | V (doc) | https://github.com/geisten/geistlib/issues/541 |

Summary of group A. All four items exist and say what the earlier pass attributed to them, with two corrections. First, The Ingestion Tax measures cross-process sharing only on Apple silicon (N up to 4) and treats GH200 with probes and module rebinding, not with a serving engine, and it has no CUDA llama.cpp path, no Tegra or GB10 platform, and no private or sealed mapping design. Second, issue #81 states the write-fault root cause of P1 for GB10 with driver line numbers, but it only asserts the private-mapping copy in one sentence and does not report the 2 MiB block upgrade or the accessed-flag trigger.

### 14.2 Group B: cross-process base-weight sharing with the GPU reading in place

Columns answer, for each work: where the weights live, whether one physical copy is shared across PROCESSES, who enforces isolation, whether adapters are private per tenant, whether the base is immutable against a tenant.

| Work (venue, year) | What it does or states (with a quote where decisive) | Overlap with S / F / M / P1 / P2 / P3 | Tag | URL |
|---|---|---|---|---|
| llama.cpp CUDA backend at commit 153d324bcf86d220b235ca010eeb11213f32b5d1 (2026-08-11), `ggml/src/ggml-cuda/ggml-cuda.cu` | Line 4802: `/* .buffer_from_host_ptr  = */ false,`. Line 4804: `/* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,`. Line 5338: `/* .buffer_from_host_ptr    = */ NULL,`. Lines 141 to 142: `GGML_CUDA_ENABLE_UNIFIED_MEMORY` selects `cudaMallocManaged`, a private managed allocation, not a file mapping. Weights live in device or managed buffers, one copy per process. | S: confirms the upstream gap exactly as the concept states it. | V (src) | https://github.com/ggml-org/llama.cpp/blob/153d324bcf86d220b235ca010eeb11213f32b5d1/ggml/src/ggml-cuda/ggml-cuda.cu |
| llama.cpp PR #26081, "llama: add default load-mode auto, which avoids mmap on iGPUs" (0cc4m, opened 2026-07-24, merged as 153d324bc) | Stated reason: "mmap is detrimental on iGPUs that have to copy the model to device-visible, but still shared memory, as that means during loading it will have the model in RAM twice, doubling the memory requirements temporarily and slowing down the process if it overflows." Scope: "I only disabled it for CUDA/ROCm and Vulkan iGPUs for now". The commit message adds "picks mmap unless a non-Metal iGPU is used". The premise is that the backend must copy. The PR does not consider reading the mapping in place. | S: the upstream decision that our in-place path reverses. It must be cited next to S(ii) so the reader sees why upstream avoids mmap and what our patch changes. | V (doc) | https://github.com/ggml-org/llama.cpp/pull/26081 |
| llama.cpp issue #28160, "Regression: --lazy-mode auto halves pp512 for qwen4exp on Vulkan (AMD iGPU)" (2026-09-01, closed) | Not about weight sharing. It reports that lazy tensor reads introduced by #27837 halve prefill on a Strix Halo iGPU (216 versus 406 t/s). The same contributor comments: "We're back to bad default behaviour for iGPUs" and "On iGPUs the optimal direction flips, most users will try to run models that can fit into memory". A second user reports the same on ROCm. | S: indirect. It shows that mapped or lazily read weights on Linux iGPU backends are currently a performance hazard upstream, so a reviewer will ask for prompt-processing numbers. Our prompt-processing parity result answers this. | V (doc) | https://github.com/ggml-org/llama.cpp/issues/28160 |
| llama.cpp PR #22120, "ggml-cuda: add mmap zero-copy buffer for integrated GPUs" (somrupp-web, 2026-04-19, closed unmerged after 6 minutes) | **The closest prior attempt at S(ii).** Body: "Implements buffer_from_host_ptr for CUDA devices to support mmap-based weight loading without a staging copy. On integrated GPUs (e.g. GB10/DGX Spark) where CPU and GPU share the same physical LPDDR5X, cudaHostRegister + cudaHostGetDevicePointer maps the mmap'd model file directly into the GPU address space — the GPU reads weights from the mmap'd pages with zero copies." Diff (1 file, +172, −2): `cudaHostRegister(ptr, size, cudaHostRegisterMapped | cudaHostRegisterPortable)` then `cudaHostGetDevicePointer`. Closed by the bot under the AI-generated-content policy. No measurement, no review, no multi-process claim, no statement that registration of a read-only file mapping succeeded on GB10. | **S(ii): the idea and a pinning-based implementation were published before us for GB10.** Our patch can still claim a measured, working implementation and, if it holds, a path without registration or pinning (pages stay evictable page cache). This must be stated as "first measured", not "first proposed". | V (doc), diff opened | https://github.com/ggml-org/llama.cpp/pull/22120 |
| llama.cpp Metal backend at 153d324bc, `ggml/src/ggml-metal/ggml-metal.cpp` and `ggml-metal-device.m` | `ggml-metal.cpp` line 682: `/* .buffer_from_host_ptr = */ true,`; line 807 wires `ggml_backend_metal_device_buffer_mapped`; `ggml-metal-device.m` line 1733: `newBufferWithBytesNoCopy:ptr length:size_aligned options:MTLResourceStorageModeShared deallocator:nil`. Weights are the mmap'ed file. Sharing across processes follows from the shared file mapping (see 14.1, maintainer quote). Isolation is the OS file mapping. No adapters-in-place design. | S: taken on Apple silicon since the Metal backend adopted host-pointer buffers. Our contribution is the NVIDIA unified-memory equivalent. | V (src) | https://github.com/ggml-org/llama.cpp/blob/153d324bcf86d220b235ca010eeb11213f32b5d1/ggml/src/ggml-metal/ggml-metal-device.m |
| llama.cpp issue #21827, "Avoid memcpy for mmap-ed weights on Unified Memory architectures (Intel Lunar Lake)" (2026-04-12, closed stale) | Request for a zero-copy SYCL path: "even if the model files are mapped into host memory via mmap, the backend still allocates device memory and performs a standard copy". A user prototype with memfd plus udmabuf plus Level Zero import "works" for loading but shows output corruption. A contributor answers that SVM "is designed for Unified Memory on Linux. It's WIP." | S: the same need on Intel UMA, unresolved. Shows memfd-backed GPU-visible weights were tried by others (Intel, dma-buf import, single process). | V (doc) | https://github.com/ggml-org/llama.cpp/issues/21827 |
| pontostroy/cuda-llm-weight-share (GitHub, created 2026-06-04, commit 15bcecae) | "A lightweight `LD_PRELOAD` library that allows multiple independent Linux processes to share a single selected CUDA allocation, typically a large machine learning model's weights, in CUDA VRAM." It intercepts `cudaMalloc`, the first process exports with `cudaIpcGetMemHandle`, later processes call `cudaIpcOpenMemHandle` (source lines 587 to 593, 1036). "Private KV cache per process". Example: two `llama-server` processes at 37,392 MiB and 6,816 MiB. Weights live in device memory, shared writable through CUDA IPC. No read-only enforcement (no `mprotect` or read-only flag in the source). Lifetime depends on a master process: after the master exits, a new process "allocates a new weights copy". Selection by exact allocation size. | **S: cross-process sharing of llama.cpp CUDA weights is taken, by a different mechanism** (device memory plus CUDA IPC). Not taken: host page table as the sharing mechanism, immutability enforced by the MMU, file-backed evictable pages, no master process, per-process divergence of extents. Per the NVIDIA Tegra note below, memory-sharing CUDA IPC is supported on Thor from CUDA 13.0, so this baseline is runnable on our platform and a reviewer can ask for it. | V (src) | https://github.com/pontostroy/cuda-llm-weight-share |
| NVIDIA, "CUDA for Tegra" application note | "In Tegra, device memory, host memory, and unified memory are allocated on the same physical SoC DRAM." "On platforms that have cudaDeviceProp::pageableMemoryAccess as 1 for the iGPU device, GPU L2 caching is enabled for system allocations created via mmap() or malloc() that can be accessed directly on the GPU without calling any CUDA APIs to register memory." "Starting CUDA 13.0, memory-sharing CUDA IPC APIs would be supported on Tegra platforms with open-source GPU driver (i.e. Jetson Thor and beyond)". "MPS ... is now available on Tegra platforms: Linux starting with CUDA 12.5". | S: the vendor documents that mmap'ed system memory is directly GPU-accessible on Thor. The mechanism is documented, not discovered by us. The note says nothing about sharing across processes, private mappings, write-fault servicing, fork, or fault containment. | V (doc) | https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html |
| NVIDIA open-gpu-kernel-modules, tag 580.173.02, `kernel-open/nvidia-uvm/uvm_ats_faults.c` | Lines 496 to 497: `if (vma->vm_flags & VM_WRITE) uvm_page_mask_or(write_fault_mask, write_fault_mask, prefetch_mask);`. Lines 632 to 634: `if (vma->vm_flags & VM_WRITE) { access_type = UVM_FAULT_ACCESS_TYPE_WRITE;`. Line 219 bounds the region by `UVM_VA_BLOCK_SIZE`. The prefetch mask is added to the write mask in a writable VMA, and every sub-region is then serviced as a write. | P1: this is the mechanism in the driver source, public since the open module release. The source is the citation for the cause. The measured consequence for model weights is the claimable part. | V (src) | https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.173.02/kernel-open/nvidia-uvm/uvm_ats_faults.c |
| LLMKube blog, "One GPU, four ways to share it: ten scenarios on a DGX Spark, and the headline finding I had to retract" (C. Maher, 2026-08-10) | Measured on GB10 with llama.cpp: cost per extra GPU-offloaded instance "GB10, 27B, --n-gpu-layers 99: 1.03x model", not shared. "mmap page-cache sharing is real, and it only helps while the weights stay CPU-resident. Once llama.cpp offloads, it allocates private CUDA device buffers and copies the weights into them. That happens on unified memory too. The page cache still holds one copy of the file; it simply is not what the GPU reads from." Time slicing with four active tenants: decode falls 4.3x, prefill 16x. "No GPU I have is MIG-capable: both GB10s and both RTX cards report nvidia.com/mig.capable=false". No MPS test. | S: independent measurement of the baseline we improve on, on GB10. P3: its statement that GB10 is not MIG-capable is a platform fact a reviewer may hold against the "MPS groups inside MIG instances" recommendation for DGX Spark. | V (doc) | https://llmkube.com/blog/gpu-sharing-four-ways |
| MLX Discussion #615, "Loading models with mmap" (2024) | Fetch-tool summary of the thread: a maintainer answers that "we can't mmap memory in a way that makes it available for the GPU as well" and that lazy loading reads weights when needed. A 2025 prototype hit Metal offset alignment constraints. The Ingestion Tax cites this thread and builds the missing producer. | S: MLX itself does not share mapped weights across processes by default. | M (tool summary of the page, quotes not checked against raw HTML) | https://github.com/ml-explore/mlx/discussions/615 |
| Ollama FAQ (docs/faq.mdx, main) | "If your system has sufficient available memory (system memory when using CPU inference, or VRAM for GPU inference) then multiple models can be loaded at the same time." `OLLAMA_NUM_PARALLEL`: "The maximum number of parallel requests each model will process at the same time". Concurrency on one model is inside one runner. Nothing on sharing one copy between runners. | S: no overlap. One process per loaded model, parallel slots inside it. | V (doc), through fetch tool | https://github.com/ollama/ollama/blob/main/docs/faq.mdx |
| Punica (Chen et al., arXiv 2310.18547, 2023) | "This allows a GPU to hold only a single copy of the underlying pre-trained model when serving multiple, different LoRA models". Base in device memory, one serving process per GPU, isolation by the runtime, adapters applied by a batched kernel (SGMV). | S: one base copy for many adapters is taken at the runtime level in one process. No process boundary, no MMU enforcement. | V (pdf) | https://arxiv.org/abs/2310.18547 |
| S-LoRA (Sheng et al., MLSys 2024, arXiv 2311.03285) | "S-LoRA stores all adapters in the main memory and fetches the adapters used by the currently running queries to the GPU memory. ... Unified Paging uses a unified memory pool to manage dynamic adapter weights with different ranks and KV cache tensors". Unmerged adapters over a shared base in one process. | S: same as Punica. The base is immutable only by runtime convention. | V (pdf) | https://arxiv.org/abs/2311.03285 |
| dLoRA (Wu et al., OSDI 2024) | "dynamically merge and unmerge LoRA adapters with the base model in each worker replica". Each replica holds its own base copy. Merging rewrites the replica's base weights in place. | S: the merged-adapter case that our extent divergence (copy+mremap) addresses exists here inside one process with a full private base per replica. No sharing of the unmodified extents across replicas. | V (pdf) | https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf |
| CaraServe (Li et al., arXiv 2401.11240, 2024) | Prior systems maintain "a shared copy of the base LLM on the GPU". CaraServe adds CPU LoRA processes: "we employ shared memory to enable fast data exchange between the base LLM process and multiple CPU LoRA processes, eliminating the need for data copying and serialization". Shared memory carries activations between processes, not the base weights. | S: multi-process LoRA serving with shared memory exists, but the base lives in one process. M: adjacent (zero-copy exchange of intermediate tensors between processes through shared memory, no revocation). | V (pdf) | https://arxiv.org/abs/2401.11240 |
| ServerlessLLM (Fu et al., OSDI 2024, arXiv 2401.14351) | "ServerlessLLM uses a model manager to load tensor data, while allowing the inference process to focus on initializing the model by setting the data pointers for each tensor. More specifically, the model manager allocates memory on GPUs and loads the binary data of the checkpoint ... [The inference process] acquires the base addresses for each GPU (i.e., CUDA IPC handles) from the model manager". | S: weights owned by one process and used by another through CUDA IPC, in device memory, for load speed. One consumer per model, no N-way sharing claim, no immutability enforcement. | V (pdf) | https://arxiv.org/abs/2401.14351 |
| Tangram (Zhu et al., arXiv 2512.01357, 2025) | "unified GPU memory pool for tensor-level parameter sharing across models, on-demand KV cache allocation". Built on ServerlessLLM and vLLM. Reuse of tensors retained in device memory across successive model loads. | S: tensor-level reuse across models in device memory, runtime-enforced. No host mapping, no OS isolation. | V (pdf), abstract and system overview only | https://arxiv.org/abs/2512.01357 |
| BlitzScale (Zhang et al., OSDI 2025, arXiv 2412.17246) | Autoscaling by loading parameters over the network or from other GPUs with "O(1) host caching". Not a sharing design. | No overlap with S beyond load-time motivation. | V (pdf), title and abstract level | https://arxiv.org/abs/2412.17246 |
| vLLM LoRA documentation | "Adapters can be efficiently served on a per-request basis with minimal overhead." Adapters are selected by the `model` field of a request and bounded by `max_loras` and `max_cpu_loras`. One server process holds the base. | S: runtime-level multi-adapter in one process. | V (doc), through fetch tool | https://docs.vllm.ai/en/latest/features/lora.html |
| MNN-LLM (arXiv 2506.10443, 2025) | Mobile engine with "DRAM-Flash hybrid storage" and a LoRA section: "using a base model in conjunction with multiple LoRA models". Single app process. | S: no cross-process sharing. | V (pdf), grep level | https://arxiv.org/abs/2506.10443 |
| ExecuTorch runtime overview | "Constant tensors point directly into the `.pte` file data, avoiding copies of that data." CPU-side zero-copy of constants from the program file. Nothing on GPU delegates reading the mapping or on sharing between processes. | S: adjacent on the CPU side only. | V (doc), through fetch tool | https://docs.pytorch.org/executorch/stable/runtime-overview.html |
| Apple, "Introducing Apple's On-Device and Server Foundation Models" (2024) | "Adapters are small collections of model weights that are overlaid onto the common base foundation model. They can be dynamically loaded and swapped". "The adapter models can be dynamically loaded, temporarily cached in memory, and swapped". "the original parameters of the base pre-trained model remain unchanged". The text does not say which process holds the base or how apps reach it. | S: a deployed product with one immutable base and per-feature adapters on a unified-memory device. The enforcement boundary is not documented. This is the product precedent a reviewer will cite. | V (doc), through fetch tool | https://machinelearning.apple.com/research/introducing-apple-foundation-models |
| Android AICore and Gemini Nano documentation | "Gemini Nano runs in Android's AICore system service". Apps do not carry the model: no "impact on your app's disk and runtime memory budget". "AICore is isolated from most other packages". The architecture figure shows a LoRA block without text. | S: the system-service alternative to S. One process owns the model and apps call it by IPC, so there is one copy by construction and isolation is by process boundary plus API. This is the design a reviewer will ask us to compare against. | V (doc), through fetch tool | https://developer.android.com/ai/gemini-nano |
| Microsoft, Phi Silica documentation (page dated 2026-10-02) | "Phi Silica is a powerful hardware-accelerated local language model ... can be integrated into your Windows apps through the Windows AI APIs". On Copilot+ PCs "Model is managed by the system". LoRA adapters "must be trained in the cloud". | S: same system-service pattern as AICore. No process-level sharing mechanism documented. | V (doc) | https://learn.microsoft.com/en-us/windows/ai/apis/phi-silica |
| Medusa (ASPLOS 2025), WarmServe (arXiv 2512.09472), Foundry (arXiv 2604.06664) | Cold-start work (CUDA graph and KV-init materialization, prewarming). From titles and search listings only. | No overlap expected with S. | NV | https://github.com/thustorage/Medusa |
| MLC-LLM, LM Studio, TGI multi-LoRA | Not opened in this pass. | Unknown. | NV | n/a |

Summary of group B. The verified set contains four distinct ways in which "one base copy" is already achieved: (1) in one process by the runtime (Punica, S-LoRA, vLLM, dLoRA), (2) by a system service that owns the model (AICore, Phi Silica, Apple), (3) across processes in device memory through CUDA IPC (cuda-llm-weight-share, ServerlessLLM), and (4) across processes through a shared file mapping that the GPU reads in place (llama.cpp Metal, The Ingestion Tax on Metal and Vulkan). For NVIDIA unified memory, PR #22120 proposed the host-pointer buffer for GB10 and was closed unreviewed, the upstream code at 153d324bc still reports `buffer_from_host_ptr = false` for CUDA, and an independent GB10 measurement finds 1.03x model size per extra instance. No verified work combines a host mapping read in place by an NVIDIA GPU with N serving processes, a hardened immutable mapping, and per-process divergence of extents.

### 14.3 Group D: MPS, MIG and time slicing isolation (P3)

| Work (venue, year) | What it does or states (with a quote where decisive) | Overlap with S / F / M / P1 / P2 / P3 | Tag | URL |
|---|---|---|---|---|
| NVIDIA MPS documentation, "When to Use MPS", section "Memory Protection and Error Containment" (current version, MPS v2 and v3) | "MPS client processes have fully isolated GPU address spaces. MPS supports a limited form of error containment: A fatal GPU fault generated by a Volta MPS client process will be contained within the subset of GPUs shared between all clients with the fatal fault-causing GPU. A fatal GPU fault generated by a Volta MPS client process will be reported to all the clients running on the subset of GPUs in which the fatal fault is contained, without indicating which client generated the error. Note that it is the responsibility of the affected clients to exit after being informed of the fatal GPU fault. Clients running on other GPUs remain unaffected by the fatal fault". The server moves from ACTIVE to FAULT and rejects new clients with `CUDA_ERROR_MPS_SERVER_NOT_READY` until the affected clients exit. Tegra limits on the same page: "MPS Client Termination, CUDA IPC and cooperative launches are not supported with MPS on Tegra platforms." "GPU compute modes are not supported on Tegra platforms." | **P3, first half: taken by the vendor documentation.** A fault of one MPS client reaches every client of the same server on that GPU. Clients on another GPU are unaffected. The text does not mention time-sliced processes outside MPS and gives no Tegra-specific containment behaviour. S: the sentence that CUDA IPC is not supported with MPS on Tegra means the CUDA IPC route to shared weights (14.2) is unavailable under MPS on our platform, which is an argument for the page-table route. | V (doc) | https://docs.nvidia.com/deploy/mps/when-to-use-mps.html |
| NVIDIA MPS documentation, "Common Tasks", "Using Static SM Partitioning" | "Starting with Driver version r610, partial error isolation is supported when static SM partitioning is enabled. Because an SM is owned only by one partition, the driver can attribute SM error state to the faulting partition/client. Clients in different SM partitions are isolated from each other's SM-triggered faults. On fault detection, the driver terminates work for the faulting client and prevents additional work from being submitted. ... It should be noted that this is a partial isolation not a guarantee that every possible GPU or system-level failure is isolated per process." Requires Ampere or newer. | **P3: a newer in-MPS alternative to "MPS groups inside MIG instances".** The documentation limits it to SM-triggered faults and calls it partial. It requires driver r610 and Ampere or newer, so whether the Thor driver in use offers it must be checked (the GB10 report in 14.1 and the paper by Liu et al. below both run 580). A reviewer will ask whether r610 static partitions remove the need for the MIG grouping. | V (doc) | https://docs.nvidia.com/deploy/mps/common-tasks.html |
| NVIDIA MPS documentation, "Architecture" | "a fatal fault from one client may bring down a different user's client that shares any GPU with the faulting client." "On pre-Volta MPS, the MPS server shuts down after encountering a fatal fault. On Volta MPS, the MPS server becomes ACTIVE again after all faulting clients have disconnected." The multi-user server option "is not supported on Tegra platforms." | P3: same as above. | V (doc) | https://docs.nvidia.com/deploy/mps/architecture.html |
| NVIDIA MIG User Guide, "Getting Started with MIG", section "MIG with CUDA MPS", and "Deployment Considerations" | "MPS and MIG can work together, potentially achieving even higher levels of utilization for certain workloads." "this mode is not supported when the GPU is in MIG mode as we use multiple MPS servers (one per MIG GPU instance)." The guide gives a script that "starts a separate MPS control daemon per MIG device". Deployment considerations: "CUDA MPS is supported on top of MIG." "CUDA IPC across GPU instances is not supported." Introduction: MIG provides "a defined quality of service (QoS) with fault isolation for different clients". | **P3, recommendation: the configuration "one MPS server per MIG instance" is a documented vendor workflow.** The guide presents it for utilization, not for bounding fault propagation, and reports no fault experiment. S: CUDA IPC cannot cross MIG GPU instances, so sharing weights across MIG instances is possible only through host memory. | V (doc) | https://docs.nvidia.com/datacenter/tesla/mig-user-guide/getting-started-with-mig.html |
| NVIDIA MIG User Guide, "Supported GPUs" | Table row "Thor iGPU, Blackwell, GB10B, TBD, Unified (N/A), 2" with the footnote "Thor iGPU supports at most two concurrent MIG instances: one compute instance and one graphics (+gfx) instance." DGX Spark is not listed. | **P3: platform limit.** On Thor the blast radius can be split into at most two groups. On DGX Spark (GB10) MIG is not listed, and an independent report finds `mig.capable=false` (14.2, LLMKube). The recommendation therefore does not transfer to "the same driver path" on GB10 as stated in the concept. | V (doc) | https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html |
| NVIDIA Jetson Linux Developer Guide r39.2.1, "Multi-Instance GPU (MIG)" | "Jetson Linux supports Multi-Instance GPU (MIG), which partitions GPU resources so that multiple workloads can run concurrently with hardware-level isolation." The procedure creates two instances on Thor with profiles 83 (`MIG 2g.0gb+gfx`) and 78 (`MIG 1g.0gb+me`). Profile names carry `0gb`: memory is not partitioned. No statement on MPS inside an instance and no fault experiment. | P3: the vendor states hardware-level isolation between the two Thor instances. Nothing on MPS groups or on which processes survive a fault. | V (doc) | https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/MiG.html |
| Liu et al., "Characterization-Guided GPU Fault Resilience in NVIDIA MPS" (arXiv 2605.26461, 2026) | "MPS has weak fault resilience: a fault in one process can terminate all co-running processes". "we systematically characterize GPU faults under MPS, classifying 19 fault scenarios and tracing their end-to-end processing paths from detection to fatality determination and propagation". It adds "a UVM-based fault isolation mechanism for MMU faults, intercepting them in the open GPU kernel module and confining their impact to the faulting client", and a standby recovery path for SM faults. Platforms: "NVIDIA RTX A6000, L40, A100, and H100 GPUs", driver 580.95.05, CUDA 13.0. It discusses r610 static SM partitions as a utilization and isolation trade-off. No Tegra, no unified-memory device, no ATS, no comparison with time-sliced survivors or with MPS per MIG instance. | **P3: the closest academic work. It takes the characterization of MPS fault propagation and goes further by fixing it in the driver.** What it leaves: the behaviour on a Tegra iGPU with ATS, the observation that time-sliced processes and the other MIG instance survive, and the latency cost of each mode on this device. | V (pdf) | https://arxiv.org/abs/2605.26461 |
| Pavlidakis et al., "Guardian: Safe GPU Sharing in Multi-Tenant Environments" (arXiv 2401.09290, 2024) | "MPS offers memory protection but has serious fault isolation issues, i.e., when a kernel performs an out-of-bounds (OOB) access, it results in crashing any other co-running application". "We have found that when a kernel of an MPS client performs an illegal memory access, both the MPS server and other co-running clients are terminated." "NVIDIA MIG resolves these issues using a hardware mechanism that provides memory and fault isolation by statically partitioning the GPU". Time sharing is described as the protected alternative. | **P3, first half: taken, with an experimental statement.** The contrast MPS (propagates) versus MIG and time sharing (isolated) is stated in the introduction. | V (pdf) | https://arxiv.org/abs/2401.09290 |
| Xing et al., "Towards Efficient and Practical GPU Multitasking in the Era of LLM" (arXiv 2508.08448, 2025) | Table 1 marks fault isolation as present for MIG, FGPU and TGS and absent for MPS, Orion, REEF, Paella, LithOS, BLESS and SGDRC. "LithOS, BLESS, and SGDRC ... like many other techniques, their implementations rely on MPS for spatial sharing, which compromises fault isolation due to the use of a shared CUDA execution context." | P3: a survey-level statement of the same trade-off. No proposal of MPS inside MIG for blast-radius control was found in the text (grep for "MPS inside", "within MIG"). | V (pdf), abstract, Table 1 and section 2 | https://arxiv.org/abs/2508.08448 |
| Coppock et al., LithOS (SOSP 2025, arXiv 2504.15465) | "In LithOS, applications run in separate address spaces and cannot access each other's memory. Illegal accesses lead to termination of the offending application. To handle other faults, LithOS enables graceful termination for common errors by intercepting signals and terminating the application without affecting other contexts." "MIG partitions the GPU's compute and memory resources along GPC boundaries, providing strong hardware isolation." | P3: adjacent. A GPU OS layer that claims per-application fault handling. No MPS-in-MIG grouping. | V (pdf), grep level | https://arxiv.org/abs/2504.15465 |
| Gilman and Walls, "Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads" (arXiv 2110.00459, 2021) | Performance study of "priority streams, time-slicing, and multi-process service (MPS)" on Ampere. "Time-slicing disallows separate applications from being executed on the GPU simultaneously". "MPS makes it possible to assign a proportional share of resources to each application". No fault propagation study (the word "fault" does not occur in a fault-containment sense). Footnote: "Jetson devices do allow time slice configuration". | P3: takes the latency side in general (MPS runs clients concurrently, time slicing serializes them). Does not take fault containment. | V (pdf) | https://arxiv.org/abs/2110.00459 |
| Orion (EuroSys 2023), REEF (OSDI 2022), TGS (NSDI 2023), Paella (SOSP 2023) | Not opened. Characterized only through Table 1 of arXiv 2508.08448 above (fault isolation: absent for Orion, REEF, Paella, present for TGS). | P3: indirect. | NV | n/a |
| KRYPTON | No GPU-sharing work of this name was located. | n/a | NV (not found) | n/a |

Summary of group D. The fact that a fatal fault of one MPS client reaches the other clients of the same server is in the NVIDIA documentation, in Guardian, and in a 2026 characterization paper that also fixes it for MMU faults. The configuration of one MPS server per MIG instance is a documented vendor workflow. What the verified set does not contain is a measurement on a Tegra unified-memory device of which processes survive a GPU fault under each sharing mode, and the framing of MPS-per-MIG-instance as a blast-radius bound. Two platform facts limit that framing: Thor supports at most two MIG instances, and GB10 is not listed as MIG-capable.

### 14.4 Group C: KV cache and session state sharing, forking and transfer (F, M)

The question asked of every row: is the sharing between sequences inside one serving process (runtime bookkeeping), or between OS processes through the MMU?

| Work (venue, year) | What it does or states (with a quote where decisive) | Overlap with S / F / M / P1 / P2 / P3 | Tag | URL |
|---|---|---|---|---|
| Wang, Ren, Gui, "ForkKV: Scaling Multi-LoRA Agent Serving via Copy-on-Write Disaggregated KV Cache" (arXiv 2604.06370, 2026) | "a novel disaggregated KV cache management mechanism inspired by the operating system (OS) primitive for subprocess creation: fork with copy-on-write (CoW) ... the massive bCache acts as the shareable and read-only memory pages of a parent process, the lightweight rCache serves as the unique CoW footprint of a child process". "When a new agent is launched, ForkKV performs a longest-prefix match to inherit the globally shared read-only bCache, forks the memory space by allocating memory exclusively for the agent's unique rCache". "we implement ForkKV on top of SGLang". The fork is a metaphor realized by a DualRadixTree inside one serving process. No OS fork, no page tables, no process boundary. Datacenter GPUs. | **F: the name, the analogy and the multi-LoRA agent motivation are taken.** The agent-forks-KV-with-CoW idea must be attributed to ForkKV. What it does not do: place the sharing boundary between OS processes, use the host MMU, or run on a unified-memory device. S: it assumes one base model in one process. | V (pdf) | https://arxiv.org/abs/2604.06370 |
| Kwon et al., vLLM / PagedAttention (SOSP 2023, arXiv 2309.06180) | "we introduce a reference count for each physical block". The block-level mechanism is "similar to the copy-on-write technique in OS virtual memory (e.g., when forking a process)". Used for parallel sampling, beam search and shared prefixes. Blocks are runtime objects in one process. | **F: block-granular CoW of KV state is taken since 2023, in one process.** Our extent-granular privatization is the same idea at the VMA level. The claim cannot be "CoW for KV". | V (pdf) | https://arxiv.org/abs/2309.06180 |
| Zheng et al., SGLang / RadixAttention (arXiv 2312.07104) | "RadixAttention, enables the automatic reuse of the KV cache across" calls, with language primitives "fork, join". In-process radix tree. | F: fork as a programming primitive over shared KV prefixes, in one process. | V (pdf), grep level | https://arxiv.org/abs/2312.07104 |
| Gim et al., Pie (SOSP 2025, arXiv 2510.24051) | API table: "export_kvpage(kv, name): Exports paged KV cache for use in other programs." "import_kvpage(name) -> list[KvPage]: Imports the paged KV cache." "copy_kvpage(q, src, dst): Copy KV cache contents at token-level." Inferlets are WebAssembly programs inside the serving system: "benefiting from its lightweight sandboxing". | **F and M: sharing of KV pages between isolated programs is taken, with isolation by a Wasm sandbox inside one serving process, not by the OS.** No revocation of the exporter's access is described in the lines read. This is the closest design to "agents exchange KV pages under an isolation boundary". | V (pdf), API table and section headers | https://arxiv.org/abs/2510.24051 |
| "AAFLOW+: Stateful Operator Abstraction with Zero-Copy Distributed KV Cache Orchestration for Multi-Agent Workflows" (arXiv 2607.10987, 2026) | "makes KV cache a first-class distributed systems object ... provides operators for KV materialization, transfer, fork, composition, and eviction. Its runtime enables zero-copy, transfer-aware execution". "The transfer operator modifies the state's location and ownership metadata but not its logical content." "A state object is moved or aliased from node n_a to node n_b". Results are "based on an analytical cost model" and a simulation profile. Cluster of A100 nodes, Arrow metadata, RDMA where available. | **F and M: the operator vocabulary (fork, transfer with ownership change, alias) is taken at the workflow level.** It is distributed and partly analytical. It does not implement a same-device, MMU-enforced move or revocation. | V (pdf), abstract and sections 2 to 5 at grep level | https://arxiv.org/abs/2607.10987 |
| Jeon and Yoo, GraniKV (arXiv 2608.15584, 2026) | "asymmetric granularity KV-cache paging layer" that splits a shared long prefix from a token-level pool for per-request suffixes in a production paged engine (SGLang, vLLM). In-process. | F: adjacent. Coarse granularity for the shared prefix and fine granularity for private suffixes is the same intuition as extent-granular privatization, at the runtime level. | V (pdf), abstract level | https://arxiv.org/abs/2608.15584 |
| Jeon et al., LRAgent (arXiv 2602.01053, 2026) | KV cache sharing for multi-LoRA agents by decomposing the cache into a "shared base component derived from pretrained" weights and adapter-dependent parts. In-process. | F: same space as ForkKV. | V (pdf), abstract level | https://arxiv.org/abs/2602.01053 |
| "Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices" (arXiv 2603.04428, 2026) | On an Apple M4 Pro: "persisting each agent's KV cache to disk in 4-bit quantized format", reload in 577 ms instead of 15.7 s re-prefill, safetensors files, a "BatchQuantizedKVCache for concurrent inference over multiple agents' quantized caches". "The individual techniques ... are" known, "This is a systems paper". One process, serialization to disk. | F and M: the on-device multi-agent KV memory problem on unified memory is taken as a problem statement. The mechanism is save and restore, not sharing or zero-copy handoff. E: must be cited for the on-device multi-agent setting. | V (pdf), abstract and introduction | https://arxiv.org/abs/2603.04428 |
| llama.cpp at 153d324bc: `include/llama.h` and `tools/server/README.md` | `llama_memory_seq_cp` (llama.h line 746): "Copy all tokens that belong to the specified sequence to another sequence", which shares KV cells between sequences of one context. `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` (line 904): "Keeps the tensor data on device buffers (i.e. not accessible in host memory, but faster save/load)". Server: "POST `/slots/{id_slot}?action=save`: Save the prompt cache of the specified slot to a file" and `action=restore`; `--cache-ram` host prompt cache; `--ctx-checkpoints`. Between processes the only path is a file round trip. | F: in-process prefix sharing between sequences is taken. M: cross-process transfer exists only by serialization to a file. No zero-copy cross-process path. | V (src) | https://github.com/ggml-org/llama.cpp/blob/153d324bcf86d220b235ca010eeb11213f32b5d1/include/llama.h |
| mlx-swift-lm issue #629, "RFC: Branchable KV cache with shared-prefix / copy-on-write semantics" (2026-09-18, open) | Request for "shared immutable prefix + copy-on-write suffix", "`fork()` returning an independent logical cache handle" on Apple silicon. Measured cost of the current deep copy: "checkpoint size: ~1.53 GB; save/copy/load round-trip: ~1.94 s" at about 63.6k tokens on a 35B model. Use cases listed: "agent branching and tool-call alternatives; speculative execution". In-process API request. | F: the same use cases and the same idea, as an open feature request on a unified-memory platform, one process. Shows the need is current and unmet there. | V (doc) | https://github.com/ml-explore/mlx-swift-lm/issues/629 |
| Yu et al., SuperInfer (MLSys 2026, arXiv 2601.20309) | "DuplexKV, an optimized rotation engine that enables full-duplex transfer over NVLink-C2C" between Hopper HBM and Grace DRAM on GH200. KV blocks are copied between tiers. One process. | M: KV movement on the NVIDIA coherent platform is taken as tier rotation by copying. Not a cross-process handoff, and not zero-copy. | V (pdf), abstract and introduction | https://arxiv.org/abs/2601.20309 |
| Liu et al., DroidSpeak (arXiv 2411.02820); KVCOMM (NeurIPS 2025, arXiv 2510.12872); KVLink (arXiv 2502.16002) | DroidSpeak: "enables KV cache reuse across distributed" LLMs that share a prefix. KVCOMM: "estimates and adjusts KV-caches for shared" content under "diverging prefixes" across agents. These are model-level reuse methods (what can be reused across models or contexts), not memory-level sharing mechanisms. KVLink was downloaded but not read beyond the title. | F: adjacent at the algorithm level. They answer when a KV is reusable, not how memory is shared. | V (pdf) abstract level for DroidSpeak and KVCOMM; NV for KVLink | https://arxiv.org/abs/2411.02820 |
| Lin et al., Parrot (OSDI 2024, arXiv 2405.19888) | "Sharing Prompt Prefix" across requests of one application through a prefix hash, on top of in-engine KV sharing. | F: prefix KV sharing by the serving engine. | V (pdf), grep level | https://arxiv.org/abs/2405.19888 |
| Mei et al., AIOS (COLM 2025, arXiv 2403.16971) | "we implement a context manager with snapshot and restoration capabilities". "the context manager snapshots the intermediate results. Upon resumption, it reloads this snapshot". An OS-named abstraction above the model server. No kernel memory mechanism. | F and M: named OS concepts for agents (context switch, snapshot). No MMU-level sharing. | V (pdf), grep level | https://arxiv.org/abs/2403.16971 |
| Gim et al., Prompt Cache (arXiv 2311.04934); Yao et al., CacheBlend (arXiv 2405.16444) | Downloaded, titles verified, not read in this pass. From memory: modular attention-state reuse and KV fusion for RAG. | F: adjacent at the algorithm level. | M | https://arxiv.org/abs/2311.04934 |
| "From Tensor Buffer to Distributed Memory Hierarchy: A Survey of KV Cache Management for LLM Serving" (arXiv 2607.02574, 2026) | Places ForkKV as "Forking semantics for branching/agentic KV" and mentions "reference counting and copy-on-write or append-only semantics" as standard. A grep for "across processes", "inter-process", "page table" and "mmap" in the survey returns no hit. | F: evidence, by absence in a 2026 survey, that process-level and MMU-level KV sharing is not an established category. Absence in one survey is weak evidence. | V (pdf), grep level | https://arxiv.org/abs/2607.02574 |
| Blog posts: A. Banerjee, "Prefill Once, Fan Out: KV Snapshot Sharing for Multi-Agent LLM Pipelines" (2026-06-09); S. Zolotukhin, "Prefix KV Cache Sharing on a Local GPU" (2026-07-21) | Fetch-tool summaries. First: llama.cpp on a GTX 1080, one process, "run prefill once, serialize the KV cache to a host buffer, `memcpy` it per branch, and restore it before decoding." Second: content-addressed KV blocks in the author's engine on an AMD GPU, eight agents sharing a 4,000-token system prompt, 4.7 GB versus 0.59 GB. | F: local multi-agent prefix sharing is being built by practitioners, in one process, by copy or by block deduplication. | M (tool summaries) | https://zolotukhin.ai/blog/2026-07-21-the-system-prompt-a-local-agent-swarm-caches-eight-times-over/ |

Summary of group C. Copy-on-write forking of KV state is taken as an idea (vLLM 2023, SGLang, ForkKV 2026, Pie's export and import of KV pages, AAFLOW+ operators). Every verified instance keeps all agents inside one serving process and implements sharing with reference counts or trees in the runtime. No verified work forks or moves KV state between OS processes through the page table on a GPU, and none revokes the sender's mapping on a move. The nearest cross-process path in the verified set is a file round trip (llama.cpp slot save and restore).

### 14.5 Group E: on-device LLM serving systems a reviewer expects (2024 to 2026)

| Work (venue, year) | What it does or states (with a quote where decisive) | Overlap with S / F / M / P1 / P2 / P3 | Tag | URL |
|---|---|---|---|---|
| Yin et al., "LLM as a System Service on Mobile Devices" (LLMaaS / LLMS, arXiv 2403.11805, 2024) | "we propose a new paradigm of mobile AI: LLM as a system service on mobile devices (LLMaaS) ... such a system service is stateful: LLMs execution often needs to maintain persistent states (mainly KV cache) across multiple invocations. To minimize the LLM context switching overhead under tight device memory budget, this work presents LLMS, which decouples the memory management of app and LLM contexts with a key idea of fine-grained, chunk-wise, globally-optimized KV cache compression and swapping." One service process owns model and contexts. | S: the system-service alternative (one copy by construction). F and M: per-app contexts are compressed and swapped by the service, not shared, forked or handed over between processes. P1 to P3: none. | V (pdf), abstract | https://arxiv.org/abs/2403.11805 |
| Yuan et al., "Mobile Foundation Model as Firmware" (MobiCom 2024, arXiv 2308.14363) | "This foundation model functions akin to firmware, unmodifiable by apps or the OS, exposed as a system service to Apps. They can invoke this foundation model through a small, offline fine-tuned 'adapter' for various downstream tasks." | **S: the immutable shared base plus private per-app adapters is taken as an architecture vision**, with immutability by placing the model in a system service on an NPU. Not by a sealed mapping and the MMU. F, M, P1 to P3: none. | V (pdf), abstract | https://arxiv.org/abs/2308.14363 |
| Shen et al., EdgeLoRA (MobiSys 2025, arXiv 2507.01438) | "An Efficient Multi-Tenant LLM Serving System on Edge Devices", built against llama.cpp, evaluated on "Jetson AGX Orin, Jetson Orin Nano and Raspberry Pi 5". "heterogeneous memory management, leveraging intelligent adapter caching and pooling"; batch LoRA inference "combines inference for pretrained weights and LoRA adapters into a unified process". Serves over 1,000 adapters on AGX Orin. | **S: multi-adapter serving over one base on a Jetson is taken, inside one process.** This is the direct edge baseline for the adapter variant of S. No process isolation between tenants, no in-place host mapping. F, M, P1 to P3: none. | V (pdf), abstract and introduction | https://arxiv.org/abs/2507.01438 |
| Su, "Execution-State Capsules: Graph-Bound Execution-State Checkpoint and Restore for Low-Latency, Small-Batch, On-Device Physical-AI Serving" (arXiv 2606.20537, 2026) | Checkpoint and restore of a closed buffer set bound to a CUDA graph, "turning restore, fork, and rollback of a session into a single copy of that buffer set". "a hierarchical planner–actor hand-off is the contract's zero-copy buffer pass": "A low-rate planner and a high-rate actor co-host in one context". Replicated on "a Jetson AGX Thor (sm_110) and a DGX Spark (GB10, sm_121)"; "GPU-resident snapshot and restore are sub-millisecond". | **F and M on our exact platforms.** Fork is a full device-to-device copy of the session state, not copy-on-write. The hand-off is zero-copy but between models in one CUDA context, with no process boundary and no revocation. This is the work a reviewer will name first for F and M on Thor. S, P1 to P3: none. | V (pdf), abstract and grep of sections 5 to 9 | https://arxiv.org/abs/2606.20537 |
| "Agent Memory Below the Prompt" (arXiv 2603.04428, 2026) | See 14.4. Multi-agent KV persistence on Apple M4 Pro, 4-bit KV on disk, one process. | F and M: problem statement overlap on a unified-memory edge device. | V (pdf) | https://arxiv.org/abs/2603.04428 |
| Si et al., The Ingestion Tax (arXiv 2608.12114, 2026) | See 14.1. | S: closest. P1: adjacent. | V (pdf) | https://arxiv.org/abs/2608.12114 |
| "Echo: Merging Host–Device Buffers to Avoid Redundant Data Movement on Unified-Memory SoCs" (arXiv 2609.05635, 2026) | "GPU applications on unified-memory (UMA) edge platforms often inherit a discrete-GPU memory abstraction in which they allocate one buffer for the CPU, another for the GPU, and copy data between them". A source and profile guided transformation that merges host and device buffers into one managed buffer, on "three NVIDIA Jetson platforms (Orin Nano, AGX Orin, and AGX Thor)". Vision and signal benchmarks, single process. | S: same platform family and the same observation (redundant copies on Jetson UMA), applied to activations and I/O buffers inside one program. No weights, no cross-process sharing, no file mapping. Must be cited as concurrent Jetson UMA work. | V (pdf), abstract and introduction | https://arxiv.org/abs/2609.05635 |
| "SiliconBench: Speed, Memory, and Fidelity for LLM Serving on Unified-Memory Desktops" (arXiv 2609.19169, 2026) | Benchmark of nine Apple Silicon serving engines with "DGX Spark provides a complementary serving-performance reference". "Explicit memory budgets do not guarantee memory headroom: two stacks complete every request while memory use approaches physical capacity". Concurrency 1 to 16 inside one engine. | S: motivates memory discipline on unified-memory machines. Does not study multi-process sharing. A reviewer will expect our throughput numbers to be placed next to its DGX Spark reference. | V (pdf), abstract | https://arxiv.org/abs/2609.19169 |
| Schieffer et al., "Harnessing Integrated CPU-GPU System Memory for HPC: a first look into Grace Hopper" (ICPP 2024, arXiv 2407.07850) | Characterizes "first-touch policy, page table entry initialization, page sizes, and page migration" for system-allocated memory under ATS on GH200. "page faults are now generated by the SMMU and can be directly handled by the operating system's page fault handling mechanism". A grep for fork, copy-on-write, write fault, read-only and file-backed returns no hit. | P1 and P2: the academic baseline for the cost of ATS faults on system memory. It does not cover write-fault upgrade in writable mappings, fork, or file mappings. | V (pdf), grep level | https://arxiv.org/abs/2407.07850 |
| vLLM blog, "vLLM on the DGX Spark: Architecture, Configuration, and Local Evaluation" (2026-06-01) | "CPU, GPU, model weights, etc. share one 128 GB pool." `--gpu-memory-utilization` "should leave room for the operating system, kernel page cache, container runtime, KV cache growth, and any other process touching that same memory." Nothing on several processes sharing weights, MPS or time slicing. | S: the vendor-side engine guidance for GB10 treats the page cache as a competitor for memory, not as the weight store. | V (doc), through fetch tool | https://vllm.ai/blog/2026-06-01-vllm-dgx-spark |
| unsloth PR #10704, "Studio: do not lose a DGX Spark's memory pool to the page cache when fitting context" (merged 2026-09-14) | Fetch-tool summary: "cudaMemGetInfo's free half on an integrated CUDA SoC is the kernel's MemFree, which counts the page cache as used." | S: an accounting hazard for our design. If weights live in the page cache, engines that size the KV cache from `cudaMemGetInfo` will see them as used memory. A reviewer familiar with GB10 will ask how our memory numbers were measured. | M (tool summary) | https://github.com/unslothai/unsloth/pull/10704 |
| Yi et al., EdgeMoE (arXiv 2308.14352) | "non-expert weights are held in device memory; while expert weights are held on external storage and fetched to" memory on demand. Single process. | S: none. Listed for completeness. | V (pdf), abstract | https://arxiv.org/abs/2308.14352 |
| PowerInfer-2, LLM in a Flash, HeteroLLM, llm.npu, MLC-LLM | Not opened in this pass. | Expected: no overlap with S, F, M beyond single-process memory management. | NV | n/a |

Summary of group E. The on-device literature has the immutable shared base with per-app adapters as a system-service architecture (firmware model, LLMaaS, AICore) and as an in-process engine on Jetson (EdgeLoRA). It has session fork and zero-copy hand-off on Thor and DGX Spark inside one CUDA context (Execution-State Capsules). It does not have the process as the unit of isolation with the OS page table as the sharing mechanism.

### Verdicts

**S (share): PARTIALLY TAKEN.**
Closest work: The Ingestion Tax (arXiv 2608.12114), with llama.cpp PR #22120 as the closest attempt on the same engine and vendor.
What is taken: file-backed weights mapped once and read in place by the GPU, N processes on one copy (Apple Metal, measured to N = 4), host-pointer import in llama.cpp on a Vulkan iGPU, in-place reads on GH200 through the NVIDIA coherent path (probes and module rebinding), the proposal of `buffer_from_host_ptr` for CUDA on GB10 (PR #22120, unmerged and unmeasured), and cross-process sharing of llama.cpp CUDA weights through CUDA IPC (cuda-llm-weight-share).
What remains claimable: the first measured in-place weight path in a real CUDA engine on an NVIDIA unified-memory SoC that shares one page-cache copy across serving processes through the host page table, together with the hardened form (sealed memfd, private read-only mapping) and per-process divergence of only the extents an adapter changes.
Citations that must stand next to the claim: The Ingestion Tax; llama.cpp Metal backend and Discussion #21223; llama.cpp PR #26081 and commit 153d324bc; llama.cpp PR #22120; cuda-llm-weight-share and ServerlessLLM (CUDA IPC route); NVIDIA CUDA for Tegra note (mmap memory is directly GPU-accessible on Thor); Punica, S-LoRA, dLoRA, EdgeLoRA (one base in one process); Mobile Foundation Model as Firmware, LLMaaS and AICore (system-service route); LLMKube GB10 measurement (baseline of 1.03x model per instance).

**F (fork): PARTIALLY TAKEN as an idea, OPEN IN THE VERIFIED SET at the process and MMU level. Not built by us.**
Closest work: ForkKV (arXiv 2604.06370).
What is taken: copy-on-write forking of KV state for branching agents and multi-LoRA agents (ForkKV, vLLM PagedAttention, SGLang fork, Pie KV page export and import), and session fork on Thor and DGX Spark by full copy (Execution-State Capsules).
What remains claimable: forking session state between OS processes with the page table as the sharing mechanism and extent-granular privatization, motivated by a measured per-page CoW cost on this platform. Since F is at the idea stage, only the cost measurement and the design argument can be claimed, not a result.
Citations: ForkKV; vLLM; SGLang; Pie; AAFLOW+; Execution-State Capsules; LRAgent; llama.cpp `llama_memory_seq_cp` and slot save and restore; mlx-swift-lm issue #629.

**M (move): OPEN IN THE VERIFIED SET at the mechanism level, PARTIALLY TAKEN at the abstraction level.**
Closest work: Execution-State Capsules (arXiv 2606.20537) for the same platform, and AAFLOW+ (arXiv 2607.10987) for the ownership-transfer operator.
What is taken: zero-copy hand-off of buffers between models in one CUDA context on Thor and DGX Spark; a transfer operator that changes ownership metadata in a distributed workflow runtime; export and import of KV pages between sandboxed programs in one serving process; shared-memory exchange of tensors between serving processes (CaraServe).
What remains claimable: a cross-process hand-off of GPU-visible state on one device in which the sender's access is revoked by the MMU, if it is built and measured.
Citations: Execution-State Capsules; AAFLOW+; Pie; CaraServe; SuperInfer (KV rotation on GH200 by copy); llama.cpp slot save and restore (file round trip as the baseline).

**P1 (writable private mapping becomes a private copy): PARTIALLY TAKEN.**
Closest work: GitHub issue #81 of MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark (2026-09-28).
What is taken: on GB10 with driver 580.173.02, UVM services every GPU fault in a writable mapping as a write, with the file and line numbers, and "a private writable mapping copies every page the GPU touches into anonymous memory". The driver source confirms the code path at tag 580.173.02. The same class of pitfall is documented for Metal (geistlib #541, The Ingestion Tax section 5.3).
What remains claimable: the amplification and its trigger for model weights, namely that the prefetch block is upgraded as a whole so one 4 KiB page costs 2 MiB and 0.2% of pages cost the whole model, the role of the accessed flag, and the correction of the statement in The Ingestion Tax that Linux keeps private file mappings file-backed.
Citations: issue #81; `uvm_ats_faults.c` at 580.173.02, lines 496 to 497 and 632 to 634; geistlib #541; The Ingestion Tax.

**P2 (post-fork GPU write cost): OPEN IN THE VERIFIED SET, with thin search coverage.**
Closest work: the nvidia-uvm source itself. `uvm_ats_faults.c` at 580.173.02, lines 545 to 565, documents an SMMU TLB invalidation on each write fault ("WAR for older kernel versions missing an SMMU invalidate on RO -> RW transition") and "use the hammer of always invalidating the GPU's TLB on each fault" when the page size is 4K. Schieffer et al. (ICPP 2024) characterize first-touch faults of system memory under ATS on GH200 without fork.
What remains claimable: the measured 2.7 s/GiB cost of GPU writes after fork in a CUDA process on Thor, its attribution to per-page faults plus invalidation, and the three fixes. No published measurement of this was found. Only two web queries targeted P2, so absence is weakly supported.
Citations: `uvm_ats_faults.c` lines 545 to 565 and `uvm_ats_sva.h` (`UVM_ATS_SMMU_WAR_REQUIRED`); Schieffer et al. 2407.07850; NVIDIA CUDA for Tegra note.

**P3 (MPS blast radius, MPS groups inside MIG instances): PARTIALLY TAKEN. The propagation fact is TAKEN.**
Closest work: Liu et al., "Characterization-Guided GPU Fault Resilience in NVIDIA MPS" (arXiv 2605.26461), with the NVIDIA MPS documentation as the primary statement.
What is taken: a fatal fault of one MPS client reaches all clients of the same server on that GPU (NVIDIA MPS documentation, Guardian, Liu et al.); MIG gives fault isolation; one MPS server per MIG instance is a documented NVIDIA workflow; r610 adds partial error isolation between static SM partitions inside MPS.
What remains claimable: the measurement on a Tegra unified-memory device of which processes survive under each mode (time-sliced processes and the other MIG instance survive, MPS co-clients do not), the latency numbers on this device, and the framing of MPS-per-MIG-instance as a blast-radius bound for on-device agents. Two limits must be stated: Thor supports at most two MIG instances, and DGX Spark is not listed as MIG-capable.
Citations: NVIDIA MPS documentation (error containment, static SM partitioning, Tegra limits); NVIDIA MIG User Guide (MIG with CUDA MPS, supported GPUs footnote for Thor); Jetson Linux Developer Guide MIG page; Liu et al. 2605.26461; Guardian 2401.09290; Xing et al. 2508.08448; Gilman and Walls 2110.00459.

### What a reviewer will ask first

- **How is S different from The Ingestion Tax and from PR #22120?** Both precede us. The honest answer is: an NVIDIA unified-memory SoC, a real CUDA engine with measured throughput, sharing across processes by private read-only mappings of a sealed object, and divergence of extents for adapters. The paper must say that mapping weights and reading them in place is known, and that our Linux and nvidia-uvm result contradicts their sentence that Linux keeps private file mappings file-backed (for writable mappings).
- **Why not CUDA IPC, or one server process with many slots or adapters?** cuda-llm-weight-share already shares llama.cpp CUDA weights across processes, and the Tegra note says memory-sharing CUDA IPC is supported on Thor from CUDA 13.0. EdgeLoRA, S-LoRA and AICore show the single-process answer. The paper needs a head-to-head with both, and the argument must rest on what the page-table route adds: immutability enforced by the MMU, no master process, evictable file pages, sharing across MIG instances (CUDA IPC cannot cross GPU instances) and under MPS on Tegra (CUDA IPC is not supported with MPS on Tegra).
- **Is P1 new, given issue #81 and the open driver source?** The cause is public. Only the amplification (2 MiB block per 4 KiB page, 0.2% of pages for the whole model), the accessed-flag trigger and the consequence for a weight-sharing design are ours. The issue and the source lines must be cited.
- **Does the P3 recommendation apply beyond Thor, and does r610 make it obsolete?** Thor has at most two MIG instances (one compute, one +gfx), GB10 is reported as not MIG-capable, and NVIDIA documents partial error isolation between static SM partitions from r610. The claim "same driver path as DGX Spark" does not extend to MIG. The paper should scope P3 to Thor and say which fault classes the SM-partition isolation does not cover (the documentation names only SM-triggered faults and calls the isolation partial).
- **F and M are not built. What is the evidence?** ForkKV, vLLM, Pie and Execution-State Capsules implement fork or hand-off inside one process, the last one on Thor and DGX Spark. A reviewer will ask what a process boundary buys for agents over a Wasm sandbox or one CUDA context, and will want at least a prototype number (fork latency and memory against a full copy of the capsule kind, 1.94 s per 1.53 GB in mlx-swift-lm #629, and the file round trip of llama.cpp slots).

### Not opened

- Orion (EuroSys 2023), REEF (OSDI 2022), TGS (NSDI 2023), Paella (SOSP 2023): characterized only through Table 1 of arXiv 2508.08448.
- Medusa (ASPLOS 2025), WarmServe, Foundry: titles only. WarmServe and Foundry PDFs were downloaded and not read.
- MLC-LLM, LM Studio, TGI multi-LoRA, PowerInfer-2, LLM in a Flash, HeteroLLM, llm.npu.
- Prompt Cache, CacheBlend, KVLink: PDFs downloaded, titles verified, text not read.
- MLX Discussion #615, unsloth PR #10704, the two KV-sharing blog posts, the NVIDIA forum post for cuda-llm-weight-share: read only through a fetch tool that returns a model-written summary. Quotes in those rows are as returned by the tool.
- Fusco et al., "Understanding Data Movement in Tightly Coupled Heterogeneous Systems" (arXiv 2408.11556), Cooper et al. (ICS 2024) on shared virtual memory, and the companion paper arXiv 2608.12103 beyond a grep.
- InfiniLoRA (arXiv 2604.07173), Flex-MIG (arXiv 2511.09143), CacheWise (arXiv 2606.16824), MemServe: located by search, not opened.
- NVIDIA forum threads on MPS client termination and on MPS sticky errors on Jetson Thor: located by search, not opened.
- llama.cpp Discussion #16578 (DGX Spark performance), llama.cpp PR #27311 beyond its description, llama.cpp issue #20697, llama.cpp issue #26448 ("run MoE expert weights from host RAM via PCIe DMA (no H2D copy)", discrete GPUs, located by search).
- NVIDIA CUDA documentation on fork after CUDA initialization: not located.
- The caller's own llama.cpp patch at commit 6f767fe96 was not inspected. The comparison with PR #22120 (registration with `cudaHostRegister` versus no registration) is therefore conditional.
- KRYPTON: no GPU-sharing work of this name was found.

### Queries tried

Direct fetches: arxiv.org/abs/2608.12114 and its PDF; api.github.com issue 81 of MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark; github.com/ggml-org/llama.cpp/discussions/21223 (HTML); api.github.com geisten/geistlib issue 541; llama.cpp PR 26081, issue 28160, issue 21827, PR 22120 (body, comments, diff), commit 153d324bc (files and patch), PRs 25863 and 27311; raw sources of llama.cpp at 153d324bc (ggml-cuda.cu, ggml-metal.cpp, ggml-metal-device.m, include/llama.h, tools/server/README.md, docs/build.md); raw nvidia-uvm sources at tag 580.173.02 (uvm_ats_faults.c, uvm_ats_sva.c, uvm_ats_sva.h); pontostroy/cuda-llm-weight-share README and source at commit 15bcecae; docs.nvidia.com MPS (ten pages), MIG User Guide (six pages), CUDA for Tegra note, Jetson Linux r39.2.1 MIG page; 40 arXiv PDFs converted with pdftotext.

GitHub issue search (repo:ggml-org/llama.cpp): "buffer_from_host_ptr cuda"; "Jetson zero-copy mmap unified memory"; "\"DGX Spark\" mmap copy"; "cuda integrated tegra host pointer weights"; "share weights between processes GPU"; "cuda iGPU mmap zero copy in place"; "cudaHostRegister mmap"; "GB10 zero-copy weights"; "Thor mmap weights CUDA unified". PR #22120 is the only CUDA host-pointer buffer PR or issue these nine queries return.

Web search: "geistlib issue 541 nvidia-uvm ATS writable mapping write fault private copy"; "Tangram serverless LLM loading GPU memory reuse arXiv"; "Medusa ASPLOS 2025 accelerating serverless LLM inference materialization arXiv"; "ForkKV KV cache fork copy-on-write multi-agent LoRA arXiv"; "copy-on-write KV cache fork across processes GPU mmap shared prefix agents branching 2026 arXiv"; "\"DGX Spark\" OR \"Jetson\" share model weights across processes zero-copy mmap GPU reads in place unified memory llama-server multiple instances one copy"; "MPS inside MIG fault isolation blast radius ... fault propagation one client crash kills other clients measurement 2025 2026 arXiv GPU sharing LLM"; "Jetson Thor MPS fault other clients killed Xid time-slicing survive MIG instance isolation forum"; "SuperInfer SLO-aware rotary scheduling ... MLSys 2026"; "Pie programmable serving system emerging LLM applications SOSP 2025"; "on-device multi-agent LLM memory unified memory Jetson Thor OR \"DGX Spark\" benchmark paper 2026 arXiv"; "multi-tenant LoRA LLM serving edge devices Jetson shared base model memory MobiSys 2025 OR 2026 EdgeLoRA"; "fork() in CUDA process copy-on-write GPU write fault cost ATS SVA SMMU \"after fork\" nvidia-uvm unified memory Jetson OR \"Grace Hopper\" MADV_DONTFORK slow"; "\"Grace Hopper\" OR GH200 ATS \"system-allocated memory\" page fault cost first touch GPU writes study"; "sealed memfd model weights GPU shared across processes MAP_PRIVATE read-only LoRA adapter merge copy-on-write mremap LLM serving unified memory"; "multiple agent processes share one copy of LLM weights on-device unified memory GPU reads page cache in place MAP_SHARED OR mmap Jetson OR DGX Spark OR GB10 CUDA 2026".

Not found, with the queries above: any work that maps a sealed memfd of weights MAP_PRIVATE|PROT_READ for an NVIDIA GPU and diverges extents with mremap; any measurement of GPU write cost after fork under ATS; any paper that proposes MPS groups inside MIG instances specifically to bound fault propagation; any process-level or MMU-level fork or move of KV state on a GPU.

## 15. What Section 14 changes, and what was measured in response

The record for this concept is `RESEARCH_LLM_SHARE_2026-10-05.md`.

1. **The four closest sources exist and were opened.** Reading weights in
   place from a shared mapping is prior art on Apple hardware and was
   proposed, without measurement, for the CUDA backend of the same engine.
   The record says so in its claim and does not claim the idea.
2. **S is now measured in a real engine.** A 59-line change to llama.cpp's
   CUDA backend makes the mapped model file the weight buffer
   (`llm_share/inplace_weights.patch`). Six paired repetitions: identical
   text, generation 0.940 times the device copy on 4 KiB pages
   [0.914, 0.967] and 0.997 times on 2 MiB pages [0.971, 1.025], load 942 to
   517 ms (270 ms on 2 MiB pages), 4.4 GiB less memory per process. Eight
   processes hold 10.0 instead of 40.8 GiB. Five processes with four
   adapters produce the texts of their device-copy counterparts in 30 of 30
   comparisons.
3. **"Why not one server process?" is measured, and the answer limits the
   claim.** One process that batches eight sequences generates about three
   times as much as eight processes (89.7 against 29.4 tokens/s in one MIG
   instance). Separate processes do not raise throughput on this GPU. The
   record therefore positions in-place sharing for agents that are separate
   programs anyway and for one batching server per MIG instance, which is
   measured at 149 tokens/s for sixteen sequences in 6.6 instead of 11.1 GiB.
4. **"Why not CUDA IPC?" is measured, and one expectation of Section 14 did
   not hold.** CUDA IPC shared a device allocation between two processes of
   one MIG instance, also when both were clients of one MPS server (5 of 5),
   which the documentation quoted in Section 14.3 describes as unsupported on
   Tegra. It did not cross MIG instances (0 of 5, `cudaErrorInvalidValue`).
   The host page table shared in all three placements. The difference that
   remains is reach across MIG instances and the absence of an owner process
   and of write access.
5. **P3 is measured on this device** with 174 runs: memory sharing is the
   same under MIG, time slicing, and MPS; a GPU fault of one MPS client ends
   the work of the other clients of that server (0 of 18 survive) and of
   nobody else (36 of 36). The propagation fact is cited, not claimed. The
   driver in use predates the partial isolation of a later release.
6. **A correction to a first attempt.** The first route check reused a probe
   that refuses two processes in one MIG instance; two of its rows were not
   measurements. That result was withdrawn and the check repeated with a new
   probe.
7. **F and M remain unbuilt for the engine.** The record reports the
   engine's own prompt cache as the baseline (a 4,081-token prefix restored
   in 1.5 s instead of 6.8 s, at 223 MiB per agent) and makes no result
   claim for either.

## 16. Addendum: key-value cache extents (publish, attach, fork across processes)

Provenance: produced by a fifth delegated search pass on 2026-10-06, after
the engine had a working path that shares the key-value cache of a prompt
prefix between serving processes by mapping (`llm_share/kv_extents.patch`).
The brief described the mechanism and asked for work that takes six claims:
C1 (one physical copy of a computed prefix cache shared by separate
processes by mapping, each with private continuation state), C2 (the same
across GPU partitions, where GPU-level sharing is unavailable), C3 (an
immutable shared extent plus a private tail chosen to avoid copy-on-write,
with the reasoning that copy-on-write is harmful on a GPU attached through
the IOMMU), C4 (state files that name rows instead of carrying them), C5 (a
cache whose memory follows the tokens in use), and C6 (fork or move of a
live serving context between processes without a copy). The text below is
the pass's own. The pass ended when its session ended; its written
deliverable was complete.

Checked by us against the saved primary texts after the pass: the Omni-Flow
passages ("binds IPC alias", "one physical copy", "immutable conditioning KV
while keeping changing denoising state in private writable pages", role
processes over ZMQ IPC, H20 GPUs); the body of llama.cpp pull request #21792
(`MAP_SHARED` file, sidecar metadata, "CPU only ... GPU device memory cannot
be mmap'd"); SGLang issue #35648 (helper process, CUDA-IPC handles, "one
physical GPU", "immutable published prefixes" as a question); issue #81
(write-fault service, `mprotect`); the vAttention passages on `cuMemMap` and
the background thread; the CUDA for Tegra note on memory-sharing IPC from
CUDA 13.0; and the Execution-State Capsules statements on copies. All are as
quoted. One phrase attributed to vLLM pull request #58439 ("shared by every
process that maps the same files") was not found in the saved body and must
not be quoted; the rest of that row is as quoted.

### 16.1 Method, and what was and was not opened

**How sources were read.** Three access paths were used; every row below is tagged with the one that applies.

- **[D] direct text**: PDF downloaded with `curl` and converted with `pdftotext`, or raw file / GitHub REST API JSON fetched with `curl`, then read or grepped by me. Quotes tagged [D] are verbatim from that text.
- **[W] WebFetch**: page fetched and summarised by the fetch tool's small model. Quotes tagged [W] are as returned by that tool; treat them as near-verbatim, not as checked against the page bytes.
- **[S] search snippet only**: the page itself was not opened. Not evidence; listed only as a lead.

Local copies of everything tagged [D] were kept in a session scratch directory (78 text files) and are not part of the repository.

**Query formulations used** (web search plus GitHub issue search API on ggml-org/llama.cpp, vllm-project/vllm, sgl-project/sglang, ml-explore/mlx-lm, ollama/ollama): "KV cache sharing across processes zero-copy mmap shared memory", "ForkKV / copy-on-write KV cache", "copy-on-write KV cache LLM process fork shared memory mmap llama.cpp", "Jetson unified memory KV cache sharing across processes", "os.fork llama.cpp after prompt evaluation copy-on-write", "LMCache multiprocess shared memory CUDA IPC", "vLLM share KV cache between instances same GPU", "Apple silicon MLX share prompt cache between processes mmap", "on-device multi-agent LLM serving shared KV cache unified memory Jetson DGX Spark", "DGX Spark multiple instances share KV cache", "lazy KV cache allocation demand paging", "KV cache memfd / shm_open / MAP_SHARED prefix", "share data between MIG instances", "SMMUv3 SVA mmu notifier invalidate cost", "SVA copy-on-write fork IOTLB invalidation", "HMM device read fault write copy-on-write", "uvm_ats_faults write fault writable mapping", "CXL shared memory KV cache".

**Could not be opened (stated, not inferred):**

| Source | Reason | What I used instead |
|---|---|---|
| patchwork.ozlabs.org/patch/972605 (Brucker, "iommu/sva: Track mm changes with an MMU notifier") | bot wall (Anubis) | [S] only: snippet says ATC invalidation "may take up to a minute according to the PCI spec". Unverified. |
| docs.nvidia.com/dynamo/.../kvbm/architecture.html | 404 | KVBM docs in the `ai-dynamo/dynamo` repo [D] |
| docs.pytorch.org multiprocessing notes | JS-only page | raw `multiprocessing.md` from pytorch repo [D] |
| pypi.org/project/pion-vllm-mlx | JS-only page | PyPI JSON API + raw `doc/shared_kv_cache.md` [D] |
| mayankbhatia.com/pdfs/IOMMU-GPU.pdf ("AI Workloads Performance with Safe IO Memory Protection") | download failed | not used |
| GraniKV (arXiv 2608.15584) | PDF not retrievable; search result says withdrawn | [S] only |
| Medusa (ASPLOS'25) | raw README path 404 | [S] only |
| ICS'24 "Shared Virtual Memory: Its Design and Performance Implications" (Cooper et al.) | not opened in full | [S] only: it is a study of AMD SVM under GPU memory oversubscription, not of IOMMU invalidation cost |
| CachyLLama (llama.cpp fork) | only a third-party article (note.com) [W]; no repo found | weak evidence |
| NVIDIA documentation of MIG on Jetson Thor | not found; the MIG user guide page I opened does not mention Jetson/Tegra | none |
| "POS" (listed next to PhoenixOS) | no separate work found; treated as the PhoenixOS abbreviation (PhOS) | PhoenixOS paper [D] |

**Limits of this audit.** Negative results ("no work found that does X") mean: not found with the queries above in the sources above. Several of the closest items are very recent (Aug-Oct 2026) and were found only by GitHub issue search, so more may exist in forks and unmerged PRs. Papers tagged [D] were read at the level of abstract plus targeted grep and the surrounding passages, not cover to cover.


### 16.2 Per-work findings (works named in the brief)

Legend for the last column: "in-proc" = sharing inside one process; "x-proc" = across OS processes; "copy" = KV bytes are copied into each consumer's cache; "map" = the same physical pages are referenced; "dev" = device (CUDA) memory; "host-PT" = host page tables.

#### 16.2.1 In-process prefix sharing

| Work | URL opened | What it actually does | Relation to C1-C6 |
|---|---|---|---|
| vLLM automatic prefix caching | https://docs.vllm.ai/en/latest/design/prefix_caching.html [W] | KV blocks hashed by tokens and prefix; requests with the same prefix reference the same blocks; per-block reference count ("The number of requests using this block now"); LRU free queue. Scoped to one engine's block pool. | In-proc, map (block table), dev. Not C1/C2. Logical form of C3's layout: shared full blocks are never rewritten, new tokens go to new blocks. |
| SGLang RadixAttention | https://arxiv.org/pdf/2312.07104 [D] | Radix tree of KV per prefix inside the runtime; frontend `fork` primitive creates parallel continuations that reuse the prefix KV. | In-proc logical fork (C6 at request level, not process level). |
| llama.cpp `llama_memory_seq_cp`, unified KV, slots | https://raw.githubusercontent.com/ggml-org/llama.cpp/master/include/llama.h and `src/llama-kv-cache.cpp` [D] | `seq_cp`: "Copy all tokens that belong to the specified sequence to another sequence"; in the unified cache this is metadata on cells. Master also has a `mem_other` constructor path: "shared cells view the source cache's K/V tensors" (one context viewing another context's KV, same process). The KV buffer is allocated whole and zeroed at construction: `ggml_backend_buffer_clear(buf, 0)`. | In-proc, map. Metadata-only sharing inside one process (C4 analogue). Upstream is not demand-grown (C5 baseline). |
| llama.cpp slot save / state files | server README (`--slot-save-path`, `POST /slots/{id}?action=save|restore`), `llama_state_seq_save_file` in llama.h [D] | State files carry the KV tensor data; restore copies it back. `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`: "Keeps the tensor data on device buffers (i.e. not accessible in host memory, but faster save/load)" (in-process checkpoint whose host part has no tensor data; still a device-side copy). | Copy. The ON_DEVICE flag is an upstream in-process precedent for "state that does not carry the rows" (C4), but it copies on device and cannot cross a process. |
| Prompt Cache (Gim et al.) | https://arxiv.org/pdf/2311.04934 [D] | Schema-defined prompt modules whose attention states are precomputed and reused; "CPU memory ... brings the overhead of host-to-device memory copying. In contrast, GPU memory does not require coping but has limited capacity." | In-proc, copy or in-place on GPU. Not C1. |
| Apple MLX prompt cache files | https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py [D for function names, W for summary] | `save_prompt_cache` -> `mx.save_safetensors`; `load_prompt_cache` -> `mx.load`. File contains the full KV arrays; each loading process gets its own arrays. | X-proc by file, copy. Not C1/C4. |

#### 16.2.2 Cross-instance KV stores that copy

| Work | URL opened | What it actually does | Relation |
|---|---|---|---|
| LMCache (paper) | https://arxiv.org/pdf/2510.09665 [D] | KV chunks offloaded to CPU memory / disk / remote and loaded back; "zero-copy operations" refers to avoiding extra copies between tiers. | X-proc, copy into engine KV. |
| LMCache multiprocess mode | https://docs.lmcache.ai/mp/index.html [W]; https://blog.lmcache.ai/en/2026/06/15/understanding-lmcache-mp-mode-transfer-paths-a-beginners-guide/ [W] | Standalone server; "Multiple vLLM instances on the same node share a single L1 cache" in `/dev/shm`. CUDA-IPC path: the worker sends a handle so the server can read the worker's GPU KV; data still ends in "LMCache-managed CPU memory". Retrieve is "(CPU->GPU copy)". | X-proc, one host copy shared, but **copy** into every engine's GPU KV. Closest *deployed* cross-process KV sharing; not page mapping into the KV tensor. Uses CUDA IPC, so the device-handle path cannot cross MIG GPU instances. |
| CacheGen | https://arxiv.org/pdf/2310.07240 [D] | Compresses KV into bitstreams and streams them over the network. | Copy + codec. |
| MemServe | https://arxiv.org/pdf/2406.17565 [D] | MemPool with memory/index/distributed-transfer APIs across instances. | Copy/transfer. |
| Mooncake | https://github.com/kvcache-ai/Mooncake [W] | Transfer Engine + Store; "zero-copy RDMA transfer"; cross-instance KV sharing in vLLM via hash-based prefix caching. | "Zero-copy" means no intermediate buffers on the transfer path; KV still lands in each engine. |
| NVIDIA Dynamo KVBM | `lib/kvbm-engine/docs/architecture.md`, `lib/kvbm-physical/README.md` in https://github.com/ai-dynamo/dynamo [D] | Tiers G1 (GPU HBM), G2 (pinned DRAM), G3 (NVMe), G4 (S3); `TransferManager` "executes transfers between heterogeneous storage tiers"; remote pull is "remote G2->local G2 via RDMA". | Copy between tiers. |
| AttentionStore / CachedAttention | https://arxiv.org/pdf/2403.19708 [D] | Hierarchical KV store (host memory, disk) for multi-turn sessions with overlapped loading. | Copy. |
| HCache | https://arxiv.org/pdf/2410.05004 [D] | Restores state from stored hidden states instead of KV. | Copy/recompute. |
| Pensieve | https://arxiv.org/pdf/2312.05516 [D] | Two-tier GPU/CPU cache with swap. Explicitly rejects in-place host access: "we choose not to use Unified Memory nor Direct Host Access because these mechanisms trigger memory transfer only ..." | Copy. Shows direct host access for KV was considered and declined on discrete GPUs. |
| llm-d | https://llm-d.ai/docs/architecture [W] | Prefix-cache-aware routing, KV-cache indexer, tiered offload, "Pulling cached prefix KV blocks from a peer's CPU tier". | Routing + copy. |

#### 16.2.3 Agent / fork oriented

| Work | URL opened | What it actually does | Relation |
|---|---|---|---|
| ForkKV | https://arxiv.org/pdf/2604.06370 [D] | Multi-LoRA serving on SGLang v0.5.6 (~3K lines Python + Triton). Splits KV into a shared base cache and per-agent residual: "analogous to a newly forked OS process mapping the read-only physical pages of its parent ... the system executes a CoW operation to allocate exclusive memory blocks for the rCache". | In-proc, dev. "Fork" and "CoW" are analogies; the mechanism is allocate-new, not page-fault CoW. Same *logical* layout as C3 (shared immutable part + private part). Not C1/C2/C6. |
| KVFlow | https://arxiv.org/pdf/2507.07400 [D] | Workflow-aware eviction and CPU->GPU preloading, built on SGLang v0.4.4. | In-proc. |
| KVCOMM | https://arxiv.org/pdf/2510.12872 [D] | Reuses KV across agents with different prefixes by offset correction from an anchor pool; "shared memory" there is a logical store; HuggingFace-based. | In-proc, approximate reuse. |
| DroidSpeak | https://arxiv.org/pdf/2411.02820 [D] | KV reuse across different fine-tuned LLMs with selective layer recompute; integrated with LMCache and vLLM; transfer over the network. | Copy. |
| Parrot | https://arxiv.org/pdf/2405.19888 [D] | Semantic Variables; shared-prefix kernel on vLLM paged memory; "context fork". | In-proc. |
| Execution-State Capsules | https://arxiv.org/pdf/2606.20537 [D] | Snapshot/restore/fork/rollback of the whole buffer set of a captured graph. Explicitly a copy: "each verb is a byte-copy of the boundary's buffer closure"; its cost table lists restore as "copy bytes back + re-bind ... Θ(bytes), bandwidth-bound copy", and the text says restore "is Θ(L) bytes of copy: the point is that copying state is far cheaper than recomputing it". **Replicated on Jetson AGX Thor and DGX Spark**; on Thor "capsule restore is a 2.5-13 ms buffer copy". Single-stream, in-process; notes APC misses on "a fresh process, a restart". | In-proc, copy. Important for C6: on the same platform a copy-based fork costs milliseconds (see section 5). |
| AIOS | https://arxiv.org/pdf/2403.16971 [D] | Context manager with "text-based" and "logits-based" snapshot/restore. | Not KV sharing. |
| LLMS (Yin et al.) | https://arxiv.org/pdf/2403.11805 [D] | One system service holds contexts for all apps; chunk-wise KV compression and swapping; evaluated on Jetson Orin NX / TX2. | In-proc (one service process), swap to disk. |

#### 16.2.4 Unified-memory in-place sharing

| Work | URL opened | What it actually does | Relation |
|---|---|---|---|
| The Ingestion Tax (arXiv 2608.12114) | https://arxiv.org/abs/2608.12114, PDF [D] | **Weights only.** "a framework-independent producer maps each tensor with MAP_SHARED, wraps the pages as a no-copy GPU buffer"; "N processes decode from one mapped copy where resident loading creates N copies". Platforms: Apple M5 Max, AMD APU (Vulkan), GH200 probes, discrete control. Notes that "A private mapping ... falls to 71% residency, because its privatized anonymous copy competes for the same memory". Related work mentions KV only via SuperInfer. | X-proc, map, host pages, read-only **weights**. No KV, no writable state, no CUDA/SMMU, no partition boundary. Establishes the building block "N processes share one host-mapped copy for GPU inference" for immutable data. |
| llama.cpp PR #22120 | https://github.com/ggml-org/llama.cpp/pull/22120 (API) [D] | "ggml-cuda: add mmap zero-copy buffer for integrated GPUs": `buffer_from_host_ptr` for CUDA via `cudaHostRegister + cudaHostGetDevicePointer` on GB10. Closed (flagged by the AI-content checker per [W]). | Weights only, single process view; pinned registration, not pageable host-PT access. |
| `cuda-llm-weight-share` | https://github.com/pontostroy/cuda-llm-weight-share [W]; forum thread [W] | LD_PRELOAD interposer on `cudaMalloc`; master exports with `cudaIpcGetMemHandle`, workers `cudaIpcOpenMemHandle`. "Only the selected model weight allocation is shared"; "Private KV cache per process". | X-proc, map, **dev via CUDA IPC**, weights only. Cannot cross MIG GPU instances. |
| Issue #81, MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark | GitHub API [D], opened 2026-09-28 | "the NVIDIA UVM driver services every GPU fault in a writable mapping as a write fault. In driver 580.173.02, `uvm_ats_faults.c` lines 496 to 497 add the faulted pages to the write mask when the mapping allows writes"; fix: `mprotect(..., PROT_READ)`; "`torch.from_file(shared=False)` is not an alternative: a private writable mapping copies every page the GPU touches into anonymous memory." | **Directly pre-publishes the write-fault-promotion half of C3(b)** and the mprotect remedy, for a read-only weight table on GB10. Does not mention the 2 MiB extension, SMMU invalidation cost, KV, or processes. |
| Apple Metal `newBufferWithBytesNoCopy` in llama.cpp | `ggml/src/ggml-metal/ggml-metal-device.m`, `ggml-metal.cpp` (master) [D] | On unified-memory devices `use_shared_buffers = has_unified_memory`; device buffers are host memory from `vm_allocate` wrapped with `newBufferWithBytesNoCopy ... MTLResourceStorageModeShared`; `buffer_from_host_ptr` wraps mmap'd model pages. Maintainer statement in Discussion #21223 [W]: "The mmap weight sharing works on Apple Silicon with GPU thanks to the unified memory". | On Apple silicon the KV cache of llama.cpp already sits in host-allocated memory used in place by the GPU (element 1 of the mechanism, other platform). Cross-process sharing exists for weights only. |

#### 16.2.5 OS fork / CoW, GPU checkpoint, cold start

| Work | URL opened | What it actually does | Relation |
|---|---|---|---|
| On-demand-fork (EuroSys'21) | https://www.cs.purdue.edu/homes/pfonseca/papers/eurosys21-odf.pdf [D] | Shares page tables between parent and child, CoW on page tables. 0 occurrences of "GPU" / "IOMMU". | CPU-only CoW; no device-visible memory. |
| Async-fork (VLDB'23) | https://arxiv.org/pdf/2301.05861 [D] | Offloads page-table copy to the child for snapshotting in-memory KV stores. 0 occurrences of "GPU" / "IOMMU". | Same. |
| CXLfork (ASPLOS'25) | https://tianyin.github.io/pub/cxlfork.pdf [D] | Remote fork: checkpointed state in CXL memory mapped into clones, "employs Copy-on-Write at runtime ... the read-only state remains in CXL memory, shared". 0 occurrences of "GPU". | CPU processes; relies on CoW. |
| TrEnv (SOSP'24) and TrEnv-X | https://arxiv.org/pdf/2509.09525 (v1 and v2) [D] | mm-templates over CXL/RDMA; "Only updated (writable) memory pages are instantiated per invocation"; agent workloads are sandboxes and browsers calling LLM APIs. 0 occurrences of "GPU". | CPU processes; CoW; the LLM is remote. |
| Catalyzer, REAP, FaaSnap | PDFs [D, abstract-level] | sfork / overlay memory; snapshot working sets. | CPU sandboxes. |
| PhoenixOS (SOSP'25) | https://arxiv.org/pdf/2405.12079 [D] | Concurrent GPU checkpoint; because GPUs lack "OS-mediated data paths (e.g., copy-on-write)" and dirty bits, it emulates them: "soft copy-on-write" by speculating on kernel write sets. Device memory, copies. | Relevant contrast for C3: on discrete GPUs CoW does not exist and is emulated; on Thor it exists (host MMU) and the claim is that it is harmful. |
| gCROP (SoCC'24) | https://ipads.se.sjtu.edu.cn/_media/publications/yang-socc24.pdf [D] | On-demand, parallel restore of GPU app checkpoints using CPU and GPU page faults. | Restore by copy. |
| cuda-checkpoint | https://github.com/NVIDIA/cuda-checkpoint [W] | "Device memory is copied to the host"; does not support "UVM memory or IPC memory created with cuMemExportToShareableHandle()". | Copy. |
| ServerlessLLM, BlitzScale, λScale, Tangram | arXiv PDFs [D] | Checkpoint loading, live autoscaling, RDMA model scaling, GPU memory reuse (Tangram's ElasticKV allocates KV on demand and "shares GPU memory handles ... through the CUDA IPC API"). | Weights/cold start; Tangram touches C5. |
| Llumnix (OSDI'24) | https://arxiv.org/pdf/2406.03243 [D, abstract] | "live migration mechanism for requests and their in-memory" state across instances. | MOVE by copy. |

#### 16.2.6 IOMMU / SVA cost and UVM fault path

| Source | URL opened | What it says | Relation to C3 |
|---|---|---|---|
| NVIDIA UVM source, `uvm_ats_faults.c` (open-gpu-kernel-modules, main) | https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/main/kernel-open/nvidia-uvm/uvm_ats_faults.c [D] | In `ats_compute_prefetch`: "Prefetch the entire region if none of the pages are resident on any node and if preferred_location is the faulting GPU" (region bounded by the VMA and `UVM_VA_BLOCK_SIZE`), then `if (vma->vm_flags & VM_WRITE) uvm_page_mask_or(write_fault_mask, write_fault_mask, prefetch_mask);`. In `uvm_ats_service_faults`, subregions in `write_fault_mask` are serviced with `UVM_FAULT_ACCESS_TYPE_WRITE` when the VMA is writable. Also: "WAR for older kernel versions missing an SMMU invalidate on RO -> RW transition". | The promotion (prefetched pages become write faults in writable VMAs) and the block-sized prefetch are in the vendor source. It is code, not an analysis; the 512x consequence is not stated anywhere I found. Note the whole-region prefetch is conditional (first touch and preferred location on the faulting GPU); otherwise a bitmap-tree prefetch mask is used. |
| CUDA for Tegra App Note 13.0 | https://docs.nvidia.com/cuda/pdf/CUDA-For-Tegra-AppNote.pdf [D] | `pageableMemoryAccess` = 1 "only possible from Thor SoC onwards"; pageable host memory "directly accessible on the GPU". Also: "Starting CUDA 13.0, memory-sharing CUDA IPC APIs would be supported on Tegra platforms with open-source GPU driver (i.e. Jetson Thor and beyond)". No mention of fork, CoW, or write-fault behaviour. | Vendor confirms the platform property. **Also says CUDA IPC memory sharing is available on Thor** (inside one GPU instance), which matters for the baseline in section 5. |
| NVIDIA MIG user guide | https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/deployment-considerations.html [W] | "CUDA IPC across GPU instances is not supported. CUDA IPC across Compute instances is supported." | Supports the premise of C2. Jetson is not mentioned on that page. |
| Linux x86 SVA doc | https://docs.kernel.org/arch/x86/sva.html [W] | "The IOMMU driver uses the mmu_notifier() support to keep the device TLB cache and the CPU cache in sync"; "On fork(2) or exec(2) the PASID is removed from the process". No cost statement. | Mechanism known; cost not documented. |
| Popple, "Invalidate secondary IOMMU TLB on permission upgrade" | https://lwn.net/Articles/938727/ [W] | SMMU "may also cache permission bits and require the same TLB invalidations"; without it "devices will fault indefinitely when writing to a PTE that was previously read-only". | A CoW break is exactly such a permission upgrade; correctness framing, no cost. |
| Gunthorpe, "[PATCH v5 0/9] Organize the SMMUv3 invalidation ..." (2026-09-01) | https://ratatoskr.run/stable/2026/09/17493409/t [W] | Reworks SVA invalidation; targets "at most 512 single invalidations" before falling back to invalidate-all; mentions soft lockups from many per-granule TLBIs. No per-invalidation latency numbers. | Kernel developers treat SVA invalidation as a latency concern; no measurement of a per-page CoW break. |
| Markuze et al., ASPLOS'16 | https://www.cs.tau.ac.il/~mad/publications/asplos2016-iommu.pdf [D] | "IOTLB invalidation is expensive, e.g., requiring ≈ 2000 cycles"; "copying is typically cheaper than invalidating the IOTLB". DMA API for NICs, not SVA. | The classic statement that an invalidation can cost more than a copy; different setting (kernel DMA mappings). |
| Kuper et al., DSA (ASPLOS'24) | https://arxiv.org/pdf/2305.02480 [D] | SVM offload to Intel DSA; device page faults are a large overhead. | SVA faults are costly; nothing on CoW. |
| Koenig et al., RISC-V SVA | https://arxiv.org/pdf/2502.17398 [D] | Measures IOTLB-miss cost of SVA on an embedded SoC. | Translation cost, not invalidation under CoW. |
| Schieffer et al., Grace Hopper system memory | https://arxiv.org/pdf/2407.07850 [D] | System-allocated memory via ATS/SMMU on GH200; first-touch, page size, migration. grep finds no "copy-on-write" or "fork". | Same driver path, no CoW analysis. |
| Linux HMM doc | https://docs.kernel.org/mm/hmm.html [W] | "It will trigger a page fault on missing or read-only entries if write access is requested." | Generic: a device fault that asks for write breaks CoW. |
| rdma-core `ibv_fork_init(3)` | raw man page [D] | Needed "to handle fork() function calls correctly and avoid data corruption"; relies on `MADV_DONTFORK`. | Long-known conflict between fork/CoW and device-visible memory (pinned case). |


### 16.3 Newly found works (not in the brief)

Ordered by how much they threaten the claims.

1. **Omni-Flow** (Meituan/PKU, arXiv 2606.31093v2, 23 Sep 2026, "EuroSys '27"). https://arxiv.org/pdf/2606.31093 [D]. Roles run as **separate processes**; a node-local `MemoryManager` owns pools and "role processes access their assigned manager through MemoryManagerClient over ZMQ IPC". The "Global Params Pool" is "the designated destination for the KV cache, using a paged, tiered storage hierarchy (L1 GPU / L2 CPU / L3 SSD)". Same-device consumers share storage without copying; Fig. 7 shows a follower that "binds IPC alias" to "one physical copy". "Compatible LLM and diffusion roles can use this mechanism to share KV storage"; "a diffusion role can reuse immutable conditioning KV while keeping changing denoising state in private writable pages". It also has a "Copy-on-Write and Partial Page Handling" step that *copies* a partial page into a writable slot. Testbed: NVIDIA H20 GPUs. The extracted text does not name the IPC API; on a discrete GPU with SGLang/PyTorch this is device-memory aliasing. **This is cross-process, no-copy sharing of KV with an immutable shared part and private writable pages, in device memory on one GPU.**

2. **SGLang RFC #35648**, "Same-GPU model replicas with managed CUDA MPS and a shared KV pool" (2026-08-20, open, 0 comments). https://github.com/sgl-project/sglang/issues/35648 [D]. "A minimal CUDA helper process owns the physical KV slab and exports CUDA-IPC handles. Schedulers map the slab and write only pages leased to them." "Identical prefixes would resolve to the same physical KV pages regardless of which DP rank receives the next request." Asks whether the first version should be "reduced to immutable published prefixes". States that "Existing CUDA-IPC weight-cache support can map one immutable daemon-owned model allocation into multiple engine processes." Scope: "one physical GPU". A proposal with acceptance criteria, not a result.

3. **llama.cpp PR #21792**, "kv: Add optional mmap kv cache" (skiz, 2026-04-12, open). https://github.com/ggml-org/llama.cpp/pull/21792 [D]. "the KV cache tensors are allocated in a `MAP_SHARED` file instead of heap memory. Cell metadata is saved to a sidecar file on close and restored on open, enabling session persistence across process restarts"; "a new process can reopen it and resume in milliseconds". Limitation stated by the author: "CPU only. `offload_kqv` must be false — GPU device memory cannot be mmap'd." One process at a time; no read-only prefix, no private tail, no concurrent attach. **This is the metadata-sidecar + file-backed KV rows idea (C4) and a zero-copy MOVE between processes (C6), for CPU inference.**

4. **vAttention** (ASPLOS'25), https://arxiv.org/pdf/2405.04437 [D]; **Prism / kvcached** (arXiv 2505.04021), https://arxiv.org/pdf/2505.04021 [D], https://github.com/ovg-project/kvcached [W]. vAttention keeps KV contiguous in virtual memory and backs it on demand through CUDA VMM (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`), "uses a background thread to allocate new page-groups when the preceding iteration is executing", plus deferred reclamation and eager allocation; it ships a modified UVM driver for 64 KB page-groups. Its discussion section says of `cudaMallocManaged` that it "lacks support for memory aliasing which prevents de-duplication of KV cache content in physical memory (de-duplication is useful when requests share a common prefix)", and that their driver changes add "page sharing with additional APIs"; the public README [W] does not mention prefix sharing. kvcached "decouples virtual and physical GPU" memory, with physical memory allocated "on demand and mapped to virtual addresses lazily" and "a pre-allocation thread". Both are single-process, device memory. **C5 as a concept is theirs.**

5. **vLLM PR #58439**, "Checkpoint-mapped PLE storage for unified-memory GPUs (DGX Spark)" (2026-09-23, open). https://github.com/vllm-project/vllm/pull/58439 [D]. "On GPUs that access pageable host memory through the host page tables (`CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES`), the lookup kernel reads a read-only mapping of the files ... clean, file-backed page-cache pages that the kernel can drop and re-read, shared by every process that maps the same files." Weights (an embedding-like table), read-only. Together with issue #81 this shows the host-page-table read-only mapping technique is in public use on GB10 for weights as of late September 2026.

6. **Host-side shared KV pools that copy into each engine** (three independent, very recent): vLLM PR #58245 "Support sharing SimpleCPUOffload prefix caches across local DP replicas" (2026-09-23) [D]: a node-local `SharedOffloadRegion`, "prefixes are published after all worker stores complete", readers pin, transfers by DMA. SGLang RFC #41514 "Same-node peer L2 sharing across DP ranks via /dev/shm" (2026-09-28) [D]: "Two DP ranks on the same node map the same L2 host pool via named shared memory"; on a hit "one memcpy within the shared SHM ... then DMAs to GPU"; "only a 288-byte pointer per page is written". SGLang RFC #37372 "Out-of-process HiCache data plane with device-memory IPC" (2026-09-01) [D]. These cross GPU boundaries (and would cross MIG) but always copy host->GPU. On a unified-memory GPU the copy is the only thing separating them from the audited mechanism.

7. **CXL shared-memory KV**: TraCT (arXiv 2512.18194) [D]: KV blocks in rack-shared CXL memory, accessed by multiple nodes/processes with offset-based addressing because "Virtual addresses differ across processes and nodes"; "a compact object store that publishes only root metadata (e.g., prefix-cache roots)"; KV is then "fully loaded to GPU memory" by DMA. Beluga (arXiv 2511.20172) [D]: GPUs reach a CXL pool via `mmap()` and `cudaMemcpy`/copy kernels. The KV survey arXiv 2607.02574 [D] classifies this as "A3 Shared memory" / "Archetype 4: Memory-pool". One shared physical copy, metadata-only publication, but a copy into each GPU.

8. **Pion** (pavelhorak/pion, `pion-vllm-mlx` on PyPI, first release 2026-10-01). PyPI JSON + `doc/shared_kv_cache.md` [D]. Apple silicon. A server stores prefix K/V "that a different process, a different model object, or a restarted server can reuse". Stage 1: other processes fetch K/V over a socket (copy). Stage 2: "K/V that never leaves Pion's process memory after the initial push"; only queries cross the wire, at "one round trip per layer per pass". Cross-process, one physical copy, **no page mapping** (RPC instead).

9. **mlxcache** (woodsonl), PR #13 [W]: MLX KV-cache daemon; disk checkpoints, "atomic blob publish (write-temp-then-rename)", `load_prompt_cache` with "mmap-lazy" loading. File-based reuse across processes and restarts; each process loads its own arrays.

10. **Nexus** (arXiv 2608.20397) [D]: Apple-silicon, GGUF/llama.cpp-based. "The compiled block is mapped via zero-copy mmap into virtual memory. Contiguous F16 keys and values are then blitted into the live cache". mmap'd KV block files, then a **copy** into the live cache.

11. **EdgeAgent** (arXiv 2610.03394, 2 Oct 2026, "ASPLOS '27") [D]: Apple M4, one process, CPU+GPU tensor parallelism. "by exploiting the physically shared addressing of the UMA architecture, the system performs an in-place metadata freeze on the suspended agent's KV cache"; "zero-overhead context switching without moving physical bytes". In-process suspend/resume; no sharing between agents or processes. Closest in *rhetoric* (UMA, in-place, metadata freeze) for an on-device multi-agent system.

12. **"Stateful Inference for Low-Latency Multi-Agent Tool Calling"** (LayerScale, arXiv 2605.26289) [D]: "metadata-only sequence aliasing in a unified KV cache ... the new sequence's page table is rewritten to point at the donor's cells, and the operation completes in constant time independent of prefix length m". In-process O(1) attach.

13. Other in-process shared-KV / fork papers, checked by grep only [D]: AAFLOW+ (arXiv 2607.10987; `Op_kv_fork`, "copy-on-write behavior: shared prefix blocks remain aliased"; evaluated with a trace-driven analytical model), TokenDance (2604.03143; master/mirror diffs), LRAgent (2602.01053), ReasonCache/MemShare (2507.21433; "Zero-Copy KV Sharing" by block-table refcount), PolyKV (2604.24971; "write-once, read-many" compressed pool injected into N agent contexts), KVTether (2609.39819; `fork` registers "a child sharing a parent prefix"), KVMem (2609.04852; GPU working set + host store), Q4 KV persistence on edge (2603.04428), QKVShare (2605.03884), SuperInfer (2601.20309; KV rotation HBM<->DRAM on GH200), "Shared KV Caching for Replicated 27B Inference" (2609.15021; two vLLM replicas on a 256 GiB LMCache pool). None is cross-process page mapping.

14. llama.cpp paged-KV efforts: Discussion #21961 [W] ("blocks that are allocated on demand ... CoW / prefix caching: the block manager has refcount infrastructure in place, but seq_cp is a no-op"), PR #17579, PR #18747, the Fenix46 fork [W]. In-process; relevant to C5.

15. Operational evidence against CUDA-IPC sharing: PyTorch notes [D]: "the sending process must stay alive as long as the consumer process has references to the tensor, and the refcounting can not save you if the consumer process exits abnormally"; and "either the spawn or forkserver start method are required to use CUDA in subprocesses". LMCache issue #5265 [D]: "Stale CUDA IPC mappings block vLLM restart after an abnormal worker exit". Useful for motivating a file-backed design.

16. Shared-KV risk literature (grep only [D]): "Bit-Flip Vulnerability of Shared KV-Cache Blocks in LLM Serving Systems" (arXiv 2604.17249; overhead measured on DGX Spark). Search also surfaced "Selective KV-Cache Sharing to Mitigate Timing Side-Channels" (arXiv 2508.08438) [S].

**Not found** (with the queries in section 1): any system, paper, PR or issue that maps the same host pages of a computed KV prefix into the KV tensors of several GPU-inference processes; any KV sharing across MIG GPU instances without a copy; any measurement of the SMMU/IOMMU invalidation cost of a CoW break in a process bound to a GPU by SVA; any use of `fork()` of a GPU-serving LLM process that keeps KV shared.


### 16.4 Verdict per claim

#### C1. One physical copy of a computed KV prefix shared by separate OS processes for GPU inference, by mapping the same pages, each with private continuation state

**Verdict: PARTIALLY TAKEN.**

- Closest: **Omni-Flow** (separate role processes, shared paged KV pool on one GPU, IPC alias, "immutable conditioning KV" + "private writable pages"), and the **SGLang RFC #35648** design (helper process owns a KV slab, exports CUDA-IPC handles, replicas "map the slab", identical prefixes "resolve to the same physical KV pages").
- What they do not do: both are **device memory through CUDA-style IPC on one GPU**, organised as a block pool with a controller; neither uses host page tables, a file, or page protections; Omni-Flow's use case is cross-role reuse inside one multimodal workflow, not N independent serving processes attaching to a system prompt; the SGLang item is an unimplemented RFC.
- On host page tables, the same-pages-in-N-processes technique is published only for **weights** (Ingestion Tax; vLLM PR #58439; llama.cpp Metal mmap).
- Residual that appears open: sharing *KV* through **host page tables** (read-only PTEs onto the same physical pages of a tmpfs file) so that the GPU of each process reads them in place, and composing one contiguous KV tensor from a shared extent and a private extent by `mmap`.
- Consequence for wording: "first to share a KV prefix across processes without copying" is not defensible. "First to do so through the host page tables of a unified-memory GPU, without GPU-level IPC" is consistent with what I found.

#### C2. Doing C1 across GPU partitions / where CUDA IPC or an MPS shared context is unavailable

**Verdict: OPEN** (with a dependency).

- Every zero-copy cross-process mechanism found rests on CUDA IPC (Omni-Flow-style alias, SGLang RFC, cuda-llm-weight-share, Tangram, LMCache's device-handle path). NVIDIA's MIG guide says "CUDA IPC across GPU instances is not supported".
- Everything that does cross such a boundary copies into each engine's KV (LMCache `/dev/shm` L1, vLLM #58245, SGLang #41514, TraCT, Beluga, Mooncake, Dynamo KVBM).
- Residual caveats: (i) C2 is a property of C1's host-page-table mechanism, not a second mechanism, so its weight depends on whether partitioned co-tenancy on Thor is a deployment a reviewer accepts; I found no NVIDIA page documenting MIG on Jetson. (ii) The copy-based host pools are the natural baseline and already "work" across partitions.

#### C3. "Immutable shared extent + private tail" chosen to avoid CoW, and the reasoning that CoW is harmful on an IOMMU/SVA-attached GPU

**Verdict: PARTIALLY TAKEN.**

- The **layout** is standard at block level: prefix caches keep shared blocks unmodified and append into private blocks (vLLM, SGLang); the KV survey describes "reference counting and copy-on-write or append-only semantics when sequences diverge"; ForkKV's shared base + exclusive residual; Omni-Flow's "immutable conditioning KV ... private writable pages"; PolyKV's "write-once, read-many"; SGLang RFC's "immutable published prefixes". Because a KV cache is append-only, none of these needs page-fault CoW, and where CoW appears it is a copy of one partial block. A reviewer will say the layout follows from append-only KV, not from the IOMMU.
- Reasoning (b), write-fault promotion: **already public.** Issue #81 (2026-09-28) states that UVM "services every GPU fault in a writable mapping as a write fault", cites the source lines, gives the `mprotect` read-only remedy, and states that "a private writable mapping copies every page the GPU touches into anonymous memory". The vendor source shows both the promotion and the block-sized prefetch. Not found anywhere: the quantified amplification (512x) and the link to KV layout.
- Reasoning (a), per-page SMMU invalidation on a CoW break: the mechanism is documented (mmu notifier keeps the device TLB in sync; permission upgrades require SMMU invalidation; kernel developers bound SVA invalidation latency), and the general lesson "an IOTLB invalidation can cost more than a copy" is Markuze et al. 2016 in the DMA setting. **No source I opened measures the cost of a single-page CoW break under SVA with a GPU bound**, on SMMUv3 or elsewhere.
- Residual that appears open: the measurement (a), the quantification of (b), and the argument that the OS-native way to give a process a private writable view of GPU-visible memory (`MAP_PRIVATE` / `fork`) is the wrong primitive there, so the split must be made explicit in the address-space layout.
- Consequence for wording: cite issue #81 and `uvm_ats_faults.c` for (b) and present the paper's part as quantification and consequence; claim (a) as a new measurement; do not claim the immutable/private layout itself.

#### C4. Metadata-only state files / O(1) publish and attach

**Verdict: PARTIALLY TAKEN** (the idea is taken on CPU; open for GPU-visible, concurrently attached state).

- Closest: **llama.cpp PR #21792** (KV rows stay in a `MAP_SHARED` file, "Cell metadata is saved to a sidecar file", another process reopens and resumes). CPU only, by the author's own limitation: "GPU device memory cannot be mmap'd".
- Related: upstream `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` (in-process state without host tensor data); `llama_memory_seq_cp` and the LayerScale paper's "metadata-only sequence aliasing ... constant time independent of prefix length" (in-process); TraCT "publishes only root metadata"; SGLang #41514 "only a 288-byte pointer per page is written"; EdgeAgent "in-place KV metadata freezing".
- Counter-example worth citing: Execution-State Capsules describe restore as "Θ(L) bytes of copy" (not constant time) on the same hardware.
- Residual: metadata-only publish/attach for KV that the GPU reads in place, with several concurrent attachers and an immutable published range.

#### C5. Demand-grown (lazily backed) KV cache for GPU inference

**Verdict: TAKEN** as a concept.

- **vAttention** (ASPLOS'25) is this idea, including preparing memory ahead of use in a background thread; **kvcached/Prism** repeats it for multi-model GPU sharing; PagedAttention and Tangram's ElasticKV allocate blocks on demand; llama.cpp has open paged-KV proposals.
- Residual difference: implemented with ordinary OS demand paging of host memory seen through host page tables, instead of `cuMemCreate`/`cuMemMap` on device memory (vAttention needs a modified driver for sub-2 MB granularity and calls stock managed memory unsuitable); upstream llama.cpp allocates and zeroes the full buffer.
- Recommendation: do not list as a contribution; present as a consequence of the host-memory design and cite vAttention and kvcached.

#### C6. Process-level fork/move of a live LLM serving context's KV state without copying it

**Verdict: PARTIALLY TAKEN for MOVE on CPU; OPEN for a live fork/move of GPU-visible KV.**

- Fork as a *request-level* abstraction is common and in-process (SGLang `fork`, Parrot, ForkKV, AAFLOW+, KVTether, Capsules by copy).
- MOVE between processes without copying exists for CPU inference (llama.cpp PR #21792, sequential handoff).
- OS fork work (On-demand-fork, Async-fork, CXLfork, TrEnv) never touches GPU-visible memory (zero occurrences of "GPU"/"IOMMU" in the texts); CUDA is documented as not fork-safe (PyTorch notes); cuda-checkpoint, PhoenixOS, gCROP and Llumnix all copy state; Capsules copy even on Thor.
- Residual: parent publishes and keeps generating while children attach and generate, across processes, with no KV copy. Not found.


### 16.5 Strongest reviewer objections and honest answers

**Objection 1 (the main one): "Every piece exists; this is an integration on a niche platform."**
Shared KV across processes on one GPU is Omni-Flow and the SGLang RFC; KV rows in a `MAP_SHARED` file with a metadata sidecar is llama.cpp PR #21792; demand paging is vAttention; read-only host-page-table mappings shared by processes are the Ingestion Tax and vLLM PR #58439; the UVM write-promotion trap and its `mprotect` fix are in a public GitHub issue; and since KV is append-only, an immutable prefix plus a private tail is what every prefix cache already is. Moreover, NVIDIA documents CUDA IPC memory sharing as supported on Thor from CUDA 13, so inside one GPU instance the Omni-Flow / SGLang design is available.

*Honest answer.* The objection is largely correct about the parts, and the paper should say so. What I could not find anywhere is the combination applied to **KV state on host page tables**: the same physical pages of a computed prefix mapped read-only into the KV tensors of several GPU-serving processes, with the tensor composed from shared and private extents by `mmap`. Three properties follow that the CUDA-IPC designs do not have and that are checkable: (1) it crosses a GPU-instance boundary, where NVIDIA states CUDA IPC is unsupported; (2) the shared state is a file whose lifetime is independent of any exporting process, whereas CUDA-IPC sharing requires the exporter to stay alive and has documented stale-mapping failures (PyTorch notes; LMCache issue #5265); (3) the memory is pageable and demand-backed by the OS rather than a pre-reserved device slab. The contribution should therefore be framed as the combination plus the measurements, with the pieces attributed. To survive this objection the evaluation needs (a) a CUDA-IPC shared-KV baseline on Thor inside one partition and (b) a shared-host-pool-plus-copy baseline (LMCache MP / vLLM #58245 style) across partitions.

**Objection 2: "Why avoid a copy that costs milliseconds?"**
On the same Jetson AGX Thor, Execution-State Capsules report a copy-based restore of "2.5-13 ms" for 2k-16k-token prefixes, and unified memory copies at memory bandwidth.

*Honest answer.* Latency is not where the mechanism wins; attach time against a memcpy of a few hundred MiB is a weak headline. The defensible benefits are memory (N attachers hold one prefix instead of N, which is what the Ingestion Tax shows for weights with "5.5 versus 0.08 tok/s" at capacity) and capability (crossing partitions; publish/attach without the publisher's cooperation or survival). The paper should lead with footprint and admission capacity at N processes, and report attach latency only as secondary.

**Objection 3: "The CoW argument is a strawman."**
No LLM serving system uses page-fault CoW for KV; block-level designs allocate new blocks. So "we deliberately avoid CoW" refutes something nobody does.

*Honest answer.* Inside one engine that is true. The argument only has force at the process level: the standard OS ways to give a second process a private writable view of existing memory are `fork()` and `MAP_PRIVATE`, both CoW, and those are what one would reach for to "fork a serving process" on a machine where the GPU reads host memory. The paper should present CoW as the natural *OS-level* baseline for cross-process fork on this platform and show measured costs (per-page invalidation; a GPU read privatising a writable private mapping), rather than as a comparison with vLLM or SGLang. It must cite issue #81 and the UVM source for the write-promotion behaviour and claim only the quantification and the SMMU-side measurement, which I did not find published.

**Objection 4: "Sharing across partitions defeats the purpose of partitioning."**
MIG exists for isolation; mapping one tenant's KV pages into another instance creates a shared channel (timing side channels on shared KV; fault propagation through shared KV blocks are both in the 2025-2026 literature).

*Honest answer.* The shared extent is read-only by page protection in the attachers, which bounds integrity risk from attachers but not from the publisher, and does nothing for timing channels. The paper needs an explicit trust model (same tenant, different partitions for performance isolation) or it will be read as weakening MIG.

**Objection 5: "The driver behaviour is version-specific."**
The whole-region prefetch in `uvm_ats_faults.c` is conditional, and kernel work on SMMUv3 SVA invalidation is active (v5 of a rework series in September 2026).

*Honest answer.* Correct; the measurements should name kernel and driver versions, and the design argument should not depend on the exact 512x or 9 microsecond figures holding on later releases.


#### Three closest works overall

1. **Omni-Flow** (arXiv 2606.31093v2): cross-process, no-copy shared KV pool with immutable KV and private writable pages, on one GPU in device memory. Closest to C1/C3.
2. **llama.cpp PR #21792** (April 2026, open): KV rows in a `MAP_SHARED` file plus metadata sidecar, resumed by another process; CPU only. Closest to C4/C6.
3. **vAttention** (ASPLOS'25): demand-paged contiguous KV with background pre-mapping. Takes C5.

Runners-up that must be cited: SGLang RFC #35648 (C1 design via CUDA IPC + MPS), issue #81 and `uvm_ats_faults.c` (C3b), the Ingestion Tax and vLLM PR #58439 (host-page-table sharing of weights), Execution-State Capsules (copy-based fork on Thor itself), LMCache MP / vLLM #58245 / SGLang #41514 (shared host pool plus copy).

## 17. What Section 16 changes, and what was measured in response

The record for the mechanism is `RESEARCH_LLM_SHARE_2026-10-05.md`,
Section 6. Everything below is in `llm_share/results/20261006-*` and passes
`llm_share/verify_llm_share_artifact.sh`.

1. **The closest works exist and bound the wording.** Sharing one copy of a
   cache between processes exists in device memory on one GPU (Omni-Flow),
   a cache in a mapped file with its metadata beside it exists for CPU
   inference (llama.cpp pull request #21792), a cache whose memory follows
   use exists for device memory (vAttention), and the write-intent fault
   service was reported (issue #81). The record claims none of these. Its
   wording is restricted to: the pages of a computed prefix mapped through
   the host page tables into the cache tensors of several GPU-serving
   processes, across the two MIG instances, and a running process that hands
   its state on without a copy.
2. **C5 is not a contribution.** The tail that follows use is reported as a
   consequence of keeping the cache in host memory. The measurement
   separates it from sharing: of the 1,810 MiB that an agent on a
   16,321-token prefix saves, 920 MiB is the prefix that is no longer
   duplicated and 890 MiB the part of the context it has not written.
3. **The two baselines that Section 16.5 asks for.** The shared host pool
   with a copy into each engine is the row called "copy" in every campaign
   (the engine's state file, read by every agent). The device-memory route
   was first probed: CUDA's virtual memory interface composes a shared
   read-only range and a private range inside one MIG instance and between
   clients of one MPS server (6 of 6 each, read-only access enforced), and
   refuses the import across the MIG instances (0 of 6,
   `CUDA_ERROR_NOT_INITIALIZED`). It was then built into the engine as a
   baseline of our own (`llm_share/kv_vmm.patch`: a cache of 2 MiB
   allocations, the ones a prefix fills exported to child processes) and
   compared with host extents and the copy for a parent and four children in
   one instance. Same texts. The device-memory cache generates at the speed
   of the copy; children on host extents are at 0.98 of it in the 12-SM
   instance and 0.92 to 0.96 in the 6-SM instance. Host extents attach in 27
   to 48 ms against 325 to 438 ms, hold 2.9 to 3.0 GiB against 3.7 to
   3.9 GiB for four children, and run with a child in the other MIG
   instance (12 of 12) where the device-memory import is refused (0 of 12).
4. **"Why avoid a copy that costs milliseconds?"** Measured as the audit
   expected. Attaching takes 25 to 65 ms against 81 to 484 ms for the copy,
   and the first token of an agent comes only 4 to 23% earlier, because
   loading the model dominates it. The record leads with memory: eight
   agents on a 16,321-token prefix hold 5.7 GiB instead of 19.9 GiB, with
   one copy of the prefix between them (7,140 MiB resident sum, 908 MiB
   proportional share).
5. **"The copy-on-write argument is a strawman."** It is measured at the
   process level in the engine. A private writable mapping of the
   publisher's cache without a CPU read pass copies on read: every agent
   ends with a private copy of the whole prefix (894 to 911 MiB for an
   893 MiB prefix), eight agents hold 12.7 GiB against 5.6 GiB, and their
   first token comes after 4.9 to 5.2 s against 1.7 s. With the read pass
   and 2 MiB pages copy-on-write comes within 15% and 160 MiB per agent of
   extents. The record states both, and rests the case for extents on the
   absence of a state in which the sharing is lost silently.
6. **C6 is measured.** A parent publishes with a pause of 3 ms against
   218 ms and generates its own continuation while four children, two in
   the other MIG instance, generate theirs. The parent writes the text of a
   process alone in 6 of 6 repetitions, every child the text of the child
   that copies in 24 of 24.
7. **Trust model and versions.** The record states that agents trust the
   publisher, that nothing addresses timing channels, and that sharing
   across MIG instances is for agents of one tenant. It names kernel and
   driver versions and does not rest the design on the exact figures.
8. **Two gates failed and are reported as failed.** "Every agent writes the
   text of an agent that recomputes the prefix" does not hold for agents in
   the other MIG instance, for the engine's own state file either. The
   cause is measured: each instance computes the same cache bits in every
   repetition, and the two instances never compute the same bits (0 of 6
   for three prefix lengths, nine of ten values differ). "Extents keep 97%
   of the generation speed of the copy" fails in two of six cells. The
   cause is measured as far as the instance: a cache in host memory, shared
   or private, generates as fast as a cache in device memory in the 12-SM
   instance and 2% (4,081-token prefix) to 6% (16,321-token prefix) slower
   in the 6-SM instance. What inside the GPU makes the small instance
   slower was not found.
9. **A finding outside the audit's scope.** The 12-SM and the 6-SM MIG
   instance of this device are each deterministic and are not bit-identical
   to each other for the same model, prompt and engine. We found no
   statement of this in the sources opened in Sections 14 and 16, and did
   not search for one.
10. **The in-process comparison.** Sharing a prefix between the sequences of
    one process is what the prefix caches of Section 16.2.1 do, and the
    engine has it. Measured for eight agents on a 16,321-token prefix: one
    batching server generates 75.6 tokens/s in 2.5 GiB, eight separate
    processes on extents 34 tokens/s in 5.7 GiB. Separate processes lose on
    both counts, and the record says so. Extents add what sharing inside a
    process cannot reach: a server in each MIG instance on one published
    prefix generates 102.9 tokens/s in 2.4 GiB, against 105.0 tokens/s in
    5.0 GiB when the second server copies the prefix and 30.6 s until all
    agents are ready when it computes the prefix again.

