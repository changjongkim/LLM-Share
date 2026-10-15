NVCC := /usr/local/cuda/bin/nvcc
NVCCFLAGS := -std=c++17 -O3 -lineinfo -arch=sm_110 \
	-Xcompiler=-Wall,-Wextra,-Werror

CXX ?= g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror
CFLAGS := -O2 -Wall -Wextra -Wpedantic -Werror
KV_ENGINE ?= llama.cpp-kv
VMM_ENGINE ?= llama.cpp-vmm
CHAIN_ENGINE ?= llama.cpp-chain
SOTA_ENGINE ?= llama.cpp-sota
STOCK_ENGINE ?= llama.cpp-stock
TUNED_ENGINE ?= llama.cpp-tuned

.PHONY: all clean

all: cuda_ipc_probe cuda_vmm_probe cuda_share_load_probe kv_fork kv_batch

cuda_ipc_probe: cuda_ipc_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

cuda_vmm_probe: cuda_vmm_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcuda

cuda_share_load_probe: cuda_share_load_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcuda

cuda_reach_probe: cuda_reach_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

cuda_tlb_probe: cuda_tlb_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

cuda_vmm_attach_probe: cuda_vmm_attach_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcuda

cuda_protect_probe: cuda_protect_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcuda

# Links against the engine with the cache patch; its headers are not ours.
kv_fork: kv_fork.cpp
	$(CXX) $(CXXFLAGS) -isystem $(KV_ENGINE)/include \
	  -isystem $(KV_ENGINE)/ggml/include $< -o $@ \
	  -L$(KV_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(KV_ENGINE)/build/bin'

kv_batch: kv_batch.cpp
	$(CXX) $(CXXFLAGS) -isystem $(KV_ENGINE)/include \
	  -isystem $(KV_ENGINE)/ggml/include $< -o $@ \
	  -L$(KV_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(KV_ENGINE)/build/bin'

# The device-memory baseline: the same child program and a parent that starts
# its children, linked against the engine with the device-memory cache.
kv_fork_vmm: kv_fork.cpp
	$(CXX) $(CXXFLAGS) -isystem $(VMM_ENGINE)/include \
	  -isystem $(VMM_ENGINE)/ggml/include $< -o $@ \
	  -L$(VMM_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(VMM_ENGINE)/build/bin'

kv_spawn: kv_spawn.cpp
	$(CXX) $(CXXFLAGS) -isystem $(VMM_ENGINE)/include \
	  -isystem $(VMM_ENGINE)/ggml/include $< -o $@ \
	  -L$(VMM_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(VMM_ENGINE)/build/bin'

# The comparison of baselines: one engine with every way to hand a prefix
# over (copy, demand-backed copy, shared device memory, copy-on-write,
# extents), the parent that starts its children and the child program.
kv_fork_sota: kv_fork.cpp
	$(CXX) $(CXXFLAGS) -isystem $(SOTA_ENGINE)/include \
	  -isystem $(SOTA_ENGINE)/ggml/include $< -o $@ \
	  -L$(SOTA_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(SOTA_ENGINE)/build/bin'

kv_spawn_sota: kv_spawn.cpp
	$(CXX) $(CXXFLAGS) -isystem $(SOTA_ENGINE)/include \
	  -isystem $(SOTA_ENGINE)/ggml/include $< -o $@ \
	  -L$(SOTA_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(SOTA_ENGINE)/build/bin'

# The baselines with the device-memory path tuned, and the extent mechanism
# with single steps that can be left out.
kv_fork_tuned: kv_fork.cpp
	$(CXX) $(CXXFLAGS) -isystem $(TUNED_ENGINE)/include \
	  -isystem $(TUNED_ENGINE)/ggml/include $< -o $@ \
	  -L$(TUNED_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(TUNED_ENGINE)/build/bin'

kv_spawn_tuned: kv_spawn.cpp
	$(CXX) $(CXXFLAGS) -isystem $(TUNED_ENGINE)/include \
	  -isystem $(TUNED_ENGINE)/ggml/include $< -o $@ \
	  -L$(TUNED_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(TUNED_ENGINE)/build/bin'

# The unmodified engine: the same program against the upstream commit with
# no patch of this repository applied.
kv_fork_stock: kv_fork.cpp
	$(CXX) $(CXXFLAGS) -isystem $(STOCK_ENGINE)/include \
	  -isystem $(STOCK_ENGINE)/ggml/include $< -o $@ \
	  -L$(STOCK_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(STOCK_ENGINE)/build/bin'

# The same program with the weights copied to device memory.
kv_fork_nommap: kv_fork_nommap.cpp kv_fork.cpp
	$(CXX) $(CXXFLAGS) -isystem $(KV_ENGINE)/include \
	  -isystem $(KV_ENGINE)/ggml/include $< -o $@ \
	  -L$(KV_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(KV_ENGINE)/build/bin'

# The tree program against the unmodified engine.
kv_tree_stock: kv_tree.cpp
	$(CXX) $(CXXFLAGS) -isystem $(STOCK_ENGINE)/include \
	  -isystem $(STOCK_ENGINE)/ggml/include $< -o $@ \
	  -L$(STOCK_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(STOCK_ENGINE)/build/bin'

# A tree of agents: the engine whose agents can publish the rows they add.
kv_tree: kv_tree.cpp
	$(CXX) $(CXXFLAGS) -isystem $(CHAIN_ENGINE)/include \
	  -isystem $(CHAIN_ENGINE)/ggml/include $< -o $@ \
	  -L$(CHAIN_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(CHAIN_ENGINE)/build/bin'

# Ollama reports a physical UUID while validating a MIG device. This preload
# adapter maps that child-runner lookup back to the selected MIG UUID.
ollama_mig_visible.so: ollama_mig_visible.c
	$(CC) $(CFLAGS) -shared -fPIC $< -o $@ -ldl

# Publishes a model file as a shared mapping on 2 MiB pages by direct reads.
weights_publish: weights_publish.c
	$(CC) $(CFLAGS) $< -o $@ -lpthread

clean:
	$(RM) cuda_ipc_probe cuda_vmm_probe cuda_share_load_probe kv_fork kv_batch kv_fork_vmm kv_spawn kv_tree kv_fork_sota kv_spawn_sota kv_fork_stock kv_fork_nommap kv_fork_tuned kv_spawn_tuned kv_tree_stock weights_publish cuda_tlb_probe cuda_reach_probe cuda_vmm_attach_probe cuda_protect_probe ollama_mig_visible.so \
	  kv_fork_linux kv_spawn_linux linux_share_probe ksm_merge_shim.so ksm_probe gpu_read_check gpu_read_trace gpu_read_map gpu_attn_probe gpu_populate_probe tp_count gpu_seal_probe kv_fork_tuned3 kv_spawn_tuned3 kv_fork3

# The same programs against the engines of kv_extents2.patch, kv_chain2.patch
# and kv_tuned2.patch, in which a publisher puts the entries of a divided
# 2 MiB page back from the CPU after it has frozen its rows.
KV2_ENGINE ?= llama.cpp-kv2
CHAIN2_ENGINE ?= llama.cpp-chain2
TUNED2_ENGINE ?= llama.cpp-tuned2

kv_fork2: kv_fork.cpp
	$(CXX) $(CXXFLAGS) -isystem $(KV2_ENGINE)/include \
	  -isystem $(KV2_ENGINE)/ggml/include $< -o $@ \
	  -L$(KV2_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(KV2_ENGINE)/build/bin'

kv_batch2: kv_batch.cpp
	$(CXX) $(CXXFLAGS) -isystem $(KV2_ENGINE)/include \
	  -isystem $(KV2_ENGINE)/ggml/include $< -o $@ \
	  -L$(KV2_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(KV2_ENGINE)/build/bin'

kv_tree2: kv_tree.cpp
	$(CXX) $(CXXFLAGS) -isystem $(CHAIN2_ENGINE)/include \
	  -isystem $(CHAIN2_ENGINE)/ggml/include $< -o $@ \
	  -L$(CHAIN2_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(CHAIN2_ENGINE)/build/bin'

# A resident agent: the steps of kv_tree with a line after each of them, so
# that a runner can release an agent that waits and time each of its steps.
kv_agent2: kv_agent.cpp
	$(CXX) $(CXXFLAGS) -isystem $(KV2_ENGINE)/include \
	  -isystem $(KV2_ENGINE)/ggml/include $< -o $@ \
	  -L$(KV2_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(KV2_ENGINE)/build/bin'

# The resident agent against the unmodified engine: every agent copies the
# weights to device memory and restores the prefix from a state file.
kv_agent_stock: kv_agent.cpp
	$(CXX) $(CXXFLAGS) -isystem $(STOCK_ENGINE)/include \
	  -isystem $(STOCK_ENGINE)/ggml/include $< -o $@ \
	  -L$(STOCK_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(STOCK_ENGINE)/build/bin'

kv_fork_tuned2: kv_fork.cpp
	$(CXX) $(CXXFLAGS) -isystem $(TUNED2_ENGINE)/include \
	  -isystem $(TUNED2_ENGINE)/ggml/include $< -o $@ \
	  -L$(TUNED2_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(TUNED2_ENGINE)/build/bin'

kv_spawn_tuned2: kv_spawn.cpp
	$(CXX) $(CXXFLAGS) -isystem $(TUNED2_ENGINE)/include \
	  -isystem $(TUNED2_ENGINE)/ggml/include $< -o $@ \
	  -L$(TUNED2_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(TUNED2_ENGINE)/build/bin'

# The same programs against the engines of kv_tuned2.patch (tuned3) and
# kv_extents2.patch (kv3) with fattn_host.patch,
# in which the attention of a decoding step reads a KV cache in host memory once
# per group of query heads (GGML_CUDA_FATTN_HOST_GQA=0 restores the choice of
# upstream).
TUNED3_ENGINE ?= llama.cpp-tuned3

kv_fork_tuned3: kv_fork.cpp
	$(CXX) $(CXXFLAGS) -isystem $(TUNED3_ENGINE)/include \
	  -isystem $(TUNED3_ENGINE)/ggml/include $< -o $@ \
	  -L$(TUNED3_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(TUNED3_ENGINE)/build/bin'

KV3_ENGINE ?= llama.cpp-kv3

kv_fork3: kv_fork.cpp
	$(CXX) $(CXXFLAGS) -isystem $(KV3_ENGINE)/include \
	  -isystem $(KV3_ENGINE)/ggml/include $< -o $@ \
	  -L$(KV3_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(KV3_ENGINE)/build/bin'

kv_spawn_tuned3: kv_spawn.cpp
	$(CXX) $(CXXFLAGS) -isystem $(TUNED3_ENGINE)/include \
	  -isystem $(TUNED3_ENGINE)/ggml/include $< -o $@ \
	  -L$(TUNED3_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(TUNED3_ENGINE)/build/bin'

# The same programs against the engine of kv_tuned2.patch and kv_linux.patch,
# which adds a baseline: an agent maps the prefix shared and writable
# (LLAMA_KV_SHARED=1). Without the variable the engine is that of tuned2.
LINUX_ENGINE ?= llama.cpp-linux

kv_fork_linux: kv_fork.cpp
	$(CXX) $(CXXFLAGS) -isystem $(LINUX_ENGINE)/include \
	  -isystem $(LINUX_ENGINE)/ggml/include $< -o $@ \
	  -L$(LINUX_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(LINUX_ENGINE)/build/bin'

kv_spawn_linux: kv_spawn.cpp
	$(CXX) $(CXXFLAGS) -isystem $(LINUX_ENGINE)/include \
	  -isystem $(LINUX_ENGINE)/ggml/include $< -o $@ \
	  -L$(LINUX_ENGINE)/build/bin -lllama -lggml -lggml-base \
	  -Wl,-rpath,'$$ORIGIN/$(LINUX_ENGINE)/build/bin'

# The sharing mechanisms of the kernel on memory that the GPU reads.
linux_share_probe: linux_share_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# KSM on memory that the GPU reads and writes.
ksm_probe: ksm_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Every word that the GPU reads when it faults on a read-only file mapping.
gpu_read_check: gpu_read_check.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# The same check, with a CPU read and a second GPU pass after a wrong read.
gpu_read_trace: gpu_read_trace.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# The same check with the file mapped shared and read-only.
gpu_read_map: gpu_read_map.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# The read pattern of attention in device memory and in host memory of several page sizes.
gpu_attn_probe: gpu_attn_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Calls of the kernel against the CPU pass of STATOR before the GPU reads or writes host memory.
gpu_populate_probe: gpu_populate_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Counts tracepoints over the whole system (the kernel has no kprobes).
tp_count: tp_count.c
	$(CC) $(CFLAGS) $< -o $@

# What memfd seals allow next to the read-only mappings of STATOR.
gpu_seal_probe: gpu_seal_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Preloaded into a process: asks KSM to merge all of its anonymous memory.
ksm_merge_shim.so: ksm_merge_shim.c
	$(CC) $(CFLAGS) -shared -fPIC $< -o $@

