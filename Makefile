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

clean:
	$(RM) cuda_ipc_probe cuda_vmm_probe cuda_share_load_probe kv_fork kv_batch kv_fork_vmm kv_spawn kv_tree kv_fork_sota kv_spawn_sota kv_fork_stock kv_fork_nommap kv_fork_tuned kv_spawn_tuned kv_tree_stock cuda_tlb_probe cuda_reach_probe cuda_vmm_attach_probe cuda_protect_probe ollama_mig_visible.so
