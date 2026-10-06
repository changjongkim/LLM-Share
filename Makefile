NVCC := /usr/local/cuda/bin/nvcc
NVCCFLAGS := -std=c++17 -O3 -lineinfo -arch=sm_110 \
	-Xcompiler=-Wall,-Wextra,-Werror

CXX ?= g++
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror
KV_ENGINE ?= llama.cpp-kv
VMM_ENGINE ?= llama.cpp-vmm

.PHONY: all clean

all: cuda_ipc_probe cuda_vmm_probe cuda_share_load_probe kv_fork kv_batch

cuda_ipc_probe: cuda_ipc_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

cuda_vmm_probe: cuda_vmm_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcuda

cuda_share_load_probe: cuda_share_load_probe.cu
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

clean:
	$(RM) cuda_ipc_probe cuda_vmm_probe cuda_share_load_probe kv_fork kv_batch kv_fork_vmm kv_spawn
