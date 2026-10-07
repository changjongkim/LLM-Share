// kv_fork with the weights read from the model file instead of mapped, so
// that they are copied to device memory although the device accepts host
// memory (GGML_CUDA_HOST_PTR): the stack that maps the key-value cache of a
// prefix and shares nothing else. The program is kv_fork.cpp unchanged; only
// the default of the model parameters differs.
#include <llama.h>

namespace {

llama_model_params model_params_without_mmap() {
  llama_model_params params = llama_model_default_params();
  params.load_mode = LLAMA_LOAD_MODE_NONE;
  return params;
}

}  // namespace

#define llama_model_default_params model_params_without_mmap
#include "kv_fork.cpp"
