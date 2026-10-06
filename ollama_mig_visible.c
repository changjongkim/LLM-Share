#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <string.h>

// Ollama validates a discovered CUDA device by launching a second runner
// with the device UUID reported by ggml. On Thor with MIG enabled, ggml
// reports the physical GPU UUID even though CUDA requires the MIG UUID in
// CUDA_VISIBLE_DEVICES. Keep Ollama's own environment unchanged and rewrite
// only that physical-UUID lookup inside dynamically loaded CUDA libraries.
typedef char *(*getenv_fn)(const char *);

static getenv_fn next_getenv;
static const char *mig_uuid;

static void resolve_getenv(void) {
    if (next_getenv == NULL) {
        *(void **) (&next_getenv) = dlsym(RTLD_NEXT, "getenv");
    }
}

__attribute__((constructor)) static void init_mig_uuid(void) {
    resolve_getenv();
    if (next_getenv != NULL) {
        mig_uuid = next_getenv("OLLAMA_MIG_VISIBLE_DEVICE");
    }
}

char *getenv(const char *name) {
    resolve_getenv();
    if (next_getenv == NULL) {
        return NULL;
    }
    char *value = next_getenv(name);
    if (mig_uuid != NULL && name != NULL && value != NULL &&
        strcmp(name, "CUDA_VISIBLE_DEVICES") == 0 &&
        strncmp(value, "GPU-", 4) == 0) {
        return (char *) mig_uuid;
    }
    return value;
}

char *secure_getenv(const char *name) {
    return getenv(name);
}
