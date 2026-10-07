"""Immutable state extents under vLLM.

A vLLM server normally copies the weights into device memory and keeps its
KV cache in a device tensor of its own. On a device whose GPU follows the
host page tables, a CUDA tensor can stand on pageable host memory. This
plugin uses that to share two kinds of state between vLLM servers, which
may run in different MIG instances:

  weights   the first server writes its loaded tensors (after vLLM has
            repacked them for its kernels) to a file; every server then
            reads them from a shared read-only mapping of that file
  prefix    the first server keeps its KV cache in a file mapping and
            publishes the blocks of a prompt prefix; a later server maps
            those blocks read-only at the start of its own cache and
            registers them as cached, so that a request with the prefix
            starts from them without computing them

Nothing in vLLM is edited. The plugin is loaded through the entry point
group vllm.general_plugins in every process and changes three places by
wrapping them, each only when its environment variable is set:

  STATOR_WEIGHTS_PUBLISH=FILE  write the weights to FILE (once) and read them from there
  STATOR_WEIGHTS_ATTACH=FILE   read the weights from FILE
  STATOR_KV_PUBLISH=FILE       the KV cache is FILE; the first finished request is published
  STATOR_KV_ATTACH=FILE        the blocks that FILE.json names are mapped read-only

The KV variables need the connector of this package, which also asks vLLM
for the layout in which a block is one contiguous range:

  --kv-transfer-config '{"kv_connector":"StatorConnector",
      "kv_connector_module_path":"stator_vllm.connector","kv_role":"kv_both"}'

Both servers of a prefix need the same PYTHONHASHSEED, because vLLM seeds
its block hashes from it.

A device id that is not an integer (the UUID of a MIG instance in
CUDA_VISIBLE_DEVICES) is accepted in every case. vLLM reads the device
before it loads its plugins, so a server in a MIG instance is started with
`python -m stator_vllm.serve` in place of `vllm serve` (serve.py).
"""
import ctypes
import json
import logging
import mmap
import os
import time

logger = logging.getLogger("stator_vllm")

PAGE = 4096
HUGE = 2 << 20
PROT_NONE = 0
MAP_FIXED = 0x10
MAP_NORESERVE = 0x4000
MADV_HUGEPAGE = 14
MADV_POPULATE_WRITE = 23
# Tensors below this size stay where vLLM put them.
SMALLEST_SHARED_BYTES = 1 << 16

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                       ctypes.c_int, ctypes.c_long]
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

# The KV cache of this process when the plugin built it: base address, bytes,
# bytes of one block over all layers, and the file.
kv_state = {}
_patched = False


def _mmap(address, size, protection, flags, fd=-1, offset=0):
    result = _libc.mmap(address, size, protection, flags, fd, offset)
    if result is None or result == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_errno(), f"mmap of {size} bytes failed")
    return result


def _reserve(size):
    """An address range of size bytes that starts on a 2 MiB boundary.

    A file with 2 MiB pages is mapped with 2 MiB entries only when its
    offsets and the addresses agree modulo 2 MiB.
    """
    start = _mmap(None, size + HUGE, PROT_NONE, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE)
    return _round_up(start, HUGE)


def _populate(address, size):
    """Puts memory behind a writable range before the device writes it."""
    if size > 0 and _libc.madvise(address, size, MADV_POPULATE_WRITE) != 0:
        ctypes.memset(address, 0, size)


def _touch(address, size):
    """A CPU read of every page: the device faults on entries that were never accessed."""
    import numpy

    view = numpy.frombuffer((ctypes.c_ubyte * size).from_address(address), dtype=numpy.uint8)
    return int(view[::PAGE].sum())


class _Range:
    """Host memory as vLLM's device sees it."""

    def __init__(self, address, size):
        self.__cuda_array_interface__ = {"shape": (size,), "typestr": "|i1", "data": (address, False),
                                         "version": 2, "strides": None}


def device_bytes(address, size, device):
    """An int8 CUDA tensor over host memory, without a copy."""
    import torch

    return torch.as_tensor(_Range(address, size), device=device)


def _round_up(value, unit):
    return (value + unit - 1) // unit * unit


# ---------------------------------------------------------------- device ids
def accept_device_names():
    """A device that is named, not numbered: the UUID of a MIG instance in CUDA_VISIBLE_DEVICES."""
    from vllm.platforms import interface

    original = interface.Platform.device_id_to_physical_device_id.__func__
    if getattr(original, "accepts_names", False):
        return

    def device_id_to_physical_device_id(cls, device_id=0):
        try:
            return original(cls, device_id)
        except ValueError:
            # One instance of the only physical GPU.
            return 0

    device_id_to_physical_device_id.accepts_names = True
    interface.Platform.device_id_to_physical_device_id = classmethod(device_id_to_physical_device_id)


# ------------------------------------------------------------------- weights
def _shared_tensors(model):
    for kind, named in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in named:
            size = tensor.numel() * tensor.element_size()
            if tensor.is_cuda and size >= SMALLEST_SHARED_BYTES:
                yield kind, name, tensor, size


def publish_weights(model, path):
    """Writes the tensors that the kernels read to one file, each at an aligned offset.

    The file is mapped and the device copies each tensor into the mapping,
    so the weights do not pass through a second copy in host memory.
    """
    import torch

    started = time.monotonic()
    entries = []
    offset = 0
    tensors = []
    for kind, name, tensor, size in _shared_tensors(model):
        # The pages of the file are 2 MiB wherever a tensor starts in them.
        offset = _round_up(offset, PAGE)
        entries.append({"kind": kind, "name": name, "offset": offset, "bytes": size,
                        "dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape)})
        tensors.append(tensor)
        offset += size
    total = _round_up(offset, HUGE)
    fd = os.open(path + ".writing", os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd, total)
    base = _mmap(_reserve(total), total, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_FIXED, fd)
    os.close(fd)
    _populate(base, total)
    target = device_bytes(base, total, tensors[0].device)
    for entry, tensor in zip(entries, tensors):
        source = tensor.detach().contiguous().view(-1).view(torch.int8)
        target[entry["offset"]:entry["offset"] + entry["bytes"]].copy_(source)
    torch.cuda.synchronize()
    del target, source
    _libc.munmap(base, total)
    os.replace(path + ".writing", path)
    with open(path + ".json.writing", "w") as index:
        json.dump({"bytes": total, "entries": entries}, index)
    os.replace(path + ".json.writing", path + ".json")
    logger.warning("stator: weights_publish tensors=%d bytes=%d ms=%.1f file=%s", len(entries), offset,
                   (time.monotonic() - started) * 1000, path)


def attach_weights(model, path):
    """Replaces the device copy of every published tensor by a view of the shared mapping."""
    import torch

    with open(path + ".json") as index:
        described = json.load(index)
    started = time.monotonic()
    before = torch.cuda.memory_allocated()
    fd = os.open(path, os.O_RDONLY)
    base = _mmap(_reserve(described["bytes"]), described["bytes"], mmap.PROT_READ,
                 mmap.MAP_SHARED | MAP_FIXED, fd)
    os.close(fd)
    _touch(base, described["bytes"])
    replaced = 0
    shared = 0
    for entry in described["entries"]:
        module_path, _, attribute = entry["name"].rpartition(".")
        module = model.get_submodule(module_path)
        current = getattr(module, attribute)
        dtype = getattr(torch, entry["dtype"])
        if list(current.shape) != entry["shape"] or current.dtype != dtype:
            raise RuntimeError(f"stator: {entry['name']} differs from the published tensor")
        view = device_bytes(base + entry["offset"], entry["bytes"], current.device).view(dtype).view(entry["shape"])
        if entry["kind"] == "parameter":
            module._parameters[attribute].data = view
        else:
            module._buffers[attribute] = view
        replaced += 1
        shared += entry["bytes"]
    del current, view
    torch.cuda.empty_cache()
    logger.warning("stator: weights_attach tensors=%d bytes=%d ms=%.1f device_before_mib=%d device_after_mib=%d "
                   "device_reserved_mib=%d file=%s", replaced, shared, (time.monotonic() - started) * 1000,
                   before >> 20, torch.cuda.memory_allocated() >> 20, torch.cuda.memory_reserved() >> 20, path)


def _share_weights():
    from vllm.model_executor.model_loader import base_loader

    original = base_loader.BaseModelLoader.load_model

    def load_model(self, vllm_config, model_config, prefix=""):
        model = original(self, vllm_config, model_config, prefix)
        publish = os.environ.get("STATOR_WEIGHTS_PUBLISH")
        attach = os.environ.get("STATOR_WEIGHTS_ATTACH")
        if publish and not os.path.exists(publish + ".json"):
            publish_weights(model, publish)
        if publish or attach:
            attach_weights(model, publish or attach)
        return model

    base_loader.BaseModelLoader.load_model = load_model


# ------------------------------------------------------------------ KV cache
def _kv_memory(total, block_bytes):
    """The memory of the KV cache of this server: (address, description)."""
    publish = os.environ.get("STATOR_KV_PUBLISH")
    attach = os.environ.get("STATOR_KV_ATTACH")
    base = _reserve(total)
    if publish:
        # The whole cache is a file that later servers map.
        fd = os.open(publish, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        os.ftruncate(fd, total)
        _mmap(base, total, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_FIXED, fd)
        os.close(fd)
        _populate(base, total)
        return base, {"role": "publisher", "file": publish, "blocks": 0}
    with open(attach + ".json") as index:
        published = json.load(index)
    if published["block_bytes"] != block_bytes:
        raise RuntimeError("stator: the published blocks have another size than the blocks of this server")
    shared = published["blocks"] * block_bytes
    if block_bytes + shared > total:
        raise RuntimeError("stator: the published prefix does not fit in the KV cache of this server")
    _mmap(base, total, mmap.PROT_READ | mmap.PROT_WRITE,
          mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED)
    _libc.madvise(base, total, MADV_HUGEPAGE)
    # Block 0 is vLLM's null block and stays private; the published blocks
    # follow it at the offsets that they have in the file.
    fd = os.open(published["file"], os.O_RDONLY)
    _mmap(base + block_bytes, shared, mmap.PROT_READ, mmap.MAP_SHARED | MAP_FIXED, fd, block_bytes)
    os.close(fd)
    _touch(base + block_bytes, shared)
    _populate(base, block_bytes)
    _populate(base + block_bytes + shared, total - block_bytes - shared)
    return base, {"role": "agent", "file": published["file"], "blocks": published["blocks"],
                  "hashes": published["hashes"]}


def _share_kv_cache():
    import torch
    from vllm.v1.worker import gpu_model_runner
    from vllm.v1.worker import kv_connector_model_runner_mixin as mixin

    owner = mixin.KVConnectorModelRunnerMixin
    original = owner.__dict__["allocate_uniform_kv_caches"].__func__
    initialize = gpu_model_runner.GPUModelRunner.initialize_kv_cache
    profiling = [False]

    def initialize_kv_cache(self, kv_cache_config, is_profiling=False):
        # vLLM may build a small cache to measure its CUDA graphs and drop it again.
        profiling[0] = is_profiling
        try:
            return initialize(self, kv_cache_config, is_profiling)
        finally:
            profiling[0] = False

    def allocate_uniform_kv_caches(kv_cache_config, attn_groups, cache_dtype, device, kernel_block_sizes):
        if profiling[0]:
            return original(kv_cache_config, attn_groups, cache_dtype, device, kernel_block_sizes)
        block_bytes = attn_groups[0][0].kv_cache_spec.page_size_bytes * len(kv_cache_config.kv_cache_tensors)
        zeros = torch.zeros

        def host_zeros(*args, **kwargs):
            # The one allocation of the function: the buffer of all layers.
            if len(args) == 1 and isinstance(args[0], int) and kwargs.get("dtype") is torch.int8:
                started = time.monotonic()
                base, description = _kv_memory(args[0], block_bytes)
                kv_state.update(description, base=base, bytes=args[0], block_bytes=block_bytes)
                logger.warning("stator: kv_cache role=%s bytes=%d block_bytes=%d blocks=%d ms=%.1f",
                               description["role"], args[0], block_bytes, description["blocks"],
                               (time.monotonic() - started) * 1000)
                return device_bytes(base, args[0], kwargs["device"])
            return zeros(*args, **kwargs)

        torch.zeros = host_zeros
        try:
            return original(kv_cache_config, attn_groups, cache_dtype, device, kernel_block_sizes)
        finally:
            torch.zeros = zeros

    gpu_model_runner.GPUModelRunner.initialize_kv_cache = initialize_kv_cache
    owner.allocate_uniform_kv_caches = staticmethod(allocate_uniform_kv_caches)


def register():
    """Entry point of vllm.general_plugins; safe to call in every process and more than once."""
    global _patched
    if _patched:
        return
    _patched = True
    accept_device_names()
    if os.environ.get("STATOR_WEIGHTS_PUBLISH") or os.environ.get("STATOR_WEIGHTS_ATTACH"):
        _share_weights()
    if os.environ.get("STATOR_KV_PUBLISH") or os.environ.get("STATOR_KV_ATTACH"):
        _share_kv_cache()
