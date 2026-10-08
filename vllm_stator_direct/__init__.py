"""Load published post-Marlin vLLM weights without a private device copy.

This experimental package complements :mod:`stator_vllm`.  A publisher uses
vLLM's normal loader, writes every registered device tensor after post-load
processing, and switches to the resulting read-only mapping.  An attacher
constructs only the model structure on the meta device and installs those
post-processed tensors directly.  It therefore never materializes checkpoint
weights or the temporary tensors used by AWQ-to-Marlin conversion.

The package is deliberately separate from ``vllm_stator`` because published
campaigns pin that package's source hash.
"""

from __future__ import annotations

import ctypes
import json
import logging
import mmap
import os
import time
from typing import Any

import stator_vllm

logger = logging.getLogger("stator_vllm_direct")

DIRECT_WEIGHTS_PUBLISH = "STATOR_DIRECT_WEIGHTS_PUBLISH"
DIRECT_WEIGHTS_ATTACH = "STATOR_DIRECT_WEIGHTS_ATTACH"

_mapping_bases: list[tuple[int, int]] = []
_device_mappings = []
_patched = False


def _round_up(value: int, unit: int) -> int:
    return (value + unit - 1) // unit * unit


def _registered_device_tensors(model):
    for kind, named in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, tensor in named:
            size = tensor.numel() * tensor.element_size()
            if tensor.is_cuda and size:
                yield kind, name, tensor, size


def publish_weights(model, path: str) -> None:
    """Publish every registered device tensor in its post-load layout."""
    import torch

    started = time.monotonic()
    entries: list[dict[str, Any]] = []
    tensors = []
    offset = 0
    for kind, name, tensor, size in _registered_device_tensors(model):
        alignment = (
            stator_vllm.HUGE
            if size >= stator_vllm.SMALLEST_SHARED_BYTES
            else stator_vllm.PAGE
        )
        offset = _round_up(offset, alignment)
        entries.append(
            {
                "kind": kind,
                "name": name,
                "offset": offset,
                "bytes": size,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
            }
        )
        tensors.append(tensor)
        offset += size
    if not tensors:
        raise RuntimeError("stator-direct: the loaded model has no device tensors")

    total = _round_up(offset, stator_vllm.HUGE)
    writing = path + ".writing"
    fd = os.open(writing, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, total)
        base = stator_vllm._mmap(
            stator_vllm._reserve(total),
            total,
            mmap.PROT_READ | mmap.PROT_WRITE,
            mmap.MAP_SHARED | stator_vllm.MAP_FIXED,
            fd,
        )
    finally:
        os.close(fd)
    stator_vllm._populate(base, total)
    target = stator_vllm.device_bytes(base, total, tensors[0].device)
    for entry, tensor in zip(entries, tensors, strict=True):
        source = tensor.detach().contiguous().view(-1).view(torch.int8)
        target[entry["offset"] : entry["offset"] + entry["bytes"]].copy_(source)
    torch.cuda.synchronize()
    del source, target
    stator_vllm._libc.munmap(base, total)
    os.replace(writing, path)
    index_writing = path + ".json.writing"
    with open(index_writing, "w", encoding="utf-8") as index:
        json.dump({"format": 1, "bytes": total, "entries": entries}, index)
    os.replace(index_writing, path + ".json")
    logger.warning(
        "stator-direct: weights_publish tensors=%d bytes=%d file_bytes=%d ms=%.1f file=%s",
        len(entries),
        sum(entry["bytes"] for entry in entries),
        total,
        (time.monotonic() - started) * 1000,
        path,
    )


def _open_mapping(path: str):
    with open(path + ".json", encoding="utf-8") as index:
        described = json.load(index)
    if described.get("format") != 1:
        raise RuntimeError("stator-direct: unsupported weights index format")
    fd = os.open(path, os.O_RDWR)
    try:
        base = stator_vllm._mmap(
            stator_vllm._reserve(described["bytes"]),
            described["bytes"],
            mmap.PROT_READ | mmap.PROT_WRITE,
            mmap.MAP_SHARED | stator_vllm.MAP_FIXED,
            fd,
        )
    finally:
        os.close(fd)
    stator_vllm._touch(base, described["bytes"])
    _mapping_bases.append((base, described["bytes"]))
    return described, base


def attach_weights(model, path: str, target_device) -> tuple[int, int]:
    """Install published tensors, allowing Marlin's final shapes to differ."""
    import torch

    described, base = _open_mapping(path)
    registered = {
        ("parameter", name)
        for name, tensor in model.named_parameters()
        if tensor.numel()
    } | {
        ("buffer", name)
        for name, tensor in model.named_buffers()
        if tensor.numel()
    }
    published = {(entry["kind"], entry["name"]) for entry in described["entries"]}
    missing = sorted(registered - published)
    unexpected = sorted(published - registered)
    if missing or unexpected:
        raise RuntimeError(
            "stator-direct: model/index names differ: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )

    cudart = torch.cuda.cudart()
    registered = cudart.cudaHostRegister(
        base, described["bytes"], 0x01 | 0x02
    )
    if registered != cudart.cudaError.success:
        raise RuntimeError(
            f"stator-direct: cudaHostRegister failed: {registered}"
        )
    if stator_vllm._libc.mprotect(
        base, described["bytes"], mmap.PROT_READ
    ) != 0:
        raise OSError(
            ctypes.get_errno(),
            "stator-direct: mprotect of registered weights failed",
        )
    device_mapping = stator_vllm.device_bytes(
        base, described["bytes"], target_device
    )
    _device_mappings.append(device_mapping)
    shared = 0
    for entry in described["entries"]:
        module_path, _, attribute = entry["name"].rpartition(".")
        module = model.get_submodule(module_path)
        dtype = getattr(torch, entry["dtype"])
        view = device_mapping[
            entry["offset"] : entry["offset"] + entry["bytes"]
        ].view(dtype).view(entry["shape"])
        if entry["kind"] == "parameter":
            module._parameters[attribute] = torch.nn.Parameter(
                view, requires_grad=False
            )
        else:
            module._buffers[attribute] = view
        shared += entry["bytes"]
    return len(described["entries"]), shared


def _restore_runtime_state(model, target_device) -> None:
    """Rebuild small non-state-dict objects normally made during post-load."""
    import torch
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
        MarlinLinearKernel,
    )
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.quantization.awq_marlin import (
        AWQMarlinLinearMethod,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_is_k_full,
        marlin_make_empty_g_idx,
        marlin_make_workspace_new,
    )

    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, AWQMarlinLinearMethod):
            kernel = quant_method.kernel
            if not isinstance(kernel, MarlinLinearKernel):
                raise RuntimeError(
                    "stator-direct: only the measured MarlinLinearKernel is supported"
                )
            config = kernel.config
            row_parallel = (
                config.partition_weight_shape[0] != config.full_weight_shape[0]
            )
            kernel.is_k_full = marlin_is_k_full(config.has_g_idx, row_parallel)
            kernel.workspace = marlin_make_workspace_new(target_device)
            if kernel.w_gidx_name is None:
                kernel.w_gidx_name = "g_idx"
            if kernel.w_zp_name is None:
                kernel.w_zp_name = "w_zp"
            module.g_idx = marlin_make_empty_g_idx(target_device)
            module.g_idx_sort_indices = marlin_make_empty_g_idx(target_device)
            module.input_global_scale = None
        if isinstance(module, Attention):
            module._q_scale_float = 1.0
            module._k_scale_float = 1.0
            module._v_scale_float = 1.0
            module._prob_scale_float = 1.0
            from vllm import envs

            module.q_range = torch.tensor(
                envs.Q_SCALE_CONSTANT, dtype=torch.float32, device="cpu"
            )
            module.k_range = torch.tensor(
                envs.K_SCALE_CONSTANT, dtype=torch.float32, device="cpu"
            )
            module.v_range = torch.tensor(
                envs.V_SCALE_CONSTANT, dtype=torch.float32, device="cpu"
            )


def direct_load_model(vllm_config, model_config, prefix: str = ""):
    """Construct the model on meta and attach final tensors from the mapping."""
    import torch
    from vllm.model_executor.model_loader.base_loader import log_model_inspection
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    path = os.environ[DIRECT_WEIGHTS_ATTACH]
    load_config = vllm_config.load_config
    load_device = (
        vllm_config.device_config.device
        if load_config.device is None
        else load_config.device
    )
    target_device = torch.device(load_device)
    started = time.monotonic()
    before = torch.cuda.memory_allocated()
    with set_default_torch_dtype(model_config.dtype):
        with torch.device("meta"):
            model = initialize_model(
                vllm_config=vllm_config,
                model_config=model_config,
                prefix=prefix,
            )
        log_model_inspection(model)
        count, shared = attach_weights(model, path, target_device)
        _restore_runtime_state(model, target_device)

    leftovers = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.device.type == "meta"
    ]
    if leftovers:
        raise RuntimeError(f"stator-direct: meta tensors remain: {leftovers[:8]}")
    logger.warning(
        "stator-direct: weights_attach tensors=%d bytes=%d ms=%.1f "
        "device_before_mib=%d device_after_mib=%d device_reserved_mib=%d file=%s",
        count,
        shared,
        (time.monotonic() - started) * 1000,
        before >> 20,
        torch.cuda.memory_allocated() >> 20,
        torch.cuda.memory_reserved() >> 20,
        path,
    )
    return model.eval()


def register() -> None:
    """Patch vLLM's loader only when a direct-share variable is present."""
    global _patched
    if _patched:
        return
    publish = os.environ.get(DIRECT_WEIGHTS_PUBLISH)
    attach = os.environ.get(DIRECT_WEIGHTS_ATTACH)
    if not publish and not attach:
        return
    if publish and attach:
        raise RuntimeError("stator-direct: publish and attach are mutually exclusive")

    from vllm.model_executor.model_loader import base_loader

    original = base_loader.BaseModelLoader.load_model
    if attach:

        def load_model(self, vllm_config, model_config, prefix=""):
            return direct_load_model(vllm_config, model_config, prefix)

    else:

        def load_model(self, vllm_config, model_config, prefix=""):
            model = original(self, vllm_config, model_config, prefix)
            if not os.path.exists(publish + ".json"):
                publish_weights(model, publish)
            started = time.monotonic()
            before = __import__("torch").cuda.memory_allocated()
            count, shared = attach_weights(
                model, publish, next(model.parameters()).device
            )
            __import__("torch").cuda.empty_cache()
            logger.warning(
                "stator-direct: weights_attach tensors=%d bytes=%d ms=%.1f "
                "device_before_mib=%d device_after_mib=%d file=%s",
                count,
                shared,
                (time.monotonic() - started) * 1000,
                before >> 20,
                __import__("torch").cuda.memory_allocated() >> 20,
                publish,
            )
            return model

    base_loader.BaseModelLoader.load_model = load_model
    _patched = True
