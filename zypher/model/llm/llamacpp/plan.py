"""Which GPU is present, and how the model is laid out on it.

Three numbers decide whether a small card is fast or unusable: how many
layers are on the GPU, how long the context window is, and how many bytes
each context token's attention state takes. They are worked out here from
the VRAM actually present, so the same code is right on a 6 GB laptop card
today and on whatever replaces it.
"""

import os
import subprocess

from zypher import config

from . import gguf

GIB = 1024 ** 3

# Windows the context is chosen from, largest first.
CONTEXTS = (32768, 24576, 16384, 12288, 8192, 6144, 4096)

# VRAM beyond weights and KV cache: the CUDA context and llama.cpp's compute
# buffers. Measured at 0.17 GiB for a 7B with flash attention at 12k context;
# planned for more than double, because the cost of guessing low is the
# driver spilling into system RAM, and the cost of guessing high is one step
# less context.
OVERHEAD_GIB = 0.4

# A full-precision KV cache is kept only when it still leaves this much
# context; below it, halving the cost per token is worth more.
F16_MIN_CONTEXT = 16384


def _query_gpu(fields):
    """One nvidia-smi query for the first GPU: a list of strings, or None.

    Asked of nvidia-smi rather than of torch: importing torch to read one
    number costs a second and several hundred MB of RAM, in a process that
    otherwise never needs it.
    """

    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, creationflags=flags,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    return [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]


def detect_gpu():
    """The first NVIDIA GPU as {name, total_gib, used_gib}, or None."""

    answer = _query_gpu("name,memory.total,memory.used")

    try:
        name, total, used = answer

        return {
            "name": name,
            "total_gib": float(total) / 1024,
            "used_gib": float(used) / 1024,
        }
    except (TypeError, ValueError):
        return None


def gpu_temperature():
    """GPU core temperature in degrees C, or None if it cannot be read."""

    answer = _query_gpu("temperature.gpu")

    try:
        return int(float(answer[0]))
    except (TypeError, ValueError, IndexError):
        return None


def _largest_context(limit_bytes, per_token, ceiling):
    for context in CONTEXTS:
        if context <= ceiling and context * per_token <= limit_bytes:
            return context

    return None


def make_plan(model_path):
    """Decide context, GPU layers, KV precision and load mode for this model
    and card.

    Returns a dict: device, gpu (or None), context, gpu_layers, kv_type,
    load_mode, fit (whether llama.cpp is left to place layers itself), and
    the estimated VRAM in GiB. Anything set explicitly in the config is taken
    as given.
    """

    shape = gguf.describe(model_path)
    weights = os.path.getsize(model_path)
    gpu = detect_gpu()
    ceiling = min(shape["context"], config.CTX_MAX)

    plan = {
        "model": shape["name"],
        "gpu": gpu,
        "device": "cuda" if gpu else "cpu",
        "weights_gib": weights / GIB,
    }

    if gpu is None or config.GPU_LAYERS == "0":
        plan.update(
            device="cpu",
            context=config.CTX or min(ceiling, 8192),
            gpu_layers="0",
            kv_type=config.KV_TYPE or "f16",
            load_mode=config.LOAD_MODE or "auto",
            fit=False,
            vram_gib=0.0,
        )

        return plan

    # What is already in use belongs to the desktop and whatever else is
    # open, and is not ours to plan with.
    reserve = max(config.VRAM_RESERVE_GB, gpu["used_gib"] + 0.1)
    for_kv = (gpu["total_gib"] - reserve - OVERHEAD_GIB) * GIB - weights

    kv_type = config.KV_TYPE
    context = config.CTX

    if not context:
        if kv_type:
            context = _largest_context(
                for_kv, gguf.kv_bytes_per_token(shape, kv_type), ceiling
            )
        else:
            context = _largest_context(
                for_kv, gguf.kv_bytes_per_token(shape, "f16"), ceiling
            )
            kv_type = "f16"

            if context is None or context < min(F16_MIN_CONTEXT, ceiling):
                kv_type = "q8_0"
                context = _largest_context(
                    for_kv, gguf.kv_bytes_per_token(shape, "q8_0"), ceiling
                )

    kv_type = kv_type or "q8_0"
    fits = context is not None and (
        context * gguf.kv_bytes_per_token(shape, kv_type) <= for_kv
    )

    if context is None:
        context = min(ceiling, CONTEXTS[-1])

    # Whole model on the GPU whenever it fits. When it cannot, llama.cpp
    # places as many layers as there is room for and runs the rest on the
    # CPU: slower, but it runs.
    gpu_layers = config.GPU_LAYERS or ("all" if fits else "auto")

    plan.update(
        context=context,
        gpu_layers=gpu_layers,
        kv_type=kv_type,
        # With everything on the card there is nothing for a memory map to
        # serve afterwards, and on Windows it cannot be let go of -- see
        # config.LOAD_MODE.
        load_mode=config.LOAD_MODE or ("none" if gpu_layers == "all" else "auto"),
        fit=gpu_layers == "auto",
        vram_gib=(
            weights + context * gguf.kv_bytes_per_token(shape, kv_type)
        ) / GIB + OVERHEAD_GIB,
    )

    return plan
