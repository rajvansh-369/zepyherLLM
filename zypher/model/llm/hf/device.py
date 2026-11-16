"""Which hardware is present, and the context budgets it can afford."""

import torch

from zypher import config
from zypher.config import SMALL_VRAM_GB, SMALL_VRAM_PROMPT_TOKENS, SMALL_VRAM_TOTAL_NEW_TOKENS


def describe_device():
    """Pick a device and report what we are working with."""

    print("=" * 60)
    print("Zephyr 7B Local LLM")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("GPU: not available -- running on CPU")
        return "cpu", 0.0

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)

    print("GPU:", props.name)
    print("VRAM:", round(vram_gb, 2), "GB")

    return "cuda", vram_gb


DEVICE, VRAM_GB = describe_device()

if DEVICE == "cuda" and 0 < VRAM_GB < SMALL_VRAM_GB:
    config.MAX_PROMPT_TOKENS = min(config.MAX_PROMPT_TOKENS, SMALL_VRAM_PROMPT_TOKENS)
    config.MAX_TOTAL_NEW_TOKENS = min(config.MAX_TOTAL_NEW_TOKENS, SMALL_VRAM_TOTAL_NEW_TOKENS)
    print("Small VRAM: prompt budget {} tokens, answer budget {} tokens.".format(
        config.MAX_PROMPT_TOKENS, config.MAX_TOTAL_NEW_TOKENS
    ))
