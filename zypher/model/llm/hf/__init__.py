"""
The transformers engine: torch + bitsandbytes on the safetensors weights.

Selected with ZYPHER_BACKEND=transformers. Needs requirements-transformers.txt
and the full 13.5 GB download; worth it only on a card with room for the
unquantized model.

    device.py      which hardware, and the budgets it can afford
    download.py    fetch and checksum the weights
    loader.py      tokenizer and model onto the device
    prompt.py      chat template and history trimming
    generation.py  streaming generation with KV-cache reuse
"""

import importlib
import importlib.util
import sys

# transformers imports torchaudio whenever it is installed, including on the
# way to loading a text-only model. One built for an older torch -- left behind
# when something upgraded torch -- fails to load its DLL with an OSError, and
# the model cannot be loaded at all. Nothing here uses audio, so a torchaudio
# that will not import is marked absent, and transformers skips it.
if importlib.util.find_spec("torchaudio") is not None:
    try:
        importlib.import_module("torchaudio")
    except Exception as error:
        sys.modules["torchaudio"] = None
        print("[ignoring torchaudio, which failed to import: {}]".format(
            type(error).__name__
        ))

import torch

from zypher import config

from ..text import OutOfMemory

# Ampere and later can run the fp32 matmuls that survive quantization -- the
# lm_head projection above all -- through the tensor cores at roughly three
# times the throughput. The precision given up is far below what 4-bit weights
# have already conceded.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from .device import DEVICE, VRAM_GB
from .download import download_model
from .generation import generate_response
from .loader import load_model, load_tokenizer, warm_up


class Engine:
    """The functions above behind the interface zypher.model.llm describes."""

    name = "transformers"

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.cache_state = {}

    def load(self):
        if not download_model():
            raise RuntimeError("model download or verification failed")

        self.tokenizer = load_tokenizer()
        self.model = load_model()

        if config.WARMUP:
            warm_up(self.tokenizer, self.model)

    def generate(self, conversation, continuing=False, on_text=None,
                 sampling=None, max_total_tokens=None, min_tokens=None):
        params = dict(sampling or config.SAMPLING["balanced"])

        # generate() rejects a zero temperature with sampling on; zero means
        # greedy.
        if params.get("temperature", 1.0) <= 0:
            params = {"do_sample": False}

        try:
            reply = generate_response(
                self.tokenizer,
                self.model,
                conversation,
                self.cache_state,
                continuing=continuing,
                on_text=on_text,
                sampling=params,
                max_total_tokens=max_total_tokens,
                min_tokens=min_tokens,
            )
        except torch.cuda.OutOfMemoryError as error:
            self.reset()
            raise OutOfMemory(str(error)) from error

        seconds = reply["seconds"]
        reply["cached_tokens"] = 0
        reply["tokens_per_second"] = reply["tokens"] / seconds if seconds > 0 else 0.0
        reply["throttled"] = None

        return reply

    def reset(self):
        self.cache_state.clear()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    def close(self):
        self.model = None
        self.cache_state.clear()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    def asleep(self):
        # This engine never unloads.
        return False

    def info(self):
        return {
            "backend": self.name,
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
            "vram_gb": round(VRAM_GB, 2),
            "context": config.MAX_PROMPT_TOKENS + config.MAX_TOTAL_NEW_TOKENS,
            "gpu_layers": "all" if DEVICE == "cuda" else "0",
            "kv_cache": "f16",
        }
