"""
The language model: download, load, prompt, generate.

    device.py      which hardware, and the budgets it can afford
    download.py    fetch and checksum the weights
    loader.py      tokenizer and model onto the device
    prompt.py      system prompt and chat template
    sampling.py    sampling profile per question
    generation.py  streaming generation with KV-cache reuse, and reply clean-up
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


# Ampere and later can run the fp32 matmuls that survive quantization -- the
# lm_head projection above all -- through the tensor cores at roughly three
# times the throughput. The precision given up is far below what 4-bit weights
# have already conceded.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# The public surface of the model layer.
from .device import DEVICE, VRAM_GB
from .download import download_model
from .generation import ClientGone, clean_reply, generate_response
from .loader import load_model, load_tokenizer, warm_up
from .prompt import system_prompt
from .sampling import pick_sampling
