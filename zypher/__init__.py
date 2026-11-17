"""
zypherLL: a local Zephyr 7B assistant with web lookup and memory.

Laid out Model / View / Presenter:

    model/      what the assistant knows and runs on -- the LLM, the memory
                store, live web search. No I/O with the user.
    presenter/  one turn, start to finish: grounding, sampling, generation,
                learning. Shared by every view.
    view/       how a person or a program talks to it -- the console, the HTTP
                API. Formatting and transport only.
    config.py   every setting.

Process-wide environment is set here, because it has to be in place before
torch, transformers or huggingface_hub are first imported.
"""

import os
import sys

# The Xet transfer backend has been observed to stream every byte of a large
# shard and then hang before moving the file into place, which looks like a
# download frozen near 100%. Plain HTTPS with range resume is the reliable
# path. Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# A long answer grows its KV cache by a block per token. Under the default
# caching allocator those blocks fragment the reserved pool, and a later turn
# fails to find a contiguous span while nvidia-smi still reports free memory.
# Expandable segments let the allocator grow a segment in place instead, which
# is what keeps a small card alive across a long chat.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# The fast tokenizer spins up a thread pool per call. Prompts here are one
# sequence at a time, so the forking costs more than the parallelism returns --
# and it prints a warning about it on every generate.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# The Windows console defaults to cp1252, which cannot encode most of what
# comes back from a web search -- currency symbols, dashes, curly quotes -- and
# printing one raises UnicodeEncodeError in the middle of a reply. Replacing
# the odd character beats losing the answer.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
