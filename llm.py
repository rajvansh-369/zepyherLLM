"""
Zephyr 7B local chat runner.

Tuned for small consumer GPUs: the model is loaded in 4-bit NF4 when the
card cannot hold it in bf16, and the weights are checksummed before use
rather than assumed good because config.json happens to be on disk.
"""

import datetime
import hashlib
import importlib.util
import json
import os
import re
import sys
import threading
import time

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

from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

import memory as memory_store
import retrieval


# Ampere and later can run the fp32 matmuls that survive quantization -- the
# lm_head projection above all -- through the tensor cores at roughly three
# times the throughput. The precision given up is far below what 4-bit weights
# have already conceded.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "richardyoung/zephyr-7b-beta-abliterated"

# Change this to your required location
MODEL_PATH = r"C:\AI\Models\zephyr-7b-beta-abliterated"

# The standing instructions. A 7B follows short, concrete rules far better
# than general exhortations ("be accurate"), and it follows the format it is
# shown: the rules below are what turns a wall of prose into a direct answer
# followed by headings, lists and fenced code. system_prompt() adds today's
# date and what is known about the user.
SYSTEM_PROMPT = (
    "You are a knowledgeable, precise and helpful AI assistant.\n"
    "\n"
    "Accuracy:\n"
    "- Answer exactly what was asked. Give the direct answer first, then the "
    "explanation.\n"
    "- If you are not sure, say so. Never invent facts, numbers, quotes, "
    "sources, links or API names.\n"
    "- Your knowledge comes from training data that ends in 2023. For anything "
    "that may have changed since, say that your information may be out of "
    "date.\n"
    "- For maths and multi-step problems, work through the steps, then give "
    "the final result on its own line.\n"
    "\n"
    "Format (Markdown):\n"
    "- Match the length to the question: a simple question gets one to three "
    "sentences and no headings.\n"
    "- For longer answers, use ## headings for separate parts, numbered lists "
    "for steps, bullet points for options, and a table to compare things.\n"
    "- Use **bold** only for the key term or the final answer.\n"
    "- Put code in fenced blocks that name the language, like ```python.\n"
    "- When asked for code, output the complete file -- every tag, every rule, "
    "closed and runnable. Never abbreviate with a placeholder such as "
    "'... rest of the code ...' and never stop early to save space.\n"
    "- No filler: do not open with \"Sure!\" or \"Great question\", and do not "
    "repeat the question back."
)

# Generation
#
# 512 was far too low for anything file-sized: a page of HTML plus its CSS is
# commonly 1500-4000 tokens, so every such answer was cut off mid-tag. The cap
# below is the budget for one generate() call; MAX_TOTAL_NEW_TOKENS is the
# budget for an answer, which auto-continue may spread over several calls.
MAX_NEW_TOKENS = 2048

# Sampling is chosen per question -- see pick_sampling. Code, maths and
# questions of fact want the most likely continuation: at 0.7 a 7B misspells
# API names and drifts off the figures in its sources. Stories and
# brainstorming read flat much below 0.8. min_p drops every token far less
# likely than the best one, which stops a sampled answer wandering into
# nonsense without flattening its word choice the way a low top_p does.
SAMPLING = {
    "precise": {"temperature": 0.3, "top_p": 0.9, "min_p": 0.05},
    "balanced": {"temperature": 0.6, "top_p": 0.9, "min_p": 0.05},
    "creative": {"temperature": 0.85, "top_p": 0.95, "min_p": 0.05},
}

# Kept mild. A higher penalty actively damages code, where repeated tokens
# (closing tags, indentation, repeated property names) are correct. Applied
# to the answer's own tokens only -- see GeneratedRepetitionPenalty.
REPETITION_PENALTY = 1.05

# Zephyr's role markers. The model occasionally writes one instead of ending
# its turn and goes on to invent the user's next message; generation stops at
# the marker, and it never reaches the screen or the stored answer.
STOP_STRINGS = ("<|user|>", "<|system|>", "<|assistant|>")

# Prompt tokens kept before the oldest turns are dropped. Bounds the KV
# cache, which is what actually runs a small GPU out of memory mid-chat.
MAX_PROMPT_TOKENS = 8192

# Ceiling for one answer, across auto-continue rounds.
MAX_TOTAL_NEW_TOKENS = 8192

# When generation stops because it hit the per-call cap rather than because
# the model emitted EOS, resume from the KV cache it just built instead of
# ending the answer. Continuing the same token stream is what keeps long
# output whole: re-prompting with "continue" makes the model restate the last
# paragraph and often re-open a tag it had already closed.
AUTO_CONTINUE = True

# KV cache for this 7B is ~128 KB/token (32 layers, 8 KV heads, GQA, fp16).
# On a small card the 4-bit weights already hold ~3.9 GB, so the budgets above
# would push it into OOM. Scale them down instead of dying mid-answer.
SMALL_VRAM_GB = 8.0
SMALL_VRAM_PROMPT_TOKENS = 4096
SMALL_VRAM_TOTAL_NEW_TOKENS = 4096

# Below this much VRAM (GB) the weights are quantized to 4-bit.
BF16_VRAM_REQUIRED_GB = 16.0
FOURBIT_VRAM_REQUIRED_GB = 5.0

# Weights are frozen at training time, so anything the model says about the
# present is a guess from 2024. When this is on, time-bound questions are
# answered from web snippets fetched at ask time instead. Toggle at runtime
# with '/web'.
RETRIEVAL_ENABLED = True

# Past exchanges are recalled by similarity and placed in the prompt, and
# stated preferences go into the system prompt, so the runner gets better at
# this particular user without the weights ever changing. Toggle at runtime
# with '/memory'.
MEMORY_ENABLED = True

# Streaming a token at a time means a flush per token, and each console flush
# is a synchronous write holding the GIL the generation thread wants. Buffering
# to this interval still looks continuous to a human and takes the writes off
# the hot path.
STREAM_FLUSH_SECONDS = 0.05

# One throwaway generate() after loading, so CUDA context setup, kernel
# autotuning and the first 4-bit dequantization of every layer are paid for up
# front rather than as several seconds of silence inside the first answer.
WARMUP = True

# Reusing fewer cached tokens than this saves less time than the bookkeeping
# around it costs.
MIN_CACHE_REUSE_TOKENS = 32

DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_WORKERS = 4

# Weight formats we never want a second copy of.
IGNORE_PATTERNS = [
    "*.bin",
    "*.pth",
    "*.msgpack",
    "*.h5",
    "*.gguf",
    "consolidated*",
    "original/*",
]


# ============================================================
# DEVICE
# ============================================================

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
    MAX_PROMPT_TOKENS = min(MAX_PROMPT_TOKENS, SMALL_VRAM_PROMPT_TOKENS)
    MAX_TOTAL_NEW_TOKENS = min(MAX_TOTAL_NEW_TOKENS, SMALL_VRAM_TOTAL_NEW_TOKENS)
    print("Small VRAM: prompt budget {} tokens, answer budget {} tokens.".format(
        MAX_PROMPT_TOKENS, MAX_TOTAL_NEW_TOKENS
    ))


# ============================================================
# VERIFY WHAT IS ON DISK
# ============================================================

def expected_weight_files(path):
    """Return (shard_names, expected_total_bytes) from the safetensors index.

    Falls back to the single-file layout when the model is not sharded.
    """

    index_file = os.path.join(path, "model.safetensors.index.json")

    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as handle:
            index = json.load(handle)

        shards = sorted(set(index["weight_map"].values()))
        total = index.get("metadata", {}).get("total_size", 0)

        return shards, total

    return ["model.safetensors"], 0


def staging_dir(path):
    """Where huggingface_hub keeps partial downloads and etag sidecars."""

    return os.path.join(path, ".cache", "huggingface", "download")


def expected_sha256(path, filename):
    """Read the expected hash from the sidecar huggingface_hub writes.

    The sidecar is three lines: commit hash, etag, timestamp. For LFS files --
    which is every weight shard -- the etag is the sha256 of the content, so
    integrity can be checked without going back to the network.
    """

    sidecar = os.path.join(staging_dir(path), filename + ".metadata")

    if not os.path.exists(sidecar):
        return None

    with open(sidecar, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    if len(lines) < 2:
        return None

    etag = lines[1].strip()

    # Non-LFS files carry a git blob sha1 instead; only sha256 is usable here.
    return etag if len(etag) == 64 else None


def file_sha256(file_path, chunk_size=16 << 20):
    """Hash a file, using the C reader when the interpreter provides one.

    hashlib.file_digest (3.11+) reads into a reusable buffer and drops the GIL
    for the whole file. The loop below allocates a fresh 16 MB bytes object per
    block, which over a 13.5 GB model is several gigabytes of garbage created
    and immediately discarded.
    """

    with open(file_path, "rb") as handle:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(handle, "sha256").hexdigest()

        digest = hashlib.sha256()

        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)

    return digest.hexdigest()


def _cache_file(path):
    return os.path.join(path, ".verified.json")


def _load_verify_cache(path):
    try:
        with open(_cache_file(path), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _save_verify_cache(path, cache):
    try:
        with open(_cache_file(path), "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2)
    except OSError:
        pass


def quarantine(path, filename):
    """Delete a bad shard and its sidecar so the next attempt refetches it."""

    targets = [
        os.path.join(path, filename),
        os.path.join(staging_dir(path), filename + ".metadata"),
    ]

    for target in targets:
        try:
            os.remove(target)
        except OSError:
            pass


def verify_model_files(path, check_hashes=True):
    """Check the weights are present and intact.

    The original code tested for config.json, but the small JSON files land
    first and the multi-GB shards land last -- so that test passed while the
    weights were still missing, and every rerun skipped the resume.

    Hashing 13.5 GB takes a couple of minutes, so a verified result is cached
    against each shard's size and mtime. In practice each shard is hashed once,
    on the run that downloads it, and startup is instant from then on.

    Returns (ok, list_of_problems).
    """

    problems = []

    if not os.path.exists(os.path.join(path, "config.json")):
        return False, ["config.json missing"]

    shards, expected_total = expected_weight_files(path)

    present = []
    actual_total = 0

    for shard in shards:
        shard_path = os.path.join(path, shard)

        if not os.path.exists(shard_path):
            problems.append(shard + " missing")
            continue

        present.append(shard)
        actual_total += os.path.getsize(shard_path)

    staging = staging_dir(path)

    if os.path.isdir(staging):
        for name in os.listdir(staging):
            if name.endswith(".incomplete"):
                size_gb = os.path.getsize(os.path.join(staging, name)) / (1024 ** 3)
                problems.append(
                    "partial file left by an interrupted download ({:.2f} GB)".format(size_gb)
                )

    # The index reports the size of the tensor data only; each shard also
    # carries a small safetensors header, so files on disk run slightly
    # larger than expected_total. Short means truncated.
    if expected_total and not problems and actual_total < expected_total:
        problems.append(
            "weights truncated: have {} bytes, need at least {} bytes".format(
                actual_total, expected_total
            )
        )

    if problems or not check_hashes:
        return (not problems), problems

    cache = _load_verify_cache(path)
    cache_dirty = False

    for shard in present:
        shard_path = os.path.join(path, shard)
        want = expected_sha256(path, shard)

        if want is None:
            # No sidecar to compare against; size checks above are all we have.
            continue

        stat = os.stat(shard_path)
        entry = cache.get(shard)

        if (
            entry
            and entry.get("sha256") == want
            and entry.get("size") == stat.st_size
            and entry.get("mtime_ns") == stat.st_mtime_ns
        ):
            continue

        print("  verifying {} ...".format(shard), end="", flush=True)
        got = file_sha256(shard_path)

        if got == want:
            print(" ok")
            cache[shard] = {
                "sha256": got,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            cache_dirty = True
        else:
            print(" CORRUPT")
            problems.append(shard + " failed checksum -- discarding it")
            cache.pop(shard, None)
            cache_dirty = True
            quarantine(path, shard)

    if cache_dirty:
        _save_verify_cache(path, cache)

    return (not problems), problems


# ============================================================
# DOWNLOAD
# ============================================================

def download_model():
    """Download or resume the model, then confirm the weights are complete."""

    os.makedirs(MODEL_PATH, exist_ok=True)

    ok, problems = verify_model_files(MODEL_PATH)

    if ok:
        print("\nModel present and complete:", MODEL_PATH)
        return True

    print("\nModel not ready:")
    for problem in problems:
        print("  -", problem)

    # Partials are always resumed, never discarded. A Xet-written partial can
    # be full-size with holes, but throwing every partial away to guard
    # against that costs gigabytes on each restart. The checksum below catches
    # a bad shard once it is finalized and quarantines it then, so the worst
    # case is one wasted refetch instead of a guaranteed one.
    print("\nDownloading (resuming from whatever is already on disk)...")
    print("Model:", MODEL_NAME)
    print("Destination:", MODEL_PATH)

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):

        try:
            snapshot_download(
                repo_id=MODEL_NAME,
                local_dir=MODEL_PATH,
                ignore_patterns=IGNORE_PATTERNS,
                max_workers=DOWNLOAD_WORKERS,
            )

        except KeyboardInterrupt:
            print("\nInterrupted. Rerun to resume from here.")
            return False

        except Exception as error:
            print("\nAttempt {}/{} failed: {}: {}".format(
                attempt, DOWNLOAD_ATTEMPTS, type(error).__name__, error
            ))

        ok, problems = verify_model_files(MODEL_PATH)

        if ok:
            print("\nDownload complete and verified.")
            return True

        if attempt < DOWNLOAD_ATTEMPTS:
            for problem in problems:
                print("  -", problem)

            # Partials are deliberately left alone here. With Xet disabled
            # they are plain append-only files, so the next attempt resumes
            # from the byte offset instead of refetching several GB. A partial
            # that does turn out to be bad is caught by the checksum once it
            # is finalized, and quarantined then.
            backoff = min(30, 2 ** attempt)
            print("Retrying in {}s...".format(backoff))
            time.sleep(backoff)

    print("\nDownload did not complete:")
    for problem in problems:
        print("  -", problem)

    return False


# ============================================================
# LOAD
# ============================================================

def load_tokenizer():

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer loaded.")

    return tokenizer


def choose_load_plan():
    """Decide how to load given the hardware actually present.

    A 7B model is ~13.5 GB in bf16. On a 6 GB card device_map="auto" does not
    fail -- it quietly spills most layers to system RAM or disk, and
    generation drops to a token or two per second. 4-bit NF4 puts the whole
    model on the GPU at roughly 3.9 GB instead.
    """

    if DEVICE == "cpu":
        return "cpu"

    if VRAM_GB >= BF16_VRAM_REQUIRED_GB:
        return "gpu-bf16"

    if VRAM_GB >= FOURBIT_VRAM_REQUIRED_GB:
        return "gpu-4bit"

    return "cpu"


def load_model():

    plan = choose_load_plan()

    print("\nLoading model [{}]...".format(plan))

    common = dict(
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    if plan == "gpu-bf16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map={"": 0},
            **common
        )

    elif plan == "gpu-4bit":
        print("VRAM is below {} GB -- quantizing weights to 4-bit NF4.".format(
            BF16_VRAM_REQUIRED_GB
        ))

        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            # Quantizes the quantization constants too; saves ~0.4 GB.
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=quant_config,
            device_map={"": 0},
            **common
        )

    else:
        # float32 would need ~28 GB of RAM. bf16 halves that and matches the
        # dtype the weights were stored in.
        print("Loading on CPU. Expect a few tokens per second.")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            **common
        )

    model.eval()

    # Settings that never change live on the model's generation config. The
    # ones chosen per question -- temperature, top_p, min_p -- are passed to
    # each generate() call instead.
    #
    # The built-in repetition penalty stays off: it counts the prompt too.
    # GeneratedRepetitionPenalty applies the same penalty to the answer alone.
    config = model.generation_config
    config.use_cache = True
    config.do_sample = True
    config.repetition_penalty = 1.0

    if plan != "cpu":
        assert_fully_on_gpu(model)

    if DEVICE == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print("Model loaded. GPU memory in use: {:.2f} GB".format(allocated))
    else:
        print("Model loaded.")

    return model


def warm_up(tokenizer, model):
    """Generate two throwaway tokens so the first real answer is not the slow one.

    The first forward pass through a freshly loaded model pays for the CUDA
    context, cuBLAS handle creation, SDPA kernel selection and -- on the 4-bit
    path -- the first dequantization of every layer. Left to happen inside the
    first answer, that is several seconds of nothing after the user has already
    pressed enter.
    """

    print("Warming up...", end="", flush=True)

    ids = tokenizer("hi", return_tensors="pt").input_ids.to(model.device)

    # The stop strings go through here too: transformers precomputes, once
    # per tokenizer, which vocabulary entries could complete each one, and
    # that scan of the vocabulary is otherwise paid inside the first answer.
    with torch.inference_mode():
        model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=2,
            pad_token_id=tokenizer.pad_token_id,
            stop_strings=list(STOP_STRINGS),
            tokenizer=tokenizer,
            use_cache=True,
        )

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    print(" done.")


def assert_fully_on_gpu(model):
    """Fail loudly if any weight landed off the GPU.

    A partial offload is the failure that looks like success: generation still
    works, the GPU sits near idle, and every token waits on a PCIe round trip
    to system RAM or -- worse -- on a disk read. Better to stop here and say
    so than to run twenty times slower without explaining why.
    """

    device_map = getattr(model, "hf_device_map", None)

    if device_map:
        stranded = {
            name: where for name, where in device_map.items()
            if where in ("cpu", "disk") or where is None
        }

        if stranded:
            raise RuntimeError(
                "{} module(s) were offloaded off the GPU (e.g. {}). "
                "Generation would run at a token or two per second. Free VRAM "
                "and retry, or lower the precision.".format(
                    len(stranded), sorted(stranded)[0]
                )
            )

    off_gpu = {
        param.device.type for param in model.parameters()
    } - {"cuda", "meta"}

    if off_gpu:
        raise RuntimeError(
            "Some parameters are on {} rather than the GPU.".format(
                ", ".join(sorted(off_gpu))
            )
        )

    print("All weights are resident on the GPU.")


# ============================================================
# PROMPT BUILDING
# ============================================================

def system_prompt(notes=()):
    """SYSTEM_PROMPT plus today's date and what is known about the user.

    The date is stated on every turn, not only on the ones that searched the
    web: without it the model reasons from its training year and gets ages,
    durations and "how long ago" wrong.

    The notes go here rather than into the user turn. A preference applies to
    every question, including the many it does not resemble, and text at the
    head of the prompt stays in the KV cache instead of being read again on
    each turn. Its wording must therefore stay stable from turn to turn.
    """

    parts = [
        SYSTEM_PROMPT,
        "",
        "Today's date is {}.".format(datetime.date.today().strftime("%d %B %Y")),
    ]

    if notes:
        parts.append("")
        parts.append(
            "What you know about the user -- follow their stated preferences, "
            "and use the facts only where they are relevant:"
        )
        parts.extend("- {}".format(note) for note in notes)

    return "\n".join(parts)


_BOS_CACHE = {}


def _wants_bos(tokenizer):
    """Whether this tokenizer's own encoding starts with BOS.

    Asked of what the tokenizer does rather than of its add_bos_token
    attribute, which transformers 5 reports as False for this model while
    encoding still prepends <s>.
    """

    key = id(tokenizer)

    if key not in _BOS_CACHE:
        bos = tokenizer.bos_token_id
        probe = tokenizer("a").input_ids if bos is not None else []
        _BOS_CACHE[key] = bool(probe) and probe[0] == bos

    return _BOS_CACHE[key]


def encode(tokenizer, text):
    """Token ids for rendered chat text, with the BOS the template leaves out.

    Zephyr's template never writes <s>, and apply_chat_template(tokenize=True)
    encodes without special tokens, so every prompt went in without the BOS
    the model saw at the head of every training sequence. Mistral-family
    models use that first position as an attention sink; without it answers
    wander more and end less cleanly. A template that writes its own BOS
    (Llama 3, Mistral Instruct) is left alone, so a model swap does not
    double it.

    Encoding the rendered text here, rather than asking apply_chat_template
    to, also sidesteps its return type: transformers 5 returns a BatchEncoding
    where 4.x returned a tensor, and ids.shape then raised AttributeError on
    every turn.
    """

    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids

    if _wants_bos(tokenizer) and (
        ids.shape[-1] == 0 or ids[0, 0].item() != tokenizer.bos_token_id
    ):
        bos = torch.full((1, 1), tokenizer.bos_token_id, dtype=ids.dtype)
        ids = torch.cat([bos, ids], dim=-1)

    return ids


def render(tokenizer, conversation, continuing=False):
    """The chat template as text, ending where the model is to write.

    With continuing=True the last message is a half-finished assistant turn
    that generation is about to resume inside, so the text has to end in the
    middle of that turn rather than after it. Rendering the earlier turns with
    add_generation_prompt gives the text up to where the assistant starts
    speaking; the partial reply is appended raw. The model then sees exactly
    what it had already written, with no closing marker in between, and
    carries on from there.
    """

    if continuing:
        prefix = tokenizer.apply_chat_template(
            conversation[:-1],
            add_generation_prompt=True,
            tokenize=False,
        )

        return prefix + conversation[-1]["content"]

    return tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )


def build_prompt_ids(tokenizer, conversation, continuing=False):
    """Render the chat template, dropping oldest turns past the token budget.

    The system message is always kept. Turns are dropped in user/assistant
    pairs so the alternation the template expects stays intact.
    """

    system = conversation[:1]
    history = list(conversation[1:])

    while True:
        ids = encode(tokenizer, render(tokenizer, system + history, continuing))

        if ids.shape[-1] <= MAX_PROMPT_TOKENS or len(history) <= 1:
            return ids, system + history

        # Drop the oldest exchange.
        del history[:2]


# ============================================================
# SAMPLING
# ============================================================

# Questions with one right answer: code, maths, definitions, conversions.
PRECISE_PATTERN = re.compile(
    r"```|(?<!\w)(?:c\+\+|c#)(?!\w)|\b(?:"
    r"code|coding|functions?|class(?:es)?|methods?|scripts?|programs?|"
    r"programming|compile|syntax|bugs?|errors?|exceptions?|traceback|debug|"
    r"regex|sql|quer(?:y|ies)|apis?|json|yaml|xml|html|css|javascript|"
    r"typescript|python|java|rust|golang|kotlin|swift|php|ruby|bash|shell|"
    r"powershell|linux|docker|git|algorithms?|calculate|compute|solve|"
    r"equations?|maths?|formulas?|convert|how many|how much|what year|"
    r"when did|when was|define|definition|difference between|facts?"
    r")\b",
    re.I,
)

# Requests where there is no right answer, only better and worse ones.
CREATIVE_PATTERN = re.compile(
    r"\b(?:story|stories|poem|poetry|lyrics|song|haiku|limerick|jokes?|"
    r"fiction|novel|screenplay|dialogue|role-?play|brainstorm|slogan|tagline|"
    r"creative|imagine|invent|name ideas|ideas for)\b",
    re.I,
)


def pick_sampling(message, grounded=False):
    """Name of the SAMPLING profile for this question.

    A grounded question is answered from sources it should copy faithfully,
    so it is always precise. Otherwise code and fact win over story: "a
    Python script that tells a joke" is code.
    """

    if grounded or PRECISE_PATTERN.search(message):
        return "precise"

    if CREATIVE_PATTERN.search(message):
        return "creative"

    return "balanced"


# ============================================================
# GENERATE
# ============================================================

def eos_ids(tokenizer):
    """EOS is a single id on some templates and a list on others."""

    eos = tokenizer.eos_token_id

    if eos is None:
        return set()

    return set(eos) if isinstance(eos, (list, tuple)) else {eos}


class GeneratedRepetitionPenalty(LogitsProcessor):
    """repetition_penalty over the answer's own tokens only.

    transformers' built-in penalty counts the prompt as well, so every token
    of the system prompt, the history, the search snippets and the recalled
    memories was marked down. That is the opposite of what grounding needs:
    the names, figures and URLs in the CONTEXT are exactly what the answer
    should copy, and they were the tokens being pushed away. Penalizing only
    what the answer itself has produced keeps the guard against loops.
    """

    def __init__(self, penalty, start):
        self.penalty = penalty
        self.start = start

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.start:]

        if self.penalty == 1.0 or generated.shape[-1] == 0:
            return scores

        picked = torch.gather(scores, 1, generated)
        picked = torch.where(picked < 0, picked * self.penalty, picked / self.penalty)

        return scores.scatter(1, generated, picked)


class CancelOnEvent(StoppingCriteria):
    """Stops generate() at its next token once the event is set."""

    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full(
            (input_ids.shape[0],),
            self.event.is_set(),
            dtype=torch.bool,
            device=input_ids.device,
        )


class StopFilter:
    """Passes streamed text through, holding back what may become a stop string.

    generate() stops on the role markers, but the tokens that spell one are
    streamed before the match completes, so "<|user" reached the screen -- and
    the stored answer -- ahead of the stop. The tail that could still turn
    into a marker is held until it either does, and is dropped, or cannot,
    and is released.
    """

    def __init__(self, stops):
        self.stops = tuple(stops)
        self.longest = max((len(stop) for stop in self.stops), default=0)
        self.pending = ""
        self.hit = False

    def feed(self, text):
        if self.hit:
            return ""

        self.pending += text

        found = [
            index for index in (self.pending.find(stop) for stop in self.stops)
            if index != -1
        ]

        if found:
            self.hit = True
            out, self.pending = self.pending[:min(found)], ""
            return out

        hold = 0

        for size in range(min(len(self.pending), self.longest - 1), 0, -1):
            tail = self.pending[-size:]

            if any(stop.startswith(tail) for stop in self.stops):
                hold = size
                break

        cut = len(self.pending) - hold
        out, self.pending = self.pending[:cut], self.pending[cut:]

        return out

    def flush(self):
        out = "" if self.hit else self.pending
        self.pending = ""

        return out


class ContextStreamer(TextIteratorStreamer):
    """TextIteratorStreamer that never decodes a token out of context.

    The Llama/Mistral decoder strips the leading space of the first token it
    is given. TextStreamer starts its decode over after every newline, so the
    first token of each new line lost its leading space -- and in code that
    token is the indentation. Every indented line came out one space short:
    test.html, produced before this fix, has its two-space steps at one,
    three, five and seven spaces. A fresh streamer on a resumed round had the
    same fault at the seam, where it glued two words together: "return value"
    came out as "returnvalue", in code a different program.

    This one keeps the last few tokens as context instead of starting over,
    and a resumed round primes it with the tokens already shown.
    """

    CONTEXT_TOKENS = 4

    def __init__(self, tokenizer, seed=()):
        super().__init__(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        self.token_cache = list(seed)
        self.print_len = len(self._decode())

    def _decode(self):
        return self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.token_cache.extend(value.tolist())
        text = self._decode()

        # A character spread over several byte tokens decodes as U+FFFD until
        # its last byte arrives.
        if text.endswith("�"):
            return

        if len(text) > self.print_len:
            self.on_finalized_text(text[self.print_len:])

        # Re-base on the last few tokens, so each decode stays a few tokens
        # long however long the answer grows.
        self.token_cache = self.token_cache[-self.CONTEXT_TOKENS:]
        self.print_len = len(self._decode())

    def end(self):
        text = self._decode() if self.token_cache else ""
        printable = text[self.print_len:]

        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

        self.on_finalized_text(printable, stream_end=True)


def stream_once(model, generate_kwargs, streamer, emit):
    """Run one generate() on a worker thread, handing its text to emit.

    Any exception on this side -- Ctrl+C, or an API client that has gone away
    raising from emit -- stops the worker at its next token. Before, it
    carried on to max_new_tokens, which on a small card is minutes, and the
    join below waited for every one of them.
    """

    result = {}
    cancel = threading.Event()

    generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
        [CancelOnEvent(cancel)]
    )

    def run():
        try:
            with torch.inference_mode():
                result["output"] = model.generate(**generate_kwargs)
        except BaseException as error:  # re-raised on the main thread below
            result["error"] = error
            streamer.end()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()

    try:
        for chunk in streamer:
            emit(chunk)
    except BaseException:
        cancel.set()
        raise
    finally:
        worker.join()

    if "error" in result:
        raise result["error"]

    return result["output"]


def common_prefix_length(left, right):
    """How many leading token ids two 1-D tensors share."""

    span = min(left.shape[-1], right.shape[-1])

    if span == 0:
        return 0

    mismatches = (left[:span] != right[:span]).nonzero()

    return span if mismatches.numel() == 0 else int(mismatches[0].item())


def reuse_cache(cache_state, input_ids):
    """Return the stored KV cache, cropped to the part of this prompt it covers.

    Consecutive turns share everything up to the new question, so recomputing
    attention over that shared span is the largest avoidable cost in the loop --
    at a 4096-token prompt it is most of the wait before the first new token.

    The previous rule kept the cache only when the last sequence was an exact
    prefix of the new one. Anything that edits history diverges partway through
    and lost the whole cache: trimming the oldest turns does it, and so does
    rewriting a grounded turn back to the bare question once its sources have
    been read. Cropping at the point of divergence keeps everything before it.
    """

    cached_ids = cache_state.get("ids")
    cache = cache_state.get("cache")

    if cached_ids is None or cache is None:
        return None

    shared = common_prefix_length(cached_ids[0], input_ids[0])

    # At least one input token has to be left uncomputed, or generate() is
    # handed a cache covering the whole prompt and no forward pass to run.
    shared = min(shared, input_ids.shape[-1] - 1)

    if shared < MIN_CACHE_REUSE_TOKENS:
        cache_state.clear()
        return None

    # The cache holds one token fewer than the ids stored with it: the last
    # token generated is never fed back through the model.
    get_length = getattr(cache, "get_seq_length", None)
    cached = int(get_length()) if get_length is not None else cached_ids.shape[-1]

    if shared < cached:
        crop = getattr(cache, "crop", None)

        if crop is None:
            cache_state.clear()
            return None

        try:
            # A negative count removes that many tokens from the end -- the
            # form both transformers 4.x and 5.x accept. The positive "crop
            # to this length" form is deprecated and goes in 5.18.
            crop(shared - cached)
        except Exception:
            # An unfamiliar cache layout, or a sliding-window layer that has
            # already dropped the states it would have to roll back to, is
            # not worth guessing at: a full recompute is slower but always
            # correct.
            cache_state.clear()
            return None

    return cache


def generate_response(tokenizer, model, conversation, cache_state,
                      continuing=False, on_text=None, sampling="balanced"):
    """Stream a reply, reusing the KV cache from the previous turn when possible.

    Each turn's prompt is the previous turn's prompt plus the new text, so the
    attention keys/values for the shared prefix do not need recomputing. If the
    prefix does not match -- history was trimmed, or the chat was cleared --
    the cache is dropped and the prompt is processed from scratch.

    One generate() call is capped at MAX_NEW_TOKENS, but hitting that cap ends
    a call, not an answer. When generation stops there without emitting EOS it
    resumes from the cache it has just built, with the tokens it already
    produced as the input, up to MAX_TOTAL_NEW_TOKENS. Continuing the same
    token stream is what keeps long output whole: nothing is recomputed, and
    the model never sees a seam to restart or repeat itself at.

    on_text, when given, is called with each piece of text as it is decoded:
    the console in the chat loop, a response stream in an API. Raising from
    it cancels the generation. sampling is a SAMPLING profile name or a dict
    of generate() sampling arguments.

    Nothing is printed; the caller reports. Returns a dict:

        text           the reply, with any role marker and what followed it
                       removed
        conversation   the conversation as sent, after any trimming
        truncated      True when the answer budget ran out before the answer
        finish_reason  "stop" or "length"
        tokens, prompt_tokens, rounds, seconds -- for reporting
    """

    input_ids, conversation = build_prompt_ids(tokenizer, conversation, continuing)
    input_ids = input_ids.to(model.device)

    past_key_values = reuse_cache(cache_state, input_ids)

    stop_ids = eos_ids(tokenizer)
    prompt_len = input_ids.shape[-1]
    params = SAMPLING[sampling] if isinstance(sampling, str) else dict(sampling)

    # One processor for the whole answer. Its start stays at the original
    # prompt length, so tokens from earlier rounds still count as the answer's
    # own when a later round resumes.
    processors = LogitsProcessorList(
        [GeneratedRepetitionPenalty(REPETITION_PENALTY, prompt_len)]
    )

    stops = StopFilter(STOP_STRINGS)
    chunks = []

    def emit(chunk):
        text = stops.feed(chunk)

        if text:
            chunks.append(text)

            if on_text is not None:
                on_text(text)

    sequences = input_ids
    attention_mask = torch.ones_like(input_ids)
    produced = 0
    rounds = 0
    truncated = False
    started = time.monotonic()

    try:
        while True:
            remaining = MAX_TOTAL_NEW_TOKENS - produced

            if remaining <= 0:
                truncated = True
                break

            budget = min(MAX_NEW_TOKENS, remaining)

            # Anything that picks up mid-answer -- a resumed round, or the
            # 'continue' command -- has to be decoded in context.
            seed = (
                input_ids[0, -ContextStreamer.CONTEXT_TOKENS:].tolist()
                if (rounds or continuing) else ()
            )
            streamer = ContextStreamer(tokenizer, seed)

            # Only what changes per call is passed here; the fixed settings
            # were written to the generation config at load time.
            generate_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                streamer=streamer,
                max_new_tokens=budget,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=processors,
                stop_strings=list(STOP_STRINGS),
                tokenizer=tokenizer,
                return_dict_in_generate=True,
                **params
            )

            if past_key_values is not None:
                generate_kwargs["past_key_values"] = past_key_values

            output = stream_once(model, generate_kwargs, streamer, emit)

            new_tokens = output.sequences.shape[-1] - input_ids.shape[-1]
            sequences = output.sequences
            produced = sequences.shape[-1] - prompt_len
            rounds += 1

            past_key_values = getattr(output, "past_key_values", None)

            # Finished on its own: EOS, a role marker, or any other stop that
            # came before the cap.
            if (sequences[0, -1].item() in stop_ids or stops.hit
                    or new_tokens < budget):
                break

            # Out of budget, or nothing to resume from.
            if not AUTO_CONTINUE or past_key_values is None:
                truncated = True
                break

            if produced >= MAX_TOTAL_NEW_TOKENS:
                truncated = True
                break

            # Resume: everything generated so far becomes the input, handed
            # back with the cache that already covers it.
            input_ids = sequences
            attention_mask = torch.ones_like(input_ids)

    except BaseException:
        cache_state.clear()
        raise

    # Whatever the filter was still holding turned out not to be a marker.
    tail = stops.flush()

    if tail:
        chunks.append(tail)

        if on_text is not None:
            on_text(tail)

    elapsed = time.monotonic() - started

    # Carry the populated cache into the next turn -- but only while it is
    # still small enough to be reusable. A cache longer than the prompt budget
    # never will be, because the next turn trims history back under that budget
    # and so diverges inside it. Keeping it holds ~128 KB of VRAM per token for
    # nothing, which at 8192 tokens is a gigabyte the next answer cannot have.
    if past_key_values is not None and sequences.shape[-1] <= MAX_PROMPT_TOKENS:
        cache_state["cache"] = past_key_values
        cache_state["ids"] = sequences.detach()
    else:
        cache_state.clear()

    text = "".join(chunks)

    return {
        # A continuation is pasted straight onto the partial reply, so its
        # leading whitespace is load-bearing: strip it and a resumed code
        # block loses the newline and indentation between two statements.
        "text": text.rstrip() if continuing else text.strip(),
        "conversation": conversation,
        "truncated": truncated,
        "finish_reason": "length" if truncated else "stop",
        "tokens": produced,
        "prompt_tokens": prompt_len,
        "rounds": rounds,
        "seconds": elapsed,
    }


# ============================================================
# OUTPUT
# ============================================================

ROLE_LABEL = re.compile(r"^\s*(?:<\|assistant\|>|assistant|ai|zephyr)\s*:\s*", re.I)
FENCE = re.compile(r"^ {0,3}(?:```|~~~)")


def clean_reply(text, finished=True):
    """Tidy a whole reply for storage and display.

    - A role label the model sometimes opens with ("Assistant:") is dropped.
    - Runs of blank lines outside code blocks collapse to one. Inside code
      they are left alone: two blank lines between functions are the style.
    - A finished reply that left a code fence open has it closed, or every
      Markdown renderer shows the rest of the page as code. A truncated one
      is left open, so 'continue' carries on inside the block.
    """

    text = ROLE_LABEL.sub("", text, count=1)

    lines = []
    in_code = False
    blank_run = 0

    for line in text.split("\n"):
        if FENCE.match(line):
            in_code = not in_code
            blank_run = 0
        elif not in_code and not line.strip():
            blank_run += 1

            if blank_run > 1:
                continue
        else:
            blank_run = 0

        lines.append(line)

    text = "\n".join(lines).strip()

    if finished and in_code:
        text += "\n```"

    return text


# ============================================================
# CHAT
# ============================================================

class ConsoleStream:
    """Writes streamed text to stdout, flushing at most every STREAM_FLUSH_SECONDS.

    A console write is synchronous and holds the GIL that the generation thread
    is waiting for, so one flush per token turns the printing into a brake on
    the decoding. Several tokens per flush still reads as continuous.
    """

    def __init__(self, interval=STREAM_FLUSH_SECONDS):
        self.interval = interval
        self.pending = []
        self.last_flush = time.monotonic()

    def __call__(self, text):
        self.pending.append(text)

        now = time.monotonic()

        if now - self.last_flush >= self.interval:
            self.flush()
            self.last_flush = now

    def flush(self):
        if self.pending:
            sys.stdout.write("".join(self.pending))
            del self.pending[:]

        sys.stdout.flush()


def new_conversation(notes=()):
    return [{"role": "system", "content": system_prompt(notes)}]


def print_memory_stats(store):

    stats = store.stats()

    if not stats["ready"]:
        print("Memory: loading its encoder in the background...")
        return

    print("Memory: {} stored ({} exchanges, {} notes, {} rated), encoder {}.".format(
        stats["total"],
        stats["exchanges"],
        stats["notes"],
        stats["rated"],
        stats["encoder"],
    ))
    print("        {}".format(stats["dir"]))

    if stats["error"] is not None:
        print("        last error: {}: {}".format(
            type(stats["error"]).__name__, stats["error"]
        ))


def print_reply_stats(reply):

    if reply["seconds"] > 0 and reply["tokens"] > 0:
        print("\n[{} tokens, {:.1f} tok/s{}]".format(
            reply["tokens"],
            reply["tokens"] / reply["seconds"],
            ", {} rounds".format(reply["rounds"]) if reply["rounds"] > 1 else "",
        ))
    else:
        print()

    if reply["truncated"]:
        print(
            "[stopped at the {}-token answer budget -- type 'continue' for "
            "more, or '/tokens N' to raise it]".format(MAX_TOTAL_NEW_TOKENS)
        )


def print_sources(sources, answer):
    """List the web sources behind an answer, so its [n] can be checked.

    Only the ones the answer cites, when it cites any; otherwise all of them,
    labelled as searched rather than as sources.
    """

    if not sources:
        return

    cited = {int(number) for number in re.findall(r"\[(\d+)\]", answer)}
    shown = [source for source in sources if source["n"] in cited]

    print("Sources:" if shown else "Searched:")

    for source in shown or sources:
        print("  [{}] {} -- {}".format(source["n"], source["title"], source["url"]))


def chat():

    global MAX_TOTAL_NEW_TOKENS
    global RETRIEVAL_ENABLED
    global MEMORY_ENABLED

    # Constructed first: it loads its sentence encoder on a background thread,
    # and the model load below takes long enough to cover that for free.
    store = memory_store.Memory()

    tokenizer = load_tokenizer()
    model = load_model()

    if WARMUP:
        warm_up(tokenizer, model)

    conversation = new_conversation()
    cache_state = {}
    last_truncated = False
    last_question = None
    last_memory = None
    last_sampling = "balanced"
    last_live = False

    # A grounded turn whose answer was cut off keeps its sources until the
    # answer is finished; this is the turn to strip once it is.
    pending_rewrite = None

    print("\n" + "=" * 60)
    print("MODEL READY")
    print("=" * 60)
    print("Answer budget: {} tokens (prompt budget: {}).".format(
        MAX_TOTAL_NEW_TOKENS, MAX_PROMPT_TOKENS
    ))
    print_memory_stats(store)
    print("-" * 60)
    print("Type 'exit' to quit.")
    print("Type 'clear' to clear conversation (memory survives it).")
    print("Type 'continue' to extend a reply that hit the budget.")
    print("Type '/tokens N' to change the answer budget.")
    print("Type '/web' to toggle live web lookup (now: {}).".format(
        "on" if RETRIEVAL_ENABLED else "off"
    ))
    print("Type '/web <question>' to force a lookup for one question.")
    print("Type '/good' or '/bad' to rate the last answer -- this is how the")
    print("     runner learns which of its own replies to lean on again.")
    print("Type '/remember <fact>' to keep something about you for good.")
    print("Type '/forget last|all|<text>' to drop memories.")
    print("Type '/memory' for what has been learned, '/memory off' to stop")
    print("     recalling it (now: {}).".format(
        "on" if MEMORY_ENABLED else "off"
    ))
    print("=" * 60)

    while True:

        try:
            user_message = input("\nYou: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_message:
            continue

        lowered = user_message.lower()

        if lowered in ("exit", "quit"):
            print("Exiting...")
            break

        if lowered == "clear":
            conversation = new_conversation()
            cache_state.clear()
            last_truncated = False
            last_question = None
            last_memory = None
            last_live = False
            pending_rewrite = None

            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            print("Conversation cleared.")
            continue

        if lowered.split()[0] == "/tokens":
            parts = user_message.split()

            if len(parts) != 2 or not parts[1].isdigit() or int(parts[1]) < 1:
                print("Usage: /tokens N   (current: {})".format(MAX_TOTAL_NEW_TOKENS))
                continue

            MAX_TOTAL_NEW_TOKENS = int(parts[1])
            print("Answer budget is now {} tokens.".format(MAX_TOTAL_NEW_TOKENS))
            print(
                "Each context token costs ~128 KB of VRAM, so if generation "
                "starts running out of memory, lower it again or 'clear' first."
            )
            continue

        command = lowered.split()[0]

        if command == "/memory":
            parts = user_message.split(None, 1)
            argument = parts[1].strip().lower() if len(parts) == 2 else None

            if argument is None:
                # Report only. '/memory' reads as a question about the state of
                # things, so it must not be the command that changes it.
                print("Memory is {}.".format("on" if MEMORY_ENABLED else "off"))
            elif argument in ("on", "off"):
                MEMORY_ENABLED = argument == "on"
                print("Memory is now {}.".format(
                    "on" if MEMORY_ENABLED else "off"
                ))
            else:
                print("Usage: /memory [on|off]")
                continue

            print_memory_stats(store)
            continue

        if command in ("/good", "/bad"):
            if last_memory is None:
                if last_live:
                    print(
                        "Answers about the present are not kept in memory -- "
                        "they go stale -- so there is nothing to rate."
                    )
                else:
                    print("Nothing to rate yet -- ask something first.")

                continue

            rated = store.rate(1 if command == "/good" else -1, last_memory)

            if command == "/good":
                print(
                    "Noted -- that answer now ranks higher when a similar "
                    "question comes up (score {}).".format(rated["score"])
                )
            else:
                # Demotion is a hard exclusion rather than a small penalty: the
                # point of a thumbs-down is that the model should stop being
                # shown that answer, not be shown it slightly less often.
                print(
                    "Noted -- that answer is now excluded from recall "
                    "(score {}). Ask again for a fresh attempt.".format(
                        rated["score"]
                    )
                )

            continue

        if command == "/remember":
            parts = user_message.split(None, 1)

            if len(parts) != 2 or not parts[1].strip():
                print("Usage: /remember <something worth keeping>")
                continue

            store.ready(30)
            record = store.note(parts[1].strip())

            if record is None:
                print("Memory is off or unavailable -- nothing stored.")
            else:
                last_memory = record
                print("Stored.")

            continue

        if command == "/forget":
            parts = user_message.split(None, 1)
            selector = parts[1].strip() if len(parts) == 2 else "last"

            store.ready(30)
            removed = store.forget(selector)

            print("Forgot {} memor{}.".format(
                removed, "y" if removed == 1 else "ies"
            ))

            if removed:
                last_memory = None

            continue

        if command.startswith("/") and command not in ("/web",):
            print("Unknown command: {}. Known: /tokens /web /memory /good "
                  "/bad /remember /forget".format(command))
            continue

        force_search = False

        if command == "/web":
            parts = user_message.split(None, 1)

            if len(parts) == 1:
                RETRIEVAL_ENABLED = not RETRIEVAL_ENABLED

                # Switching it back on is also how a user says "the network is
                # back" after repeated failures paused lookups.
                if RETRIEVAL_ENABLED:
                    retrieval.reset_failures()

                print("Live web lookup is now {}.".format(
                    "on" if RETRIEVAL_ENABLED else "off"
                ))
                continue

            # '/web <question>' searches for that one question even when the
            # router would have skipped it, and even while the toggle is off.
            user_message = parts[1].strip()
            lowered = user_message.lower()
            force_search = True

        if lowered == "continue":
            if not last_truncated or conversation[-1]["role"] != "assistant":
                print("Nothing to continue -- the last reply finished on its own.")
                continue

            partial = conversation[-1]["content"]

            print("\nAI: ", end="", flush=True)

            stream = ConsoleStream()

            try:
                try:
                    # Resumes inside the existing assistant turn, so the model
                    # picks up where it stopped instead of restarting the answer.
                    reply = generate_response(
                        tokenizer,
                        model,
                        conversation,
                        cache_state,
                        continuing=True,
                        on_text=stream,
                        sampling=last_sampling,
                    )
                finally:
                    stream.flush()

                conversation = reply["conversation"]
                last_truncated = reply["truncated"]

                conversation[-1]["content"] = clean_reply(
                    partial + reply["text"], finished=not last_truncated
                )

                print_reply_stats(reply)

                if not last_truncated and pending_rewrite is not None:
                    turn, bare = pending_rewrite
                    turn["content"] = bare
                    pending_rewrite = None

                # Only the finished answer is worth remembering, so the
                # extended reply replaces whatever the truncated one stored.
                if (MEMORY_ENABLED and not last_truncated and last_question
                        and not last_live):
                    last_memory = store.remember(
                        last_question, conversation[-1]["content"]
                    )

            except KeyboardInterrupt:
                cache_state.clear()
                print("\n[interrupted]")

            except torch.cuda.OutOfMemoryError:
                cache_state.clear()
                torch.cuda.empty_cache()
                print(
                    "\nOut of GPU memory. Lower the budget with '/tokens N', "
                    "or type 'clear' to reset the conversation."
                )

            except Exception as error:
                cache_state.clear()
                print("\nGeneration error: {}: {}".format(type(error).__name__, error))

            continue

        context = None
        sources = []

        if force_search or (RETRIEVAL_ENABLED
                            and retrieval.needs_live_data(user_message)):

            print("[searching the web...]", end="", flush=True)
            context, sources = retrieval.fetch_context(user_message)
            print(" done." if context else " nothing found.")

        # Web context is evidence about the world; memory is evidence about
        # this user and what has already been said to them. Both wrap the
        # question, memory outermost, so the instructions for using each block
        # sit next to the block they govern.
        grounded = (
            retrieval.ground(user_message, context) if context else user_message
        )

        notes = ()

        if MEMORY_ENABLED:
            # Only ever actually waits on the first question: the encoder is
            # loading in the background while the model loads.
            store.ready(15)

            hits = store.recall(user_message)

            if hits:
                print("[recalled {} memor{}]".format(
                    len(hits), "y" if len(hits) == 1 else "ies"
                ))

            grounded = memory_store.ground(grounded, store.block(hits))
            notes = store.profile()

        # Rebuilt every turn, for the date and for notes captured since. The
        # same text as last turn keeps the whole KV cache; a change costs one
        # recompute of the prompt.
        conversation[0]["content"] = system_prompt(notes)

        # The stored turn carries the retrieved snippets and the recalled
        # memories only while the model is reading them; it is rewritten back
        # to the bare question below, once the answer is in.
        conversation.append({"role": "user", "content": grounded})

        augmented = grounded != user_message
        last_question = user_message
        last_sampling = pick_sampling(user_message, grounded=bool(context))

        # An answer about the present goes stale. Recalled later as "what you
        # told this user", it would put last month's price into today's
        # answer, so such exchanges are never stored.
        last_live = (
            force_search or bool(context) or retrieval.needs_live_data(user_message)
        )

        print("\nAI: ", end="", flush=True)

        stream = ConsoleStream()

        # Only generation is undone on failure. Everything after it works on
        # a turn that is already in the conversation, and popping "the user
        # turn" there would remove the answer instead.
        try:
            try:
                reply = generate_response(
                    tokenizer,
                    model,
                    conversation,
                    cache_state,
                    on_text=stream,
                    sampling=last_sampling,
                )
            finally:
                stream.flush()

        except KeyboardInterrupt:
            cache_state.clear()
            conversation.pop()
            last_truncated = False
            pending_rewrite = None
            print("\n[interrupted]")
            continue

        except torch.cuda.OutOfMemoryError:
            cache_state.clear()
            conversation.pop()
            last_truncated = False
            pending_rewrite = None
            torch.cuda.empty_cache()
            print(
                "\nOut of GPU memory. Lower MAX_PROMPT_TOKENS, or use "
                "'/tokens N' with a smaller N, or type 'clear' to reset."
            )
            continue

        except Exception as error:
            cache_state.clear()
            conversation.pop()
            last_truncated = False
            pending_rewrite = None
            print("\nGeneration error: {}: {}".format(type(error).__name__, error))
            continue

        conversation = reply["conversation"]
        last_truncated = reply["truncated"]

        response = clean_reply(reply["text"], finished=not last_truncated)
        conversation.append({"role": "assistant", "content": response})

        print_reply_stats(reply)
        print_sources(sources, response)

        # Drop the injected blocks now that they have been read. Left in
        # history they would spend a large share of the prompt budget on
        # every later turn re-showing text the model has already used, and
        # push real conversation out of the window that much sooner.
        #
        # They stay while the answer is truncated: 'continue' re-renders
        # this same turn, and the model has to still see its sources. The
        # rewrite then happens when the continued answer finishes.
        #
        # The rewrite makes this turn diverge from what the KV cache holds,
        # but the cache is no longer discarded for it: reuse_cache() crops
        # to the divergence point, so every turn before this one is still
        # reused on the next question.
        pending_rewrite = None

        if augmented and conversation[-2]["role"] == "user":
            if last_truncated:
                pending_rewrite = (conversation[-2], user_message)
            else:
                conversation[-2]["content"] = user_message

        last_memory = None

        if not MEMORY_ENABLED:
            continue

        try:
            # Learn from the turn. A statement the user made about themselves
            # is worth keeping however the reply turned out, so note capture
            # is not gated on the answer finishing. Each note is shown,
            # because it sits in the system prompt from now on and a wrong one
            # should be seen -- and forgotten -- straight away.
            for record in store.capture_notes(user_message):
                print("[noted: {}]".format(record["a"]))

            # The exchange itself is only stored once it is complete: half an
            # answer is not worth being reminded of later. Until then there is
            # nothing to rate, and leaving the previous target in place would
            # point '/good' and '/bad' at an unrelated older memory.
            if response and not last_truncated and not last_live:
                last_memory = store.remember(user_message, response)

        except Exception as error:  # a memory failure must not end the chat
            print("[memory: {}: {}]".format(type(error).__name__, error))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if not download_model():
        sys.exit(1)

    chat()
