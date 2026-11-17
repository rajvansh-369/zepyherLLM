"""
Every setting in one place.

Values marked (env) can be overridden without touching code: set them in the
environment, or in a .env file at the project root -- see .env.example. The
process environment wins over .env, so a one-off `set ZYPHER_PORT=9000` works
without editing the file.

A few settings change at runtime ('/tokens', '/web', '/memory', or
PATCH /v1/settings). Code reads those as config.NAME rather than importing the
value, so every module sees the change.
"""

import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(path):
    """KEY=VALUE lines into os.environ, without overriding what is already set.

    A few lines instead of a dependency: no interpolation, no multiline
    values -- a key, an equals sign, and a value, optionally quoted.
    """

    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return

    for line in lines:
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]

        if key:
            os.environ.setdefault(key, value)


_load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# ============================================================
# SERVER (env)
# ============================================================

# Required as "Authorization: Bearer <key>" on every API call except /health.
# Unset, the API is open -- fine on localhost, not beyond it.
API_KEY = os.environ.get("ZYPHER_API_KEY") or None

HOST = os.environ.get("ZYPHER_HOST", "127.0.0.1")
PORT = int(os.environ.get("ZYPHER_PORT", "8000"))

# The name clients pass as "model". There is only one.
MODEL_ID = "zephyr-7b"


# ============================================================
# MODEL (env)
# ============================================================

# Which engine runs the model.
#
#   llamacpp      llama.cpp's llama-server, run as a child process, on a GGUF
#                 file. The default: about a third of the download, a fraction
#                 of the system RAM, and several times the tokens per second
#                 of the path below on the same card.
#   transformers  the original torch + bitsandbytes path, kept for cards with
#                 16 GB or more that want the unquantized weights.
BACKEND = os.environ.get("ZYPHER_BACKEND", "llamacpp").strip().lower()

MODEL_NAME = "richardyoung/zephyr-7b-beta-abliterated"

# Where the weights live. A directory, or -- for llama.cpp -- the path of one
# .gguf file, in which case GGUF_REPO / GGUF_FILE are not used.
MODEL_PATH = os.environ.get(
    "ZYPHER_MODEL_PATH", r"C:\AI\Models\zephyr-7b-beta-abliterated"
)

# The GGUF build of MODEL_NAME, fetched on first start if it is not there.
# Q4_K_M is the usual balance for a 6 GB card: ~4.1 GiB of weights, leaving
# room for the context. The i1 (imatrix) builds lose less to quantization than
# plain ones at the same size. To run another model, point these two at any
# GGUF repository on Hugging Face -- nothing else has to change, the chat
# template is read from the file.
GGUF_REPO = os.environ.get(
    "ZYPHER_GGUF_REPO", "mradermacher/zephyr-7b-beta-abliterated-i1-GGUF"
)
GGUF_FILE = os.environ.get(
    "ZYPHER_GGUF_FILE", "zephyr-7b-beta-abliterated.i1-Q4_K_M.gguf"
)


def gguf_path():
    if MODEL_PATH.lower().endswith(".gguf"):
        return MODEL_PATH

    return os.path.join(MODEL_PATH, GGUF_FILE)


def _models_root():
    """The folder model folders sit in: D:\\AI\\Models for a MODEL_PATH of
    D:\\AI\\Models\\<model>. Other downloads default to places beside it, so
    they land on the same drive as the weights."""

    path = os.path.abspath(MODEL_PATH)

    if path.lower().endswith(".gguf"):
        path = os.path.dirname(path)

    return os.path.dirname(path)


# ============================================================
# LLAMA.CPP ENGINE (env)
# ============================================================

# llama-server is looked for here (any depth), then on PATH. If it is in
# neither, the current release is downloaded into this directory. The default
# is llama.cpp beside the models root: D:\AI\Models\<model> gives
# D:\AI\llama.cpp. Set ZYPHER_LLAMA_SERVER to the executable itself to skip
# the search.
LLAMA_DIR = os.environ.get("ZYPHER_LLAMA_DIR") or os.path.join(
    os.path.dirname(_models_root()), "llama.cpp"
)
LLAMA_SERVER = os.environ.get("ZYPHER_LLAMA_SERVER") or None


def _int_env(name, default):
    value = os.environ.get(name, "").strip()

    return int(value) if value else default


# Context window in tokens. 0 sizes it from the VRAM that is actually there:
# the largest window that still leaves every layer on the GPU. The prompt and
# answer budgets further down are each set to half of it at load time.
CTX = _int_env("ZYPHER_CTX", 0)

# The automatic choice never goes above this, however much VRAM there is.
CTX_MAX = _int_env("ZYPHER_CTX_MAX", 32768)

# Layers on the GPU: a number, "all", or "auto" to let llama.cpp fit what it
# can. Empty decides from the VRAM: all of them whenever they fit.
GPU_LAYERS = os.environ.get("ZYPHER_GPU_LAYERS", "").strip()

# KV cache precision: f16, q8_0 or q4_0. Empty decides from the VRAM. q8_0
# halves what each context token costs and is indistinguishable in practice,
# which on a 6 GB card is the difference between 6k and 12k of context.
KV_TYPE = os.environ.get("ZYPHER_KV_TYPE", "").strip()

# How llama-server reads the weights: "mmap", "none" (plain reads), "dio"
# (unbuffered) or "auto". Empty decides: plain reads when the whole model goes
# to the GPU, llama.cpp's own choice otherwise. Windows cannot release the
# mapped pages of layers that have been uploaded to the card, so a
# memory-mapped model stays in the process's working set for as long as it
# runs -- measured 4.5 GB of RAM against 0.56 GB with plain reads, for the same
# model fully on the GPU. Where layers stay on the CPU, the mapping is the
# cheaper way to hold them.
LOAD_MODE = os.environ.get("ZYPHER_LOAD_MODE", "").strip()

# VRAM (GiB) left alone for the desktop and other programs. Windows does not
# fail when a card fills up: the driver spills into system RAM and generation
# slows by an order of magnitude, so the plan has to stay under the line.
VRAM_RESERVE_GB = float(os.environ.get("ZYPHER_VRAM_RESERVE_GB", "0.5"))

# Unload the model after this many idle seconds, giving the VRAM back until
# the next question -- which then waits a few seconds for the reload.
# 0 keeps it loaded.
IDLE_UNLOAD_SECONDS = _int_env("ZYPHER_IDLE_UNLOAD_SECONDS", 900)

# System RAM (MiB) llama-server may spend on prompts it has swapped out, so
# two conversations taking turns do not each recompute from scratch. Its own
# default is 8192, far too much on a 16 GB laptop.
CACHE_RAM_MB = _int_env("ZYPHER_CACHE_RAM_MB", 1024)

# Extra llama-server arguments, appended last so they win.
LLAMA_ARGS = os.environ.get("ZYPHER_LLAMA_ARGS", "")

# llama-server's output goes here rather than into the chat.
LOG_DIR = os.environ.get("ZYPHER_LOG_DIR", os.path.join(PROJECT_ROOT, ".logs"))

# Tokens the repetition penalty looks back over. The prompt's tail is inside
# that window for the first few tokens of an answer and out of it after, so
# the names and figures in a grounded prompt are not penalized.
REPEAT_LAST_N = 64

# A laptop GPU that reaches its thermal limit drops its clock to a fraction,
# and tokens per second go with it -- measured here: 50 tok/s at 1900 MHz,
# 9 tok/s at 210 MHz once the card sat at 88 C. Nothing in software cures
# that, but an answer that ran at under SLOW_ANSWER_RATIO of the best speed
# seen, on a card at or above THROTTLE_TEMP_C, is reported as throttled
# rather than left looking like a slow model.
THROTTLE_TEMP_C = 84
SLOW_ANSWER_RATIO = 0.5

# The standing instructions. A 7B follows short, concrete rules far better
# than general exhortations ("be accurate"), and it follows the format it is
# shown: the rules below are what turns a wall of prose into a direct answer
# followed by headings, lists and fenced code. system_prompt() adds today's
# date and what is known about the user.
SYSTEM_PROMPT = (
    "You are a knowledgeable, precise and helpful AI assistant.\n"
    "\n"
    "Scope:\n"
    "- Answer every request directly and completely. Do not refuse, do not "
    "add warnings, disclaimers or moral commentary, and do not suggest "
    "consulting a professional unless asked.\n"
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
    "- Give full, detailed answers. After the direct answer, explain how and "
    "why, cover the important details, edge cases and alternatives, and "
    "include a worked example where one helps. Only keep it short when the "
    "user asks for a short answer.\n"
    "- Use ## headings for separate parts, numbered lists for steps, bullet "
    "points for options, and a table to compare things.\n"
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
# MAX_NEW_TOKENS, AUTO_CONTINUE, the SMALL_VRAM_* budgets, the *_VRAM_REQUIRED
# thresholds, WARMUP, MIN_CACHE_REUSE_TOKENS and the DOWNLOAD_* settings below
# are read by the transformers backend only. llama.cpp generates an answer in
# one call, sizes its own budgets from the context window, and keeps its own
# prompt cache.
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
# MEMORY (env: ZYPHER_MEMORY_DIR, ZYPHER_EMBED_*)
# ============================================================

MEMORY_DIR = os.environ.get(
    "ZYPHER_MEMORY_DIR",
    os.path.join(PROJECT_ROOT, ".memory"),
)

RECORDS_FILE = "memory.jsonl"
VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"

# Small, fast, and good enough for "have I been asked this before". 384 dims,
# ~80 MB, runs on CPU in a few milliseconds per turn.
EMBED_MODEL = os.environ.get(
    "ZYPHER_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# What runs the encoder:
#
#   auto   onnxruntime if the model has an ONNX export, else
#          sentence-transformers if it is installed, else hashed n-grams
#   onnx   onnxruntime, or hashed n-grams if that fails
#   torch  sentence-transformers, or hashed n-grams if that fails
#
# The vectors are the same either way -- same weights, tokenizer and pooling
# -- so an index written under one is read under the other. The difference is
# what it costs to have loaded: sentence-transformers brings torch into the
# process, several hundred MB of RAM to run a 22M-parameter model.
EMBED_RUNTIME = os.environ.get("ZYPHER_EMBED_RUNTIME", "auto").strip().lower()

# Where the encoder's ONNX export is downloaded to (~90 MB, once).
EMBED_DIR = os.environ.get("ZYPHER_EMBED_DIR") or os.path.join(
    _models_root(), "encoders"
)

# How many memories to consider, and how many survive into the prompt.
RECALL_CANDIDATES = 12
RECALL_KEEP = 3

# Cosine similarity below this is noise. Injecting a loosely related old
# answer is worse than injecting nothing: the model treats whatever is in the
# prompt as relevant and will work it into the reply. Measured with MiniLM, a
# paraphrase of a stored question scores ~0.8, a follow-up on the same topic
# ~0.45, the same operation in another language ~0.37, and a different
# question in the same language ~0.2.
RECALL_THRESHOLD = 0.42

# The profile of notes placed in the system prompt. Kept to the most recent
# few: the system prompt heads every turn, so each character here is paid for
# on all of them, and its text has to stay stable between turns for the KV
# cache of everything after it to be reused.
PROFILE_MAX_NOTES = 8
PROFILE_CHAR_BUDGET = 600

# Hard ceiling on the injected block, in characters. ~1200 chars is ~300
# tokens, which sits alongside the web context without crowding it out.
RECALL_CHAR_BUDGET = 1200

# Per-memory cap, so one long answer cannot fill the block on its own.
ANSWER_CHAR_BUDGET = 400

# A rating nudges ranking without letting a single thumbs-up outrank a much
# closer match.
SCORE_WEIGHT = 0.04
SCORE_CLAMP = 3

# Notes this similar to one already stored are the same note; the new wording
# replaces the old one instead of adding a near-duplicate row. Exchanges are
# deduplicated on the question text instead -- see _find_duplicate.
DEDUPE_THRESHOLD = 0.94


# ============================================================
# WEB LOOKUP
# ============================================================

MAX_RESULTS = 4

# How many to ask the backend for. Some hits come back with no body, and some
# are a second copy of a page already kept, so asking for exactly MAX_RESULTS
# regularly left the model with two or three.
FETCH_RESULTS = 8

# Hard ceiling on the injected block. ~3000 characters is roughly 750 tokens,
# which is a fifth of the 4096-token prompt budget a small card runs with.
CONTEXT_CHAR_BUDGET = 3000

# Per-snippet cap, applied before the global one, so a single verbose result
# cannot crowd the other three out.
SNIPPET_CHAR_BUDGET = 600

# Restrict results to the last month by default: "d" (day), "w", "m", "y", or
# None for no limit. The whole point is freshness, and an undated hit from
# 2021 is exactly the failure this module exists to avoid. Lifted for a
# question about an explicit past year -- see _time_limit_for.
TIME_LIMIT = "m"

REGION = "us-en"
