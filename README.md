# zypherLL

A local chat runner for Zephyr 7B that keeps working on a small GPU, answers
time-bound questions from the live web instead of from 2024, and gets better at
the person using it without the weights ever changing.

Everything runs on your machine. Nothing is sent anywhere except the search
queries, and those only when a question needs them.

```
You: What indentation do I use?
[recalled 2 memories]

AI: Based on what you have told me before, you use four spaces for indentation
in Python code, in line with PEP 8.
```

---

## Requirements

- Python 3.11 (3.9+ works; 3.11 enables the faster weight checksum)
- ~14 GB of disk for the model
- A GPU with 5 GB of VRAM or more. Less than that falls back to CPU, which
  works but answers at a few tokens per second.

## Install

torch on PyPI ships the **CPU** build on Windows, which silently disables the
only path that is fast. Install it from the CUDA index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Match the index to your driver (`cu121`, `cu124`, ...). Check it took:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## Run

```bash
python llm.py
```

The first run downloads `richardyoung/zephyr-7b-beta-abliterated` (~13.5 GB) to
the path set in `MODEL_PATH` at the top of `llm.py`. Change that line to put it
somewhere else.

The download resumes if interrupted -- rerun the same command. Every shard is
checksummed against the hash the Hub recorded for it before the model is
loaded, and a corrupt one is deleted so the next run refetches just that shard.
The result is cached, so startup after the first run is instant rather than
spending two minutes rehashing 13.5 GB.

---

## How it loads

The plan is chosen from the card actually present, because `device_map="auto"`
does not fail on a small GPU -- it quietly spills layers to system RAM and
generation drops to a token or two per second while the GPU sits idle. That
failure looks like success, so the loader refuses it and says so instead.

| VRAM | Plan | Notes |
|---|---|---|
| 16 GB or more | bf16 on GPU | full precision |
| 5 GB to 16 GB | 4-bit NF4 on GPU | ~3.8 GB resident, double-quantized |
| under 5 GB, or no GPU | bf16 on CPU | a few tokens per second |

Under 8 GB the prompt and answer budgets are halved automatically (8192 to
4096 tokens each). The KV cache costs roughly 128 KB per token on this model,
so on a small card the cache -- not the weights -- is what runs out first.

Measured on a 6 GB card: 3.84 GB resident, 8-9 tok/s, ~55 s to load the shards.

---

## Commands

| Command | Does |
|---|---|
| `exit` / `quit` | leave |
| `clear` | reset the conversation. Memory survives it |
| `continue` | extend a reply that stopped at the answer budget |
| `/tokens N` | change the answer budget |
| `/good` / `/bad` | rate the last answer -- this is how it learns |
| `/remember <fact>` | keep something about you permanently |
| `/forget last\|all\|<text>` | drop memories |
| `/memory` | show what has been learned |
| `/memory on\|off` | toggle recall |
| `/web` | toggle live web lookup |
| `/web <question>` | force a lookup for one question |

Anything else beginning with `/` is rejected rather than sent to the model, so
a typo'd `/good` does not silently become a question.

---

## Self-learning

There is no fine-tuning here and no training step. The model learns the way a
colleague does: by remembering what has already been said and being told when
it got something wrong.

**Every finished exchange is stored** in `.memory/`, embedded with a small
sentence encoder that runs on the CPU in a few milliseconds and costs no VRAM.
When a later question resembles one already answered, the closest few memories
are placed in the prompt before the question.

**Statements about you are captured automatically.** "My name is Sneha", "I
always use 4-space indentation", "I prefer tabs" -- these become notes, which
are recalled on a weaker match than a whole exchange and survive `clear`,
restarts and everything else.

**Ratings close the loop.** `/good` makes an answer surface earlier next time.
`/bad` excludes it from recall outright, because the point of a thumbs-down is
that the model should stop being shown that answer, not be shown it slightly
less often. A demotion always lands below zero, so an answer praised once and
criticised later does not drift back to neutral.

Asking the same question again replaces the stored answer rather than adding a
near-duplicate, so the store holds your best answer to each question instead of
every attempt at it.

Recall is capped at 3 memories and ~1200 characters, roughly 300 tokens. It
shares the prompt with web context, and on a 4096-token budget neither is
allowed to crowd the other out.

The store is plain files you can read, edit or delete:

```
.memory/memory.jsonl   one JSON record per line
.memory/vectors.npy    the index, rebuilt automatically if it drifts
```

Move it with the `ZYPHER_MEMORY_DIR` environment variable. Delete the directory
to start over.

If `sentence-transformers` is missing, memory falls back to hashed n-grams so
the runner still works offline on first start -- but paraphrases stop matching,
so recall is noticeably worse.

---

## Live web lookup

The weights are frozen at training time, so anything the model says about the
present is a guess from 2024. Questions containing time words (`today`,
`latest`, `price`, `who won`, ...) or a year at or after 2024 trigger a search,
and four dated snippets are injected with instructions to prefer them over
memory of training data.

The router is string matching, not a model call: it runs before every turn, and
a second forward pass to classify the question would cost more than the search
it is trying to avoid. False negatives are recoverable with `/web <question>`.

With no network, the search backend takes ~17 s to fail. After two failures in
a row lookups pause for the session so they stop appearing in front of answers
the weights could have given immediately; `/web` re-arms them.

Retrieved snippets and recalled memories are stripped out of the history once
the answer is in. Left there, they would re-spend a large part of the prompt
budget on every later turn showing the model text it has already used.

---

## Long output

`MAX_NEW_TOKENS` caps one `generate()` call; `MAX_TOTAL_NEW_TOKENS` caps an
answer. Hitting the first ends a call, not an answer -- generation resumes from
the KV cache it just built, with the tokens it already produced as the input.

That matters for code. Re-prompting with "continue" makes the model restate the
last paragraph and often re-open a tag it had already closed; continuing the
same token stream leaves no seam to restart at. `test.html` is a full portfolio
page the runner produced this way, in one answer.

When an answer does hit the total budget it says so, and `continue` resumes
inside the same assistant turn.

---

## Configuration

All at the top of each file.

| Setting | File | Default |
|---|---|---|
| `MODEL_PATH` | `llm.py` | `C:\AI\Models\zephyr-7b-beta-abliterated` |
| `SYSTEM_PROMPT` | `llm.py` | instructs complete, non-abbreviated code |
| `MAX_PROMPT_TOKENS` | `llm.py` | 8192 (4096 under 8 GB VRAM) |
| `MAX_TOTAL_NEW_TOKENS` | `llm.py` | 8192 (4096 under 8 GB VRAM) |
| `TEMPERATURE` / `TOP_P` | `llm.py` | 0.7 / 0.9 |
| `REPETITION_PENALTY` | `llm.py` | 1.05 -- kept mild, higher damages code |
| `MEMORY_ENABLED` | `llm.py` | `True` |
| `RETRIEVAL_ENABLED` | `llm.py` | `True` |
| `RECALL_KEEP` / `RECALL_THRESHOLD` | `memory.py` | 3 / 0.42 |
| `EMBED_MODEL` | `memory.py` | `all-MiniLM-L6-v2` |
| `MAX_RESULTS` / `TIME_LIMIT` | `retrieval.py` | 4 / last month |

---

## Files

```
llm.py            model download + verification, loading, generation, chat loop
memory.py         persistent semantic memory and the feedback loop
retrieval.py      live web lookup: routing, fetching, grounding
requirements.txt  dependencies, with the reason for each floor
test.html         sample long output from the runner
```

---

## Troubleshooting

**Out of GPU memory mid-chat.** The KV cache grew past what is left. `/tokens N`
with a smaller N, or `clear`. The allocator is already configured with
expandable segments, which prevents the common case of failing while free
memory is still reported.

**A download that hangs near 100%.** The Xet transfer backend has been seen to
stream every byte of a shard and then stall before moving it into place. It is
disabled in `llm.py` before `huggingface_hub` is imported; plain HTTPS with
range resume is the reliable path.

**"modules were offloaded off the GPU".** Something else is holding VRAM. Close
it and retry -- this error exists so you find out now rather than wondering why
generation runs twenty times slower than it should.

**Answers are slow on a machine with a GPU.** `torch.cuda.is_available()` is
almost certainly `False`, which means the CPU build of torch got installed. See
**Install**.

**Garbled characters in web results.** The Windows console defaults to cp1252
and cannot encode most of what comes back from a search. stdout is reconfigured
to UTF-8 with replacement, so the odd character is lost rather than the answer.
#   z e p y h e r L L M  
 