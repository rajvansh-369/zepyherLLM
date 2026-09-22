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

## Setup

Settings that differ per machine live in a `.env` file at the project root,
next to `main.py`. It is git-ignored, so your API key never gets committed.

```bash
copy .env.example .env      # Windows  (cp on macOS / Linux)
```

Then open `.env` and fill in what you need:

| Variable | What | Default |
|---|---|---|
| `ZYPHER_API_KEY` | Clients must send it as `Authorization: Bearer <key>`. Leave it empty to run the API with no key (only safe on localhost) | empty |
| `ZYPHER_HOST` / `ZYPHER_PORT` | where the API listens | `127.0.0.1` / `8000` |
| `ZYPHER_MODEL_PATH` | where the weights are downloaded to | `C:\AI\Models\zephyr-7b-beta-abliterated` |
| `ZYPHER_MEMORY_DIR` | the memory store | `.memory` |
| `ZYPHER_EMBED_MODEL` | recall encoder | `all-MiniLM-L6-v2` |

To make a key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

A variable already set in the shell overrides `.env`.

## Run

```bash
python main.py              # console chat
python main.py serve        # HTTP API -- see below
```

The first run downloads `richardyoung/zephyr-7b-beta-abliterated` (~13.5 GB) to
`ZYPHER_MODEL_PATH`.

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

## Answer quality

Most of what makes a 7B answer well is how it is prompted, sampled and
decoded, not the weights. On every turn:

| | What | Why |
|---|---|---|
| BOS | `<s>` is put at the head of the prompt | Zephyr's template leaves it out, but every training sequence began with it; without it answers wander and end less cleanly |
| System prompt | direct answer first, Markdown structure sized to the question, fenced code that names its language, no invented facts, today's date, what is known about you | a 7B follows short concrete rules far better than "be accurate" |
| Sampling | chosen per question: `precise` (code, maths, facts, anything answered from search results), `creative` (stories, poems, brainstorming), `balanced` otherwise; all with `min_p` 0.05 | at 0.7 a 7B misspells API names and drifts off the figures in its sources, and much below 0.8 a story reads flat |
| Repetition penalty | 1.05, on the answer's own tokens only | the built-in one also marks down every prompt token -- including the names and numbers in search results the answer is meant to copy |
| Stop strings | `<\|user\|>`, `<\|system\|>`, `<\|assistant\|>` | the model sometimes writes a role marker and goes on to invent your next message; it now stops there, and the marker never reaches the screen or the history |
| Streaming | every token decoded in context | the stock streamer restarts after each newline, and this tokenizer drops the first token's leading space -- so every indented line came out one space short, and words were glued together at every auto-continue seam |
| Clean-up | a stray `Assistant:` label is dropped, blank-line runs outside code collapse, a code fence left open is closed | the stored and returned text renders as Markdown |

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

## HTTP API

`python main.py serve` runs the same pipeline as the console (web lookup,
memory recall, notes, learning) behind an OpenAI-compatible HTTP API.

```bash
python main.py serve                              # http://127.0.0.1:8000
python main.py serve --host 0.0.0.0 --port 8080   # set ZYPHER_API_KEY first
```

**Postman:** import both files in `postman/`: the collection and the
`zypherLL local` environment. Select the environment, then set its `apiKey` to
the same value as `ZYPHER_API_KEY` in `.env`. Leave `apiKey` empty if the
server runs without a key. All requests except Health send the key as a bearer
token. The chat requests save the stored exchange's id to `{{memoryId}}`, and
the Rate requests use it.

The server starts answering straight away; `/health` reports `loading` until
the model is up, and chat calls return 503 until then.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="<ZYPHER_API_KEY, or anything if unset>")
reply = client.chat.completions.create(
    model="zephyr-7b",
    messages=[{"role": "user", "content": "Who won the match last night?"}],
    stream=True,
)
```

| Endpoint | Does |
|---|---|
| `GET /health` | load state, device, whether a key is required (no auth) |
| `GET /v1/models` | the one model |
| `GET /v1/settings` | server defaults and budgets |
| `POST /v1/chat/completions` | answer; `stream: true` for server-sent events |
| `GET /v1/memory` | memory stats and the notes in the system prompt |
| `GET /v1/memory/records?kind=note\|exchange` | stored records, newest first |
| `POST /v1/memory/notes` | `{"text": ...}` -- same as `/remember` |
| `POST /v1/memory/{id}/rate` | `{"rating": "good"\|"bad"}` -- same as `/good` / `/bad` |
| `DELETE /v1/memory?selector=last\|all\|<text>` | same as `/forget` |
| `PATCH /v1/settings` | `{"web": bool, "memory": bool, "max_tokens": N}`, the server defaults |

Chat requests take the usual `messages`, `max_tokens`, `temperature`,
`top_p` and `stream`, plus:

- `web`: `"auto"` (default, the router decides), `true` (force a lookup), `false`
- `memory`: recall and learn on this request; defaults to the server setting
- `sampling`: `precise`, `balanced` or `creative`, instead of the per-question pick

The response carries a `zypher` object with the cited `sources`, how many
memories were recalled, notes captured, and the `memory_id` of the stored
exchange -- pass that to `/rate`. When streaming it arrives on the last chunk.

A client system message is appended to the runner's own system prompt, not
swapped in for it. Ending `messages` with a partial assistant message continues
that reply, like `continue` in the console. `max_tokens` is capped at the
answer budget.

Requests are served one at a time -- one model, one GPU. Disconnecting mid-
stream stops the generation at the next token. Set `ZYPHER_API_KEY` to require
`Authorization: Bearer <key>`; without it, anyone who can reach the port can
use the model and read its memory.

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
always use 4-space indentation", "I prefer tabs" -- these become notes, and
survive `clear`, restarts and everything else. The most recent eight (your
name always among them) go into the system prompt of every turn rather than
being recalled by similarity: "I always use 4-space indentation" has to apply
to "write a function that merges two lists", which it barely resembles.

Because a note is shown to the model on every turn, capture is strict. It
matches only at the start of a sentence, ignores code and questions, and skips
sentences about the problem at hand ("I use this function but it throws", "I
am a bit lost"). Each captured note is printed as `[noted: ...]`, so a wrong
one can be dropped with `/forget` straight away.

**Ratings close the loop.** `/good` makes an answer surface earlier next time.
`/bad` excludes it from recall outright, because the point of a thumbs-down is
that the model should stop being shown that answer, not be shown it slightly
less often. A demotion always lands below zero, so an answer praised once and
criticised later does not drift back to neutral.

Asking the same question again replaces the stored answer rather than adding a
near-duplicate, so the store holds your best answer to each question instead of
every attempt at it.

Answers about the present -- anything that searched the web, or looked like it
should have -- are not stored. Recalled later as "what you told this user",
they would put last month's price into today's answer.

Recall is capped at 3 exchanges and ~1200 characters, roughly 300 tokens. It
shares the prompt with web context, and on a 4096-token budget neither is
allowed to crowd the other out. Each recalled exchange carries its date, and
its code blocks are left out: squeezed onto one line they are broken syntax,
which a 7B then imitates.

The store is plain files you can read, edit or delete:

```
.memory/memory.jsonl   one JSON record per line
.memory/vectors.npy    the index, rebuilt automatically if it drifts
```

Move it with the `ZYPHER_MEMORY_DIR` environment variable. Delete the directory
to start over.

If `sentence-transformers` is missing or fails to import, memory falls back to
hashed n-grams so the runner still works offline on first start -- but
paraphrases stop matching, so recall is noticeably worse.

---

## Live web lookup

The weights are frozen at training time, so anything the model says about the
present is a guess from 2024. Questions containing time words (`today`,
`latest`, `price`, `who won`, ...) or a year at or after 2024 trigger a search,
and four dated snippets are injected with instructions to prefer them over
memory of training data, cite them by number, and copy figures exactly. The
sources an answer cites are listed under it, so each `[n]` can be checked.

The router is pattern matching, not a model call: it runs before every turn,
and a second forward pass to classify the question would cost more than the
search it is trying to avoid. Time words are matched as whole words -- as
substrings, "now" fired inside "know" and "score" inside "underscore", and most
ordinary questions went to the web. Requests to produce or change something
("write", "fix", "convert", ...) and messages containing code never search.
False negatives are recoverable with `/web <question>`.

Results are limited to the last month, except for a question about a named
past year; a filtered search that finds nothing is retried without the filter.
Duplicate pages are dropped, and snippets end on a whole sentence.

With no network, the search backend takes ~17 s to fail. After two failures in
a row lookups pause for the session so they stop appearing in front of answers
the weights could have given immediately; `/web` re-arms them. A search that
simply finds nothing is not a failure.

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
page the runner produced this way, in one answer -- before the streaming fix
under **Answer quality**, which is why its indentation runs one space short.

When an answer does hit the total budget it says so, and `continue` resumes
inside the same assistant turn.

---

## Configuration

Every setting is in `zypher/config.py`, each with a note on why it has its
value. Per-machine ones come from `.env` (see **Setup**).

| Setting | Default |
|---|---|
| `SYSTEM_PROMPT` | direct answers, no refusals or disclaimers, Markdown formatting rules, complete code |
| `MAX_PROMPT_TOKENS` | 8192 (4096 under 8 GB VRAM) |
| `MAX_TOTAL_NEW_TOKENS` | 8192 (4096 under 8 GB VRAM) |
| `SAMPLING` | temperature 0.3 / 0.6 / 0.85 for precise / balanced / creative |
| `REPETITION_PENALTY` | 1.05 on the answer only -- kept mild, higher damages code |
| `STOP_STRINGS` | Zephyr's role markers |
| `MEMORY_ENABLED` / `RETRIEVAL_ENABLED` | `True` / `True` |
| `RECALL_KEEP` / `RECALL_THRESHOLD` | 3 / 0.42 |
| `PROFILE_MAX_NOTES` | 8 notes in the system prompt |
| `MAX_RESULTS` / `TIME_LIMIT` | 4 / last month |

---

## Files

The code is split into three layers: Model, View and Presenter (MVP). Each
layer only talks to the one next to it.

```
main.py                   entry point: `chat` (default) or `serve`
.env.example              per-machine settings -- copy to .env
zypher/
  config.py               every setting, and the .env loader
  model/                  MODEL -- what it knows and runs on; no user I/O
    llm/
      device.py           GPU detection, budgets for small cards
      download.py         resumable download, shard checksums
      loader.py           bf16 / 4-bit / CPU load plan, warm-up
      prompt.py           system prompt, chat template, history trimming
      sampling.py         precise / balanced / creative per question
      generation.py       streaming, KV-cache reuse, auto-continue, clean-up
    memory.py             semantic memory, notes, ratings
    retrieval.py          web routing, search, grounding
  presenter/
    assistant.py          PRESENTER -- one turn: ground, generate, learn
  view/                   VIEW -- transport and formatting only
    cli.py                console chat and its commands
    api/
      app.py              FastAPI app and server
      routes.py           endpoints, SSE streaming, auth
      schemas.py          request bodies
postman/                  collection + local environment
requirements.txt          dependencies, with the reason for each floor
test.html                 sample long output from the runner
```

Both views call `Assistant.answer()`, so a change to how a turn works (a new
grounding source, a different learning rule) is made in one place.

---

## Troubleshooting

**Out of GPU memory mid-chat.** The KV cache grew past what is left. `/tokens N`
with a smaller N, or `clear`. The allocator is already configured with
expandable segments, which prevents the common case of failing while free
memory is still reported.

**A download that hangs near 100%.** The Xet transfer backend has been seen to
stream every byte of a shard and then stall before moving it into place. It is
disabled in `zypher/__init__.py` before `huggingface_hub` is imported; plain HTTPS with
range resume is the reliable path.

**"modules were offloaded off the GPU".** Something else is holding VRAM. Close
it and retry -- this error exists so you find out now rather than wondering why
generation runs twenty times slower than it should.

**Answers are slow on a machine with a GPU.** `torch.cuda.is_available()` is
almost certainly `False`, which means the CPU build of torch got installed. See
**Install**. Upgrading anything that depends on torch (`pip install -U ...`)
can quietly replace a CUDA build with the CPU one from PyPI.

**`OSError: Could not load this library: ...libtorchaudio.pyd`.** A torchaudio
built for an older torch is still installed. transformers imports it whenever
it is present, so loading the model fails, and memory falls back to hashed
embeddings. This project does not use it: `pip uninstall torchaudio`, or
reinstall it from the same index and version line as torch.

**Garbled characters in web results.** The Windows console defaults to cp1252
and cannot encode most of what comes back from a search. stdout is reconfigured
to UTF-8 with replacement, so the odd character is lost rather than the answer.
