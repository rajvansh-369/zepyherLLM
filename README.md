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

- Windows on x64, Python 3.11 (3.9+ works). On other platforms install
  llama.cpp yourself and point `ZYPHER_LLAMA_SERVER` at `llama-server`
- ~6 GB of disk: the model (4.4 GB), llama.cpp with its CUDA runtime (~1.1 GB)
  and the recall encoder (90 MB)
- An NVIDIA GPU with 6 GB of VRAM runs the whole model on the card. With less,
  llama.cpp keeps what fits on the GPU and runs the rest on the CPU; with no
  GPU at all it runs on the CPU, at a few tokens per second.

## Install

```bash
pip install -r requirements.txt
```

That is all of it. The model runs in llama.cpp's `llama-server`, which is not
a Python package: it is downloaded on first start, with the weights. There is
no torch in the default install and no CUDA build of anything to get right.

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
| `ZYPHER_MODEL_PATH` | folder the weights are downloaded to, or the path of one `.gguf` file. llama.cpp and the encoder are installed beside it | `C:\AI\Models\zephyr-7b-beta-abliterated` |
| `ZYPHER_MEMORY_DIR` | the memory store | `.memory` |
| `ZYPHER_EMBED_MODEL` | recall encoder | `sentence-transformers/all-MiniLM-L6-v2` |

The engine settings (context window, GPU layers, KV cache precision, idle
unload, ...) are all optional and listed in `.env.example`; left unset, each is
worked out from the hardware. `projectDetail.md` has the full table.

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

For a ChatGPT-style chat in the browser, run `serve` and then the app in
`web/` -- step by step in `HOW_TO_RUN.md`.

The first run downloads three things, each once:

| What | Size | Where |
|---|---|---|
| the model, `zephyr-7b-beta-abliterated.i1-Q4_K_M.gguf` | 4.4 GB | `ZYPHER_MODEL_PATH` |
| the current llama.cpp release and its CUDA runtime | ~1.1 GB | `llama.cpp` beside the models folder (`ZYPHER_LLAMA_DIR`) |
| the recall encoder | 90 MB | `encoders` beside the model folder (`ZYPHER_EMBED_DIR`) |

Downloads resume if interrupted -- rerun the same command. The model and the
encoder are checksummed against the hash the Hub recorded for them.

To run a different model, set `ZYPHER_GGUF_REPO` and `ZYPHER_GGUF_FILE` to any
GGUF on Hugging Face. Nothing else has to change: the layer count and the chat
template are read from the file.

---

## How it loads

The model runs in a private `llama-server` child process, on a loopback port
nobody else is told, behind a key made up for that run. The Python process
holds no tensors: it plans the load, sends prompts over HTTP and streams the
answer back. If Python dies, however it dies, the server is taken down with it
rather than left holding the GPU.

Three numbers decide whether a small card is fast or unusable, and they are
chosen from the VRAM actually free at start, not from a table:

| Decision | Rule |
|---|---|
| GPU layers | all of them whenever weights + context fit; otherwise llama.cpp places what fits and the CPU runs the rest |
| KV cache precision | full (`f16`) if that still leaves 16k of context, else `q8_0` -- half the bytes per token, indistinguishable in practice |
| Context window | the largest of 4k ... 32k that fits after `ZYPHER_VRAM_RESERVE_GB` and whatever other programs already hold |

The prompt and answer budgets are each half the context window. Windows does
not fail when a card fills up -- the driver spills into system RAM and
generation slows by an order of magnitude -- so the plan stays under the line
instead of finding it.

Measured on a 6 GB RTX 3060 Laptop GPU:

| | |
|---|---|
| Plan | 12288-token context, every layer on the GPU, `q8_0` KV cache |
| VRAM | 5.6 GB with the desktop's share |
| System RAM | ~0.5 GB for `llama-server`, ~0.15 GB for Python |
| Speed | 45-55 tok/s; ~43 tok/s with 5,400 tokens of prompt |
| Follow-up turns | only the new text is evaluated -- the server keeps the rest cached |
| Load | 2-5 s once the file is in the OS cache; a minute or two for the first read off a hard disk |

For comparison, the engine this replaced (torch + bitsandbytes, 4-bit) ran the
same model on the same card at 8-9 tok/s with a 4096-token budget.

**Idle unload.** After 15 minutes without a question the model is unloaded and
the VRAM handed back; the next question waits a few seconds while it loads
again. `ZYPHER_IDLE_UNLOAD_SECONDS=0` keeps it loaded.

**Heat.** A laptop GPU at its thermal limit drops its clock to a fraction and
tokens per second go with it -- here, 50 tok/s fell to 9 once the card sat at
88 C. Software cannot cure that, but it is reported: an answer that ran at
under half the session's best speed on a card at 84 C or more is marked as
throttled, in the console and in the API's `zypher.throttled`.

**The original engine** is still there behind `ZYPHER_BACKEND=transformers`,
for cards with 16 GB or more that want the unquantized weights. It needs
`requirements-transformers.txt` and the 13.5 GB safetensors download.

---

## Answer quality

Most of what makes a 7B answer well is how it is prompted, sampled and
decoded, not the weights. On every turn:

| | What | Why |
|---|---|---|
| BOS | `<s>` is put at the head of the prompt | Zephyr's template leaves it out, but every training sequence began with it; without it answers wander and end less cleanly |
| System prompt | direct answer first, Markdown structure sized to the question, fenced code that names its language, no invented facts, today's date, what is known about you | a 7B follows short concrete rules far better than "be accurate" |
| Sampling | chosen per question: `precise` (code, maths, facts, anything answered from search results), `creative` (stories, poems, brainstorming), `balanced` otherwise; all with `min_p` 0.05 | at 0.7 a 7B misspells API names and drifts off the figures in its sources, and much below 0.8 a story reads flat |
| Repetition penalty | 1.05, over the last 64 tokens | a penalty over the whole prompt marks down the names and numbers in search results the answer is meant to copy; a short window has moved past the prompt after the first few tokens of an answer |
| Stop strings | `<\|user\|>`, `<\|system\|>`, `<\|assistant\|>` | the model sometimes writes a role marker and goes on to invent your next message; it now stops there, and the marker never reaches the screen or the history |
| Streaming | text is sent as the engine decodes it, and a continued reply keeps its leading whitespace | stripped, a resumed code block loses the newline between two statements and words are glued together at the seam |
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
| `GET /health` | load state, engine, GPU, context window, whether a key is required (no auth) |
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
- `min_tokens`: no end of turn before the answer is this many tokens long, for
  when Zephyr stops too soon. Capped at the answer budget; set it higher than
  the model has things to say and it pads with repetition

The response carries a `zypher` object with the cited `sources`, how many
memories were recalled, notes captured, and the `memory_id` of the stored
exchange -- pass that to `/rate`. It also reports `tokens_per_second`, and
`throttled` when a hot GPU cost the answer its speed. When streaming it arrives
on the last chunk. `usage.prompt_tokens_details.cached_tokens` says how much of
the prompt the engine already had and did not evaluate again.

A client system message is appended to the runner's own system prompt, not
swapped in for it. Ending `messages` with a partial assistant message continues
that reply, like `continue` in the console. `max_tokens` is capped at the
answer budget.

A browser app on another origin calling the API directly needs that origin in
`ZYPHER_CORS_ORIGINS` (comma separated); unset, no CORS headers are sent.

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
It runs under onnxruntime rather than torch -- the same vectors, in about
160 MB of RAM instead of 800.
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
shares the prompt with web context, and neither is allowed to crowd the other
out of the prompt budget. Each recalled exchange carries its date, and
its code blocks are left out: squeezed onto one line they are broken syntax,
which a 7B then imitates.

The store is plain files you can read, edit or delete:

```
.memory/memory.jsonl   one JSON record per line
.memory/vectors.npy    the index, rebuilt automatically if it drifts
```

Move it with the `ZYPHER_MEMORY_DIR` environment variable. Delete the directory
to start over.

The encoder's ONNX export is downloaded on first start (~90 MB). If
`onnxruntime` or `tokenizers` is missing, or the encoder has no ONNX export,
`sentence-transformers` is used when it is installed; `ZYPHER_EMBED_RUNTIME`
(`auto`, `onnx`, `torch`) forces one or the other. Both give the same vectors,
so the index on disk is shared. With neither available, memory falls back to
hashed n-grams so the runner still works offline -- but paraphrases stop
matching, so recall is noticeably worse.

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

An answer is one streamed completion, up to the answer budget: half the
context window by default, `/tokens N` or `max_tokens` to change it. What the
prompt leaves of the window is the hard limit either way.

When an answer hits the budget it says so, and `continue` resumes inside the
same assistant turn: the chat template is applied to the turns before it and
the partial reply is appended raw, so the model carries on mid-sentence
instead of opening a new turn. That matters for code. Re-prompting with
"continue" makes a model restate the last paragraph and often re-open a tag it
had already closed; continuing the same token stream leaves no seam to restart
at. The engine still has the whole prompt cached, so resuming costs nothing.

However long the reply being continued has grown, it and its question are
never the turns dropped to make room. Once an answer reaches the end of the
context window itself, it says so and asks for the rest in a new message.

`test.html` is a full portfolio page the runner produced in one answer, under
the original engine.

---

## Configuration

Every setting is in `zypher/config.py`, each with a note on why it has its
value. Per-machine ones come from `.env` (see **Setup**).

| Setting | Default |
|---|---|
| `SYSTEM_PROMPT` | direct answers, no refusals or disclaimers, Markdown formatting rules, complete code |
| `MAX_PROMPT_TOKENS` | half the context window (6144 on a 6 GB card) |
| `MAX_TOTAL_NEW_TOKENS` | half the context window (6144 on a 6 GB card) |
| `SAMPLING` | temperature 0.3 / 0.6 / 0.85 for precise / balanced / creative |
| `REPETITION_PENALTY` / `REPEAT_LAST_N` | 1.05 over the last 64 tokens -- kept mild, higher damages code |
| `STOP_STRINGS` | Zephyr's role markers |
| `MEMORY_ENABLED` / `RETRIEVAL_ENABLED` | `True` / `True` |
| `RECALL_KEEP` / `RECALL_THRESHOLD` | 3 / 0.42 |
| `PROFILE_MAX_NOTES` | 8 notes in the system prompt |
| `MAX_RESULTS` / `TIME_LIMIT` | 4 / last month |
| `THROTTLE_TEMP_C` / `SLOW_ANSWER_RATIO` | 84 / 0.5 -- when a slow answer is reported as GPU throttling |

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
      __init__.py         the Engine interface; picks the engine
      text.py             system prompt, reply clean-up, shared errors
      sampling.py         precise / balanced / creative per question
      llamacpp/           DEFAULT ENGINE -- GGUF on llama.cpp
        __init__.py       history trimming, streaming generation, throttle check
        gguf.py           layers / heads / context read from the file header
        plan.py           GPU detection; context, GPU layers, KV precision
        runtime.py        fetches llama-server and the GGUF
        server.py         runs llama-server as a child process
        client.py         HTTP to it
      hf/                 ORIGINAL ENGINE -- transformers + bitsandbytes
    fetch.py              resumable, checksummed downloads (stdlib only)
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
requirements-transformers.txt   extras for the original engine
projectDetail.md          architecture, flows, full API and settings reference
test.html                 sample long output from the runner
```

Both views call `Assistant.answer()`, so a change to how a turn works (a new
grounding source, a different learning rule) is made in one place.

---

## Troubleshooting

The engine's own output is in `.logs/llama-server.log`, rewritten at every
start. When it fails to come up, the last lines of that file are printed.

**Answers slow down as a chat goes on, then recover after a pause.** The GPU is
thermally throttling -- the console says so under the answer, and
`nvidia-smi --query-gpu=temperature.gpu,clocks.current.graphics --format=csv`
shows a card in the high 80s with its clock at a few hundred MHz. Give the
laptop airflow (raised at the back, vents clear, on mains power); a cooler
card is the only fix.

**Slow from the first answer.** Look at the `Plan:` line at start, or
`gpu_layers` in `/health`. Anything but `all` means part of the model is on
the CPU because something else was holding VRAM when it loaded. Close it and
restart, lower `ZYPHER_CTX`, or pick a smaller quantization with
`ZYPHER_GGUF_FILE`.

**The first load takes minutes.** The 4.4 GB file is being read off the disk
for the first time since a restart. On a hard disk that is a minute or two; an
SSD, or any later start, takes seconds.

**Out of GPU memory mid-chat.** Another program took VRAM after the model was
planned around what was free. Close it; the engine restarts itself on the next
question. If it keeps happening, raise `ZYPHER_VRAM_RESERVE_GB` or set a
smaller `ZYPHER_CTX`.

**"no prebuilt llama.cpp for this platform".** The automatic install covers
Windows x64. Elsewhere, install llama.cpp yourself and set
`ZYPHER_LLAMA_SERVER` to the `llama-server` executable.

**Memory says it is using hashed embeddings.** The message names the reason.
Usually `onnxruntime` or `tokenizers` is not installed (`pip install -r
requirements.txt`), or the encoder could not be downloaded on first start.

**Garbled characters in web results.** The Windows console defaults to cp1252
and cannot encode most of what comes back from a search. stdout is reconfigured
to UTF-8 with replacement, so the odd character is lost rather than the answer.

**With `ZYPHER_BACKEND=transformers`:** slow answers on a machine with a GPU
almost always mean the CPU build of torch got installed -- install torch from
the CUDA wheel index first, as `requirements-transformers.txt` says.
`OSError: Could not load ... libtorchaudio.pyd` means a torchaudio built for an
older torch is still installed; this project does not use it, so
`pip uninstall torchaudio`.
