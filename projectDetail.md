# zypherLL — Project Detail

A local Zephyr 7B assistant (`richardyoung/zephyr-7b-beta-abliterated`) with
live web lookup and self-learning memory, available as a console chat and as an
OpenAI-compatible HTTP API. Both front ends share one pipeline.

- [1. Overview](#1-overview)
- [2. Project structure](#2-project-structure)
- [3. Architecture (MVP)](#3-architecture-mvp)
- [4. Flows](#4-flows)
- [5. HTTP API reference](#5-http-api-reference)
- [6. Console commands](#6-console-commands)
- [7. Configuration](#7-configuration)
- [8. Data storage](#8-data-storage)

---

## 1. Overview

| Item | Value |
|---|---|
| Language | Python 3.11 (3.9+ works) |
| Model | `richardyoung/zephyr-7b-beta-abliterated`, exposed as model id `zephyr-7b` |
| Inference | llama.cpp (`llama-server` child process) on a GGUF file — Q4_K_M by default. The original `transformers` + `bitsandbytes` engine is kept behind `ZYPHER_BACKEND=transformers` |
| API server | FastAPI + uvicorn |
| Memory encoder | `sentence-transformers/all-MiniLM-L6-v2` on the CPU, run by onnxruntime (no torch in the process). Falls back to `sentence-transformers` if installed, then to hashed n-grams |
| Web search | `ddgs` (`DDGS().text`), region `us-en`, last-month results by default |
| Entry point | `main.py` — `chat` (default) or `serve` |

```bash
python main.py                                # console chat
python main.py serve                          # HTTP API on ZYPHER_HOST:ZYPHER_PORT
python main.py serve --host 0.0.0.0 --port 8080
```

---

## 2. Project structure

```
zypherLL/
├── main.py                        entry point: argparse -> cli.main() or api.app.serve()
├── .env                           per-machine settings (git-ignored)
├── .env.example                   template for it, every setting commented
├── requirements.txt               dependencies (no torch)
├── requirements-transformers.txt  extras for the original torch engine
├── .logs/llama-server.log         the engine's own output (git-ignored)
├── README.md                      user guide
├── projectDetail.md               this file
├── test.html                      sample long output produced by the runner
├── web/                           browser chat app (Vite + React), API client only -- see web/README.md
├── postman/
│   ├── zypherLL.postman_collection.json          every endpoint, ready to send
│   └── zypherLL.local.postman_environment.json   baseUrl, apiKey, memoryId
└── zypher/
    ├── __init__.py                process-wide env (Xet off, CUDA allocator, UTF-8 stdout)
    ├── config.py                  every setting + the .env loader
    ├── model/                     MODEL: LLM, memory, web — no user I/O
    │   ├── llm/
    │   │   ├── __init__.py        the Engine interface and create_engine()
    │   │   ├── text.py            system prompt, reply clean-up, shared errors
    │   │   ├── sampling.py        pick precise / balanced / creative per question
    │   │   ├── llamacpp/          DEFAULT ENGINE: GGUF on llama.cpp
    │   │   │   ├── __init__.py    Engine: load, prompt trimming, streaming generation
    │   │   │   ├── gguf.py        read layers / heads / trained context from the file header
    │   │   │   ├── plan.py        GPU detection; context, GPU layers, KV precision
    │   │   │   ├── runtime.py     find or install llama-server; fetch the GGUF
    │   │   │   ├── server.py      spawn / stop llama-server; tied to this process's life
    │   │   │   └── client.py      HTTP: /health, /apply-template, /tokenize, /completion
    │   │   └── hf/                ORIGINAL ENGINE: transformers + bitsandbytes
    │   │       ├── __init__.py    Engine adapter over the modules below
    │   │       ├── device.py      GPU detection, VRAM, small-card budgets
    │   │       ├── download.py    resumable download, SHA-256 shard verification
    │   │       ├── loader.py      load plan (bf16 / 4-bit / CPU), warm-up, offload check
    │   │       ├── prompt.py      chat template, BOS, history trimming
    │   │       └── generation.py  streaming, KV-cache reuse, auto-continue, stop strings
    │   ├── fetch.py               resumable, SHA-256 checked downloads (stdlib only)
    │   ├── memory.py              semantic memory: exchanges, notes, ratings, recall, forget
    │   └── retrieval.py           live-data router, web search, grounding, source list
    ├── presenter/
    │   ├── __init__.py            exports Assistant, ModelNotReady
    │   └── assistant.py           Assistant: load, answer (one turn), memory operations
    └── view/
        ├── cli.py                 console loop and slash commands
        └── api/
            ├── app.py             FastAPI app factory, background model load, uvicorn
            ├── routes.py          endpoints, bearer auth, SSE streaming
            └── schemas.py         pydantic request bodies
```

### Module responsibilities

| Module | Key functions / classes | Role |
|---|---|---|
| `config.py` | `_load_dotenv`, constants | Loads `.env` without overriding the shell. Runtime-mutable settings are read as `config.NAME` so changes propagate |
| `model/llm/__init__.py` | `create_engine` | Returns the engine named by `ZYPHER_BACKEND`. Imports no inference library until asked |
| `model/llm/text.py` | `system_prompt`, `clean_reply`, `ClientGone`, `OutOfMemory`, `PromptTooLong` | What both engines share: system prompt (date + notes), reply clean-up, errors |
| `model/llm/sampling.py` | `pick_sampling` | Code/maths/facts/grounded → `precise`; stories → `creative`; otherwise `balanced` |
| `model/llm/llamacpp/__init__.py` | `Engine` (`load`, `generate`, `reset`, `close`, `asleep`, `info`) | Trims old turns to `MAX_PROMPT_TOKENS`, streams one `/completion` per answer, restarts a dead server, flags answers slowed by a hot GPU |
| `model/llm/llamacpp/gguf.py` | `read_metadata`, `describe`, `kv_bytes_per_token` | Reads the GGUF header so the memory plan fits any model |
| `model/llm/llamacpp/plan.py` | `detect_gpu`, `gpu_temperature`, `make_plan` | VRAM and temperature via `nvidia-smi`; picks context, GPU layers, KV precision and load mode |
| `model/llm/llamacpp/runtime.py` | `ensure_model`, `ensure_server` | Finds the GGUF and `llama-server`, downloading either the first time |
| `model/fetch.py` | `download`, `download_verified`, `hub_url` | Resumable downloads with the standard library, checked against the SHA-256 the Hub records. Used for the GGUF, the llama.cpp release and the recall encoder |
| `model/llm/llamacpp/server.py` | `LlamaServer` | Child process on a random loopback port with a per-run key; killed with the parent (Windows job object) |
| `model/llm/llamacpp/client.py` | `Client`, `ServerError` | stdlib HTTP; closing a stream cancels the generation |
| `model/llm/hf/*` | `Engine`, `generate_response`, `load_model`, ... | The original torch engine, unchanged apart from sitting behind the Engine interface |
| `model/memory.py` | `Memory` (`remember`, `note`, `capture_notes`, `recall`, `block`, `profile`, `rate`, `forget`, `stats`), `ground` | Persistent JSONL + vector store |
| `model/retrieval.py` | `needs_live_data`, `fetch_context`, `ground`, `reset_failures` | Decides when to search, fetches and numbers snippets, backs off after two failures |
| `presenter/assistant.py` | `Assistant.load`, `.status`, `.answer`, `.reset`, `.close`, `.engine_info`, `.memory_stats`, `.profile`, `.records`, `.note`, `.rate`, `.forget` | The single place where a turn is run |
| `view/cli.py` | `main` | Console I/O and slash commands |
| `view/api/app.py` | `create_app`, `serve` | Builds the app; loads the model on a background thread |
| `view/api/routes.py` | `authorize`, endpoint handlers, `stream`, `chunk` | HTTP ↔ `Assistant` translation |
| `view/api/schemas.py` | `ChatRequest`, `Message`, `NoteRequest`, `RateRequest`, `SettingsRequest` | Validation |

---

## 3. Architecture (MVP)

```
            ┌──────────────────────┐        ┌──────────────────────────────┐
  person ──▶│  view/cli.py         │        │  view/api (FastAPI)          │◀── HTTP client
            │  console + commands  │        │  auth, JSON, SSE             │
            └──────────┬───────────┘        └──────────────┬───────────────┘
                       │  Assistant.answer(messages, ...)  │
                       └───────────────┬───────────────────┘
                                       ▼
                     ┌─────────────────────────────────────┐
                     │ presenter/assistant.py  (Assistant) │
                     │ ground → generate → learn           │
                     │ threading.Lock: one generation/time │
                     └───┬──────────────┬──────────────┬───┘
                         ▼              ▼              ▼
                ┌──────────────┐ ┌─────────────┐ ┌──────────────┐
                │ model/llm    │ │ model/      │ │ model/       │
                │ Engine       │ │ memory.py   │ │ retrieval.py │
                │ (HTTP client)│ │ .memory/    │ │ web search   │
                └──────┬───────┘ └─────────────┘ └──────────────┘
                       │ 127.0.0.1:<random port>, per-run key
                       ▼
                ┌───────────────────────────────┐
                │ llama-server (child process)  │
                │ GGUF weights + KV cache, GPU  │
                └───────────────────────────────┘
```

Rules:

- Views never touch the model layer directly (the API reads
  `Assistant.engine_info()` for `/health` and calls `retrieval.reset_failures()`).
- The Python process holds no tensors. Weights, KV cache and prompt cache all
  live in `llama-server`, which only this process can reach.
- A change to how a turn works is made once, in `Assistant.answer()`.
- Generations are serialised by `Assistant.lock` — one model, one GPU.

---

## 4. Flows

### 4.1 Startup

```
main.py
 ├─ chat  → view/cli.main()      → Assistant.load() (blocking)  → prompt loop
 └─ serve → view/api/app.serve() → uvicorn(create_app())
                                     └─ lifespan: Thread(Assistant.load)
                                        /health = "loading" until ready
```

`Assistant.load()`:

1. `Memory()` — starts loading the sentence encoder on a background thread:
   its ONNX export under onnxruntime (downloaded to `ZYPHER_EMBED_DIR` the
   first time), else `sentence-transformers`, else hashed n-grams.
2. `llm.create_engine()` → `Engine.load()` (llama.cpp engine):
   1. `runtime.ensure_model()` — the GGUF, downloaded with resume and SHA-256
      checked the first time.
   2. `plan.make_plan()` — from `nvidia-smi` and the GGUF header:

      | Decision | Rule |
      |---|---|
      | GPU layers | `all` when weights + context fit, else `auto` (llama.cpp places what fits) |
      | KV cache | `f16` if that still allows 16k context, else `q8_0` (half the bytes per token) |
      | Context | largest of 4k…32k that fits after `ZYPHER_VRAM_RESERVE_GB` and VRAM already in use |
      | Load mode | plain reads (`none`) when every layer is on the GPU, else llama.cpp's own choice. A memory-mapped model fully on the card still sat in the working set on Windows: 4.5 GB of RAM against 0.56 GB |

   3. `runtime.ensure_server()` — find `llama-server`, or download the release.
   4. `LlamaServer.start()` — spawn with `--flash-attn on`, `--parallel 1`,
      `--cache-reuse 256`, `--cache-ram`, `--load-mode`, `--sleep-idle-seconds`;
      wait for `/health`.
   5. `MAX_PROMPT_TOKENS` and `MAX_TOTAL_NEW_TOKENS` are each set to half the
      context window.
3. `ready` is set; on failure `error` is set and `/health` reports `error`.

### 4.2 One turn — `Assistant.answer()`

```
messages ─▶ split system text / history
         ─▶ continuing?  (last message is a partial assistant reply)
         │
         ├─ 1. WEB      web=True, or web="auto" + RETRIEVAL_ENABLED + needs_live_data(q)
         │              → status {"stage":"searching"}
         │              → retrieval.fetch_context(q) → numbered snippets + sources
         │              → retrieval.ground(q, context)
         │
         ├─ 2. MEMORY   (if memory on) store.recall(q) → ≤3 hits, ≤1200 chars
         │              → memory.ground(grounded, block)
         │              → status {"stage":"recalled","count":n}
         │              → store.profile() → ≤8 notes for the system prompt
         │
         ├─ 3. PROMPT   system = SYSTEM_PROMPT + date + notes (+ client system text)
         │              question replaced by its grounded version (this turn only)
         │
         ├─ 4. GENERATE sampling profile (request > pick_sampling), temp/top_p overrides
         │              budget = min(max_tokens, MAX_TOTAL_NEW_TOKENS)
         │              with lock: engine.generate(...)
         │                • /apply-template + /tokenize; drop oldest turns past the budget
         │                • one streamed /completion; the server reuses its prompt
         │                  cache for the shared prefix
         │                • stream text → on_event("text", piece)
         │                • stop at EOS, a role marker, or the budget
         │              llm.clean_reply(...)
         │
         └─ 5. LEARN    capture_notes(q)            → notes_captured
                        if finished and not live:   store.remember(q, answer) → memory_id
                        cited sources = those whose [n] appears in the answer
```

Returned dict:

```json
{
  "text": "...",
  "finish_reason": "stop | length",
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0}},
  "meta": {
    "sampling": "precise | balanced | creative",
    "live": false,
    "sources": [{"n": 1, "title": "...", "url": "..."}],
    "cited": false,
    "recalled": 0,
    "notes_captured": [],
    "memory_id": 12,
    "seconds": 3.214,
    "tokens_per_second": 45.0,
    "throttled": null,
    "rounds": 1,
    "history_kept": 4
  }
}
```

`history_kept` is used by the console only and is removed from API responses.

`throttled` is `null`, or `{"gpu_temp_c": 88, "usual_tokens_per_second": 50.1}`
when the answer ran at under `SLOW_ANSWER_RATIO` (0.5) of the best speed this
session has seen while the GPU was at `THROTTLE_TEMP_C` (84) or above. It is
judged on speed first and only then confirmed with the thermometer, so an
ordinary answer costs no `nvidia-smi` call. The console prints a notice under
such an answer.

If the idle timer had unloaded the model, a `{"stage": "waking"}` status is
sent before generation while it loads back in.

### 4.3 Continue

When the last message is an assistant message, the answer is continued in place.
If it is the answer this process just produced, the same grounded question,
web context and sources are reused, so the prompt bytes match and the whole
prompt cache is kept. A truncated answer (`finish_reason: "length"`) is not stored in
memory.

### 4.4 Streaming (API)

```
client ──POST stream:true──▶ routes.stream()
                              ├─ worker thread: assistant.answer(on_event=...)
                              │     on_event → loop.call_soon_threadsafe(queue.put)
                              └─ async generator reads queue:
                                   first   → delta {"role":"assistant","content":""}
                                   status  → delta {}, "zypher": {...}
                                   text    → delta {"content": piece}
                                   done    → finish_reason, usage, "zypher": meta
                                   error   → {"error":{"message":...}}
                                   then    → data: [DONE]
client disconnects → gone.set() → next on_event raises ClientGone → generation stops
```

### 4.5 Learning

| Event | Effect |
|---|---|
| Finished, non-live exchange | Stored as `kind: "exchange"`; same question again replaces the old answer |
| Statement about the user ("My name is…", "I always use…") | Captured as `kind: "note"`, shown in every system prompt |
| `/good` or `POST /v1/memory/{id}/rate {"rating":"good"}` | `score += 1` → ranks higher in recall |
| `/bad` or `rating: "bad"` | `score` set ≤ −1 → excluded from recall |
| Live / web-grounded answer | Never stored (would go stale) |

---

## 5. HTTP API reference

Base URL: `http://{ZYPHER_HOST}:{ZYPHER_PORT}` (default `http://127.0.0.1:8000`).

### 5.1 Authentication

If `ZYPHER_API_KEY` is set, every endpoint except `GET /health` requires:

```
Authorization: Bearer <ZYPHER_API_KEY>
```

The key is compared in constant time. Missing or wrong key → `401 {"detail": "invalid or missing API key"}`.
If the key is unset the API is open; `serve` prints a warning when bound to a
non-localhost address without a key.

### 5.2 Endpoint summary

| # | Method | Path | Auth | Purpose |
|---|---|---|---|---|
| 1 | GET | `/health` | no | Load state, engine, device, VRAM, context, whether auth is on |
| 2 | GET | `/v1/models` | yes | The single model |
| 3 | GET | `/v1/settings` | yes | Server-wide defaults |
| 4 | PATCH | `/v1/settings` | yes | Change defaults |
| 5 | POST | `/v1/chat/completions` | yes | Answer (JSON or SSE) |
| 6 | GET | `/v1/memory` | yes | Memory stats and profile notes |
| 7 | GET | `/v1/memory/records` | yes | Stored records, newest first |
| 8 | POST | `/v1/memory/notes` | yes | Add a note (`/remember`) |
| 9 | POST | `/v1/memory/{record_id}/rate` | yes | Rate a record (`/good` / `/bad`) |
| 10 | DELETE | `/v1/memory` | yes | Forget records (`/forget`) |

FastAPI also serves interactive docs at `/docs` and the schema at `/openapi.json`.

---

### 5.3 `GET /health`

No auth. Available immediately after the server starts.

**Response 200**

```json
{
  "status": "loading | ready | error",
  "error": null,
  "backend": "llama.cpp",
  "device": "cuda",
  "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU",
  "vram_gb": 6.0,
  "context": 12288,
  "gpu_layers": "all",
  "kv_cache": "q8_0",
  "auth": true
}
```

| Field | Meaning |
|---|---|
| `status` | `loading` until the model is up, `error` if loading failed |
| `error` | Load error text, else `null` |
| `backend` | `llama.cpp` or `transformers` |
| `device` | `cuda` or `cpu`; `null` until the hardware has been looked at |
| `gpu` / `vram_gb` | Card name and its total VRAM (GiB, 2 dp) |
| `context` | Context window in tokens; prompt and answer budgets are each half |
| `gpu_layers` | `all`, a number, or `auto` — anything but `all` means part of the model is on the CPU |
| `kv_cache` | KV cache precision: `f16`, `q8_0`, ... |
| `auth` | `true` when `ZYPHER_API_KEY` is set |

---

### 5.4 `GET /v1/models`

**Response 200**

```json
{
  "object": "list",
  "data": [{
    "id": "zephyr-7b",
    "object": "model",
    "created": 0,
    "owned_by": "local",
    "root": "richardyoung/zephyr-7b-beta-abliterated"
  }]
}
```

---

### 5.5 `GET /v1/settings`

**Response 200**

```json
{
  "web": true,
  "memory": true,
  "max_tokens": 6144,
  "max_prompt_tokens": 6144
}
```

Each is half the context window the engine chose at load — 6144 for the
12288-token window a 6 GB card gets.

### 5.6 `PATCH /v1/settings`

Changes server-wide defaults (what a request gets when it does not specify).
In-memory only — reset on restart. All fields optional.

**Request**

| Field | Type | Effect |
|---|---|---|
| `web` | bool | `RETRIEVAL_ENABLED`. Setting `true` also clears the search-failure back-off |
| `memory` | bool | `MEMORY_ENABLED` |
| `max_tokens` | int ≥ 1 | `MAX_TOTAL_NEW_TOKENS` (answer budget ceiling) |

```json
{"web": true, "memory": true, "max_tokens": 2048}
```

**Response 200** — the new settings, same shape as `GET /v1/settings`.
**422** — validation error.

---

### 5.7 `POST /v1/chat/completions`

OpenAI-compatible. Unknown OpenAI fields (`n`, `stop`, `seed`, …) are accepted
and ignored.

**Request**

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | string | `zephyr-7b` | Ignored; there is one model |
| `messages` | array, min 1 | — | `{role: system\|user\|assistant, content: string \| [{type:"text", text}]}`. Multi-part content is flattened to its text parts |
| `stream` | bool | `false` | `true` → server-sent events |
| `max_tokens` | int ≥ 1 | server setting | Capped at `MAX_TOTAL_NEW_TOKENS` |
| `max_completion_tokens` | int ≥ 1 | — | Takes precedence over `max_tokens` |
| `temperature` | float 0–2 | from profile | `0` → greedy decoding |
| `top_p` | float (0, 1] | from profile | |
| `web` | `"auto"` \| bool | `"auto"` | `"auto"`: router decides; `true`: force search; `false`: never |
| `memory` | bool | server setting | Recall and learn on this request |
| `sampling` | `precise` \| `balanced` \| `creative` | per question | Overrides automatic choice |
| `min_tokens` | int ≥ 1 | — | Blocks end-of-turn until this many tokens; capped at the budget |

Message semantics:

- Client `system` messages are **appended** to the runner's own system prompt.
- The last `user` message is the question.
- If `messages` ends with an `assistant` message, that reply is continued.

**Example**

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $ZYPHER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Who won the match last night?"}],"web":"auto"}'
```

**Response 200 (`stream: false`)**

```json
{
  "id": "chatcmpl-3f2a...",
  "object": "chat.completion",
  "created": 1790000000,
  "model": "zephyr-7b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 812, "completion_tokens": 240, "total_tokens": 1052,
            "prompt_tokens_details": {"cached_tokens": 390}},
  "zypher": {
    "sampling": "precise",
    "live": true,
    "sources": [{"n": 1, "title": "...", "url": "https://..."}],
    "cited": true,
    "recalled": 0,
    "notes_captured": [],
    "memory_id": null,
    "seconds": 5.4,
    "tokens_per_second": 47.2,
    "throttled": null,
    "rounds": 1
  }
}
```

`usage.prompt_tokens_details.cached_tokens` is how much of the prompt the
engine already held from the previous request and did not evaluate again.

| `zypher` field | Meaning |
|---|---|
| `sampling` | Profile used |
| `live` | Answer is about the present (searched or should have) — not stored |
| `sources` | Cited sources if any `[n]` appears in the answer, else all fetched |
| `cited` | Whether the answer cites any source |
| `recalled` | Number of past exchanges placed in the prompt |
| `notes_captured` | Notes captured from this question |
| `memory_id` | Id of the stored exchange (use with `/rate`), or `null` |
| `seconds` | Generation time |
| `tokens_per_second` | Generation speed of this answer |
| `throttled` | `null`, or `{gpu_temp_c, usual_tokens_per_second}` when a hot GPU cost this answer its speed (see 4.2) |
| `rounds` | Completion calls used: 1, or 2 when `min_tokens` forces a first round |

`finish_reason: "length"` means the budget was hit; resend with the partial
answer as the last assistant message to continue.

**Response (`stream: true`)** — `text/event-stream`:

```
data: {"id":"chatcmpl-..","object":"chat.completion.chunk","created":..,"model":"zephyr-7b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {..."choices":[{"index":0,"delta":{},"finish_reason":null}],"zypher":{"stage":"searching"}}

data: {..."choices":[{"index":0,"delta":{},"finish_reason":null}],"zypher":{"stage":"recalled","count":2}}

data: {..."choices":[{"index":0,"delta":{"content":"The "},"finish_reason":null}]}

data: {..."choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{...},"zypher":{...}}

data: [DONE]
```

A failure mid-stream is sent as `data: {"error": {"message": "..."}}` followed by `data: [DONE]`.
Disconnecting stops generation at the next token.

**Errors**

| Status | When |
|---|---|
| 400 | No user message, or last user message empty |
| 401 | Bad or missing API key |
| 422 | Body fails validation |
| 500 | Model failed to load (`"model failed to load: ..."`) |
| 503 | Model still loading, or `"out of GPU memory -- send less history or lower max_tokens"` |

---

### 5.8 `GET /v1/memory`

**Response 200**

```json
{
  "stats": {
    "ready": true,
    "enabled": true,
    "total": 42,
    "notes": 5,
    "exchanges": 37,
    "rated": 6,
    "encoder": "sentence-transformers/all-MiniLM-L6-v2",
    "dir": "D:\\GIT\\zypherLL\\.memory",
    "error": null
  },
  "profile": ["My name is Sneha", "I always use 4-space indentation"]
}
```

`stats` is `null` if memory is not initialised. `profile` is the list of notes
currently placed in the system prompt (≤ 8, name first). **503** if memory is
not initialised yet.

### 5.9 `GET /v1/memory/records`

**Query**

| Param | Type | Default | Notes |
|---|---|---|---|
| `kind` | `note` \| `exchange` | all | Filter |
| `limit` | int | 50 | Page size |
| `offset` | int | 0 | Start index |

**Response 200**

```json
{
  "total": 37,
  "records": [
    {"id": 42, "ts": "2026-09-23T10:15:02+05:30", "kind": "exchange",
     "q": "How do I sort a dict by value?", "a": "...", "score": 1, "uses": 3}
  ]
}
```

Record fields: `id`, `ts` (ISO-8601 with offset), `kind`, `q` (question; notes
use `"About the user"`), `a` (answer or note text), `score` (rating), `uses`.

### 5.10 `POST /v1/memory/notes`

Same as `/remember <fact>`.

**Request** — `{"text": "I prefer tabs over spaces"}` (1–200 chars)

**Response 200** — the stored record (`kind: "note"`). A near-duplicate note
replaces the existing one.
**503** — memory off or unavailable. **422** — text empty or too long.

### 5.11 `POST /v1/memory/{record_id}/rate`

Same as `/good` / `/bad`.

**Request** — `{"rating": "good"}` or `{"rating": "bad"}`

**Response 200** — the updated record. `good` adds 1 to `score`; `bad` sets
`score` to at most −1, which removes it from recall.
**404** — `"no memory with id N"`. **503** — memory not ready.

### 5.12 `DELETE /v1/memory?selector=...`

Same as `/forget`.

| `selector` | Removes |
|---|---|
| `last` (default) | The most recently stored record |
| `all` | Everything |
| any other text | Every record with similarity ≥ `RECALL_THRESHOLD` (0.42) to the text |

**Response 200** — `{"removed": 3}`

### 5.13 Postman

Import both files in `postman/`. Select the **zypherLL local** environment and
set `apiKey` to the value of `ZYPHER_API_KEY` (empty if the server runs without
a key). Chat requests save `zypher.memory_id` into `{{memoryId}}`; the Rate
requests use it.

### 5.14 OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="<ZYPHER_API_KEY>")
reply = client.chat.completions.create(
    model="zephyr-7b",
    messages=[{"role": "user", "content": "Explain Python decorators"}],
    stream=True,
    extra_body={"web": False, "sampling": "precise"},
)
for chunk in reply:
    print(chunk.choices[0].delta.content or "", end="")
```

---

## 6. Console commands

| Command | API equivalent | Does |
|---|---|---|
| `exit` / `quit` | — | Leave |
| `clear` | — (send fresh `messages`) | Reset the conversation; memory survives |
| `continue` | End `messages` with the partial assistant reply | Extend a truncated answer |
| `/tokens N` | `PATCH /v1/settings {"max_tokens": N}` | Answer budget |
| `/good` / `/bad` | `POST /v1/memory/{id}/rate` | Rate last answer |
| `/remember <fact>` | `POST /v1/memory/notes` | Store a note |
| `/forget last\|all\|<text>` | `DELETE /v1/memory?selector=` | Drop memories |
| `/memory` | `GET /v1/memory` | Show what has been learned |
| `/memory on\|off` | `PATCH /v1/settings {"memory": bool}` | Toggle recall |
| `/web` | `PATCH /v1/settings {"web": bool}` | Toggle web lookup |
| `/web <question>` | `"web": true` on a chat request | Force a lookup |

Unknown `/commands` are rejected rather than sent to the model.

---

## 7. Configuration

### 7.1 Environment (`.env` or shell; shell wins)

| Variable | Default | Purpose |
|---|---|---|
| `ZYPHER_API_KEY` | empty (no auth) | Bearer key for the API |
| `ZYPHER_HOST` | `127.0.0.1` | Bind address |
| `ZYPHER_PORT` | `8000` | Port |
| `ZYPHER_CORS_ORIGINS` | empty (no CORS) | Comma-separated browser origins allowed to call the API directly |
| `ZYPHER_MODEL_PATH` | `C:\AI\Models\zephyr-7b-beta-abliterated` | Weights directory, or the path of one `.gguf` file |
| `ZYPHER_GGUF_REPO` | `mradermacher/zephyr-7b-beta-abliterated-i1-GGUF` | Hugging Face repository the GGUF is fetched from |
| `ZYPHER_GGUF_FILE` | `zephyr-7b-beta-abliterated.i1-Q4_K_M.gguf` | Which file in it |
| `ZYPHER_BACKEND` | `llamacpp` | `llamacpp` or `transformers` |
| `ZYPHER_LLAMA_DIR` | `llama.cpp` beside the models folder | Where llama.cpp is found or installed |
| `ZYPHER_LLAMA_SERVER` | unset | Path of a `llama-server` executable to use as is |
| `ZYPHER_CTX` | `0` (from VRAM) | Context window in tokens |
| `ZYPHER_CTX_MAX` | `32768` | Ceiling for the automatic choice |
| `ZYPHER_GPU_LAYERS` | unset (from VRAM) | A number, `all` or `auto` |
| `ZYPHER_KV_TYPE` | unset (from VRAM) | `f16`, `q8_0`, `q4_0` |
| `ZYPHER_LOAD_MODE` | unset (`none` when all layers are on the GPU) | How the weights are read: `mmap`, `none`, `dio`, `auto` |
| `ZYPHER_VRAM_RESERVE_GB` | `0.5` | VRAM left for other programs |
| `ZYPHER_IDLE_UNLOAD_SECONDS` | `900` | Unload the model when idle; `0` never |
| `ZYPHER_CACHE_RAM_MB` | `1024` | System RAM for swapped-out prompt caches |
| `ZYPHER_LLAMA_ARGS` | empty | Extra `llama-server` arguments |
| `ZYPHER_LOG_DIR` | `<project>/.logs` | `llama-server.log` |
| `ZYPHER_MEMORY_DIR` | `<project>/.memory` | Memory store |
| `ZYPHER_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Recall encoder: a Hugging Face repository, or a local folder laid out the same way |
| `ZYPHER_EMBED_RUNTIME` | `auto` | `auto` (onnxruntime, else sentence-transformers, else hashed n-grams), `onnx`, `torch` |
| `ZYPHER_EMBED_DIR` | `encoders` beside the model folder | Where the encoder's ONNX export is downloaded (~90 MB) |

Generate a key: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

### 7.2 `zypher/config.py` (main settings)

| Setting | Default | Notes |
|---|---|---|
| `MAX_TOTAL_NEW_TOKENS` | half the context window | Per answer; runtime-mutable |
| `MAX_PROMPT_TOKENS` | half the context window | Older turns dropped beyond this |
| `SAMPLING` | temp 0.3 / 0.6 / 0.85, `min_p` 0.05 | precise / balanced / creative |
| `REPETITION_PENALTY` / `REPEAT_LAST_N` | 1.05 / 64 | Over the last 64 tokens |
| `STOP_STRINGS` | `<\|user\|>`, `<\|system\|>`, `<\|assistant\|>` | |
| `THROTTLE_TEMP_C` / `SLOW_ANSWER_RATIO` | 84 / 0.5 | When a slow answer is reported as GPU throttling |
| `MAX_NEW_TOKENS`, `AUTO_CONTINUE`, `WARMUP` | 2048 / `True` / `True` | transformers engine only |
| `RETRIEVAL_ENABLED` / `MEMORY_ENABLED` | `True` / `True` | Runtime-mutable |
| `RECALL_KEEP` / `RECALL_THRESHOLD` | 3 / 0.42 | |
| `RECALL_CHAR_BUDGET` / `ANSWER_CHAR_BUDGET` | 1200 / 400 | |
| `PROFILE_MAX_NOTES` / `PROFILE_CHAR_BUDGET` | 8 / 600 | |
| `DEDUPE_THRESHOLD` | 0.94 | Note de-duplication |
| `MAX_RESULTS` / `FETCH_RESULTS` | 4 / 8 | Web snippets kept / requested |
| `CONTEXT_CHAR_BUDGET` / `SNIPPET_CHAR_BUDGET` | 3000 / 600 | |
| `TIME_LIMIT` | `"m"` | Last month; lifted for named past years |

---

## 8. Data storage

| Path | Content |
|---|---|
| `ZYPHER_MODEL_PATH/*.gguf` | The model llama.cpp runs (4.4 GB) |
| `ZYPHER_MODEL_PATH/*.safetensors` | The original 13.5 GB shards — only the transformers engine reads them |
| `ZYPHER_LLAMA_DIR/<release>/` | llama.cpp binaries and the CUDA runtime (~1.1 GB) |
| `ZYPHER_EMBED_DIR/<encoder>/` | The recall encoder's ONNX export, tokenizer and pooling config (~90 MB) |
| `.logs/llama-server.log` | The engine's output from the last start |
| `.memory/memory.jsonl` | One JSON record per line (see 5.9) |
| `.memory/vectors.npy` | Embedding matrix, row-aligned with the records; rebuilt if it drifts |
| `.memory/meta.json` | Encoder metadata (rebuilds vectors when the encoder changes) |

Delete `.memory/` to start over. Nothing leaves the machine except search
queries, and only when a question needs the web.
