"""
HTTP API for the Zephyr 7B runner.

OpenAI-compatible where it can be -- POST /v1/chat/completions, streaming or
not, works with the openai client and anything built on it -- so the model
can sit behind existing tools. What the console chat adds on top of the raw
model comes with it: live web lookup, memory recall, profile notes, and the
learning from each finished exchange.

    python api.py                      # 127.0.0.1:8000
    python api.py --host 0.0.0.0 --port 8080

Set ZYPHER_API_KEY to require "Authorization: Bearer <key>" on every call
except /health. Unset, the API is open, which is why it binds to localhost
unless told otherwise.

One model and one GPU: generations are served one at a time, in arrival
order. The KV cache is shared across requests and cropped to whatever prefix
a new prompt has in common with the last one, so a client resending its
growing history reuses nearly all of it.
"""

import argparse
import asyncio
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

import llm
import memory as memory_store
import retrieval


API_KEY = os.environ.get("ZYPHER_API_KEY") or None

MODEL_ID = "zephyr-7b"


# ============================================================
# ENGINE
# ============================================================

class ClientGone(Exception):
    """Raised from the token callback to stop a generation nobody is reading."""


class Engine:
    """The loaded model, the memory store, and the lock that serialises them.

    The KV cache follows whichever request ran last. reuse_cache() crops it to
    the prefix a new prompt shares with it, so interleaved clients cost a
    recompute, never a wrong answer.
    """

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.store = None
        self.cache_state = {}
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.error = None

    def load(self):
        try:
            # Constructed first: its encoder loads in the background while the
            # model does.
            self.store = memory_store.Memory()

            if not llm.download_model():
                raise RuntimeError("model download or verification failed")

            self.tokenizer = llm.load_tokenizer()
            self.model = llm.load_model()

            if llm.WARMUP:
                llm.warm_up(self.tokenizer, self.model)

            self.ready.set()

        except BaseException as error:
            self.error = error
            raise

    def turn(self, messages, options, on_event=None):
        """Answer one request. Runs on a worker thread, under the GPU lock.

        messages is the client's conversation. Its last message is either the
        user's question, or a partial assistant reply to continue.

        on_event, when given, receives ("text", str) for each decoded piece,
        and ("status", dict) for search and recall before generation starts.
        Raising from it cancels the generation.
        """

        def event(kind, payload):
            if on_event is not None:
                on_event(kind, payload)

        store = self.store
        use_memory = options["memory"] and store is not None

        system_extra, history = split_system(messages)
        continuing = history[-1]["role"] == "assistant"

        question = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"), ""
        )

        context = None
        sources = []
        recalled = 0
        notes = ()

        if not continuing:
            web = options["web"]

            if web is True or (web == "auto" and llm.RETRIEVAL_ENABLED
                               and retrieval.needs_live_data(question)):
                event("status", {"stage": "searching"})
                context, sources = retrieval.fetch_context(question)

        grounded = retrieval.ground(question, context) if context else question

        if use_memory:
            store.ready(15)

            if not continuing:
                hits = store.recall(question)
                recalled = len(hits)
                grounded = memory_store.ground(grounded, store.block(hits))

            notes = store.profile()

        if recalled:
            event("status", {"stage": "recalled", "count": recalled})

        system = llm.system_prompt(notes)

        if system_extra:
            system += "\n\n" + system_extra

        conversation = [{"role": "system", "content": system}] + [dict(m) for m in history]

        if not continuing:
            conversation[-1]["content"] = grounded

        sampling = options["sampling"] or llm.pick_sampling(
            question, grounded=bool(context)
        )
        params = dict(llm.SAMPLING[sampling])

        for key in ("temperature", "top_p"):
            if options.get(key) is not None:
                params[key] = options[key]

        # Greedy decoding: generate() rejects a zero temperature with
        # sampling on.
        if params["temperature"] <= 0:
            params = {"do_sample": False}

        live = bool(context) or options["web"] is True or retrieval.needs_live_data(question)

        with self.lock:
            try:
                reply = llm.generate_response(
                    self.tokenizer,
                    self.model,
                    conversation,
                    self.cache_state,
                    continuing=continuing,
                    on_text=lambda text: event("text", text),
                    sampling=params,
                    max_total_tokens=options["max_tokens"],
                )
            except torch.cuda.OutOfMemoryError:
                self.cache_state.clear()
                torch.cuda.empty_cache()
                raise

        finished = not reply["truncated"]

        if continuing:
            text = reply["text"]
            full_answer = llm.clean_reply(history[-1]["content"] + text, finished)
        else:
            text = llm.clean_reply(reply["text"], finished)
            full_answer = text

        captured = []
        memory_id = None

        if use_memory:
            try:
                if not continuing:
                    captured = [r["a"] for r in store.capture_notes(question)]

                if full_answer and finished and not live and question:
                    record = store.remember(question, full_answer)
                    memory_id = record["id"] if record else None

            except Exception as error:  # a memory failure must not fail the answer
                print("[memory: {}: {}]".format(type(error).__name__, error))

        return {
            "text": text,
            "finish_reason": reply["finish_reason"],
            "usage": {
                "prompt_tokens": reply["prompt_tokens"],
                "completion_tokens": reply["tokens"],
                "total_tokens": reply["prompt_tokens"] + reply["tokens"],
            },
            "zypher": {
                "sampling": sampling,
                "live": live,
                "sources": cited_sources(sources, full_answer),
                "recalled": recalled,
                "notes_captured": captured,
                "memory_id": memory_id,
                "seconds": round(reply["seconds"], 3),
                "rounds": reply["rounds"],
            },
        }


def split_system(messages):
    """Client system text, and the user/assistant turns in order.

    The runner's own system prompt always leads -- it carries the date, the
    formatting rules and the user's notes. A client's system message is
    appended to it rather than replacing it.
    """

    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    history = [m for m in messages if m["role"] != "system"]

    return system, history


def cited_sources(sources, answer):
    """The sources the answer cites, or all of them when it cites none."""

    cited = {source["n"] for source in sources if "[{}]".format(source["n"]) in answer}

    return [s for s in sources if s["n"] in cited] or list(sources)


engine = Engine()


# ============================================================
# SCHEMAS
# ============================================================

class TextPart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    text: Optional[str] = None


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant"]
    content: Union[str, List[TextPart]]

    def text(self):
        if isinstance(self.content, str):
            return self.content

        return "".join(part.text or "" for part in self.content if part.type == "text")


class ChatRequest(BaseModel):
    # Unknown OpenAI fields (n, stop, seed, ...) are accepted and ignored, so
    # an existing client does not have to be trimmed down to talk to this.
    model_config = ConfigDict(extra="ignore")

    model: str = MODEL_ID
    messages: List[Message] = Field(min_length=1)
    stream: bool = False
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, gt=0, le=1)

    # Runner extensions.
    web: Union[bool, Literal["auto"]] = "auto"
    memory: Optional[bool] = None
    sampling: Optional[Literal["precise", "balanced", "creative"]] = None


class NoteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=memory_store.NOTE_MAX_CHARS)


class RateRequest(BaseModel):
    rating: Literal["good", "bad"]


class SettingsRequest(BaseModel):
    web: Optional[bool] = None
    memory: Optional[bool] = None


# ============================================================
# APP
# ============================================================

@asynccontextmanager
async def lifespan(app):
    # Loading takes a minute; /health answers meanwhile and reports "loading".
    threading.Thread(target=engine.load, daemon=True).start()
    yield


app = FastAPI(title="zypherLL", version="1.0", lifespan=lifespan)


def authorize(request: Request):
    if API_KEY is None:
        return

    if request.headers.get("authorization", "") != "Bearer " + API_KEY:
        raise HTTPException(401, "invalid or missing API key")


def require_model():
    if engine.error is not None:
        raise HTTPException(500, "model failed to load: {}".format(engine.error))

    if not engine.ready.is_set():
        raise HTTPException(503, "model is still loading")


def require_memory():
    if engine.store is None:
        raise HTTPException(503, "memory is not initialised yet")

    engine.store.ready(30)


def memory_stats():
    if engine.store is None:
        return None

    stats = dict(engine.store.stats())
    error = stats.pop("error")
    stats["error"] = None if error is None else "{}: {}".format(type(error).__name__, error)

    return stats


@app.get("/health")
def health():
    return {
        "status": (
            "error" if engine.error is not None
            else "ready" if engine.ready.is_set()
            else "loading"
        ),
        "device": llm.DEVICE,
        "vram_gb": round(llm.VRAM_GB, 2),
        "max_prompt_tokens": llm.MAX_PROMPT_TOKENS,
        "max_answer_tokens": llm.MAX_TOTAL_NEW_TOKENS,
        "web": llm.RETRIEVAL_ENABLED,
        "memory": llm.MEMORY_ENABLED,
        "memory_stats": memory_stats(),
    }


@app.get("/v1/models", dependencies=[Depends(authorize)])
def models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": 0,
            "owned_by": "local",
            "root": llm.MODEL_NAME,
        }],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
async def chat_completions(body: ChatRequest):
    require_model()

    messages = [{"role": m.role, "content": m.text()} for m in body.messages]
    history = [m for m in messages if m["role"] != "system"]

    if not history:
        raise HTTPException(400, "messages needs at least one user message")

    if history[-1]["role"] == "user" and not history[-1]["content"].strip():
        raise HTTPException(400, "the last user message is empty")

    options = {
        "web": body.web,
        "memory": llm.MEMORY_ENABLED if body.memory is None else body.memory,
        "sampling": body.sampling,
        "temperature": body.temperature,
        "top_p": body.top_p,
        # Capped at the server's budget: past it the KV cache is what runs a
        # small card out of memory.
        "max_tokens": min(
            body.max_completion_tokens or body.max_tokens or llm.MAX_TOTAL_NEW_TOKENS,
            llm.MAX_TOTAL_NEW_TOKENS,
        ),
    }

    completion_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())

    if not body.stream:
        try:
            result = await asyncio.to_thread(engine.turn, messages, options)
        except torch.cuda.OutOfMemoryError:
            raise HTTPException(
                503, "out of GPU memory -- send less history or lower max_tokens"
            )

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": result["finish_reason"],
            }],
            "usage": result["usage"],
            "zypher": result["zypher"],
        }

    return StreamingResponse(
        stream_completion(messages, options, completion_id, created),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def chunk(completion_id, created, delta, finish_reason=None, **extra):
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    payload.update(extra)

    return sse(payload)


def sse(payload):
    return "data: {}\n\n".format(json.dumps(payload, ensure_ascii=False))


async def stream_completion(messages, options, completion_id, created):
    """Server-sent events in the OpenAI chunk format.

    Search and recall progress is sent as chunks with an empty delta and a
    "zypher" field, which OpenAI clients ignore. The last chunk carries the
    finish reason, usage and sources.
    """

    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    gone = threading.Event()

    def on_event(kind, payload):
        if gone.is_set():
            raise ClientGone()

        loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

    def run():
        try:
            result = engine.turn(messages, options, on_event)
            loop.call_soon_threadsafe(queue.put_nowait, ("done", result))
        except ClientGone:
            pass
        except BaseException as error:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", error))

    loop.run_in_executor(None, run)

    try:
        yield chunk(completion_id, created, {"role": "assistant", "content": ""})

        while True:
            kind, payload = await queue.get()

            if kind == "text":
                yield chunk(completion_id, created, {"content": payload})

            elif kind == "status":
                yield chunk(completion_id, created, {}, zypher=payload)

            elif kind == "done":
                yield chunk(
                    completion_id, created, {},
                    finish_reason=payload["finish_reason"],
                    usage=payload["usage"],
                    zypher=payload["zypher"],
                )
                break

            else:
                message = (
                    "out of GPU memory -- send less history or lower max_tokens"
                    if isinstance(payload, torch.cuda.OutOfMemoryError)
                    else "{}: {}".format(type(payload).__name__, payload)
                )
                yield sse({"error": {"message": message}})
                break

        yield "data: [DONE]\n\n"

    finally:
        # The client disconnected, or the stream ended: either way the next
        # token callback stops the generation instead of running to the cap
        # with the GPU lock held. The worker winds itself down.
        gone.set()


# -- memory ---------------------------------------------------

@app.get("/v1/memory", dependencies=[Depends(authorize)])
def memory_overview():
    require_memory()

    return {"stats": memory_stats(), "profile": engine.store.profile()}


@app.get("/v1/memory/records", dependencies=[Depends(authorize)])
def memory_records(kind: Optional[Literal["note", "exchange"]] = None,
                   limit: int = 50, offset: int = 0):
    require_memory()

    records = [
        r for r in reversed(engine.store.records)
        if kind is None or r.get("kind", "exchange") == kind
    ]

    return {"total": len(records), "records": records[offset:offset + max(0, limit)]}


@app.post("/v1/memory/notes", dependencies=[Depends(authorize)])
def memory_note(body: NoteRequest):
    require_memory()

    record = engine.store.note(body.text)

    if record is None:
        raise HTTPException(503, "memory is off or unavailable")

    return record


@app.post("/v1/memory/{record_id}/rate", dependencies=[Depends(authorize)])
def memory_rate(record_id: int, body: RateRequest):
    require_memory()

    record = next((r for r in engine.store.records if r.get("id") == record_id), None)

    if record is None:
        raise HTTPException(404, "no memory with id {}".format(record_id))

    return engine.store.rate(1 if body.rating == "good" else -1, record)


@app.delete("/v1/memory", dependencies=[Depends(authorize)])
def memory_forget(selector: str = "last"):
    """Drop memories: 'last', 'all', or free text matched by similarity."""

    require_memory()

    return {"removed": engine.store.forget(selector)}


# -- settings -------------------------------------------------

@app.patch("/v1/settings", dependencies=[Depends(authorize)])
def settings(body: SettingsRequest):
    """Server-wide defaults: what a request gets when it does not say."""

    if body.web is not None:
        llm.RETRIEVAL_ENABLED = body.web

        # Switching it back on is how a client says the network is back
        # after repeated failures paused lookups.
        if body.web:
            retrieval.reset_failures()

    if body.memory is not None:
        llm.MEMORY_ENABLED = body.memory

    return {"web": llm.RETRIEVAL_ENABLED, "memory": llm.MEMORY_ENABLED}


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the Zephyr 7B runner over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost") and API_KEY is None:
        print("[warning: listening on {} with no ZYPHER_API_KEY set -- anyone who "
              "can reach this port can use the model and read its memory]".format(args.host))

    uvicorn.run(app, host=args.host, port=args.port)
