"""
Endpoints. Each one translates HTTP to an Assistant call and back.

    GET    /health                      load state, engine, device (no auth)
    GET    /v1/models                   the one model
    POST   /v1/chat/completions         answer, optionally as server-sent events
    GET    /v1/memory                   stats and the notes in the system prompt
    GET    /v1/memory/records           stored records, newest first
    POST   /v1/memory/notes             keep a fact about the user
    GET    /v1/memory/about             what the user wrote about themselves
    PUT    /v1/memory/about             change it
    POST   /v1/memory/{id}/rate         good / bad
    DELETE /v1/memory                   forget last | all | <text>
    GET    /v1/settings                 server-wide defaults
    PATCH  /v1/settings                 change them
"""

import asyncio
import hmac
import json
import threading
import time
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from zypher import config
from zypher.model import retrieval
from zypher.model.llm import ClientGone, OutOfMemory
from zypher.presenter import ModelNotReady

from .schemas import AboutRequest, ChatRequest, NoteRequest, RateRequest, SettingsRequest

OOM_MESSAGE = "out of GPU memory -- send less history or lower max_tokens"


def authorize(request: Request):
    if config.API_KEY is None:
        return

    # Constant-time compare, so the key cannot be recovered byte by byte from
    # response timings.
    supplied = request.headers.get("authorization", "")

    if not hmac.compare_digest(supplied.encode(), ("Bearer " + config.API_KEY).encode()):
        raise HTTPException(401, "invalid or missing API key")


def assistant_of(request: Request):
    return request.app.state.assistant


public = APIRouter()
router = APIRouter(dependencies=[Depends(authorize)])


# -- system ---------------------------------------------------

@public.get("/health")
def health(assistant=Depends(assistant_of)):
    # backend, device, gpu, vram_gb, context, gpu_layers, kv_cache -- filled
    # in once the engine has looked at the hardware.
    info = {"backend": config.BACKEND, "device": None, "vram_gb": 0.0}
    info.update(assistant.engine_info())

    return dict(
        info,
        # Every configured model; "active" is the one on the GPU now.
        models=assistant.models(),
        switching=assistant.switching,
        status=assistant.status(),
        error=None if assistant.error is None else str(assistant.error),
        auth=config.API_KEY is not None,
    )


@router.get("/v1/models")
def models(assistant=Depends(assistant_of)):
    return {
        "object": "list",
        "data": [{
            "id": model["id"],
            "object": "model",
            "created": 0,
            "owned_by": "local",
            "slot": model["slot"],
            "active": model["active"],
        } for model in assistant.models()],
    }


def current_settings(assistant):
    return {
        "web": config.RETRIEVAL_ENABLED,
        "memory": config.MEMORY_ENABLED,
        "max_tokens": config.MAX_TOTAL_NEW_TOKENS,
        "max_prompt_tokens": config.MAX_PROMPT_TOKENS,
        "model": assistant.active,
        "models": assistant.models(),
        "switching": assistant.switching,
        "switch_error": assistant.switch_error,
    }


@router.get("/v1/settings")
def get_settings(assistant=Depends(assistant_of)):
    return current_settings(assistant)


@router.patch("/v1/settings")
def patch_settings(body: SettingsRequest, assistant=Depends(assistant_of)):
    """Server-wide defaults: what a request gets when it does not say."""

    if body.model is not None and body.model != assistant.active:
        if body.model not in assistant.engines:
            raise HTTPException(
                400, "no {} model is configured -- set ZYPHER_CODE_MODEL_PATH".format(body.model)
            )

        if assistant.status() != "ready":
            raise HTTPException(409, "a model is still loading -- try again when it is ready")

        # The load takes up to a minute: answer now, and let /health say
        # "loading" until it is done.
        assistant.switching = body.model
        threading.Thread(target=assistant.switch, args=(body.model,), daemon=True).start()

    if body.web is not None:
        config.RETRIEVAL_ENABLED = body.web

        # Switching it back on is how a client says the network is back
        # after repeated failures paused lookups.
        if body.web:
            retrieval.reset_failures()

    if body.memory is not None:
        config.MEMORY_ENABLED = body.memory

    if body.max_tokens is not None:
        config.MAX_TOTAL_NEW_TOKENS = body.max_tokens

    return current_settings(assistant)


# -- chat -----------------------------------------------------

@router.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest, assistant=Depends(assistant_of)):
    if assistant.status() != "ready":
        raise HTTPException(
            500 if assistant.error is not None else 503,
            "model failed to load: {}".format(assistant.error)
            if assistant.error is not None else "model is still loading",
        )

    messages = [message.as_dict() for message in body.messages]

    options = dict(
        web=body.web,
        memory=body.memory,
        sampling=body.sampling,
        temperature=body.temperature,
        top_p=body.top_p,
        max_tokens=body.max_completion_tokens or body.max_tokens,
        min_tokens=body.min_tokens,
        model=body.model,
    )

    meta = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time())}

    if body.stream:
        return StreamingResponse(
            stream(assistant, messages, options, meta),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await asyncio.to_thread(assistant.answer, messages, **options)
    except ValueError as error:
        raise HTTPException(400, str(error))
    except ModelNotReady as error:
        raise HTTPException(503, str(error))
    except OutOfMemory:
        raise HTTPException(503, OOM_MESSAGE)

    return {
        "id": meta["id"],
        "object": "chat.completion",
        "created": meta["created"],
        "model": result["meta"]["model"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["text"]},
            "finish_reason": result["finish_reason"],
        }],
        "usage": result["usage"],
        "zypher": public_meta(result["meta"]),
    }


def public_meta(meta):
    return {key: value for key, value in meta.items() if key != "history_kept"}


def sse(payload):
    return "data: {}\n\n".format(json.dumps(payload, ensure_ascii=False))


def chunk(meta, delta, finish_reason=None, **extra):
    payload = {
        "id": meta["id"],
        "object": "chat.completion.chunk",
        "created": meta["created"],
        "model": config.MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    payload.update(extra)

    return sse(payload)


async def stream(assistant, messages, options, meta):
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
            result = assistant.answer(messages, on_event=on_event, **options)
            loop.call_soon_threadsafe(queue.put_nowait, ("done", result))
        except ClientGone:
            pass
        except BaseException as error:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", error))

    loop.run_in_executor(None, run)

    try:
        yield chunk(meta, {"role": "assistant", "content": ""})

        while True:
            kind, payload = await queue.get()

            if kind == "text":
                yield chunk(meta, {"content": payload})

            elif kind == "status":
                yield chunk(meta, {}, zypher=payload)

            elif kind == "done":
                yield chunk(
                    meta, {},
                    finish_reason=payload["finish_reason"],
                    usage=payload["usage"],
                    zypher=public_meta(payload["meta"]),
                )
                break

            else:
                message = (
                    OOM_MESSAGE if isinstance(payload, OutOfMemory)
                    else str(payload) if isinstance(payload, (ValueError, ModelNotReady))
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

def memory_call(function, *args):
    try:
        return function(*args)
    except ModelNotReady as error:
        raise HTTPException(503, str(error))


@router.get("/v1/memory")
def memory_overview(assistant=Depends(assistant_of)):
    return {
        "stats": assistant.memory_stats(),
        "profile": memory_call(assistant.profile),
    }


@router.get("/v1/memory/records")
def memory_records(kind: Optional[Literal["note", "exchange"]] = None,
                   limit: int = 50, offset: int = 0,
                   assistant=Depends(assistant_of)):
    records = memory_call(assistant.records, kind)
    offset = max(0, offset)

    return {"total": len(records), "records": records[offset:offset + max(0, limit)]}


@router.post("/v1/memory/notes")
def memory_note(body: NoteRequest, assistant=Depends(assistant_of)):
    record = memory_call(assistant.note, body.text)

    if record is None:
        raise HTTPException(503, "memory is off or unavailable")

    return record


@router.get("/v1/memory/about")
def memory_about(assistant=Depends(assistant_of)):
    return memory_call(assistant.about)


@router.put("/v1/memory/about")
def memory_set_about(body: AboutRequest, assistant=Depends(assistant_of)):
    return memory_call(assistant.set_about, body.model_dump())


@router.post("/v1/memory/{record_id}/rate")
def memory_rate(record_id: int, body: RateRequest, assistant=Depends(assistant_of)):
    record = memory_call(assistant.rate, record_id, body.rating == "good")

    if record is None:
        raise HTTPException(404, "no memory with id {}".format(record_id))

    return record


@router.delete("/v1/memory")
def memory_forget(selector: str = "last", assistant=Depends(assistant_of)):
    """Drop memories: 'last', 'all', or free text matched by similarity."""

    return {"removed": memory_call(assistant.forget, selector)}
