"""
The FastAPI app, and the server that runs it.

The server answers as soon as it starts: /health reports "loading" while the
model loads on a background thread, and chat calls return 503 until it is
ready.
"""

import os
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from zypher import config
from zypher.presenter import Assistant

from .routes import public, router

# Browser apps on other origins allowed to call the API directly, comma
# separated (e.g. http://localhost:3000). Empty -- the default -- sends no
# CORS headers, so browsers refuse cross-origin calls; the web/ app does not
# need this, it reaches the API through its own server's proxy. Read here,
# after config has loaded .env into the environment.
CORS_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.environ.get("ZYPHER_CORS_ORIGINS", "").split(",")
    if origin.strip()
]


def create_app(assistant=None, load=True):
    """Build the app around an Assistant. load=False skips loading the model,
    for tests that stub it."""

    assistant = assistant or Assistant()

    @asynccontextmanager
    async def lifespan(app):
        if load:
            threading.Thread(target=assistant.load, daemon=True).start()

        yield

        # Server shutting down: stop the engine and give the VRAM back.
        assistant.close()

    app = FastAPI(title="zypherLL", version="1.0", lifespan=lifespan)
    app.state.assistant = assistant

    if CORS_ORIGINS:
        # No cookies are used -- auth is the bearer key -- so credentials
        # stay off, and "*" is honoured as "any origin".
        app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_ORIGINS,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

    app.include_router(public)
    app.include_router(router)

    return app


def serve(host=None, port=None):
    host = host or config.HOST
    port = port or config.PORT

    if host not in ("127.0.0.1", "localhost", "::1") and config.API_KEY is None:
        print("[warning: listening on {} with no ZYPHER_API_KEY set -- anyone who "
              "can reach this port can use the model and read its memory]".format(host))

    uvicorn.run(create_app(), host=host, port=port)
