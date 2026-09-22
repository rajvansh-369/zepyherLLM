"""
The FastAPI app, and the server that runs it.

The server answers as soon as it starts: /health reports "loading" while the
model loads on a background thread, and chat calls return 503 until it is
ready.
"""

import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from zypher import config
from zypher.presenter import Assistant

from .routes import public, router


def create_app(assistant=None, load=True):
    """Build the app around an Assistant. load=False skips loading the model,
    for tests that stub it."""

    assistant = assistant or Assistant()

    @asynccontextmanager
    async def lifespan(app):
        if load:
            threading.Thread(target=assistant.load, daemon=True).start()
        yield

    app = FastAPI(title="zypherLL", version="1.0", lifespan=lifespan)
    app.state.assistant = assistant
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
