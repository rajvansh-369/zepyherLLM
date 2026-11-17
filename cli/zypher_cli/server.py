"""
Starting the zypherLL API when it is not running.

The CLI only ever talks to the API over HTTP. This is the one place it knows
the server is a program in the same checkout: it can start `python main.py
serve` in the background, so `zy` works from a cold machine. The server is
left running afterwards -- the web app shares it, and it gives its VRAM back
by itself when idle.
"""

import os
import subprocess
import sys
import time

from .api import ApiError, Offline


# The server this run started, if any: lets wait_ready notice it dying
# before it ever listened, instead of polling until the timeout.
_process = None


def can_start(settings):
    return settings.repo is not None


def start(settings):
    """Launch the API detached from this terminal. Returns its log path."""

    repo = settings.repo
    logs = repo / ".logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / "api.log"

    python = os.environ.get("ZYPHER_PYTHON") or sys.executable
    options = {}

    if os.name == "nt":
        # Its own hidden console and process group: Ctrl+C in this terminal,
        # or closing it, must not take the model down with it.
        options["creationflags"] = (
            subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        options["start_new_session"] = True

    global _process

    with open(str(log_path), "ab") as log:
        _process = subprocess.Popen(
            [python, "main.py", "serve"], cwd=str(repo),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            **options
        )

    return log_path


def wait_ready(api, timeout=900, on_wait=None):
    """Poll /health until the model is up. Returns the health dict.

    Raises ApiError if the load fails, or nothing answers in time. The first
    start downloads the model, which is why the limit is generous.
    """

    deadline = time.monotonic() + timeout
    seen = False

    while time.monotonic() < deadline:
        try:
            health = api.health()
            seen = True
        except Offline:
            # Not listening yet -- or it started and died.
            if seen:
                raise ApiError(0, "the API stopped while loading")

            if _process is not None and _process.poll() is not None:
                raise ApiError(0, "the API exited while starting (see .logs/api.log)")

            health = None
        except ApiError:
            health = None

        if health is not None:
            if health.get("status") == "ready":
                return health

            if health.get("status") == "error":
                raise ApiError(500, "the model failed to load: {}".format(health.get("error")))

        if on_wait is not None:
            on_wait(health)

        time.sleep(1.0)

    raise ApiError(0, "the API did not come up within {} s".format(timeout))
