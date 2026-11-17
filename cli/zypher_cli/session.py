"""The last conversation of each project folder, kept so `zy -c` can pick it up."""

import hashlib
import json
import os
import time


def _path(settings, folder):
    digest = hashlib.sha1(os.path.normcase(str(folder)).encode("utf-8")).hexdigest()[:16]

    return settings.sessions_dir / (digest + ".json")


def save(settings, folder, data):
    """Best effort: a session that cannot be saved must not stop the work."""

    if not data.get("messages"):
        return

    try:
        path = _path(settings, folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"folder": str(folder), "saved": time.time(), "data": data}),
            encoding="utf-8",
        )
        os.replace(str(temporary), str(path))
    except OSError:
        pass


def load(settings, folder):
    """The saved conversation for this folder, or None."""

    try:
        return json.loads(_path(settings, folder).read_text(encoding="utf-8"))["data"]
    except (OSError, ValueError, KeyError):
        return None
