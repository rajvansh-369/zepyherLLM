"""
Where things are: the API, its key, and this tool's own folder.

The shell environment wins, then cli/.env, then the zypherLL .env -- so on the
machine that runs the model there is nothing to configure: the key and the
port are read from the same file the server reads.
"""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CLI_DIR = PACKAGE_DIR.parent


def repo_root():
    """The zypherLL checkout this CLI belongs to, or None."""

    explicit = os.environ.get("ZYPHER_HOME")
    candidates = ([Path(explicit)] if explicit else []) + [CLI_DIR.parent]

    for folder in candidates:
        if (folder / "main.py").is_file() and (folder / "zypher").is_dir():
            return folder

    return None


def read_env(path):
    """KEY=VALUE lines of a .env file. Missing file: {}."""

    values = {}

    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return values

    for line in lines:
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]

        values[key.strip()] = value

    return values


class Settings:

    def __init__(self, url=None):
        self.repo = repo_root()
        self.home = Path(os.environ.get("ZYPHER_CLI_HOME") or Path.home() / ".zypher")

        files = {}

        if self.repo is not None:
            files.update(read_env(self.repo / ".env"))

        files.update(read_env(self.home / ".env"))
        files.update(read_env(CLI_DIR / ".env"))

        def get(name, default=""):
            return os.environ.get(name) or files.get(name) or default

        host = get("ZYPHER_HOST", "127.0.0.1")

        # A server bound to every interface is reached through loopback.
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"

        self.url = (
            url or get("ZYPHER_API_URL")
            or "http://{}:{}".format(host, get("ZYPHER_PORT", "8000"))
        ).rstrip("/")
        self.key = get("ZYPHER_API_KEY")
        self.shell = get("ZYPHER_SHELL").lower()

    @property
    def history_file(self):
        return self.home / "history"

    @property
    def sessions_dir(self):
        return self.home / "sessions"

    def ensure_home(self):
        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            return True
        except OSError:
            return False
