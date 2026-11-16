"""Get what the engine runs on: the llama-server binary and the GGUF weights.

Both are fetched once, with resume, into the folders named in the config,
and found there on every later start without touching the network.
"""

import json
import os
import platform
import shutil
import sys
import urllib.request
import zipfile

from zypher import config
from zypher.model import fetch

RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


def _server_name():
    return "llama-server.exe" if os.name == "nt" else "llama-server"


# ============================================================
# WEIGHTS
# ============================================================

def ensure_model():
    """Path of the GGUF file, downloading it on first start."""

    path = config.gguf_path()

    if os.path.exists(path):
        return path

    if config.MODEL_PATH.lower().endswith(".gguf"):
        raise FileNotFoundError(
            "ZYPHER_MODEL_PATH names {}, which does not exist".format(path)
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)

    url = fetch.hub_url(config.GGUF_REPO, config.GGUF_FILE)

    print("\nDownloading the model (once):")
    print("  from", url)
    print("  to  ", path)

    fetch.download_verified(url, path, "model")

    return path


# ============================================================
# LLAMA-SERVER
# ============================================================

def find_server():
    """An existing llama-server: the configured one, the newest under
    LLAMA_DIR, or one on PATH. None if there is none."""

    if config.LLAMA_SERVER:
        if not os.path.exists(config.LLAMA_SERVER):
            raise FileNotFoundError(
                "ZYPHER_LLAMA_SERVER names {}, which does not exist".format(
                    config.LLAMA_SERVER
                )
            )

        return config.LLAMA_SERVER

    found = []

    for root, _dirs, files in os.walk(config.LLAMA_DIR):
        if _server_name() in files:
            candidate = os.path.join(root, _server_name())
            found.append((os.path.getmtime(candidate), candidate))

    if found:
        return max(found)[1]

    return shutil.which("llama-server")


def _pick_assets(assets, has_nvidia):
    """Release archives for this machine: the build, plus the CUDA runtime
    DLLs it needs. Empty if this platform is not one we know how to pick for."""

    if os.name != "nt" or platform.machine().lower() not in ("amd64", "x86_64"):
        return []

    names = {asset["name"]: asset["browser_download_url"] for asset in assets}

    def first(*needles):
        for name in sorted(names):
            if all(needle in name for needle in needles) and name.endswith(".zip"):
                return [(name, names[name])]

        return []

    if has_nvidia:
        # CUDA 12 runs on every driver from the last few years; the CUDA 13
        # build wants a recent one.
        build = first("llama-", "bin-win-cuda-12", "x64")
        runtime = first("cudart-", "win-cuda-12", "x64")

        if build and runtime:
            return build + runtime

    return first("llama-", "bin-win-vulkan-x64") or first("llama-", "bin-win-cpu-x64")


def install_server(has_nvidia):
    """Download the current llama.cpp release into LLAMA_DIR."""

    print("\nllama-server not found -- downloading llama.cpp (once)...")

    request = urllib.request.Request(RELEASE_API, headers={"User-Agent": fetch.USER_AGENT})

    with urllib.request.urlopen(request, timeout=60) as response:
        release = json.load(response)

    chosen = _pick_assets(release.get("assets", []), has_nvidia)

    if not chosen:
        raise RuntimeError(
            "no prebuilt llama.cpp for this platform ({} {}). Install it "
            "yourself -- https://github.com/ggml-org/llama.cpp -- and set "
            "ZYPHER_LLAMA_SERVER to the llama-server executable.".format(
                sys.platform, platform.machine()
            )
        )

    target = os.path.join(config.LLAMA_DIR, release.get("tag_name", "current"))
    os.makedirs(target, exist_ok=True)

    for name, url in chosen:
        archive = os.path.join(target, name)
        fetch.download(url, archive, name)

        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target)

        os.remove(archive)

    server = find_server()

    if server is None:
        raise RuntimeError("llama.cpp was downloaded but llama-server is not in it")

    print("llama.cpp {} installed in {}".format(release.get("tag_name", ""), target))

    return server


def ensure_server(has_nvidia):
    return find_server() or install_server(has_nvidia)
