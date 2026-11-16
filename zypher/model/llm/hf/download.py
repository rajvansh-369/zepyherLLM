"""Fetch the weights, resuming, and checksum them before they are trusted."""

import hashlib
import json
import os
import time

from huggingface_hub import snapshot_download

from zypher.config import (
    DOWNLOAD_ATTEMPTS,
    DOWNLOAD_WORKERS,
    IGNORE_PATTERNS,
    MODEL_NAME,
    MODEL_PATH,
)


def expected_weight_files(path):
    """Return (shard_names, expected_total_bytes) from the safetensors index.

    Falls back to the single-file layout when the model is not sharded.
    """

    index_file = os.path.join(path, "model.safetensors.index.json")

    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as handle:
            index = json.load(handle)

        shards = sorted(set(index["weight_map"].values()))
        total = index.get("metadata", {}).get("total_size", 0)

        return shards, total

    return ["model.safetensors"], 0


def staging_dir(path):
    """Where huggingface_hub keeps partial downloads and etag sidecars."""

    return os.path.join(path, ".cache", "huggingface", "download")


def expected_sha256(path, filename):
    """Read the expected hash from the sidecar huggingface_hub writes.

    The sidecar is three lines: commit hash, etag, timestamp. For LFS files --
    which is every weight shard -- the etag is the sha256 of the content, so
    integrity can be checked without going back to the network.
    """

    sidecar = os.path.join(staging_dir(path), filename + ".metadata")

    if not os.path.exists(sidecar):
        return None

    with open(sidecar, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    if len(lines) < 2:
        return None

    etag = lines[1].strip()

    # Non-LFS files carry a git blob sha1 instead; only sha256 is usable here.
    return etag if len(etag) == 64 else None


def file_sha256(file_path, chunk_size=16 << 20):
    """Hash a file, using the C reader when the interpreter provides one.

    hashlib.file_digest (3.11+) reads into a reusable buffer and drops the GIL
    for the whole file. The loop below allocates a fresh 16 MB bytes object per
    block, which over a 13.5 GB model is several gigabytes of garbage created
    and immediately discarded.
    """

    with open(file_path, "rb") as handle:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(handle, "sha256").hexdigest()

        digest = hashlib.sha256()

        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)

    return digest.hexdigest()


def _cache_file(path):
    return os.path.join(path, ".verified.json")


def _load_verify_cache(path):
    try:
        with open(_cache_file(path), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _save_verify_cache(path, cache):
    try:
        with open(_cache_file(path), "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2)
    except OSError:
        pass


def quarantine(path, filename):
    """Delete a bad shard and its sidecar so the next attempt refetches it."""

    targets = [
        os.path.join(path, filename),
        os.path.join(staging_dir(path), filename + ".metadata"),
    ]

    for target in targets:
        try:
            os.remove(target)
        except OSError:
            pass


def verify_model_files(path, check_hashes=True):
    """Check the weights are present and intact.

    The original code tested for config.json, but the small JSON files land
    first and the multi-GB shards land last -- so that test passed while the
    weights were still missing, and every rerun skipped the resume.

    Hashing 13.5 GB takes a couple of minutes, so a verified result is cached
    against each shard's size and mtime. In practice each shard is hashed once,
    on the run that downloads it, and startup is instant from then on.

    Returns (ok, list_of_problems).
    """

    problems = []

    if not os.path.exists(os.path.join(path, "config.json")):
        return False, ["config.json missing"]

    shards, expected_total = expected_weight_files(path)

    present = []
    actual_total = 0

    for shard in shards:
        shard_path = os.path.join(path, shard)

        if not os.path.exists(shard_path):
            problems.append(shard + " missing")
            continue

        present.append(shard)
        actual_total += os.path.getsize(shard_path)

    staging = staging_dir(path)

    if os.path.isdir(staging):
        for name in os.listdir(staging):
            if name.endswith(".incomplete"):
                size_gb = os.path.getsize(os.path.join(staging, name)) / (1024 ** 3)
                problems.append(
                    "partial file left by an interrupted download ({:.2f} GB)".format(size_gb)
                )

    # The index reports the size of the tensor data only; each shard also
    # carries a small safetensors header, so files on disk run slightly
    # larger than expected_total. Short means truncated.
    if expected_total and not problems and actual_total < expected_total:
        problems.append(
            "weights truncated: have {} bytes, need at least {} bytes".format(
                actual_total, expected_total
            )
        )

    if problems or not check_hashes:
        return (not problems), problems

    cache = _load_verify_cache(path)
    cache_dirty = False

    for shard in present:
        shard_path = os.path.join(path, shard)
        want = expected_sha256(path, shard)

        if want is None:
            # No sidecar to compare against; size checks above are all we have.
            continue

        stat = os.stat(shard_path)
        entry = cache.get(shard)

        if (
            entry
            and entry.get("sha256") == want
            and entry.get("size") == stat.st_size
            and entry.get("mtime_ns") == stat.st_mtime_ns
        ):
            continue

        print("  verifying {} ...".format(shard), end="", flush=True)
        got = file_sha256(shard_path)

        if got == want:
            print(" ok")
            cache[shard] = {
                "sha256": got,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            cache_dirty = True
        else:
            print(" CORRUPT")
            problems.append(shard + " failed checksum -- discarding it")
            cache.pop(shard, None)
            cache_dirty = True
            quarantine(path, shard)

    if cache_dirty:
        _save_verify_cache(path, cache)

    return (not problems), problems


def download_model():
    """Download or resume the model, then confirm the weights are complete."""

    os.makedirs(MODEL_PATH, exist_ok=True)

    ok, problems = verify_model_files(MODEL_PATH)

    if ok:
        print("\nModel present and complete:", MODEL_PATH)
        return True

    print("\nModel not ready:")
    for problem in problems:
        print("  -", problem)

    # Partials are always resumed, never discarded. A Xet-written partial can
    # be full-size with holes, but throwing every partial away to guard
    # against that costs gigabytes on each restart. The checksum below catches
    # a bad shard once it is finalized and quarantines it then, so the worst
    # case is one wasted refetch instead of a guaranteed one.
    print("\nDownloading (resuming from whatever is already on disk)...")
    print("Model:", MODEL_NAME)
    print("Destination:", MODEL_PATH)

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):

        try:
            snapshot_download(
                repo_id=MODEL_NAME,
                local_dir=MODEL_PATH,
                ignore_patterns=IGNORE_PATTERNS,
                max_workers=DOWNLOAD_WORKERS,
            )

        except KeyboardInterrupt:
            print("\nInterrupted. Rerun to resume from here.")
            return False

        except Exception as error:
            print("\nAttempt {}/{} failed: {}: {}".format(
                attempt, DOWNLOAD_ATTEMPTS, type(error).__name__, error
            ))

        ok, problems = verify_model_files(MODEL_PATH)

        if ok:
            print("\nDownload complete and verified.")
            return True

        if attempt < DOWNLOAD_ATTEMPTS:
            for problem in problems:
                print("  -", problem)

            # Partials are deliberately left alone here. With Xet disabled
            # they are plain append-only files, so the next attempt resumes
            # from the byte offset instead of refetching several GB. A partial
            # that does turn out to be bad is caught by the checksum once it
            # is finalized, and quarantined then.
            backoff = min(30, 2 ** attempt)
            print("Retrying in {}s...".format(backoff))
            time.sleep(backoff)

    print("\nDownload did not complete:")
    for problem in problems:
        print("  -", problem)

    return False
