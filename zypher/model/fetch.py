"""Resumable, checksummed downloads with the standard library.

Used for everything this project fetches -- the GGUF weights, the llama.cpp
release, the recall encoder -- so none of them needs huggingface_hub, and a
download interrupted at 4 GB of 4.4 picks up where it stopped.
"""

import hashlib
import os
import time
import urllib.error
import urllib.request

USER_AGENT = "zypherLL"
CHUNK = 1 << 20
ATTEMPTS = 5


def hub_url(repo, filename):
    """Download URL of one file in a Hugging Face repository."""

    return "https://huggingface.co/{}/resolve/main/{}".format(repo, filename)


def _open(url, start=0):
    headers = {"User-Agent": USER_AGENT}

    if start:
        headers["Range"] = "bytes={}-".format(start)

    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)


def download(url, destination, label=None):
    """Fetch url to destination, resuming a partial file.

    Written to <destination>.part and renamed when complete, so a file with
    the final name is always a whole one. With a label, progress is printed
    on one line; without, nothing is printed -- for small files, and for
    downloads on a background thread that must not write over the prompt.
    """

    partial = destination + ".part"

    for attempt in range(1, ATTEMPTS + 1):
        have = os.path.getsize(partial) if os.path.exists(partial) else 0

        try:
            try:
                response = _open(url, have)
            except urllib.error.HTTPError as error:
                # 416: the partial file is already the whole thing.
                if error.code == 416 and have:
                    break

                # Missing or forbidden will not be cured by trying again.
                if error.code in (401, 403, 404):
                    raise FileNotFoundError("{} ({})".format(url, error.code)) from error

                raise

            with response:
                # A server that ignores Range sends the file from the start.
                if have and response.status != 206:
                    have = 0

                total = have + int(response.headers.get("Content-Length") or 0)
                shown = 0.0

                with open(partial, "ab" if have else "wb") as handle:
                    while True:
                        block = response.read(CHUNK)

                        if not block:
                            break

                        handle.write(block)
                        have += len(block)

                        now = time.monotonic()

                        if label and now - shown >= 1.0:
                            shown = now
                            print("\r{}: {:.0f} / {:.0f} MB".format(
                                label, have / 1e6, total / 1e6), end="", flush=True)

            if total and have < total:
                raise OSError("connection closed at {} of {} bytes".format(have, total))

            if label:
                print("\r{}: {:.0f} MB -- done.{}".format(label, have / 1e6, " " * 16))

            break

        except FileNotFoundError:
            raise

        except (OSError, urllib.error.URLError) as error:
            if label:
                print("\n[{}: attempt {}/{} failed: {}]".format(label, attempt, ATTEMPTS, error))

            if attempt == ATTEMPTS:
                raise

            time.sleep(min(30, 3 * attempt))

    os.replace(partial, destination)


class _NoRedirect(urllib.request.HTTPRedirectHandler):

    def redirect_request(self, *args, **kwargs):
        return None


def linked_sha256(url):
    """The sha256 Hugging Face advertises for a large file, or None.

    It is on the redirect, not on the storage host the redirect leads to, so
    the request must stop at the first response.
    """

    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})

    try:
        response = urllib.request.build_opener(_NoRedirect).open(request, timeout=60)
        headers = response.headers
    except urllib.error.HTTPError as error:
        headers = error.headers
    except (OSError, urllib.error.URLError):
        return None

    etag = (headers.get("X-Linked-Etag") or headers.get("ETag") or "").strip('"')

    return etag if len(etag) == 64 else None


def sha256(path):
    with open(path, "rb") as handle:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(handle, "sha256").hexdigest()

        digest = hashlib.sha256()

        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)

        return digest.hexdigest()


def download_verified(url, destination, label=None):
    """download(), then check the file against the hash the Hub records.

    Checked once, here, rather than on every start: a file that passed its
    hash does not change on disk. One that fails is set aside as
    <destination>.corrupt, so the next run fetches it again.
    """

    expected = linked_sha256(url)

    download(url, destination, label)

    if not expected:
        return

    if label:
        print("Verifying checksum...", end="", flush=True)

    if sha256(destination) != expected:
        os.replace(destination, destination + ".corrupt")

        raise RuntimeError(
            "{} failed its checksum after download; it was set aside as "
            "{}.corrupt -- start again to re-download".format(
                os.path.basename(destination), destination
            )
        )

    if label:
        print(" ok.")
