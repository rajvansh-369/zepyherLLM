"""
Every call to the zypherLL HTTP API (projectDetail.md, section 5).

A streamed answer is read on a worker thread and handed over through a queue.
The caller polls that queue, so Ctrl+C is noticed within a tenth of a second
even while the server is silent -- reloading the model, or reading a long
prompt -- instead of when the next byte happens to arrive.
"""

import json
import queue
import threading

import httpx


class ApiError(Exception):

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class Offline(ApiError):
    """Nothing is listening at the API address."""

    def __init__(self, url):
        super().__init__(0, "no zypherLL API at {}".format(url))


def _detail(response):
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()[:300] or "HTTP {}".format(response.status_code)

    detail = body.get("detail") if isinstance(body, dict) else None

    if isinstance(detail, list):  # FastAPI validation errors
        return "; ".join(str(item.get("msg", item)) for item in detail)

    return str(detail or body)


class Stream:
    """One streamed answer.

    Iterating yields (kind, payload):
        ("status", {"stage": ...})     searching / recalled / waking
        ("text", str)                  the next piece of the answer
        ("done", {"finish_reason", "usage", "meta"})
    and raises ApiError if the server reports one. close() abandons the
    answer; the server stops generating at its next token.
    """

    def __init__(self, http, url, body):
        self.http = http
        self.url = url
        self.body = body
        self.queue = queue.Queue()
        self.response = None
        self.closed = False

        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        put = self.queue.put

        try:
            with self.http.stream("POST", "/v1/chat/completions", json=self.body) as response:
                self.response = response

                if response.status_code >= 400:
                    response.read()
                    put(("error", ApiError(response.status_code, _detail(response))))
                    return

                for line in response.iter_lines():
                    if self.closed:
                        return

                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()

                    if data == "[DONE]":
                        break

                    payload = json.loads(data)

                    if "error" in payload:
                        put(("error", ApiError(0, payload["error"].get("message", "error"))))
                        return

                    choice = payload["choices"][0]
                    text = (choice.get("delta") or {}).get("content")
                    extra = payload.get("zypher")

                    if text:
                        put(("text", text))

                    if choice.get("finish_reason"):
                        put(("done", {
                            "finish_reason": choice["finish_reason"],
                            "usage": payload.get("usage") or {},
                            "meta": extra or {},
                        }))
                    elif extra and "stage" in extra:
                        put(("status", extra))

        except httpx.TransportError as error:
            if not self.closed:
                put(("error", Offline(self.url) if isinstance(error, httpx.ConnectError)
                     else ApiError(0, "connection lost: {}".format(error) or "connection lost")))

        except Exception as error:  # a malformed chunk must not hang the reader
            if not self.closed:
                put(("error", ApiError(0, "{}: {}".format(type(error).__name__, error))))

        finally:
            put(("end", None))

    def __iter__(self):
        while True:
            try:
                kind, payload = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if kind == "end":
                return

            if kind == "error":
                raise payload

            yield kind, payload

    def close(self):
        self.closed = True

        if self.response is not None:
            try:
                self.response.close()
            except Exception:
                pass


class Api:

    def __init__(self, url, key=""):
        self.url = url

        # trust_env off: a system-wide HTTP proxy must not be handed requests
        # (and the key) meant for a server on this machine.
        self.http = httpx.Client(
            base_url=url,
            headers={"Authorization": "Bearer " + key} if key else {},
            timeout=httpx.Timeout(10.0, read=900.0),
            trust_env=False,
        )

    def _call(self, method, path, **kwargs):
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.ConnectError:
            raise Offline(self.url)
        except httpx.TransportError as error:
            raise ApiError(0, "{}: {}".format(type(error).__name__, error))

        if response.status_code >= 400:
            raise ApiError(response.status_code, _detail(response))

        return response.json()

    def health(self, timeout=3.0):
        return self._call("GET", "/health", timeout=timeout)

    def settings(self):
        return self._call("GET", "/v1/settings")

    def memory(self):
        return self._call("GET", "/v1/memory")

    def note(self, text):
        return self._call("POST", "/v1/memory/notes", json={"text": text})

    def rate(self, record_id, good):
        return self._call(
            "POST", "/v1/memory/{}/rate".format(record_id),
            json={"rating": "good" if good else "bad"},
        )

    def forget(self, selector):
        return self._call("DELETE", "/v1/memory", params={"selector": selector})

    def chat(self, messages, **options):
        """Start a streamed answer. options: web, memory, sampling, max_tokens, model."""

        body = {"messages": messages, "stream": True}
        body.update({key: value for key, value in options.items() if value is not None})

        return Stream(self.http, self.url, body)

    def close(self):
        self.http.close()
