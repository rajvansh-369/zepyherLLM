"""HTTP to llama-server, with the standard library.

Requests are few and local, so there is no session to keep and no pool to
manage: one connection per call. Closing the connection of a streaming call
is how a generation is cancelled -- the server stops at its next token.
"""

import http.client
import json


class ServerError(Exception):
    """llama-server answered with an error."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Client:

    def __init__(self, port, key, timeout=600):
        self.port = port
        self.key = key
        self.timeout = timeout

    def _connect(self, timeout=None):
        return http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout=timeout or self.timeout
        )

    def _headers(self):
        return {
            "Authorization": "Bearer " + self.key,
            "Content-Type": "application/json",
        }

    def _send(self, connection, method, path, body):
        payload = None if body is None else json.dumps(body).encode("utf-8")
        connection.request(method, path, body=payload, headers=self._headers())

        return connection.getresponse()

    @staticmethod
    def _error(status, raw):
        try:
            detail = json.loads(raw).get("error", {})
            message = detail.get("message") if isinstance(detail, dict) else str(detail)
        except (ValueError, AttributeError):
            message = None

        return ServerError(status, message or raw.decode("utf-8", "replace")[:500])

    def request(self, method, path, body=None, timeout=None):
        connection = self._connect(timeout)

        try:
            response = self._send(connection, method, path, body)
            raw = response.read()
        finally:
            connection.close()

        if response.status != 200:
            raise self._error(response.status, raw)

        return json.loads(raw) if raw else {}

    def ready(self):
        """True once the model is loaded; False while it loads or before the
        port is open."""

        try:
            self.request("GET", "/health", timeout=5)
        except (OSError, ServerError, http.client.HTTPException):
            return False

        return True

    def render(self, messages):
        """The chat template applied to messages, ending where the model is
        to write."""

        return self.request("POST", "/apply-template", {"messages": messages})["prompt"]

    def tokenize(self, text):
        return self.request(
            "POST", "/tokenize", {"content": text, "add_special": True}
        )["tokens"]

    def stream(self, body):
        """POST /completion with stream on; yields each event as a dict.

        A generator: closing it (or abandoning it to an exception) closes
        the connection, which stops the generation.
        """

        connection = self._connect()

        try:
            response = self._send(connection, "POST", "/completion", dict(body, stream=True))

            if response.status != 200:
                raise self._error(response.status, response.read())

            for raw in response:
                line = raw.strip()

                if not line.startswith(b"data:"):
                    # Errors after the stream has begun arrive on their own
                    # line, as "error: {...}".
                    if line.startswith(b"error:"):
                        raise self._error(500, line[len(b"error:"):].strip())

                    continue

                data = line[len(b"data:"):].strip()

                if data == b"[DONE]":
                    return

                event = json.loads(data)

                if "error" in event:
                    raise self._error(500, data)

                yield event
        finally:
            connection.close()
