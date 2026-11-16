"""
The llama.cpp engine: a GGUF model served by a private llama-server.

    gguf.py     read the model's shape from the file header
    plan.py     which GPU, and the context / layers / KV precision it affords
    runtime.py  fetch llama-server and the weights
    server.py   run llama-server as a child process
    client.py   HTTP to it

Why this rather than torch + bitsandbytes on a small card: the weights are a
4 GB download instead of 13.5 GB, they are memory-mapped straight onto the GPU
instead of being unpacked through system RAM, the KV cache can be held at
8 bits, and generation runs several times faster. This Python process holds
no tensors at all.
"""

import http.client
import threading
import time

from zypher import config

from ..text import OutOfMemory, PromptTooLong
from . import plan as planner
from . import runtime
from .client import Client, ServerError
from .server import LlamaServer

# Tokens of the context window kept free, so a prompt measured here and a
# prompt counted by the server can differ by a few without overflowing.
CONTEXT_MARGIN = 16

# The smallest answer worth starting. A prompt that leaves less than this is
# reported as too long rather than answered with a clipped sentence.
MIN_ANSWER_TOKENS = 64


class Engine:
    """Loads the model and generates. One generation at a time; the caller
    holds the lock."""

    name = "llama.cpp"

    def __init__(self):
        self.server = None
        self.client = None
        self.plan = None
        self.model_path = None
        self._start_lock = threading.Lock()
        self._best_speed = 0.0

    # -- lifecycle --------------------------------------------

    def load(self):
        self.model_path = runtime.ensure_model()
        self.plan = planner.make_plan(self.model_path)

        executable = runtime.ensure_server(self.plan["gpu"] is not None)

        self._report_plan()
        self._start(executable)

        # Half the window for the conversation, half for the answer. The
        # answer is further limited, per turn, to what the prompt leaves.
        context = self.plan["context"]
        config.MAX_PROMPT_TOKENS = context // 2
        config.MAX_TOTAL_NEW_TOKENS = context // 2

    def _report_plan(self):
        plan = self.plan

        print("=" * 60)
        print("zypherLL -- llama.cpp engine")
        print("=" * 60)
        print("Model:   {} ({:.2f} GiB)".format(plan["model"], plan["weights_gib"]))

        if plan["gpu"]:
            print("GPU:     {} ({:.1f} GiB, {:.1f} GiB in use by other programs)".format(
                plan["gpu"]["name"], plan["gpu"]["total_gib"], plan["gpu"]["used_gib"]
            ))
        else:
            print("GPU:     none found -- running on CPU")

        print("Plan:    context {} tokens, GPU layers {}, KV cache {}{}".format(
            plan["context"], plan["gpu_layers"], plan["kv_type"],
            ", ~{:.1f} GiB VRAM".format(plan["vram_gib"]) if plan["vram_gib"] else "",
        ))

    def _start(self, executable):
        self.server = LlamaServer(executable, self.model_path, self.plan)
        self.client = Client(self.server.port, self.server.key)

        print("Loading model...", end="", flush=True)
        started = time.monotonic()

        def slow():
            # Seconds when the file is in the OS cache or on an SSD; minutes
            # for a first read off a hard disk. Worth saying which this is.
            print(" (reading {:.1f} GiB from disk -- the first load after a "
                  "restart is the slow one)...".format(self.plan["weights_gib"]),
                  end="", flush=True)

        self.server.start()
        self.server.wait_ready(self.client.ready, on_slow=slow)

        print(" ready in {:.1f} s.".format(time.monotonic() - started))

    def _ensure_running(self):
        """Restart llama-server if it has died -- a driver reset, or the
        out-of-memory killer -- rather than failing every later question."""

        with self._start_lock:
            if self.server is not None and self.server.alive():
                return

            executable = self.server.executable
            self.server.stop()

            print("\n[llama-server stopped -- restarting]")
            self._start(executable)

    def reset(self):
        """Drop the cached prompt, as 'clear' asks."""

        if self.client is None:
            return

        try:
            self.client.request("POST", "/slots/0?action=erase", timeout=10)
        except Exception:
            # Only a hint: the next prompt replaces the cache either way.
            pass

    def close(self):
        if self.server is not None:
            self.server.stop()

    def asleep(self):
        """True when the idle timer has unloaded the model, so the next
        question will wait for it to load again."""

        if self.client is None or config.IDLE_UNLOAD_SECONDS <= 0:
            return False

        try:
            return bool(self.client.request("GET", "/props", timeout=5).get("is_sleeping"))
        except Exception:
            return False

    def info(self):
        plan = self.plan or {}
        gpu = plan.get("gpu") or {}

        return {
            "backend": self.name,
            "device": plan.get("device"),
            "gpu": gpu.get("name"),
            "vram_gb": round(gpu.get("total_gib", 0.0), 2),
            "context": plan.get("context"),
            "gpu_layers": plan.get("gpu_layers"),
            "kv_cache": plan.get("kv_type"),
        }

    def _throttled(self, speed, tokens):
        """The GPU's temperature if this answer was slowed by heat, else None.

        Judged against the best speed this session has seen, and only then
        confirmed with the thermometer, so an ordinary answer costs no
        nvidia-smi call. A long prompt also slows generation, but not by half.
        """

        if self.plan["device"] != "cuda" or tokens < 32 or speed <= 0:
            return None

        best, self._best_speed = self._best_speed, max(self._best_speed, speed)

        if speed >= best * config.SLOW_ANSWER_RATIO:
            return None

        temperature = planner.gpu_temperature()

        if temperature is None or temperature < config.THROTTLE_TEMP_C:
            return None

        return temperature

    # -- prompt -----------------------------------------------

    def _render(self, conversation, continuing):
        """Chat template as text. With continuing, the last message is a
        partial assistant reply: the template is applied to the turns before
        it and the partial reply appended raw, so the model resumes inside
        its own turn instead of starting a new one."""

        if continuing:
            return self.client.render(conversation[:-1]) + conversation[-1]["content"]

        return self.client.render(conversation)

    def _build_prompt(self, conversation, continuing):
        """Prompt tokens, dropping the oldest exchanges past the budget.

        The system message is always kept, and so is the turn being answered:
        the question, and with continuing the partial reply after it. A long
        reply being continued is over the prompt budget by itself, and
        dropping it -- with its question -- would leave nothing to continue.
        Turns go in user/assistant pairs, so the alternation the template
        expects stays intact.
        """

        system = conversation[:1]
        history = list(conversation[1:])
        keep = 2 if continuing else 1

        while True:
            tokens = self.client.tokenize(self._render(system + history, continuing))

            if len(tokens) <= config.MAX_PROMPT_TOKENS or len(history) <= keep:
                return tokens, system + history

            del history[:2]

    # -- generation -------------------------------------------

    def generate(self, conversation, continuing=False, on_text=None,
                 sampling=None, max_total_tokens=None, min_tokens=None):
        """Stream a reply. See zypher.model.llm for the contract.

        The prompt cache needs no handling here: llama-server compares each
        prompt with the last one it saw and evaluates only what differs, so a
        conversation resent with one more turn costs that turn alone.
        """

        self._ensure_running()

        started = time.monotonic()
        context = self.plan["context"]

        prompt, conversation = self._build_prompt(conversation, continuing)
        prompt_len = len(prompt)

        room = context - prompt_len - CONTEXT_MARGIN

        if room < MIN_ANSWER_TOKENS:
            if continuing:
                raise PromptTooLong(
                    "this answer has reached the end of the {}-token context "
                    "window and cannot be continued -- ask for the rest in a "
                    "new message".format(context)
                )

            raise PromptTooLong(
                "the prompt is {} tokens and the context window is {} -- send "
                "less text, or raise ZYPHER_CTX".format(prompt_len, context)
            )

        budget = min(max_total_tokens or config.MAX_TOTAL_NEW_TOKENS, room)
        params = dict(sampling or config.SAMPLING["balanced"])

        body = {
            "temperature": params.get("temperature", 0.6),
            "top_p": params.get("top_p", 1.0),
            "min_p": params.get("min_p", 0.0),
            # The profiles are defined by temperature, top_p and min_p alone.
            "top_k": 0,
            "repeat_penalty": config.REPETITION_PENALTY,
            "repeat_last_n": config.REPEAT_LAST_N,
            "stop": list(config.STOP_STRINGS),
            "cache_prompt": True,
        }

        chunks = []
        generated = []
        produced = 0
        cached = 0
        rounds = 0
        speed = 0.0
        stop_type = "limit"

        # min_tokens is a first round with EOS switched off, then a second
        # that carries on from the same tokens with it back on. The server's
        # cache makes the seam free.
        forced = min(min_tokens or 0, budget)
        stages = ([(forced, True)] if forced else []) + [(None, False)]

        try:
            for limit, ignore_eos in stages:
                remaining = budget - produced

                if remaining <= 0:
                    stop_type = "limit"
                    break

                request = dict(
                    body,
                    prompt=prompt + generated,
                    n_predict=limit or remaining,
                    ignore_eos=ignore_eos,
                )

                final = None
                events = self.client.stream(request)

                try:
                    for event in events:
                        generated.extend(event.get("tokens") or ())
                        text = event.get("content") or ""

                        if text:
                            chunks.append(text)

                            if on_text is not None:
                                on_text(text)

                        if event.get("stop"):
                            final = event
                finally:
                    # Also the cancel path: an on_text that raises lands
                    # here, and closing the connection stops the server at
                    # its next token.
                    events.close()

                rounds += 1

                if final is None:
                    raise ConnectionError("llama-server ended the stream early")

                timings = final.get("timings") or {}
                produced += int(final.get("tokens_predicted") or 0)
                speed = float(timings.get("predicted_per_second") or speed)
                stop_type = final.get("stop_type") or "limit"

                # How much of the prompt the server already had. Taken from
                # the first round: by the second, everything is cached.
                if rounds == 1:
                    cached = int(timings.get("cache_n") or 0)

                # The forced round ends at its limit by design; anything
                # else -- a stop string -- ends the answer.
                if not (ignore_eos and stop_type == "limit"):
                    break

        except ServerError as error:
            if "memory" in str(error).lower():
                raise OutOfMemory(str(error)) from error

            if error.status == 400:
                raise PromptTooLong(str(error)) from error

            raise

        except (OSError, http.client.HTTPException) as error:
            # The connection dropped mid-answer. If the process is gone too,
            # that was almost certainly the GPU running out; the next
            # question starts a fresh server.
            if not self.server.alive():
                raise OutOfMemory(
                    "llama-server stopped mid-answer, most likely out of "
                    "memory.\n{}".format(self.server.log_tail(8))
                ) from error

            raise

        truncated = stop_type == "limit"
        text = "".join(chunks)
        best = self._best_speed
        hot = self._throttled(speed, produced)

        return {
            # A continuation is pasted straight onto the partial reply, so
            # its leading whitespace is load-bearing: strip it and a resumed
            # code block loses the newline between two statements.
            "text": text.rstrip() if continuing else text.strip(),
            "conversation": conversation,
            "truncated": truncated,
            "finish_reason": "length" if truncated else "stop",
            "tokens": produced,
            "prompt_tokens": prompt_len,
            "cached_tokens": min(cached, prompt_len),
            "tokens_per_second": speed,
            # Set only when heat cost this answer its speed: the temperature,
            # and what the same card managed earlier.
            "throttled": (
                {"gpu_temp_c": hot, "usual_tokens_per_second": round(best, 1)}
                if hot is not None else None
            ),
            "rounds": rounds,
            "seconds": time.monotonic() - started,
        }
