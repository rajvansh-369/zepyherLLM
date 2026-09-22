"""
One turn, start to finish.

Both views -- the console and the HTTP API -- hand this a conversation and get
back an answer. Everything in between happens here, once:

    1. decide whether the question needs the web, and fetch it
    2. recall similar past exchanges and the user's standing notes
    3. wrap the question in both, and build the system prompt
    4. pick sampling, generate, clean up
    5. learn: capture notes, store the finished exchange

The conversation a view passes in holds bare turns only. The grounded version
of the question -- search results and recalled memories wrapped round it --
exists for the one generation that reads it and is never stored in history.
Left there, it would re-spend a large part of the prompt budget on every later
turn re-showing text the model has already used.
"""

import threading

import torch

from zypher import config
from zypher.model import llm
from zypher.model import memory as memory_store
from zypher.model import retrieval


class ModelNotReady(Exception):
    """The model is still loading, or failed to."""


class Assistant:
    """The loaded model and memory store, and the lock that serialises them.

    One model on one GPU: generations run one at a time. The KV cache follows
    whichever turn ran last, and reuse_cache() crops it to the prefix the next
    prompt shares with it -- so a view resending its growing history reuses
    nearly all of it, and interleaved API clients cost a recompute, never a
    wrong answer.
    """

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.store = None
        self.cache_state = {}
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.error = None

        # The grounded text of the last question answered, so continuing a
        # truncated answer shows the model the same sources it started from
        # -- and the same prompt bytes, which keeps the whole KV cache.
        self._last_grounded = None

    # -- lifecycle --------------------------------------------

    def load(self):
        """Download if needed, load and warm up. Blocks for about a minute."""

        try:
            # Constructed first: it loads its encoder on a background thread,
            # and the model load below covers that for free.
            self.store = memory_store.Memory()

            if not llm.download_model():
                raise RuntimeError("model download or verification failed")

            self.tokenizer = llm.load_tokenizer()
            self.model = llm.load_model()

            if config.WARMUP:
                llm.warm_up(self.tokenizer, self.model)

            self.ready.set()

        except BaseException as error:
            self.error = error
            raise

    def status(self):
        if self.error is not None:
            return "error"

        return "ready" if self.ready.is_set() else "loading"

    def reset(self):
        """Forget the KV cache and the pending sources, as 'clear' does."""

        self.cache_state.clear()
        self._last_grounded = None

        if llm.DEVICE == "cuda":
            torch.cuda.empty_cache()

    # -- the turn ---------------------------------------------

    def answer(self, messages, web="auto", memory=None, sampling=None,
               temperature=None, top_p=None, max_tokens=None, on_event=None):
        """Answer a conversation. Blocking; call it off the event loop.

        messages: [{"role": "system"|"user"|"assistant", "content": str}].
            The last non-system message is either the question, or a partial
            assistant reply to continue in place.
        web: "auto" lets the router decide, True forces a search, False
            never searches.
        memory: recall and learn on this turn; None means the server default.
        sampling: a SAMPLING profile name, or None to pick per question.
        temperature, top_p: override the profile's values.
        max_tokens: answer budget, capped at config.MAX_TOTAL_NEW_TOKENS.
        on_event(kind, payload): "status" with a dict before generation
            ({"stage": "searching"} or {"stage": "recalled", "count": n}),
            then "text" with each decoded piece. Raising from it -- ClientGone
            for a vanished reader -- cancels the generation.

        Returns a dict: text, finish_reason, usage, and meta -- sampling,
        live, sources, cited, recalled, notes_captured, memory_id, seconds,
        rounds, history_kept.
        """

        if self.error is not None:
            raise ModelNotReady("model failed to load: {}".format(self.error))

        if not self.ready.is_set():
            raise ModelNotReady("model is still loading")

        def event(kind, payload):
            if on_event is not None:
                on_event(kind, payload)

        store = self.store
        use_memory = (config.MEMORY_ENABLED if memory is None else memory) and store is not None

        system_extra, history = _split_system(messages)

        if not history:
            raise ValueError("messages needs at least one user message")

        continuing = history[-1]["role"] == "assistant"

        question = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"), ""
        )

        if not continuing and not question.strip():
            raise ValueError("the last user message is empty")

        # -- 1-3: ground the question --------------------------

        context = None
        sources = []
        recalled = 0
        notes = ()

        if continuing:
            # The same sources as the answer being continued, if it is the
            # one this process just gave.
            previous = self._last_grounded
            grounded = previous[1] if previous and previous[0] == question else question
            context = previous[2] if previous and previous[0] == question else None
            sources = previous[3] if previous and previous[0] == question else []
        else:
            if web is True or (web == "auto" and config.RETRIEVAL_ENABLED
                               and retrieval.needs_live_data(question)):
                event("status", {"stage": "searching"})
                context, sources = retrieval.fetch_context(question)

            grounded = retrieval.ground(question, context) if context else question

        if use_memory:
            # Only ever waits on the first question: the encoder loads in the
            # background while the model does.
            store.ready(15)

            if not continuing:
                hits = store.recall(question)
                recalled = len(hits)
                grounded = memory_store.ground(grounded, store.block(hits))

            notes = store.profile()

        if recalled:
            event("status", {"stage": "recalled", "count": recalled})

        # Rebuilt every turn, for the date and for notes captured since. The
        # same text as last turn keeps the whole KV cache.
        system = llm.system_prompt(notes)

        if system_extra:
            system += "\n\n" + system_extra

        conversation = [{"role": "system", "content": system}]
        conversation += [dict(m) for m in history]

        # The question sits last, or just before the partial answer.
        question_at = len(conversation) - (2 if continuing else 1)

        if conversation[question_at]["role"] == "user":
            conversation[question_at]["content"] = grounded

        # -- 4: generate ----------------------------------------

        profile = sampling or llm.pick_sampling(question, grounded=bool(context))
        params = dict(config.SAMPLING[profile])

        if temperature is not None:
            params["temperature"] = temperature

        if top_p is not None:
            params["top_p"] = top_p

        # generate() rejects a zero temperature with sampling on; zero means
        # greedy.
        if params["temperature"] <= 0:
            params = {"do_sample": False}

        budget = min(max_tokens or config.MAX_TOTAL_NEW_TOKENS, config.MAX_TOTAL_NEW_TOKENS)

        # An answer about the present goes stale. Recalled later as "what you
        # told this user", it would put last month's price into today's
        # answer, so such exchanges are never stored.
        live = bool(context) or web is True or retrieval.needs_live_data(question)

        with self.lock:
            try:
                reply = llm.generate_response(
                    self.tokenizer,
                    self.model,
                    conversation,
                    self.cache_state,
                    continuing=continuing,
                    on_text=lambda text: event("text", text),
                    sampling=params,
                    max_total_tokens=budget,
                )
            except torch.cuda.OutOfMemoryError:
                self.cache_state.clear()
                torch.cuda.empty_cache()
                raise

        finished = not reply["truncated"]

        if continuing:
            text = reply["text"]
            full_answer = llm.clean_reply(history[-1]["content"] + text, finished)
        else:
            text = llm.clean_reply(reply["text"], finished)
            full_answer = text

        self._last_grounded = (
            None if finished else (question, grounded, context, sources)
        )

        # -- 5: learn --------------------------------------------

        captured = []
        memory_id = None

        if use_memory:
            try:
                # A statement about the user is worth keeping however the
                # reply turned out, so capture is not gated on it finishing.
                if not continuing:
                    captured = [record["a"] for record in store.capture_notes(question)]

                # Half an answer is not worth being reminded of later.
                if full_answer and finished and not live and question:
                    record = store.remember(question, full_answer)
                    memory_id = record["id"] if record else None

            except Exception as error:  # a memory failure must not fail the answer
                print("[memory: {}: {}]".format(type(error).__name__, error))

        cited = _cited(sources, full_answer)

        return {
            "text": text,
            "finish_reason": reply["finish_reason"],
            "usage": {
                "prompt_tokens": reply["prompt_tokens"],
                "completion_tokens": reply["tokens"],
                "total_tokens": reply["prompt_tokens"] + reply["tokens"],
            },
            "meta": {
                "sampling": profile,
                "live": live,
                "sources": cited or list(sources),
                "cited": bool(cited),
                "recalled": recalled,
                "notes_captured": captured,
                "memory_id": memory_id,
                "seconds": round(reply["seconds"], 3),
                "rounds": reply["rounds"],
                # Turns (excluding system) that still fit the prompt budget. A
                # view keeping its own history can drop the rest: they will
                # never be sent again, and re-encoding them each turn only
                # costs time.
                "history_kept": len(reply["conversation"]) - 1,
            },
        }

    # -- memory -----------------------------------------------

    def _memory(self):
        if self.store is None:
            raise ModelNotReady("memory is not initialised yet")

        self.store.ready(30)

        return self.store

    def memory_stats(self):
        if self.store is None:
            return None

        stats = dict(self.store.stats())
        error = stats.pop("error")
        stats["error"] = None if error is None else "{}: {}".format(type(error).__name__, error)

        return stats

    def profile(self):
        return self._memory().profile()

    def records(self, kind=None):
        """Stored records, newest first."""

        return [
            record for record in reversed(self._memory().records)
            if kind is None or record.get("kind", "exchange") == kind
        ]

    def note(self, text):
        """Keep a fact about the user. Returns the record, or None."""

        return self._memory().note(text)

    def rate(self, record_id, good):
        """Promote or demote a stored record. Returns it, or None if unknown."""

        store = self._memory()
        record = next((r for r in store.records if r.get("id") == record_id), None)

        if record is None:
            return None

        return store.rate(1 if good else -1, record)

    def forget(self, selector="last"):
        """Drop memories: 'last', 'all', or free text. Returns how many."""

        return self._memory().forget(selector)


def _split_system(messages):
    """Client system text, and the user/assistant turns in order.

    The runner's own system prompt always leads -- it carries the date, the
    formatting rules and the user's notes. A client's system message is
    appended to it rather than replacing it.
    """

    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    history = [m for m in messages if m["role"] != "system"]

    return system, history


def _cited(sources, answer):
    """The sources an answer cites by [n]."""

    return [s for s in sources if "[{}]".format(s["n"]) in answer]
