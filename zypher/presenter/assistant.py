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

import re
import threading

from zypher import config
from zypher.model import llm
from zypher.model import memory as memory_store
from zypher.model import retrieval


class ModelNotReady(Exception):
    """The model is still loading, or failed to."""


class Assistant:
    """The engine and memory store, and the lock that serialises them.

    One model on one GPU: generations run one at a time. The engine's prompt
    cache follows whichever turn ran last and is reused up to the point where
    the next prompt differs from it -- so a view resending its growing history
    reuses nearly all of it, and interleaved API clients cost a recompute,
    never a wrong answer.
    """

    def __init__(self):
        # One engine per configured model slot ("general", and "code" when
        # ZYPHER_CODE_MODEL_PATH is set). Only the selected one is loaded:
        # two 7B models do not fit a 6 GB card, and swapping per question
        # cost a minute or more on a hard disk. See switch().
        self.engines = {}
        self.active = "general"
        self.switching = None     # the slot being loaded by switch(), if any
        self.switch_error = None  # why the last switch failed, if it did
        self.store = None
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.error = None

        # The grounded text of the last question answered, so continuing a
        # truncated answer shows the model the same sources it started from
        # -- and the same prompt bytes, which keeps the whole KV cache.
        self._last_grounded = None

    # -- lifecycle --------------------------------------------

    def load(self):
        """Fetch what is missing and load the selected model. Blocks: seconds
        once everything is on disk, a download's worth the first time."""

        try:
            # Constructed first: it loads its encoder on a background thread,
            # and the model load below covers that for free.
            self.store = memory_store.Memory()

            slots = config.model_slots()

            if not llm.can_swap():
                slots = {"general": slots["general"]}

            self.engines = {slot: llm.create_engine(spec) for slot, spec in slots.items()}

            chosen = config.selected_model()
            self.active = chosen if chosen in self.engines else "general"
            self.engines[self.active].load()

            self.ready.set()

        except BaseException as error:
            self.error = error
            raise

    @property
    def engine(self):
        """The engine of the model loaded now."""

        return self.engines.get(self.active)

    def models(self):
        """[{"slot", "id", "active"}] for each configured model."""

        loaded = self.engines or {"general": None}

        return [
            {"slot": slot, "id": spec["id"], "active": slot == self.active}
            for slot, spec in config.model_slots().items()
            if slot in loaded
        ]

    def switch(self, slot, save=True):
        """Unload the current model and load slot's in its place. Blocks for
        the load; run it off the event loop. With save, the choice is kept,
        so the next start loads the same model.

        Waits for an answer in progress to finish first. If the new model
        will not load, the previous one is loaded back and switch_error says
        why.
        """

        if slot not in self.engines:
            raise ValueError("no model {!r} is configured".format(slot))

        with self.lock:
            if slot == self.active:
                self.switching = None
                return

            previous = self.active
            self.switching = slot
            self.switch_error = None

            try:
                self.engines[previous].stop()

                try:
                    self.engines[slot].load()
                except Exception as error:
                    print("[could not load the {} model: {}; loading {} back]".format(
                        slot, error, previous))
                    self.switch_error = "{}: {}".format(type(error).__name__, error)
                    self.engines[slot].stop()
                    self.engines[previous].start()
                    return

                self.active = slot
                self._last_grounded = None
                if save:
                    config.save_selected_model(slot)

            finally:
                self.switching = None

    def _slot_named(self, model):
        """The configured slot a request's model field names, or None."""

        name = (model or "").strip().lower()

        for slot, spec in config.model_slots().items():
            if slot in self.engines and name in (slot, spec["id"].lower()):
                return slot

        return None

    def status(self):
        if self.error is not None:
            return "error"

        if self.switching is not None:
            return "loading"

        return "ready" if self.ready.is_set() else "loading"

    def reset(self):
        """Forget the cached prompt and the pending sources, as 'clear' does."""

        self._last_grounded = None

        if self.engine is not None:
            self.engine.reset()

    def close(self):
        """Release the model. The views call this on the way out."""

        for engine in self.engines.values():
            engine.close()

    def engine_info(self):
        """Backend, device and context window, or {} before the load."""

        return self.engine.info() if self.engine is not None else {}

    # -- the turn ---------------------------------------------

    def answer(self, messages, web="auto", memory=None, sampling=None,
               temperature=None, top_p=None, max_tokens=None, min_tokens=None,
               on_event=None, model=None):
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
        min_tokens: no end of turn before this many tokens, capped at the
            answer budget.
        model: a slot ("general", "code") or model id this request needs
            -- the zy CLI asks for "code". Anything else (None, "auto", an
            unknown name) means the model chosen in the settings, so the
            web chat stays on that choice even after zy borrowed the GPU.
            Whichever it is, it is loaded first if it is not on the GPU.
        on_event(kind, payload): "status" with a dict before generation
            ({"stage": "searching"}, {"stage": "recalled", "count": n}, or
            {"stage": "waking"} when the model has to be loaded back in),
            then "text" with each decoded piece. Raising from it -- ClientGone
            for a vanished reader -- cancels the generation.

        Returns a dict: text, finish_reason, usage, and meta -- sampling,
        live, sources, cited, recalled, notes_captured, memory_id, seconds,
        tokens_per_second, throttled, rounds, history_kept.
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
        about = None

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
                earlier = [m["content"] for m in history[:-1] if m["role"] == "user"]
                hits = store.recall(_recall_query(question, earlier), visible=earlier)
                recalled = len(hits)
                grounded = memory_store.ground(grounded, store.block(hits))

            notes = store.profile()
            about = store.about()

        if recalled:
            event("status", {"stage": "recalled", "count": recalled})

        # Rebuilt every turn, for the date and for notes captured since. The
        # same text as last turn keeps the whole KV cache.
        system = llm.system_prompt(notes, about)

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

        # An answer about the present goes stale. Recalled later as "what you
        # told this user", it would put last month's price into today's
        # answer, so such exchanges are never stored.
        live = bool(context) or web is True or retrieval.needs_live_data(question)

        wanted = self._slot_named(model) or config.selected_model()

        if wanted in self.engines and wanted != self.active:
            event("status", {"stage": "switching", "model": config.model_slots()[wanted]["id"]})
            # Not saved: the choice in the settings stays what loads at start.
            self.switch(wanted, save=False)

        with self.lock:
            # Set by the model just loaded: its context decides the budget.
            budget = min(max_tokens or config.MAX_TOTAL_NEW_TOKENS, config.MAX_TOTAL_NEW_TOKENS)

            # Unloaded for being idle: this answer starts with the model
            # being read back in, and the reader should know why it waits.
            if self.engine.asleep():
                event("status", {"stage": "waking"})

            try:
                reply = self.engine.generate(
                    conversation,
                    continuing=continuing,
                    on_text=lambda text: event("text", text),
                    sampling=params,
                    max_total_tokens=budget,
                    min_tokens=min_tokens,
                )
            except llm.OutOfMemory:
                self.engine.reset()
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
                # How much of the prompt came out of the cache rather than
                # being read again -- the OpenAI field for it.
                "prompt_tokens_details": {"cached_tokens": reply["cached_tokens"]},
            },
            "meta": {
                "model": self.engine.spec["id"] if hasattr(self.engine, "spec") else config.MODEL_ID,
                "sampling": profile,
                "live": live,
                "sources": cited or list(sources),
                "cited": bool(cited),
                "recalled": recalled,
                "notes_captured": captured,
                "memory_id": memory_id,
                "seconds": round(reply["seconds"], 3),
                "tokens_per_second": round(reply["tokens_per_second"], 1),
                # None, or the GPU temperature and usual speed when heat
                # slowed this answer.
                "throttled": reply["throttled"],
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

    def about(self):
        return self._memory().about()

    def set_about(self, fields):
        return self._memory().set_about(fields)

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


# Openers that make a message a follow-up: its subject is in the turn before.
FOLLOW_UP = re.compile(
    r"^(?:and|but|also|so|then|what about|how about|why|why not|it|that|this|"
    r"those|these|same|ok|okay|now|more|another|instead)\b",
    re.I,
)


def _recall_query(question, earlier):
    """What to search memory with for this question.

    A follow-up -- "and how do I test it?" -- names nothing. Searched alone it
    matches whatever else is about testing, or nothing, so a short or
    referring question is searched together with the question before it.
    """

    if not earlier:
        return question

    if len(question.split()) >= 12 and not FOLLOW_UP.match(question.strip()):
        return question

    return "{}\n{}".format(earlier[-1][-600:], question)


def _cited(sources, answer):
    """The sources an answer cites by [n]."""

    return [s for s in sources if "[{}]".format(s["n"]) in answer]
