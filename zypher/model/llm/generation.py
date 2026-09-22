"""Streaming generation with KV-cache reuse, and clean-up of the finished reply."""

import re
import threading
import time

import torch
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from zypher import config
from zypher.config import (
    AUTO_CONTINUE,
    MAX_NEW_TOKENS,
    MIN_CACHE_REUSE_TOKENS,
    REPETITION_PENALTY,
    SAMPLING,
    STOP_STRINGS,
)

from .prompt import build_prompt_ids


class ClientGone(Exception):
    """Raised from an on_text callback to stop a generation nobody is reading."""


def eos_ids(tokenizer):
    """EOS is a single id on some templates and a list on others."""

    eos = tokenizer.eos_token_id

    if eos is None:
        return set()

    return set(eos) if isinstance(eos, (list, tuple)) else {eos}


class GeneratedRepetitionPenalty(LogitsProcessor):
    """repetition_penalty over the answer's own tokens only.

    transformers' built-in penalty counts the prompt as well, so every token
    of the system prompt, the history, the search snippets and the recalled
    memories was marked down. That is the opposite of what grounding needs:
    the names, figures and URLs in the CONTEXT are exactly what the answer
    should copy, and they were the tokens being pushed away. Penalizing only
    what the answer itself has produced keeps the guard against loops.
    """

    def __init__(self, penalty, start):
        self.penalty = penalty
        self.start = start

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.start:]

        if self.penalty == 1.0 or generated.shape[-1] == 0:
            return scores

        picked = torch.gather(scores, 1, generated)
        picked = torch.where(picked < 0, picked * self.penalty, picked / self.penalty)

        return scores.scatter(1, generated, picked)


class CancelOnEvent(StoppingCriteria):
    """Stops generate() at its next token once the event is set."""

    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full(
            (input_ids.shape[0],),
            self.event.is_set(),
            dtype=torch.bool,
            device=input_ids.device,
        )


class StopFilter:
    """Passes streamed text through, holding back what may become a stop string.

    generate() stops on the role markers, but the tokens that spell one are
    streamed before the match completes, so "<|user" reached the screen -- and
    the stored answer -- ahead of the stop. The tail that could still turn
    into a marker is held until it either does, and is dropped, or cannot,
    and is released.
    """

    def __init__(self, stops):
        self.stops = tuple(stops)
        self.longest = max((len(stop) for stop in self.stops), default=0)
        self.pending = ""
        self.hit = False

    def feed(self, text):
        if self.hit:
            return ""

        self.pending += text

        found = [
            index for index in (self.pending.find(stop) for stop in self.stops)
            if index != -1
        ]

        if found:
            self.hit = True
            out, self.pending = self.pending[:min(found)], ""
            return out

        hold = 0

        for size in range(min(len(self.pending), self.longest - 1), 0, -1):
            tail = self.pending[-size:]

            if any(stop.startswith(tail) for stop in self.stops):
                hold = size
                break

        cut = len(self.pending) - hold
        out, self.pending = self.pending[:cut], self.pending[cut:]

        return out

    def flush(self):
        out = "" if self.hit else self.pending
        self.pending = ""

        return out


class ContextStreamer(TextIteratorStreamer):
    """TextIteratorStreamer that never decodes a token out of context.

    The Llama/Mistral decoder strips the leading space of the first token it
    is given. TextStreamer starts its decode over after every newline, so the
    first token of each new line lost its leading space -- and in code that
    token is the indentation. Every indented line came out one space short:
    test.html, produced before this fix, has its two-space steps at one,
    three, five and seven spaces. A fresh streamer on a resumed round had the
    same fault at the seam, where it glued two words together: "return value"
    came out as "returnvalue", in code a different program.

    This one keeps the last few tokens as context instead of starting over,
    and a resumed round primes it with the tokens already shown.
    """

    CONTEXT_TOKENS = 4

    def __init__(self, tokenizer, seed=()):
        super().__init__(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        self.token_cache = list(seed)
        self.print_len = len(self._decode())

    def _decode(self):
        return self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.token_cache.extend(value.tolist())
        text = self._decode()

        # A character spread over several byte tokens decodes as U+FFFD until
        # its last byte arrives.
        if text.endswith("�"):
            return

        if len(text) > self.print_len:
            self.on_finalized_text(text[self.print_len:])

        # Re-base on the last few tokens, so each decode stays a few tokens
        # long however long the answer grows.
        self.token_cache = self.token_cache[-self.CONTEXT_TOKENS:]
        self.print_len = len(self._decode())

    def end(self):
        text = self._decode() if self.token_cache else ""
        printable = text[self.print_len:]

        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

        self.on_finalized_text(printable, stream_end=True)


def stream_once(model, generate_kwargs, streamer, emit):
    """Run one generate() on a worker thread, handing its text to emit.

    Any exception on this side -- Ctrl+C, or an API client that has gone away
    raising from emit -- stops the worker at its next token. Before, it
    carried on to max_new_tokens, which on a small card is minutes, and the
    join below waited for every one of them.
    """

    result = {}
    cancel = threading.Event()

    generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
        [CancelOnEvent(cancel)]
    )

    def run():
        try:
            with torch.inference_mode():
                result["output"] = model.generate(**generate_kwargs)
        except BaseException as error:  # re-raised on the main thread below
            result["error"] = error
            streamer.end()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()

    try:
        for chunk in streamer:
            emit(chunk)
    except BaseException:
        cancel.set()
        raise
    finally:
        worker.join()

    if "error" in result:
        raise result["error"]

    return result["output"]


def common_prefix_length(left, right):
    """How many leading token ids two 1-D tensors share."""

    span = min(left.shape[-1], right.shape[-1])

    if span == 0:
        return 0

    mismatches = (left[:span] != right[:span]).nonzero()

    return span if mismatches.numel() == 0 else int(mismatches[0].item())


def reuse_cache(cache_state, input_ids):
    """Return the stored KV cache, cropped to the part of this prompt it covers.

    Consecutive turns share everything up to the new question, so recomputing
    attention over that shared span is the largest avoidable cost in the loop --
    at a 4096-token prompt it is most of the wait before the first new token.

    The previous rule kept the cache only when the last sequence was an exact
    prefix of the new one. Anything that edits history diverges partway through
    and lost the whole cache: trimming the oldest turns does it, and so does
    rewriting a grounded turn back to the bare question once its sources have
    been read. Cropping at the point of divergence keeps everything before it.
    """

    cached_ids = cache_state.get("ids")
    cache = cache_state.get("cache")

    if cached_ids is None or cache is None:
        return None

    shared = common_prefix_length(cached_ids[0], input_ids[0])

    # At least one input token has to be left uncomputed, or generate() is
    # handed a cache covering the whole prompt and no forward pass to run.
    shared = min(shared, input_ids.shape[-1] - 1)

    if shared < MIN_CACHE_REUSE_TOKENS:
        cache_state.clear()
        return None

    # The cache holds one token fewer than the ids stored with it: the last
    # token generated is never fed back through the model.
    get_length = getattr(cache, "get_seq_length", None)
    cached = int(get_length()) if get_length is not None else cached_ids.shape[-1]

    if shared < cached:
        crop = getattr(cache, "crop", None)

        if crop is None:
            cache_state.clear()
            return None

        try:
            # A negative count removes that many tokens from the end -- the
            # form both transformers 4.x and 5.x accept. The positive "crop
            # to this length" form is deprecated and goes in 5.18.
            crop(shared - cached)
        except Exception:
            # An unfamiliar cache layout, or a sliding-window layer that has
            # already dropped the states it would have to roll back to, is
            # not worth guessing at: a full recompute is slower but always
            # correct.
            cache_state.clear()
            return None

    return cache


def generate_response(tokenizer, model, conversation, cache_state,
                      continuing=False, on_text=None, sampling="balanced",
                      max_total_tokens=None):
    """Stream a reply, reusing the KV cache from the previous turn when possible.

    Each turn's prompt is the previous turn's prompt plus the new text, so the
    attention keys/values for the shared prefix do not need recomputing. If the
    prefix does not match -- history was trimmed, or the chat was cleared --
    the cache is dropped and the prompt is processed from scratch.

    One generate() call is capped at MAX_NEW_TOKENS, but hitting that cap ends
    a call, not an answer. When generation stops there without emitting EOS it
    resumes from the cache it has just built, with the tokens it already
    produced as the input, up to MAX_TOTAL_NEW_TOKENS. Continuing the same
    token stream is what keeps long output whole: nothing is recomputed, and
    the model never sees a seam to restart or repeat itself at.

    on_text, when given, is called with each piece of text as it is decoded:
    the console in the chat loop, a response stream in an API. Raising from
    it cancels the generation. sampling is a SAMPLING profile name or a dict
    of generate() sampling arguments. max_total_tokens overrides
    MAX_TOTAL_NEW_TOKENS for this answer only.

    Nothing is printed; the caller reports. Returns a dict:

        text           the reply, with any role marker and what followed it
                       removed
        conversation   the conversation as sent, after any trimming
        truncated      True when the answer budget ran out before the answer
        finish_reason  "stop" or "length"
        tokens, prompt_tokens, rounds, seconds -- for reporting
    """

    input_ids, conversation = build_prompt_ids(tokenizer, conversation, continuing)
    input_ids = input_ids.to(model.device)

    past_key_values = reuse_cache(cache_state, input_ids)

    stop_ids = eos_ids(tokenizer)
    prompt_len = input_ids.shape[-1]
    params = SAMPLING[sampling] if isinstance(sampling, str) else dict(sampling)
    answer_budget = max_total_tokens or config.MAX_TOTAL_NEW_TOKENS

    # One processor for the whole answer. Its start stays at the original
    # prompt length, so tokens from earlier rounds still count as the answer's
    # own when a later round resumes.
    processors = LogitsProcessorList(
        [GeneratedRepetitionPenalty(REPETITION_PENALTY, prompt_len)]
    )

    stops = StopFilter(STOP_STRINGS)
    chunks = []

    def emit(chunk):
        text = stops.feed(chunk)

        if text:
            chunks.append(text)

            if on_text is not None:
                on_text(text)

    sequences = input_ids
    attention_mask = torch.ones_like(input_ids)
    produced = 0
    rounds = 0
    truncated = False
    started = time.monotonic()

    try:
        while True:
            remaining = answer_budget - produced

            if remaining <= 0:
                truncated = True
                break

            budget = min(MAX_NEW_TOKENS, remaining)

            # Anything that picks up mid-answer -- a resumed round, or the
            # 'continue' command -- has to be decoded in context.
            seed = (
                input_ids[0, -ContextStreamer.CONTEXT_TOKENS:].tolist()
                if (rounds or continuing) else ()
            )
            streamer = ContextStreamer(tokenizer, seed)

            # Only what changes per call is passed here; the fixed settings
            # were written to the generation config at load time.
            generate_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                streamer=streamer,
                max_new_tokens=budget,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=processors,
                stop_strings=list(STOP_STRINGS),
                tokenizer=tokenizer,
                return_dict_in_generate=True,
                **params
            )

            if past_key_values is not None:
                generate_kwargs["past_key_values"] = past_key_values

            output = stream_once(model, generate_kwargs, streamer, emit)

            new_tokens = output.sequences.shape[-1] - input_ids.shape[-1]
            sequences = output.sequences
            produced = sequences.shape[-1] - prompt_len
            rounds += 1

            past_key_values = getattr(output, "past_key_values", None)

            # Finished on its own: EOS, a role marker, or any other stop that
            # came before the cap.
            if (sequences[0, -1].item() in stop_ids or stops.hit
                    or new_tokens < budget):
                break

            # Out of budget, or nothing to resume from.
            if not AUTO_CONTINUE or past_key_values is None:
                truncated = True
                break

            if produced >= answer_budget:
                truncated = True
                break

            # Resume: everything generated so far becomes the input, handed
            # back with the cache that already covers it.
            input_ids = sequences
            attention_mask = torch.ones_like(input_ids)

    except BaseException:
        cache_state.clear()
        raise

    # Whatever the filter was still holding turned out not to be a marker.
    tail = stops.flush()

    if tail:
        chunks.append(tail)

        if on_text is not None:
            on_text(tail)

    elapsed = time.monotonic() - started

    # Carry the populated cache into the next turn -- but only while it is
    # still small enough to be reusable. A cache longer than the prompt budget
    # never will be, because the next turn trims history back under that budget
    # and so diverges inside it. Keeping it holds ~128 KB of VRAM per token for
    # nothing, which at 8192 tokens is a gigabyte the next answer cannot have.
    if past_key_values is not None and sequences.shape[-1] <= config.MAX_PROMPT_TOKENS:
        cache_state["cache"] = past_key_values
        cache_state["ids"] = sequences.detach()
    else:
        cache_state.clear()

    text = "".join(chunks)

    return {
        # A continuation is pasted straight onto the partial reply, so its
        # leading whitespace is load-bearing: strip it and a resumed code
        # block loses the newline and indentation between two statements.
        "text": text.rstrip() if continuing else text.strip(),
        "conversation": conversation,
        "truncated": truncated,
        "finish_reason": "length" if truncated else "stop",
        "tokens": produced,
        "prompt_tokens": prompt_len,
        "rounds": rounds,
        "seconds": elapsed,
    }


# ============================================================
# OUTPUT
# ============================================================

ROLE_LABEL = re.compile(r"^\s*(?:<\|assistant\|>|assistant|ai|zephyr)\s*:\s*", re.I)
FENCE = re.compile(r"^ {0,3}(?:```|~~~)")


def clean_reply(text, finished=True):
    """Tidy a whole reply for storage and display.

    - A role label the model sometimes opens with ("Assistant:") is dropped.
    - Runs of blank lines outside code blocks collapse to one. Inside code
      they are left alone: two blank lines between functions are the style.
    - A finished reply that left a code fence open has it closed, or every
      Markdown renderer shows the rest of the page as code. A truncated one
      is left open, so 'continue' carries on inside the block.
    """

    text = ROLE_LABEL.sub("", text, count=1)

    lines = []
    in_code = False
    blank_run = 0

    for line in text.split("\n"):
        if FENCE.match(line):
            in_code = not in_code
            blank_run = 0
        elif not in_code and not line.strip():
            blank_run += 1

            if blank_run > 1:
                continue
        else:
            blank_run = 0

        lines.append(line)

    text = "\n".join(lines).strip()

    if finished and in_code:
        text += "\n```"

    return text
