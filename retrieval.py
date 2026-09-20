"""
Live web retrieval for the Zephyr chat runner.

The weights are frozen at training time, so the only way the model can answer
a question about today is to be handed today's text in its prompt. This module
does three things: decide whether a question is time-bound at all, fetch a few
search snippets when it is, and wrap them into a grounded user turn.

Retrieval is deliberately cheap and deliberately small. Every character that
goes into the prompt costs KV cache, and on a 6 GB card the cache is what runs
out first.
"""

import datetime
import re


# ============================================================
# CONFIG
# ============================================================

# How many search hits to keep. A 7B reading four snippets answers better than
# the same 7B reading twenty: past a handful it starts blending sources
# together instead of picking one.
MAX_RESULTS = 4

# Hard ceiling on the injected block. ~3000 characters is roughly 750 tokens,
# which is a fifth of the 4096-token prompt budget a small card runs with.
CONTEXT_CHAR_BUDGET = 3000

# Per-snippet cap, applied before the global one, so a single verbose result
# cannot crowd the other three out.
SNIPPET_CHAR_BUDGET = 600

# Restrict results to the last month by default: "d" (day), "w", "m", "y", or
# None for no limit. The whole point is freshness, and an undated hit from
# 2021 is exactly the failure this module exists to avoid.
TIME_LIMIT = "m"

REGION = "us-en"

# Words that make an answer depend on when it is asked.
TIME_WORDS = (
    "today", "tonight", "yesterday", "tomorrow", "now", "currently",
    "current", "latest", "recent", "recently", "this week", "this month",
    "this year", "right now", "up to date", "up-to-date", "as of",
    "news", "headline", "score", "weather", "forecast", "price",
    "stock", "exchange rate", "release date", "who won", "what happened",
    "still alive", "release", "version",
)

# Any year at or after the training cutoff is a request for something the
# weights cannot contain.
YEAR_PATTERN = re.compile(r"\b(20[2-9][0-9])\b")
CUTOFF_YEAR = 2024

# Failed searches in a row before retrieval gives up for the session. Offline,
# every attempt costs ~17s of backend retries before it fails.
MAX_CONSECUTIVE_FAILURES = 2

_failures = 0


# ============================================================
# ROUTER
# ============================================================

def needs_live_data(text):
    """True when the question is time-bound enough to be worth a search.

    Kept as string matching rather than a model call on purpose: the router
    runs before every turn, and a second forward pass to classify the question
    would cost more than the search it is trying to avoid. False negatives are
    recoverable -- the user can force a search with '/web <question>'.
    """

    lowered = text.lower()

    if any(word in lowered for word in TIME_WORDS):
        return True

    for match in YEAR_PATTERN.findall(text):
        if int(match) >= CUTOFF_YEAR:
            return True

    return False


# ============================================================
# FETCH
# ============================================================

def reset_failures():
    """Re-arm retrieval after the breaker below has tripped."""

    global _failures

    _failures = 0


def fetch_context(query, max_results=MAX_RESULTS,
                  char_budget=CONTEXT_CHAR_BUDGET, timelimit=TIME_LIMIT):
    """Search the web and return a compact, dated block of text.

    Returns None when there is nothing usable, so the caller can fall back to
    answering from the weights instead of injecting an empty CONTEXT section
    that the model would then dutifully report as "no information".
    """

    global _failures

    # With no network, ddgs works through its backend list before giving up,
    # which costs about 17 seconds -- on every question, in front of an answer
    # the weights could have given immediately. After a couple of failures in
    # a row, stop asking until something resets the count.
    if _failures >= MAX_CONSECUTIVE_FAILURES:
        return None

    try:
        from ddgs import DDGS

        hits = DDGS().text(
            query,
            region=REGION,
            timelimit=timelimit,
            max_results=max_results,
        )

    except ImportError:
        _failures = MAX_CONSECUTIVE_FAILURES
        print("\n[retrieval unavailable: pip install ddgs]")
        return None

    except Exception as error:
        # Rate limits and transient network failures are normal here. They
        # must not take the chat down with them.
        _failures += 1

        print("\n[search failed: {}: {}]".format(type(error).__name__, error))

        if _failures >= MAX_CONSECUTIVE_FAILURES:
            print("[live lookup paused -- answering offline. '/web' retries.]")

        return None

    _failures = 0

    if not hits:
        return None

    stamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    lines = ["Retrieved from the web at {}.".format(stamp), ""]
    used = 0

    for index, hit in enumerate(hits, 1):
        # Titles come back with duplicate pages' headlines concatenated,
        # which can run to several hundred characters of near-repetition.
        title = (hit.get("title") or "").strip()[:120]
        body = (hit.get("body") or "").strip()[:SNIPPET_CHAR_BUDGET]
        href = (hit.get("href") or "").strip()

        if not body:
            continue

        entry = "[{}] {}\n{}\nsource: {}\n".format(index, title, body, href)

        # Stop at a whole snippet rather than slicing one in half: a truncated
        # sentence is what the model will quote back, mid-word.
        if used + len(entry) > char_budget:
            break

        lines.append(entry)
        used += len(entry)

    if used == 0:
        return None

    return "\n".join(lines)


# ============================================================
# GROUNDING
# ============================================================

def ground(user_message, context):
    """Wrap the question in retrieved text plus instructions for using it.

    The instructions sit here rather than in the system prompt because they
    only make sense on turns that actually carry a CONTEXT block. Told on
    every turn to "answer from the context above", the model starts refusing
    ordinary questions that never had one.
    """

    today = datetime.date.today().strftime("%d %B %Y")

    # Stating the date twice, and forbidding the model's own, is not belt and
    # braces: left to itself it dates the answer from its training data and
    # reports today's prices "as of September 2021", which is worse than no
    # answer at all because it reads as sourced.
    return (
        "Today's date is {}.\n\n"
        "CONTEXT -- live search results, fetched moments ago:\n"
        "<<<\n"
        "{}\n"
        ">>>\n"
        "END CONTEXT\n\n"
        "Answer the question below using the CONTEXT. The CONTEXT is newer "
        "than your training data, so where the two disagree the CONTEXT is "
        "right. Any date you remember from training is wrong -- today is {}. "
        "Cite the sources you used by their [number]. If the CONTEXT does not "
        "answer the question, say so plainly instead of guessing.\n\n"
        "Question: {}"
    ).format(today, context, today, user_message)
