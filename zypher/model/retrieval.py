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
from urllib.parse import urlparse

from zypher.config import (
    CONTEXT_CHAR_BUDGET,
    FETCH_RESULTS,
    MAX_RESULTS,
    REGION,
    SNIPPET_CHAR_BUDGET,
    TIME_LIMIT,
)


# ============================================================
# ROUTING PATTERNS
# ============================================================

# Phrases that make an answer depend on when it is asked. Matched as whole
# words: as bare substrings "now" fired inside "know", "score" inside
# "underscore", "news" inside "newsletter" and "as of" inside "has often", so
# most ordinary questions went to the web and came back with a CONTEXT block
# telling the model to answer from search results that were about nothing.
TIME_PATTERN = re.compile(
    r"\b(?:"
    r"today|tonight|yesterday|tomorrow|right now|as of (?:now|today)|"
    r"currently|latest|newest|recent|recently|upcoming|"
    r"this (?:week|month|year|season)|last (?:night|week|month)|"
    r"up[- ]to[- ]date|what(?:'s| is) new|news|headlines?|breaking|"
    r"weather|forecast|"
    r"prices?|stocks|stock (?:price|market)s?|share prices?|"
    r"exchange rates?|market cap|"
    r"release date|released|launched|new version|"
    r"who won|election|scores?|standings|"
    r"what happened|still alive"
    r")\b"
    # "current" on its own is mostly programming -- "current directory",
    # "current user" -- and time-bound only in front of a role or a quantity.
    r"|\bcurrent (?:\w+ )?(?:president|ceo|prime minister|leader|champion|"
    r"price|rate|version|status|events|affairs|situation|weather|record)s?\b"
    # A trailing "now?" asks about the present; "now add tests" does not.
    r"|\bnow\W*$",
    re.I,
)

# A request to produce or change something is answered from the model's own
# skill, not from the news, even when it mentions "today" or "latest": "write
# a function that returns today's date" needs no search. Only the start of the
# message is checked, because that is where the instruction sits.
TASK_PATTERN = re.compile(
    r"^\W*(?:(?:hey|hi|ok|okay|so|now|also|then|and|please|pls|kindly)\W+)*"
    r"(?:(?:can|could|would|will) you\s+(?:please\s+)?|"
    r"i (?:want|need) you to\s+|help me\s+(?:to\s+)?)?"
    r"(?:write|create|build|generate|draft|compose|design|implement|code|"
    r"refactor|rewrite|fix|debug|convert|translate|format|optimi[sz]e|"
    r"add|modify|change|edit|remove|rename|explain this|summari[sz]e this)\b",
    re.I,
)

# Pasted code is a question about that code.
CODE_PATTERN = re.compile(
    r"```|^\s*(?:def |class |import |from \S+ import |function |const |let |"
    r"var |public |private |#include|<[a-z!/][^>]*>)",
    re.I | re.M,
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

    Kept as pattern matching rather than a model call on purpose: the router
    runs before every turn, and a second forward pass to classify the question
    would cost more than the search it is trying to avoid. False negatives are
    recoverable -- the user can force a search with '/web <question>'.
    """

    if CODE_PATTERN.search(text) or TASK_PATTERN.search(text):
        return False

    if TIME_PATTERN.search(text):
        return True

    return any(int(year) >= CUTOFF_YEAR for year in YEAR_PATTERN.findall(text))


def _time_limit_for(query):
    """The freshness filter to search with.

    A question about a named past year -- "who won the 2024 election" -- is
    answered by pages from that year, which a last-month filter hides.
    """

    years = [int(year) for year in YEAR_PATTERN.findall(query)]

    if years and max(years) < datetime.date.today().year:
        return None

    return TIME_LIMIT


# ============================================================
# FETCH
# ============================================================

def reset_failures():
    """Re-arm retrieval after the breaker below has tripped."""

    global _failures

    _failures = 0


def _search(query, timelimit):
    """Raw hits from the backend; [] when the search found nothing.

    ddgs raises DDGSException("No results found.") for an empty result set.
    That is an answer, not an outage -- counted as a failure, two niche
    questions in a row would trip the breaker and pause lookups for the whole
    session on a working network.
    """

    from ddgs import DDGS
    from ddgs.exceptions import DDGSException

    try:
        return DDGS().text(
            query,
            region=REGION,
            timelimit=timelimit,
            max_results=FETCH_RESULTS,
        ) or []
    except DDGSException as error:
        if "no results" in str(error).lower():
            return []
        raise


def _clip(text, limit):
    """Shorten to at most limit characters, ending on a whole sentence.

    Falls back to a whole word when no sentence ends in the back half of the
    budget. A snippet cut mid-word is what the model then quotes back,
    mid-word.
    """

    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))

    if end >= limit // 2:
        return cut[:end + 1]

    space = cut.rfind(" ")

    return (cut[:space] if space > 0 else cut).rstrip(" ,;:-") + " ..."


def _domain(url):
    host = urlparse(url).netloc.lower()

    return host[4:] if host.startswith("www.") else host


def fetch_context(query, max_results=MAX_RESULTS,
                  char_budget=CONTEXT_CHAR_BUDGET, timelimit=None):
    """Search the web and return (block, sources).

    block is a compact, dated, numbered text for the prompt. sources is the
    matching list of {"n", "title", "url"}, so a [2] in the answer can be
    shown to the user as a link they can check.

    Returns (None, []) when there is nothing usable, so the caller can fall
    back to answering from the weights instead of injecting an empty CONTEXT
    section that the model would then dutifully report as "no information".
    """

    global _failures

    # With no network, ddgs works through its backend list before giving up,
    # which costs about 17 seconds -- on every question, in front of an answer
    # the weights could have given immediately. After a couple of failures in
    # a row, stop asking until something resets the count.
    if _failures >= MAX_CONSECUTIVE_FAILURES:
        return None, []

    if timelimit is None:
        timelimit = _time_limit_for(query)

    try:
        hits = _search(query, timelimit)

        # A freshness filter that leaves nothing is worse than none: current
        # but niche questions ("latest version of X") often have no page from
        # the last month at all.
        if not hits and timelimit is not None:
            hits = _search(query, None)

    except ImportError:
        _failures = MAX_CONSECUTIVE_FAILURES
        print("\n[retrieval unavailable: pip install ddgs]")
        return None, []

    except Exception as error:
        # Rate limits and transient network failures are normal here. They
        # must not take the chat down with them.
        _failures += 1

        print("\n[search failed: {}: {}]".format(type(error).__name__, error))

        if _failures >= MAX_CONSECUTIVE_FAILURES:
            print("[live lookup paused -- answering offline. '/web' retries.]")

        return None, []

    _failures = 0

    stamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    lines = ["Retrieved from the web at {}.".format(stamp), ""]
    sources = []
    seen_urls = set()
    seen_bodies = set()
    used = 0

    for hit in hits:
        if len(sources) >= max_results:
            break

        # Titles come back with duplicate pages' headlines concatenated,
        # which can run to several hundred characters of near-repetition.
        title = _clip(hit.get("title") or "", 120)
        href = (hit.get("href") or "").strip()

        # Snippets are cut from the middle of a page and can open on the
        # tail of a clause (", Python 3.14.6 is ...").
        body = _clip(hit.get("body") or "", SNIPPET_CHAR_BUDGET).lstrip(",;: ")

        # The same page, or the same text syndicated under two URLs, is one
        # source. Shown twice, a 7B reads it as two sources agreeing.
        url_key = href.rstrip("/").lower()
        body_key = body[:80].lower()

        if not body or body_key in seen_bodies or (url_key and url_key in seen_urls):
            continue

        number = len(sources) + 1

        entry = "[{}] {}\n{}\nsource: {}\n".format(
            number, title or _domain(href), body, href
        )

        # Stop at a whole snippet rather than slicing one in half.
        if used + len(entry) > char_budget:
            break

        seen_urls.add(url_key)
        seen_bodies.add(body_key)
        lines.append(entry)
        sources.append({"n": number, "title": title or _domain(href), "url": href})
        used += len(entry)

    if not sources:
        return None, []

    return "\n".join(lines), sources


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
    #
    # The numbers-and-names rule is the one a 7B breaks most: it reads a
    # price or a date in a snippet and writes a nearby, plausible, invented
    # one. Stating it as its own line is what makes it stick.
    return (
        "Today's date is {}.\n\n"
        "CONTEXT -- live search results, fetched moments ago:\n"
        "<<<\n"
        "{}\n"
        ">>>\n"
        "END CONTEXT\n\n"
        "Answer the question below using the CONTEXT. It is newer than your "
        "training data, so where the two disagree the CONTEXT is right, and "
        "any date you remember from training is wrong -- today is {}.\n"
        "- Put the direct answer first, then the supporting detail.\n"
        "- Cite each fact you take from the CONTEXT with its number, like "
        "[1] or [2][3].\n"
        "- Copy numbers, names and dates exactly as the CONTEXT gives them. "
        "Never invent one it does not contain.\n"
        "- If the sources disagree, say so and give each figure with its "
        "source.\n"
        "- If the CONTEXT does not answer the question, say so plainly in "
        "one sentence instead of guessing.\n\n"
        "Question: {}"
    ).format(today, context, today, user_message)
