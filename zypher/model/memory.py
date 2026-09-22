"""
Persistent semantic memory for the Zephyr chat runner.

The weights never change, so "learning" here means the runner accumulates its
own history and hands the relevant parts back to the model as context. A
question asked today is embedded, matched against every exchange the model has
already had, and the closest ones are injected into the prompt the same way a
web snippet is.

Three things make it a feedback loop rather than a log:

  * Every finished exchange is stored automatically.
  * The user rates an answer ('/good', '/bad'). A bad answer is demoted out of
    recall, so the model stops being reminded of its own mistakes; a good one
    is promoted and surfaces earlier.
  * Facts the user states about themselves are captured without a rating, so
    preferences survive 'clear' and restarts. They form a short profile that
    goes into the system prompt on every turn rather than being recalled by
    similarity: a preference such as "I always use 4-space indentation" has to
    apply to "write a function that merges two lists", which it barely
    resembles.

Everything is CPU-side and additive: embedding a turn costs milliseconds and
no VRAM, which matters because on a small card the KV cache is what runs out
first.
"""

import datetime
import json
import os
import re
import threading
import zlib

import numpy as np

from zypher.config import (
    ANSWER_CHAR_BUDGET,
    DEDUPE_THRESHOLD,
    EMBED_MODEL,
    MEMORY_DIR,
    META_FILE,
    PROFILE_CHAR_BUDGET,
    PROFILE_MAX_NOTES,
    RECALL_CANDIDATES,
    RECALL_CHAR_BUDGET,
    RECALL_KEEP,
    RECALL_THRESHOLD,
    RECORDS_FILE,
    SCORE_CLAMP,
    SCORE_WEIGHT,
    VECTORS_FILE,
)


# ============================================================
# NOTE CAPTURE
# ============================================================

# Sentences that state something durable about the user. Captured as notes so
# they survive 'clear', which a plain exchange does not: exchanges are only
# recalled when a later question resembles them, whereas a preference should
# apply to everything.
#
# Every pattern is anchored to the start of the sentence. Matched anywhere,
# "i am a" caught "so I am a bit lost", "i use" caught "I use this function
# but it throws", and a bare "don't" caught "I don't understand" -- and each of
# those then sat in the prompt of every later turn as a fact about the user.
#
# Explicit statements are kept as they are. The inferred ones -- a sentence
# that merely sounds like self-description -- are also run past NOTE_REJECT.
NOTE_EXPLICIT = (
    re.compile(r"^(?:please )?remember(?: that|:)? (?!to\b)", re.I),
    re.compile(r"^my (?:name|job|role|title|time ?zone|os|stack|editor|ide|setup) is\b", re.I),
    re.compile(r"^(?:you can |please )?call me\b", re.I),
    re.compile(r"^(?:from now on|going forward|in (?:the )?future)\b", re.I),
    re.compile(r"^(?:please )?(?:always|never) (?!mind\b)\w+", re.I),
)

NOTE_INFERRED = (
    re.compile(r"^i(?:'m| am) (?:a|an) \w+", re.I),
    re.compile(r"^i (?:work|live|study) (?:as|at|in|on|for)\b", re.I),
    re.compile(r"^i(?:'m| am) (?:working|building) (?:on|with)\b", re.I),
    re.compile(r"^i (?:prefer|like|love|hate|dislike)\b", re.I),
    re.compile(r"^i (?:always|usually|normally|mostly|mainly) \w+", re.I),
    re.compile(r"^i (?:use|code in|program in|write in) \w+", re.I),
)

# Signs that an inferred sentence is about the problem in front of the user
# rather than about the user: it points at something ("this", "it"), or it is
# a complaint or a state of mind.
NOTE_REJECT = re.compile(
    r"\b(?:this|these|those|it|here|above|below|"
    r"errors?|bugs?|issues?|problems?|exception|traceback|crash\w*|fail\w*|"
    r"broken|wrong|confused|lost|stuck|sorry|unsure|trying|getting|seeing|"
    r"bit|little)\b",
    re.I,
)

# Notes that say who the user is. profile() keeps these ahead of newer ones.
IDENTITY = re.compile(r"^(?:my name is|(?:you can |please )?call me)\b", re.I)

# Openers stripped before matching, so "Also, I prefer tabs" and "Hi, my name
# is Sam" are seen from where the statement starts.
NOTE_LEAD = re.compile(
    r"^(?:(?:hi|hello|hey|also|and|btw|by the way|oh|ok|okay|so|but|fyi|"
    r"just so you know|note)\b[\s,:;!-]*)+",
    re.I,
)

# Code is quoted, not said: a comment reading "# never call this twice" is not
# the user stating a preference.
CODE_BLOCK = re.compile(r"```.*?(?:```|\Z)|^(?: {4}|\t)[^\n]*", re.S | re.M)

NOTE_MAX_CHARS = 200

# At most this many notes are taken from one message; a message that yields
# more is a document being pasted, not a user describing themselves.
NOTES_PER_MESSAGE = 3


# ============================================================
# EMBEDDING
# ============================================================

class _HashingEmbedder:
    """Deterministic fallback used when sentence-transformers is unavailable.

    Hashes word and character 4-grams into a fixed vector. It is weaker than a
    trained encoder at matching paraphrases, but it needs no download and no
    model load, so memory still works on a machine that is offline the first
    time it runs.

    The bucket comes from crc32, not hash(). Python salts str hashes per
    process, so vectors written by one run landed in different buckets from
    the queries of the next, and every restart silently turned recall into
    noise. The name carries a version so indexes written that way are rebuilt.
    """

    name = "hashing-512-v2"
    dim = 512

    _token = re.compile(r"[a-z0-9]+")

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)

        for row, text in enumerate(texts):
            lowered = text.lower()
            words = self._token.findall(lowered)

            grams = list(words)
            grams += [lowered[i:i + 4] for i in range(len(lowered) - 3)]

            for gram in grams:
                out[row, zlib.crc32(gram.encode("utf-8")) % self.dim] += 1.0

        return _normalize(out)


class _SentenceEmbedder:

    def __init__(self, model):
        self._model = model
        self.name = EMBED_MODEL
        self.dim = int(model.get_sentence_embedding_dimension())

    def encode(self, texts):
        vectors = self._model.encode(
            texts,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(vectors, dtype=np.float32)


def _normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return matrix / norms


def _build_embedder():
    # Not just ImportError: a torchaudio left behind by an older torch fails
    # to load its DLL with an OSError from inside the import, and that used to
    # escape to _load and skip reading the history altogether.
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as error:
        if not isinstance(error, ImportError):
            print("\n[memory: sentence-transformers failed to import ({}: {}); "
                  "using hashed embeddings]".format(type(error).__name__, error))

        return _HashingEmbedder()

    try:
        model = SentenceTransformer(EMBED_MODEL, device="cpu")
    except Exception as error:
        print("\n[memory: falling back to hashed embeddings: {}: {}]".format(
            type(error).__name__, error
        ))
        return _HashingEmbedder()

    return _SentenceEmbedder(model)


# ============================================================
# STORE
# ============================================================

class Memory:
    """Append-only record store with an in-process vector index.

    Records and vectors are kept row-aligned: record i is described by row i of
    the matrix. Deletes rewrite both. At personal-chat scale -- thousands of
    rows, not millions -- a dense matmul over the whole matrix is faster than
    any index would be, and it removes a dependency and a file format that can
    drift out of sync with the records.
    """

    def __init__(self, directory=MEMORY_DIR):
        self.dir = directory
        self.enabled = True
        self.records = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.last_recall = []

        self._embedder = None
        self._lock = threading.Lock()
        self._loaded = threading.Event()
        self._error = None

        os.makedirs(self.dir, exist_ok=True)

        # Loading the encoder takes a second or two and is not needed until
        # the first question, which is always after a multi-second model load.
        # Doing it here in the background makes it free.
        threading.Thread(target=self._load, daemon=True).start()

    # -- paths ------------------------------------------------

    def _path(self, name):
        return os.path.join(self.dir, name)

    # -- loading ----------------------------------------------

    def _load(self):
        try:
            try:
                self._embedder = _build_embedder()
            except Exception as error:  # never take the chat down with it
                self._error = error
                self._embedder = _HashingEmbedder()

            try:
                self._read_records()
                self._read_vectors()
            except Exception as error:
                # The records on disk could not be read. Every write path
                # rewrites the whole file from self.records, so carrying on
                # would replace the user's history with this session's few
                # rows. Nothing is written until the file can be read again.
                self._error = error
                self.enabled = False
        finally:
            self._loaded.set()

    def _read_records(self):
        path = self._path(RECORDS_FILE)

        if not os.path.exists(path):
            return

        records = []

        # A stray non-UTF-8 byte fails the iterator, not json.loads, and so
        # would lose every line rather than just its own.
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()

                if not line:
                    continue

                try:
                    records.append(json.loads(line))
                except ValueError:
                    # One corrupt line must not cost the whole history.
                    continue

        self.records = records

    def _read_vectors(self):
        meta_path = self._path(META_FILE)
        vec_path = self._path(VECTORS_FILE)

        meta = {}

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    meta = json.load(handle)
            except (OSError, ValueError):
                meta = {}

        usable = (
            meta.get("embedder") == self._embedder.name
            and os.path.exists(vec_path)
        )

        if usable:
            try:
                vectors = np.load(vec_path)
            except (OSError, ValueError):
                vectors = None

            if vectors is not None and len(vectors) == len(self.records):
                self.vectors = np.asarray(vectors, dtype=np.float32)
                return

        # Either the encoder changed or the files drifted apart. Re-embedding
        # is the only way back to a consistent index.
        self._reembed_all()

    def _reembed_all(self):
        if not self.records:
            self.vectors = np.zeros((0, self._embedder.dim), dtype=np.float32)
            self._write_vectors()
            return

        print("[memory: indexing {} memories...]".format(len(self.records)))

        self.vectors = self._embedder.encode(
            [_embed_text(record) for record in self.records]
        )

        self._write_vectors()

    def ready(self, timeout=None):
        """Block until the encoder and index are up. Returns False on timeout."""

        return self._loaded.wait(timeout)

    # -- persistence ------------------------------------------

    def _write_vectors(self):
        try:
            np.save(self._path(VECTORS_FILE), self.vectors)

            with open(self._path(META_FILE), "w", encoding="utf-8") as handle:
                json.dump(
                    {"embedder": self._embedder.name, "dim": self._embedder.dim},
                    handle,
                    indent=2,
                )
        except OSError as error:
            print("\n[memory: could not save index: {}]".format(error))

    def _append_record(self, record):
        try:
            with open(self._path(RECORDS_FILE), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as error:
            print("\n[memory: could not save: {}]".format(error))

    def _rewrite_records(self):
        try:
            tmp = self._path(RECORDS_FILE + ".tmp")

            with open(tmp, "w", encoding="utf-8") as handle:
                for record in self.records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            os.replace(tmp, self._path(RECORDS_FILE))
        except OSError as error:
            print("\n[memory: could not save: {}]".format(error))

    # -- writing ----------------------------------------------

    def remember(self, question, answer, kind="exchange"):
        """Store one exchange, replacing a near-identical earlier one.

        Returns the stored record, or None when memory is off or not ready.
        """

        if not self.enabled or not question or not answer:
            return None

        if not self._loaded.is_set():
            return None

        with self._lock:
            vector = self._embedder.encode([_join(question, answer, kind)])[0]

            duplicate = self._find_duplicate(question, vector, kind)

            if duplicate is not None:
                record = self.records[duplicate]
                record["q"] = question
                record["a"] = answer
                record["ts"] = _now()
                record["revisions"] = record.get("revisions", 0) + 1

                self.vectors[duplicate] = vector

                self._rewrite_records()
                self._write_vectors()

                return record

            record = {
                "id": self._next_id(),
                "ts": _now(),
                "kind": kind,
                "q": question,
                "a": answer,
                "score": 0,
                "uses": 0,
            }

            self.records.append(record)

            self.vectors = (
                vector.reshape(1, -1) if len(self.vectors) == 0
                else np.vstack([self.vectors, vector])
            )

            self._append_record(record)
            self._write_vectors()

            return record

    def note(self, text):
        """Store a standalone fact about the user."""

        text = text.strip()

        if not text:
            return None

        return self.remember("About the user", text[:NOTE_MAX_CHARS], kind="note")

    def capture_notes(self, user_message):
        """Pull durable statements out of an ordinary turn.

        Pattern matching rather than a model call: this runs on every turn, and
        a second forward pass to classify the sentence would cost more than the
        whole memory layer. A wrong note is not free, though -- it sits in the
        system prompt of every later turn -- so the patterns lean towards
        missing a statement, which '/remember' can add, over inventing one.
        The caller shows what was captured, so a wrong note is seen and
        '/forget' can remove it.
        """

        if not self.enabled or not self._loaded.is_set():
            return []

        captured = []
        prose = CODE_BLOCK.sub(" ", user_message)

        for sentence in re.split(r"(?<=[.!?\n])\s+", prose):
            sentence = NOTE_LEAD.sub("", sentence.strip())

            if not 8 <= len(sentence) <= NOTE_MAX_CHARS:
                continue

            # A question states nothing; it asks. "Do you remember that ..."
            # is not the user telling us a fact.
            if sentence.rstrip().endswith("?"):
                continue

            if any(pattern.search(sentence) for pattern in NOTE_EXPLICIT):
                text = _note_text(sentence)
            elif (any(pattern.search(sentence) for pattern in NOTE_INFERRED)
                    and not NOTE_REJECT.search(sentence)):
                text = sentence
            else:
                continue

            record = self.note(text)

            if record is not None:
                captured.append(record)

            if len(captured) >= NOTES_PER_MESSAGE:
                break

        return captured

    def _next_id(self):
        return max((record.get("id", 0) for record in self.records), default=0) + 1

    def _find_duplicate(self, question, vector, kind):
        """Index of the record this one supersedes, or None.

        Exchanges are matched on the question alone. They are indexed on the
        question *and* the answer, which is what makes recall work, but it is
        exactly the wrong test here: asking the same thing again and getting a
        better answer is the case worth collapsing, and a changed answer is
        what pushes the combined vectors apart.

        Notes have no question, so they fall back to similarity on their text.
        """

        if not self.records:
            return None

        if kind != "note":
            key = _question_key(question)

            for index in range(len(self.records) - 1, -1, -1):
                record = self.records[index]

                if record.get("kind") == "note":
                    continue

                if _question_key(record.get("q", "")) == key:
                    return index

            return None

        if len(self.vectors) == 0:
            return None

        scores = self.vectors @ vector

        for index in np.argsort(-scores)[:5]:
            index = int(index)

            if scores[index] < DEDUPE_THRESHOLD:
                return None

            if self.records[index].get("kind") == "note":
                return index

        return None

    # -- reading ----------------------------------------------

    def recall(self, query, keep=RECALL_KEEP):
        """Return the past exchanges worth putting in front of the model.

        Ranked by cosine similarity, nudged by rating. Records the user marked
        bad are excluded outright: the point of a thumbs-down is that the model
        should stop seeing that answer, not see it ranked slightly lower.

        Notes are not recalled here; they reach the model through profile(),
        on every turn, and showing one twice only makes it louder.
        """

        if not self.enabled or not query:
            return []

        if not self._loaded.is_set() or len(self.vectors) == 0:
            return []

        with self._lock:
            vector = self._embedder.encode([query])[0]
            scores = self.vectors @ vector

            candidates = np.argsort(-scores)[:RECALL_CANDIDATES]

            hits = []

            for index in candidates:
                index = int(index)
                record = self.records[index]

                if record.get("score", 0) < 0 or record.get("kind") == "note":
                    continue

                similarity = float(scores[index])

                if similarity < RECALL_THRESHOLD:
                    continue

                boost = SCORE_WEIGHT * max(
                    -SCORE_CLAMP, min(SCORE_CLAMP, record.get("score", 0))
                )

                hits.append((similarity + boost, similarity, index, record))

            hits.sort(key=lambda item: -item[0])
            hits = hits[:keep]

            self.last_recall = [index for _, _, index, _ in hits]

            for index in self.last_recall:
                self.records[index]["uses"] = self.records[index].get("uses", 0) + 1

            return [(similarity, record) for _, similarity, _, record in hits]

    def block(self, hits):
        """Render recalled exchanges as a prompt section, or None if empty.

        Each one carries the date it happened, so the model can tell an answer
        from last week from one it gave a year ago.
        """

        if not hits:
            return None

        entries = []
        used = 0

        for _, record in hits:
            entry = "- On {}, asked: {}\n  You answered: {}".format(
                (record.get("ts") or "")[:10] or "an earlier day",
                _clip(record["q"], 200),
                _clip(_prose(record["a"]), ANSWER_CHAR_BUDGET),
            )

            if used + len(entry) > RECALL_CHAR_BUDGET:
                break

            entries.append(entry)
            used += len(entry)

        if not entries:
            return None

        return "\n".join(["Earlier exchanges with this user:"] + entries)

    def profile(self):
        """The notes to state in the system prompt, oldest first.

        The most recent PROFILE_MAX_NOTES that fit PROFILE_CHAR_BUDGET, minus
        any the user rated down -- except that who the user is outranks what
        they said last: a name pushed out by eight newer preferences is the
        note most missed. Returned in a stable order, because this text heads
        the prompt and any change to it costs the whole KV cache.
        """

        if not self.enabled or not self._loaded.is_set():
            return []

        with self._lock:
            notes = [
                record for record in self.records
                if record.get("kind") == "note" and record.get("score", 0) >= 0
            ]

        def when(record):
            return (record.get("ts", ""), record.get("id", 0))

        notes.sort(key=when)

        identity = [record for record in notes if IDENTITY.match(record.get("a", ""))]
        others = [record for record in notes if not IDENTITY.match(record.get("a", ""))]

        chosen = []
        used = 0

        for record in identity[::-1] + others[::-1]:
            size = len(_clip(record.get("a", ""), NOTE_MAX_CHARS))

            if len(chosen) >= PROFILE_MAX_NOTES or used + size > PROFILE_CHAR_BUDGET:
                break

            chosen.append(record)
            used += size

        chosen.sort(key=when)

        return [_clip(record.get("a", ""), NOTE_MAX_CHARS) for record in chosen]

    # -- feedback ---------------------------------------------

    def rate(self, delta, target=None):
        """Adjust the score of a record. Defaults to the last one stored.

        A demotion always lands below zero rather than simply subtracting one.
        Scores accumulate, so an answer praised once and then criticised would
        otherwise come back to neutral and go on being recalled -- when what
        the user said, most recently and unambiguously, was "not this one".
        Promotion stays additive: it only affects ranking.

        Returns the record, or None when there is nothing to rate.
        """

        if not self.records:
            return None

        with self._lock:
            record = target if target is not None else self.records[-1]
            score = record.get("score", 0) + delta

            record["score"] = min(-1, score) if delta < 0 else score

            self._rewrite_records()

            return record

    def forget(self, selector):
        """Drop memories. Selector is 'last', 'all', or free text to match.

        Returns the number removed.
        """

        if not self._loaded.is_set():
            return 0

        with self._lock:
            if selector == "all":
                removed = len(self.records)
                self.records = []
                self.vectors = np.zeros((0, self._embedder.dim), dtype=np.float32)

            elif selector == "last":
                if not self.records:
                    return 0

                self.records.pop()
                self.vectors = self.vectors[:-1]
                removed = 1

            else:
                if len(self.vectors) == 0:
                    return 0

                vector = self._embedder.encode([selector])[0]
                scores = self.vectors @ vector
                doomed = set(int(i) for i in np.where(scores >= RECALL_THRESHOLD)[0])

                if not doomed:
                    return 0

                keep = [i for i in range(len(self.records)) if i not in doomed]

                self.records = [self.records[i] for i in keep]
                self.vectors = (
                    self.vectors[keep] if keep
                    else np.zeros((0, self._embedder.dim), dtype=np.float32)
                )
                removed = len(doomed)

            self._rewrite_records()
            self._write_vectors()

            return removed

    # -- reporting --------------------------------------------

    def stats(self):
        kinds = {}
        rated = 0

        for record in self.records:
            kinds[record.get("kind", "exchange")] = kinds.get(
                record.get("kind", "exchange"), 0
            ) + 1

            if record.get("score", 0):
                rated += 1

        return {
            "ready": self._loaded.is_set(),
            "enabled": self.enabled,
            "total": len(self.records),
            "notes": kinds.get("note", 0),
            "exchanges": kinds.get("exchange", 0),
            "rated": rated,
            "encoder": getattr(self._embedder, "name", "loading"),
            "dir": self.dir,
            "error": self._error,
        }


# ============================================================
# HELPERS
# ============================================================

def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _question_key(text):
    """Normalized question, for deciding two turns asked the same thing.

    Case, spacing and trailing punctuation are noise here: "how do I sort a
    dict?" and "How do I sort a dict" are one question asked twice.
    """

    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _clip(text, limit):
    text = " ".join(text.split())

    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


def _prose(answer):
    """An answer with its code blocks elided, for recall.

    _clip collapses whitespace, which turns a code block into one long line
    of broken syntax -- and a 7B shown broken code imitates it. The prose
    around the code is what records what was said.
    """

    return CODE_BLOCK.sub(" [code omitted] ", answer)


def _note_text(sentence):
    """An explicit statement as the fact it states.

    "Remember that I work nights." is stored as "I work nights."; the command
    wrapped around the fact means nothing once it is in the profile.
    """

    text = re.sub(r"^(?:please )?remember(?: that|:)?\s+", "", sentence, flags=re.I)

    return text[:1].upper() + text[1:] if text else sentence


def _join(question, answer, kind):
    """Text an exchange is indexed by.

    Notes are indexed on their content alone; their question field is a
    constant placeholder and would only add noise. Exchanges are indexed on the
    question plus the opening of the answer, because the question alone misses
    follow-ups phrased as "and the other one?" while the whole answer drowns
    the question in boilerplate.
    """

    if kind == "note":
        return answer

    return "{}\n{}".format(question, answer[:ANSWER_CHAR_BUDGET])


def _embed_text(record):
    return _join(record.get("q", ""), record.get("a", ""), record.get("kind"))


def ground(user_message, memory_block, web_context=None):
    """Prepend what the model has learned to the turn it is about to answer.

    Kept separate from retrieval.ground so the two can compose: web context is
    evidence about the world, memory is evidence about this user and this
    conversation, and the model is told to weigh them differently.
    """

    if not memory_block:
        return user_message

    return (
        "MEMORY -- what you have learned from earlier conversations with this "
        "user:\n"
        "<<<\n"
        "{}\n"
        ">>>\n"
        "END MEMORY\n\n"
        "Use the MEMORY when it is relevant: stay consistent with what you "
        "already told this user, and correct it if it was wrong. If it is not "
        "relevant to the question, ignore it silently -- never mention having "
        "a memory, and never answer a question the user did not ask.\n\n"
        "{}"
    ).format(memory_block, user_message)
