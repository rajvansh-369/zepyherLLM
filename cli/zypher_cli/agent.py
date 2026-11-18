"""
The agent: one user request, worked through to an answer.

    ask -> [ generate -> first tool call -> permission -> run it -> result ]* -> answer

The conversation lives here as bare turns, and the client keeps it inside the
server's prompt budget itself. The server would do that too, but by dropping
the oldest turns first -- which for an agent is the task it was given. So old
tool output is cut down before anything else, then whole earlier requests,
then the oldest steps of the current one; the request being worked on stays.

Three things are deliberately off for agent turns: memory (a tool result is
not a question worth recalling or storing), automatic web lookup (same), and
free sampling -- tool calls want the most likely continuation.
"""

import os
import re
import time

from . import protocol
from .api import ApiError, Offline
from .tools import IGNORED_DIRS, AlreadyDone, Result, ToolError, diff_stats, merge_loose

MAX_STEPS = 25
MAX_FAILURES = 4

# Failed changes to one file in one request before giving up on it. Counted
# across the whole request, not in a row: a model alternating a failed edit
# with a successful read never fails MAX_FAILURES times running, and looped
# until the context window was trimmed and it started over.
MAX_FILE_MISSES = 4

# The same call this many times in one request, anywhere in it, is a loop.
MAX_SAME_CALL = 3
MAX_CONTINUES = 3
RUN_TIMEOUT = 120

# Tokens the server's own system prompt adds in front of ours.
SERVER_PROMPT_TOKENS = 450

STAGES = {
    "searching": "Searching the web",
    "waking": "Loading the model back in",
    "recalled": "Thinking",
}

MODES = {
    "ask": "ask before changes",
    "edits": "edits allowed, ask before commands",
    "all": "everything allowed",
}

CHAINING = (";", "&", "|", "`", "$(", "\n", ">", "<")

NO_CHANGE = (
    "Nothing changed: {} already has exactly this. If an error is still "
    "there, this is not the fix -- read the line the error points to and "
    "change that line."
)

# Commands that install software.
INSTALLS = re.compile(
    r"\b(?:pip3?|uv\s+pip|poetry|pipx|conda|npm|pnpm|yarn|bun|gem|cargo|go|choco|winget|scoop|apt(?:-get)?|brew|composer)"
    r"(?:\s+-\S+)*\s+(?:install|i|add|get)\b",
    re.I,
)

# The model has stopped working on the task and is showing how each tool is
# used, one call after another ("To create a file, use the following tool call:").
TUTORIAL = re.compile(r"\b(?:use|make|send) (?:the following|this) tool call\b", re.I)

# Commands that rewrite files in place. Not run: the file tools do it safely.
WRITES_FILES = re.compile(
    r"\b(?:Set-Content|Add-Content|Out-File|WriteAllText|WriteAllLines|WriteAllBytes)\b"
    r"|(?:^|[|;&]\s*)(?:sc|ac)\s"
    r"|\bsed\s+(?:-\w*\s+)*-i|\bperl\s+-\w*i|\bpython[\d.]*\s+-c\s+.*\bopen\s*\([^)]*['\"][wa]",
    re.I,
)


class Declined(Exception):

    def __init__(self, feedback=""):
        super().__init__(feedback)
        self.feedback = feedback


def _whitespace_run(text):
    """The model sometimes falls into emitting spaces without end."""

    return len(text) > 400 and not text[-400:].strip()


def _looping(text):
    """The same non-empty line 15 times running."""

    lines = [line for line in text.split("\n")[-17:-1] if line.strip()]

    return len(lines) >= 15 and len(set(lines[-15:])) == 1


class Agent:

    def __init__(self, api, workspace, ui, auto="ask", tools=True):
        self.api = api
        self.ws = workspace
        self.ui = ui
        self.auto = auto
        self.tools = tools

        self.messages = []
        self.seen = set()        # files the model has been shown
        self.undo = []           # [(path, old text or None)]
        self.allowed = set()     # commands approved for the session, by first word
        self.pending = ""        # goes in front of the next user message

        self.web = None          # None: off for the agent, auto for plain chat
        self.max_tokens = None
        self.budget = 6144       # prompt tokens the server will accept
        self.ratio = 3.0         # characters per token, measured as we go
        self.last = None         # the last finished generation's usage and meta
        self.truncated = False   # the last answer hit the token budget
        self.vary = False        # the model is repeating itself: sample more freely
        self.system = ""
        self.profile = ()

    # -- setup ------------------------------------------------

    def refresh(self):
        """Read the server's limits and the user's notes; rebuild the system text."""

        try:
            self.budget = int(self.api.settings().get("max_prompt_tokens") or self.budget)
        except ApiError:
            pass

        try:
            self.profile = tuple(self.api.memory().get("profile") or ())
        except ApiError:
            self.profile = ()

        # A single tool result may take about a third of the prompt budget.
        self.ws.result_chars = max(2000, min(16000, int(self.budget * 3 * 0.3)))
        self.rebuild_system()

    def rebuild_system(self):
        root = self.ws.root
        names = []

        try:
            entries = sorted(os.scandir(str(root)), key=lambda e: (not e.is_dir(), e.name.lower()))
        except OSError:
            entries = []

        for entry in entries:
            if entry.is_dir() and entry.name in IGNORED_DIRS:
                continue

            names.append(entry.name + ("/" if entry.is_dir() else ""))

        listing = ", ".join(names[:40]) + (", ..." if len(names) > 40 else "")
        branch = None

        try:
            head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()

            if head.startswith("ref: refs/heads/"):
                branch = head[len("ref: refs/heads/"):]
        except OSError:
            pass

        notes = ""

        try:
            notes = (root / "ZYPHER.md").read_text(encoding="utf-8", errors="replace").strip()[:2000]
        except OSError:
            pass

        self.system = protocol.system_prompt(
            str(root),
            "Windows" if os.name == "nt" else os.uname().sysname,
            self.ws.shell.name,
            listing, branch, notes, self.profile,
        )

    # -- the conversation -------------------------------------

    def wire(self):
        """The messages as sent."""

        turns = [{"role": m["role"], "content": m["content"]} for m in self.messages]

        if not self.tools:
            return turns

        return [{"role": "system", "content": self.system}] + protocol.shots() + turns

    def tokens(self):
        """Estimated prompt size of the conversation as it stands."""

        chars = sum(len(m["content"]) for m in self.wire())

        return SERVER_PROMPT_TOKENS + int(chars / self.ratio)

    def usage(self):
        """How full the prompt budget is, 0-1."""

        return min(1.0, self.tokens() / float(self.budget))

    def _calibrate(self, usage, chars):
        prompt = (usage or {}).get("prompt_tokens") or 0

        if prompt > SERVER_PROMPT_TOKENS + 200 and chars > 0:
            measured = chars / float(prompt - SERVER_PROMPT_TOKENS)
            # A little under what was measured: overestimating the prompt
            # only compacts early, underestimating loses the task.
            self.ratio = max(2.2, min(4.5, measured * 0.95))

    def _add(self, role, kind, content, **extra):
        message = dict(role=role, kind=kind, content=content, **extra)
        self.messages.append(message)

        return message

    def _close_turn(self, text="(stopped)"):
        """Keep user and assistant turns alternating, as the chat template
        and the server's own trimming both expect."""

        if self.messages and self.messages[-1]["role"] == "user":
            self._add("assistant", "answer", text)

    def _forget_failure(self):
        """Drop a request that was given up on, before the next one.

        Its wrong guesses, and the "(stopped ...)" line that ended it, are
        the nearest thing in the conversation for the model to copy -- and it
        does, word for word.
        """

        if not self.messages or not self.messages[-1].get("gave_up"):
            return

        start = max((i for i, m in enumerate(self.messages) if m["kind"] == "user"), default=0)
        del self.messages[start:]

    def clear(self):
        self.messages = []
        self.seen = set()
        self.pending = ""
        self.last = None
        self.truncated = False

    def compact(self, force=False):
        """Bring the conversation back inside the prompt budget.

        Returns True if anything was cut. Done rarely and deeply rather than
        a little every step: each cut changes the prompt from that point on,
        and the server has to read it again from there.
        """

        if not force and self.tokens() <= self.budget * 0.9:
            return False

        target = self.budget * 0.55
        messages = self.messages
        before = len(messages), self.tokens()

        # 1. Old tool traffic: results, and the bodies of old writes.
        results = [i for i, m in enumerate(messages) if m["kind"] == "result"]
        recent = set(results[-2:])
        calls = [i for i, m in enumerate(messages) if m["kind"] == "call"]
        recent.update(calls[-2:])

        for index, message in enumerate(messages):
            if self.tokens() <= target:
                break

            if index in recent or message.get("elided") or len(message["content"]) < 300:
                continue

            if message["kind"] == "result":
                message["content"] = protocol.result_message(
                    message["title"],
                    "[removed to save space -- make the call again if you need this]",
                )
                message["elided"] = True

            elif message["kind"] == "call" and message["name"] in ("write", "edit"):
                call = protocol.parse(message["call"], final=True).call
                message["content"] = (
                    (message["prose"] + "\n\n" if message["prose"] else "") + call.render(elide=True)
                )
                message["elided"] = True

        # 2. Whole earlier requests, oldest first.
        def task_index():
            return max((i for i, m in enumerate(messages) if m["kind"] == "user"), default=0)

        while self.tokens() > target and task_index() >= 2:
            del messages[:2]

        # 3. The oldest steps of the current request, keeping its last two.
        while self.tokens() > target and len(messages) - (task_index() + 1) > 4:
            start = task_index() + 1
            del messages[start:start + 2]

        return (len(messages), self.tokens()) != before

    # -- one request ------------------------------------------

    def ask(self, text, web=None):
        """Answer a request. Returns True if it ran to an answer."""

        self._close_turn()
        self._forget_failure()

        content = self.pending + text
        self.pending = ""

        if self.tools:
            content += protocol.USER_REMINDER

        self._add("user", "user", content, text=text)

        return self._work(web=web)

    def resume(self):
        """Carry on an answer that hit the token budget."""

        if not self.truncated or not self.messages or self.messages[-1]["kind"] != "answer":
            return self.ask("continue")

        carry = self.messages.pop()["content"]

        return self._work(carry=carry)

    def _work(self, web=None, carry=""):
        ui = self.ui
        steps = 0
        failures = 0
        previous = None
        repeats = 0
        made = {}      # rendered call -> times made in this request
        misses = {}    # file -> failed changes to it in this request
        produced = 0
        started = time.monotonic()
        finished = False
        changed = tried = 0

        while True:
            if self.compact():
                ui.note("  (older parts of the conversation were trimmed to fit the context window)")

            reply = self._generate(web if steps == 0 else False, carry)
            carry = ""

            if reply is None:
                break

            text, parsed, interrupted, tokens = reply
            produced += tokens
            prose = protocol.clean_prose(parsed.prose) if self.tools else text.strip()
            call = parsed.call

            if interrupted:
                self._add("assistant", "answer", prose or "(interrupted)")
                ui.note("  Interrupted.")
                break

            if call is None:
                self._add("assistant", "answer", prose or "(no reply)")

                if not prose:
                    ui.note("  (the model gave no reply)")

                finished = True
                break

            if TUTORIAL.search(prose):
                ui.warn("Stopped: the model drifted off the task (it began demonstrating tools).")
                self._add("assistant", "answer", "(stopped: drifted off the task)", gave_up=True)
                break

            steps += 1
            rendered = call.render()
            self._add(
                "assistant", "call", (prose + "\n\n" if prose else "") + rendered,
                name=call.name, call=rendered, prose=prose,
            )

            stop = None
            gave_up = False

            try:
                result = self._execute(call)
            except Declined as declined:
                if declined.feedback:
                    result = Result(
                        "The user declined this and said: {}".format(declined.feedback),
                        "declined", ok=False,
                    )
                else:
                    result = Result("The user declined this action.", "declined", ok=False)
                    stop = "Okay, I did not do that."
            except KeyboardInterrupt as signal:
                # A command stopped half way still hands over what it printed.
                result = getattr(signal, "result", None) or Result(
                    "The user interrupted this.", "interrupted", ok=False
                )
                stop = "(interrupted)"
                ui.note("  Interrupted.")

            body = result.body
            failures = failures + 1 if getattr(result, "failed", False) else 0

            if call.name in ("write", "edit", "replace"):
                tried += 1
                target = call.path or ""

                if result.ok and result.summary not in ("no change", "not written", "failed"):
                    changed += 1
                    misses.pop(target, None)
                elif getattr(result, "failed", False):
                    misses[target] = misses.get(target, 0) + 1

            made[rendered] = made.get(rendered, 0) + 1

            if rendered == previous or made[rendered] > 1:
                # The most likely continuation led here once and will again:
                # say so briefly, without the same long result a second
                # time, and let the next reply be sampled more freely.
                repeats += 1
                self.vary = True
                body = (
                    "{}\n\nYou already made this exact call and got this same "
                    "result. Do NOT send it again. {}".format(
                        body.split("\n")[0],
                        "Read the file, or search it, and use the text that is "
                        "really there." if getattr(result, "failed", False)
                        else "Make a different call, or give the final answer.",
                    )
                )
            else:
                repeats = 0
                self.vary = False

            previous = rendered
            self._add("user", "result", protocol.result_message(call.title(), body), title=call.title())

            if stop is None and failures >= MAX_FAILURES:
                ui.warn("Stopped: {} tool calls in a row failed.".format(failures))
                stop = "(stopped after repeated failed tool calls)"
                gave_up = True
            elif stop is None and max(misses.values(), default=0) >= MAX_FILE_MISSES:
                stuck = max(misses, key=misses.get)
                ui.warn("Stopped: {} changes to {} failed. If the change is already "
                        "there, nothing more is needed.".format(misses[stuck], stuck))
                stop = "(stopped: changes to {} kept failing)".format(stuck)
                gave_up = True
            elif stop is None and (repeats >= 3 or made[rendered] >= MAX_SAME_CALL):
                ui.warn("Stopped: the model kept repeating the same call.")
                stop = "(stopped: repeating the same call)"
                gave_up = True
            elif stop is None and steps >= MAX_STEPS:
                ui.warn("Stopped after {} steps. Say 'continue' to carry on.".format(steps))
                stop = "(paused after {} steps)".format(steps)

            if stop is not None:
                self._add("assistant", "answer", stop, gave_up=gave_up)
                break

        # A small model reports "done" whether or not its change went in.
        if tried and not changed:
            ui.warn("No file was changed: every change the model tried failed or was declined.")

        self._footer(steps, produced, time.monotonic() - started)

        return finished

    def _footer(self, steps, produced, seconds):
        parts = []

        if steps:
            parts.append("{} step{}".format(steps, "" if steps == 1 else "s"))

        if produced:
            parts.append("{} tokens".format(produced))

        meta = (self.last or {}).get("meta") or {}

        if meta.get("tokens_per_second"):
            parts.append("{:.0f} tok/s".format(meta["tokens_per_second"]))

        parts.append("{:.0f}s".format(seconds))
        parts.append("context {:.0f}%".format(self.usage() * 100))

        self.ui.gap()
        self.ui.note("  " + " · ".join(parts))

        hot = meta.get("throttled")

        if hot:
            self.ui.note(
                "  The GPU is at {} C and slowed itself down; give it a minute "
                "or more airflow.".format(hot.get("gpu_temp_c"))
            )

    # -- generating -------------------------------------------

    def _generate(self, web, carry=""):
        """Stream one reply, stopping at its first complete tool call.

        Returns (text, parsed, interrupted, tokens), or None after an error
        that has been reported.
        """

        ui = self.ui
        view = ui.stream("Thinking")
        text = carry
        continues = 0
        tokens = 0
        interrupted = False
        finish = None

        if web is None:
            web = self.web if self.web is not None else (False if self.tools else "auto")

        try:
            while True:
                messages = self.wire()
                continuing = bool(text)

                if continuing:
                    messages.append({"role": "assistant", "content": text})

                sent = sum(len(m["content"]) for m in messages)
                stream = self.api.chat(
                    messages,
                    web=False if continuing else web,
                    memory=False if self.tools else None,
                    sampling=("balanced" if self.vary else "precise") if self.tools else None,
                    max_tokens=self.max_tokens,
                )

                finish = None
                stopped = False
                start_length = len(text)

                try:
                    for kind, payload in stream:
                        if kind == "status":
                            view.status(STAGES.get(payload.get("stage"), "Thinking"))

                        elif kind == "done":
                            finish = payload

                        elif kind == "text":
                            text += payload

                            if _whitespace_run(text) or _looping(text):
                                text = text.rstrip()
                                stopped = True
                                break

                            if not self.tools:
                                view.prose(text)
                                continue

                            parsed = protocol.parse(text)
                            view.prose(protocol.displayable(parsed.prose))

                            if parsed.call is not None:
                                view.calling(self._progress(parsed.call))

                                if parsed.call.complete:
                                    stopped = True
                                    break

                            elif protocol.ran_away(text):
                                stopped = True
                                break

                except KeyboardInterrupt:
                    interrupted = True
                finally:
                    stream.close()

                if finish is not None:
                    self.last = finish
                    self._calibrate(finish["usage"], sent)
                    tokens += finish["usage"].get("completion_tokens") or 0
                else:
                    tokens += int((len(text) - start_length) / self.ratio)

                cut_short = finish is not None and finish["finish_reason"] == "length"
                parsed = protocol.parse(text) if self.tools else protocol.Parsed(text, None)
                mid_call = parsed.call is not None and not parsed.call.complete

                # A write that ran out of tokens half way: carry straight on
                # inside it rather than hand the model half a file.
                if cut_short and mid_call and not interrupted and continues < MAX_CONTINUES:
                    continues += 1
                    continue

                break

            self.truncated = cut_short and not mid_call and not interrupted

            if self.tools:
                parsed = protocol.parse(text, final=not interrupted)

                if interrupted and parsed.call is not None and not parsed.call.complete:
                    parsed = protocol.Parsed(parsed.prose, None)

                view.prose(protocol.clean_prose(parsed.prose), final=True)
            else:
                view.prose(text.strip(), final=True)

        except ApiError as error:
            view.end()
            self.report(error)
            return None

        finally:
            view.end()

        if self.truncated:
            ui.note("  (stopped at the token budget -- /continue for more)")

        self._sources()

        return text, parsed, interrupted, tokens

    def _progress(self, call):
        if call.name == "replace":
            return call.label() if call.complete else "Replace…"

        if call.name in ("write", "edit") and call.path:
            return "{}({}) · {} lines".format(
                call.name.capitalize(), call.path, call.body.count("\n") + 1
            )

        return call.label() if call.complete else call.name.capitalize() + "…"

    def _sources(self):
        meta = (self.last or {}).get("meta") or {}
        sources = meta.get("sources") or []

        if not sources:
            return

        self.ui.gap()
        self.ui.note("  Sources:" if meta.get("cited") else "  Searched:")

        for source in sources:
            self.ui.note("  [{}] {} -- {}".format(source.get("n"), source.get("title"), source.get("url")))

        meta["sources"] = []  # shown once, not again after the next tool step

    def report(self, error):
        if isinstance(error, Offline):
            self.ui.error("{}. Start it with: python main.py serve".format(error.message))
        elif error.status == 401:
            self.ui.error("The API rejected the key. ZYPHER_API_KEY must match the server's .env.")
        else:
            self.ui.error(error.message)

    # -- tools ------------------------------------------------

    def _execute(self, call):
        ui = self.ui
        ui.call(call.label())

        if call.error:
            ui.outcome("malformed call", ok=False)
            return self._failure(call.error)

        try:
            return getattr(self, "_do_" + call.name)(call)
        except ToolError as error:
            ui.outcome(str(error).split("\n")[0][:140], ok=False)
            return self._failure(str(error))

    def _failure(self, text):
        result = Result(text, "failed", ok=False)
        result.failed = True

        return result

    def _key(self, path):
        return os.path.normcase(str(path))

    def _permit(self, kind, question, command=""):
        """Raise Declined unless this may go ahead."""

        # Installing changes the machine, not the project, and a small model
        # will happily install a package it made up the name of. Always asked,
        # whatever the mode; never done unattended.
        installing = kind == "run" and INSTALLS.search(command)

        if self.auto == "all" and not installing:
            return

        always = None

        if installing:
            if not self.ui.interactive:
                self.ui.outcome("installing needs an interactive session", ok=False)
                raise Declined("installing packages is not allowed in a non-interactive run")

        elif kind == "edit":
            if self.auto == "edits":
                return

            always = "yes, and stop asking for edits"

        elif kind == "run":
            word = command.split()[0] if command.split() else ""
            simple = word and not any(mark in command for mark in CHAINING)

            if simple and word in self.allowed:
                return

            if simple:
                always = "always allow `{}` commands".format(word)

        if not self.ui.interactive:
            self.ui.outcome("not allowed without --yes", ok=False)
            raise Declined("this is a non-interactive run; changes need the --yes flag")

        answer, feedback = self.ui.confirm(question, always)

        if answer == "always":
            if kind == "edit":
                self.auto = "edits"
            else:
                self.allowed.add(command.split()[0])

        elif answer == "no":
            self.ui.outcome("declined" + (": " + feedback if feedback else ""), ok=False)
            raise Declined(feedback)

    def _outside(self, path, verb):
        if not self.ws.inside(path):
            self._permit("outside", "{} {} -- outside the project folder?".format(verb, path))

    def _do_read(self, call):
        path = self.ws.resolve(call.path)
        self._outside(path, "Read")

        result = self.ws.read(path, call.args.get("lines"))
        self.seen.add(self._key(path))
        self.ui.outcome(result.summary)

        return result

    def _do_list(self, call):
        path = self.ws.resolve(call.path or ".")
        self._outside(path, "List")

        result = self.ws.list(path)
        self.ui.outcome(result.summary)

        return result

    def _do_search(self, call):
        path = self.ws.resolve(call.path or ".")
        self._outside(path, "Search")

        result = self.ws.search(call.args["pattern"], path)
        self.ui.outcome(result.summary)

        return result

    def _show_file(self, path):
        """A file's content for the model, after a change to it went wrong."""

        try:
            shown = self.ws.read(path)
        except ToolError:
            return ""

        self.seen.add(self._key(path))

        return "\n\nCurrent content of {}:\n{}".format(self.ws.show(path), shown.body)

    def _do_write(self, call):
        ws, ui = self.ws, self.ui
        path = ws.resolve(call.path)
        shown = ws.show(path)
        old, new = ws.plan_write(path, call.body)

        if old is not None and self._key(path) not in self.seen:
            # Overwriting a file it has never looked at is nearly always the
            # model inventing the file from its name.
            ui.outcome("exists and was not read: showing it to the model first")

            return Result(
                "{0} already exists and you have not read it, so nothing was "
                "written.{1}\n\nTo change part of it use <edit>. To replace all "
                "of it, send the <write> again.".format(shown, self._show_file(path)),
                "not written",
            )

        if old == new:
            ui.outcome("no change")
            return Result(NO_CHANGE.format(shown), "no change")

        ui.diff(old, new)
        self._outside(path, "Write")
        self._permit("edit", "{} {}?".format("Create" if old is None else "Overwrite", shown))

        ws.commit(path, new)
        self.undo.append((path, old))
        self.seen.add(self._key(path))

        count = new.count("\n")

        if old is None:
            summary = "created, {} line{}".format(count, "" if count == 1 else "s")
        else:
            summary = "+{} -{}".format(*diff_stats(old, new))

        ui.outcome(summary)

        return self._checked(path, old, new, Result("Wrote {} ({} lines).".format(shown, count), summary))

    def _do_edit(self, call):
        ws, ui = self.ws, self.ui
        path = ws.resolve(call.path)
        shown = ws.show(path)

        if not path.exists():
            raise ToolError(
                "there is no file {}. To create it use <write path=\"{}\">.".format(shown, shown)
            )

        if not call.blocks():
            return self._edit_loose(call)

        try:
            old, new = ws.plan_edit(path, call.blocks())
        except AlreadyDone as done:
            ui.outcome("already applied")
            self.seen.add(self._key(path))

            return Result(
                "{} already has this change ({}). The edit is done. If that was "
                "the last change, give the final answer now.".format(shown, done),
                "already applied",
            )
        except ToolError as error:
            ui.outcome(str(error).split("\n")[0][:140], ok=False)

            return self._failure(self._edit_help(path, call, error))

        ui.diff(old, new)
        self._outside(path, "Edit")
        self._permit("edit", "Apply this edit to {}?".format(shown))

        ws.commit(path, new)
        self.undo.append((path, old))
        self.seen.add(self._key(path))

        added, removed = diff_stats(old, new)
        summary = "+{} -{}".format(added, removed)
        ui.outcome(summary)

        return self._checked(path, old, new, Result(
            "Edited {} ({} added, {} removed).".format(shown, added, removed), summary
        ))

    def _edit_loose(self, call):
        """An edit sent as the new code itself, with no SEARCH/REPLACE block.

        Each part is worked into its file by merge_loose. A body may also
        carry other files as "[file NAME]" sections, the way attached files
        are shown to the model; each is applied to its own file.
        """

        ws, ui = self.ws, self.ui
        parts = _file_sections(call.path, call.body)
        bodies = []
        failed = False

        for index, (raw, text) in enumerate(parts):
            path = ws.resolve(raw)
            shown = ws.show(path)

            if index:
                ui.call("Edit({})".format(shown))

            try:
                if path.is_file():
                    if self._key(path) not in self.seen:
                        raise ToolError("you have not read {} yet{}".format(shown, self._show_file(path)))

                    old = ws.load(path)[0]
                    new = merge_loose(old, text)
                    new = new if new.endswith("\n") else new + "\n"
                else:
                    old, new = ws.plan_write(path, protocol._unfence(text))

                if old == new:
                    ui.outcome("no change")
                    bodies.append(NO_CHANGE.format(shown))
                    continue

                ui.diff(old, new)
                self._outside(path, "Edit")
                self._permit("edit", "Apply this edit to {}?".format(shown))

            except ToolError as error:
                ui.outcome(str(error).split("\n")[0][:140], ok=False)
                failed = True
                bodies.append(
                    "{}: {}. Nothing was changed in it.\n\nSend the change as a block:\n"
                    "<edit path=\"{}\">\n<<<<<<< SEARCH\nexisting lines, copied from the "
                    "file\n=======\nnew lines\n>>>>>>> REPLACE\n</edit>".format(shown, error, shown)
                )
                continue

            ws.commit(path, new)
            self.undo.append((path, old))
            self.seen.add(self._key(path))

            added, removed = diff_stats(old, new)
            summary = "+{} -{}".format(added, removed)
            ui.outcome(summary)

            result = self._checked(path, old, new, Result(
                "Edited {} ({} added, {} removed).".format(shown, added, removed), summary))
            failed = failed or getattr(result, "failed", False)
            bodies.append(result.body)

        result = Result("\n\n".join(bodies), "failed" if failed else "edited", ok=not failed)
        result.failed = failed

        return result

    def _edit_help(self, path, call, error):
        """What the model is told when an edit's SEARCH did not match."""

        shown = self.ws.show(path)
        text = "{}. Nothing was changed.".format(str(error).rstrip("."))
        blocks = call.blocks()

        if " in the file. Add the lines" in str(error) and len(blocks) == 1 and "\n" not in blocks[0][0]:
            # One line that is in the file several times: almost always a
            # rename. Hand over the call that does it.
            return (
                "{}\n\nTo change every one of them, send exactly this:\n{}".format(
                    text, protocol.replace_call(shown, blocks[0][0].strip(), blocks[0][1].strip()))
            )

        if self._key(path) not in self.seen:
            # It never looked: the SEARCH text was made up.
            return (
                "{} You have not read {}, so the SEARCH text was a guess.{}\n\n"
                "Use text that is really in the file. To change a word or name "
                'everywhere, use <replace path="{}" old="TEXT" new="TEXT"/>.'.format(
                    text, shown, self._show_file(path), shown)
            )

        close = "\n".join(self.ws.near(path, search) for search, _ in blocks).strip()

        if close:
            return (
                "{}\n\nThe closest lines in {} are:\n{}\n\nCopy the SEARCH lines "
                "exactly from these and send the edit again.".format(text, shown, close)
            )

        return (
            "{} No line in {} looks like it.\n\nFind the real text first with "
            '<search pattern="TEXT" path="{}"/> or <read path="{}"/>.'.format(
                text, shown, shown, shown)
        )

    def _do_replace(self, call):
        ws, ui = self.ws, self.ui
        path = ws.resolve(call.path)
        shown = ws.show(path)
        find, put = call.args["old"], call.args["new"]

        try:
            old, new, places = ws.plan_replace(path, find, put)
        except ToolError as error:
            ui.outcome(str(error).split("\n")[0][:140], ok=False)
            unread = ""

            if self._key(path) not in self.seen:
                unread = " You have not read the file.{}".format(self._show_file(path))

            return self._failure(
                "{}. Nothing was changed.{}\n\nUse text that is really in the "
                "file.".format(error, unread)
            )

        ui.diff(old, new)
        self._outside(path, "Change")
        self._permit("edit", "Replace {!r} with {!r} in {} place{} in {}?".format(
            find, put, places, "" if places == 1 else "s", shown))

        ws.commit(path, new)
        self.undo.append((path, old))
        self.seen.add(self._key(path))

        summary = "{} place{}".format(places, "" if places == 1 else "s")
        ui.outcome(summary)

        return self._checked(path, old, new, Result(
            "Replaced {!r} with {!r} in {} of {}.".format(find, put, summary, shown), summary
        ))

    def _checked(self, path, old, new, result):
        """result, with a syntax error the change brought in added to it.

        An error the file already had before the change is not reported: it
        is not the model's to fix, and would send it off course.
        """

        problem = self.ws.check(path, new)

        if not problem or (old is not None and self.ws.check(path, old)):
            return result

        self.ui.outcome("problem: {}".format(problem.split("\n")[0])[:140], ok=False)

        broken = Result(
            "{}\n\nBut {} now has an error: {}\nFix it before anything else: read "
            "the lines around it, then edit them.".format(result.body, self.ws.show(path), problem),
            "error in the result", ok=False,
        )
        broken.failed = True

        return broken

    def _do_run(self, call):
        ws, ui = self.ws, self.ui
        command = call.body

        if WRITES_FILES.search(command):
            # Windows PowerShell reads and writes files in the ANSI code
            # page: a -replace piped into Set-Content mangles every character
            # outside it. The file tools keep encoding and line ends intact.
            ui.outcome("refused: changes files from the shell", ok=False)

            return self._failure(
                "That command was not run: it changes files from the shell, which "
                "can corrupt their text. Change files only with the tools:\n"
                '<replace path="FILE" old="TEXT" new="TEXT"/> for text in many places, '
                "<edit> for a few lines, <write> for a whole new file."
            )

        self._permit("run", "Run this command?", command)

        started = time.monotonic()
        output, code, note = ws.run(command, RUN_TIMEOUT, ui.output)
        ui.output_end()
        seconds = time.monotonic() - started

        if note == "interrupted":
            ui.outcome("interrupted after {:.0f}s".format(seconds), ok=False)
            status = "[interrupted by the user after {:.0f}s]".format(seconds)
        elif note == "timed out":
            ui.outcome("stopped after {}s".format(RUN_TIMEOUT), ok=False)
            status = (
                "[stopped: still running after {}s. Commands that do not exit, "
                "such as servers, cannot be run here.]".format(RUN_TIMEOUT)
            )
        else:
            ui.outcome("exit {} · {:.1f}s".format(code, seconds), ok=code == 0)
            status = "[exit code {}]".format(code)

        body = ws.clip(output).rstrip("\n")
        result = Result((body + "\n" if body else "(no output)\n") + status, status, ok=code == 0)

        if note == "interrupted":
            raise _Interrupted(result)

        return result

    # -- outside a turn ---------------------------------------

    def revert(self):
        """Undo the last file change. Returns the path shown, or None."""

        if not self.undo:
            return None

        path, old = self.undo.pop()
        self.ws.restore(path, old)
        shown = self.ws.show(path)

        self.pending += "[The user undid your last change to {}.]\n\n".format(shown)
        self.seen.discard(self._key(path))

        return shown

    def attach(self, text):
        """Text with the files it mentions added below it.

        A file is attached when it is named with @ (@src/app.py), or named
        plainly (src/app.py, index.html) and exists in the project. The model
        then starts from what is really in the file instead of from a guess.
        A big file comes as its map and its first part. Files beyond the
        first stop once they would take more than about a tool result's
        worth of room; the model is told to read those.

        Returns (text, [names attached]).
        """

        names = []
        extra = []
        skipped = []
        used = 0
        done = set()

        for word in text.split():
            explicit = word.startswith("@")
            raw = word.lstrip("@").strip("\"'`()[]{}<>").rstrip(",.;:!?")

            if not raw or "://" in raw or not (explicit or "." in raw or "/" in raw or "\\" in raw):
                continue

            try:
                path = self.ws.resolve(raw)
            except ToolError:
                continue

            key = self._key(path)

            if key in done or not path.is_file() or not (explicit or self.ws.inside(path)):
                continue

            done.add(key)
            shown = self.ws.show(path)

            try:
                result = self.ws.summary(path) or self.ws.read(path)
            except ToolError:
                continue

            if names and used + len(result.body) > self.ws.result_chars * 1.2:
                skipped.append(shown)
                continue

            used += len(result.body)

            if result.summary != "map":   # a map is not the file: still unread
                self.seen.add(key)

            names.append(shown)
            extra.append("[file {}]\n{}".format(shown, result.body))

        if skipped:
            extra.append("[not attached, to save room -- read when needed: {}]".format(", ".join(skipped)))

        if extra:
            text += "\n\n" + "\n\n".join(extra)

        return text, names

    # -- saving -----------------------------------------------

    def export(self):
        return {"messages": self.messages, "seen": sorted(self.seen), "tools": self.tools}

    def restore(self, data):
        self.messages = list(data.get("messages") or [])
        self.seen = set(data.get("seen") or [])
        self._close_turn()


_SECTION = re.compile(r"^\[file ([^\]\n]+)\][ \t]*$", re.M)


def _file_sections(path, body):
    """[(path, text)]: the body for path, then each "[file NAME]" section."""

    marks = list(_SECTION.finditer(body))

    if not marks:
        return [(path, body)]

    parts = []
    head = body[:marks[0].start()]

    if head.strip():
        parts.append((path, head))

    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(body)
        text = body[mark.end():end].strip("\n")

        if text.strip():
            parts.append((mark.group(1).strip(), text))

    return parts


class _Interrupted(KeyboardInterrupt):
    """A command the user stopped: its output still goes to the model."""

    def __init__(self, result):
        super().__init__()
        self.result = result
