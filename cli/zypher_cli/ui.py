"""
What the terminal shows.

Terminal draws the interactive session with rich: answers as Markdown that
streams in, tool calls as one line each with their outcome under it, diffs,
and the permission question. Plain is the same interface for `zy -p`: bare
text on stdout that can be piped, everything else on stderr.

A streamed answer is printed block by block. Finished paragraphs and code
blocks are printed for good and scroll away normally; only the block still
being written is redrawn in place. Redrawing the whole answer on every token
breaks as soon as it is taller than the window.
"""

import difflib
import sys
import time

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Heading, Markdown
from rich.panel import Panel
from rich.segment import Segment
from rich.table import Table
from rich.text import Text

ACCENT = "#8b8cf6"
GOOD = "#34d399"
BAD = "#f87171"
WARN = "#fbbf24"
DIM = "grey50"

FRAMES = ("·", "•", "●", "•")
MAX_DIFF_LINES = 60
MAX_OUTPUT_LINES = 12


class _Heading(Heading):
    """Headings on the left: rich centres them, which reads badly in a log."""

    def __rich_console__(self, console, options):
        text = self.text
        text.justify = "left"
        text.stylize("bold")

        if self.tag == "h1":
            text.stylize("underline")

        yield text


class _Markdown(Markdown):
    elements = dict(Markdown.elements, heading_open=_Heading)


def markdown(text):
    # ansi_dark has no background of its own, so code blocks sit on whatever
    # the terminal's is -- dark or light.
    return _Markdown(text, code_theme="ansi_dark")


class _Tail:
    """The last lines of a renderable: the block in progress, never taller
    than the window."""

    def __init__(self, renderable, height):
        self.renderable = renderable
        self.height = max(3, height)

    def __rich_console__(self, console, options):
        lines = console.render_lines(self.renderable, options, pad=False)

        for line in lines[-self.height:]:
            yield from line
            yield Segment.line()


def _block_end(text):
    """Index just past the last finished block of streaming Markdown, or 0.

    A block ends at a blank line outside a code fence. Lists are kept whole:
    a blank line followed by another list item does not end the block.
    """

    in_fence = False
    cut = 0
    position = 0
    blank_at = None

    for line in text.split("\n"):
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            if blank_at is not None and not in_fence:
                cut = blank_at

            in_fence = not in_fence
            blank_at = None
        elif not in_fence:
            if not stripped:
                blank_at = position + len(line) + 1
            elif blank_at is not None:
                if not _continues(stripped, line):
                    cut = blank_at

                blank_at = None

        position += len(line) + 1

    return min(cut, len(text))


def _continues(stripped, line):
    """True when a line after a blank one still belongs to the block above:
    a list item, or text indented under one."""

    if line[:1] in (" ", "\t"):
        return True

    head = stripped.split(" ", 1)[0]

    return head in ("-", "*", "+") or (head[:-1].isdigit() and head[-1:] in ".)")


def diff_lines(old, new, context=2):
    """[(kind, old number, new number, text)] for a change. kind: ' ', '+', '-', '@'."""

    before = (old or "").splitlines()
    after = new.splitlines()
    rows = []

    for group in difflib.SequenceMatcher(None, before, after, autojunk=False).get_grouped_opcodes(context):
        if rows:
            rows.append(("@", None, None, ""))

        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for offset in range(i2 - i1):
                    rows.append((" ", i1 + offset + 1, j1 + offset + 1, before[i1 + offset]))
                continue

            for index in range(i1, i2):
                rows.append(("-", index + 1, None, before[index]))

            for index in range(j1, j2):
                rows.append(("+", None, index + 1, after[index]))

    return rows


# ============================================================
# INTERACTIVE
# ============================================================

class StreamView:
    """One answer while it streams."""

    def __init__(self, ui, label):
        self.ui = ui
        self.console = ui.console
        self.label = label
        self.started = time.monotonic()
        self.shown = 0          # characters of prose printed for good
        self.tail = ""          # the block still being written
        self.tool = None        # "Write(app.py) · 12 lines" while a call streams
        self.printed = False
        self._cache = (None, None)
        self.live = Live(
            get_renderable=self._render, console=self.console,
            refresh_per_second=8, transient=True,
        )
        self.live.start()

    def status(self, label):
        self.label = label

    def prose(self, text, final=False):
        """Show the answer so far. text is all of it, not the new part."""

        if len(text) < self.shown:
            return

        pending = text[self.shown:]
        cut = len(pending) if final else _block_end(pending)

        if cut:
            block = pending[:cut].strip("\n")

            if block.strip():
                self._print(block)

            self.shown += cut

        self.tail = text[self.shown:]

    def calling(self, label):
        self.tool = label

    def end(self):
        self.tail = ""
        self.tool = None
        self.live.stop()

    def _print(self, block):
        if self.printed:
            self.console.print()

        self.console.print(self.ui.bulleted(markdown(block), not self.printed))
        self.printed = True

    def _spinner(self, text, hint=True):
        elapsed = time.monotonic() - self.started
        line = Text()
        line.append(FRAMES[int(elapsed * 5) % len(FRAMES)] + " ", style=ACCENT)
        line.append(text, style=ACCENT)

        if hint:
            line.append("  {:.0f}s · ctrl+c to interrupt".format(elapsed), style=DIM)

        return line

    def _render(self):
        parts = []

        if self.tail.strip():
            key = (self.tail, self.console.size)

            if self._cache[0] != key:
                body = self.ui.bulleted(markdown(self.tail), not self.printed)
                self._cache = (key, _Tail(body, self.console.size.height - 4))

            if self.printed:
                parts.append(Text(""))

            parts.append(self._cache[1])

        if self.tool:
            if parts:
                parts.append(Text(""))

            parts.append(self._spinner(self.tool))
        elif not parts and not self.printed:
            parts.append(self._spinner(self.label + "…"))

        return Group(*parts)


class Terminal:

    interactive = True

    def __init__(self):
        self.console = Console(highlight=False)
        self._output_lines = 0
        self._session = None

    # -- pieces -----------------------------------------------

    def bulleted(self, renderable, first=True, style=ACCENT):
        grid = Table.grid(padding=0, expand=True)
        grid.add_column(width=2, no_wrap=True)
        grid.add_column(ratio=1)
        grid.add_row(Text("●" if first else " ", style=style), renderable)

        return grid

    def banner(self, title, lines, tips):
        body = Text()
        body.append("● ", style=ACCENT)
        body.append(title, style="bold")

        for line in lines:
            body.append("\n  " + line, style=DIM)

        self.console.print()
        self.console.print(Panel(body, border_style=ACCENT, expand=False, padding=(0, 2)))
        self.console.print(Text("  " + tips, style=DIM))
        self.console.print()

    def stream(self, label="Thinking"):
        return StreamView(self, label)

    def gap(self):
        self.console.print()

    # -- tool calls -------------------------------------------

    def call(self, label):
        line = Text()
        line.append("● ", style=ACCENT)
        name, _, rest = label.partition("(")
        line.append(name, style="bold")
        line.append("(" + rest)
        self.console.print()
        self.console.print(line, soft_wrap=True)
        self._output_lines = 0

    def outcome(self, summary, ok=True):
        line = Text("  └ ", style=DIM)
        line.append(summary, style=DIM if ok else BAD)
        self.console.print(line)

    def output(self, line):
        """A line of a running command's output, as it arrives."""

        self._output_lines += 1

        if self._output_lines <= MAX_OUTPUT_LINES:
            self.console.print(Text("  │ " + line.rstrip("\n"), style=DIM), soft_wrap=True)

    def output_end(self):
        hidden = self._output_lines - MAX_OUTPUT_LINES

        if hidden > 0:
            self.console.print(Text("  │ … +{} lines".format(hidden), style=DIM))

    def diff(self, old, new):
        rows = diff_lines(old, new)
        width = len(str(max([r[1] or 0 for r in rows] + [r[2] or 0 for r in rows] + [1])))
        text = Text()

        for index, (kind, before, after, line) in enumerate(rows):
            if index >= MAX_DIFF_LINES:
                text.append("    … +{} more lines\n".format(len(rows) - index), style=DIM)
                break

            if kind == "@":
                text.append("    " + " " * width + " ⋮\n".replace("⋮", "..."), style=DIM)
                continue

            number = str(after if kind != "-" else before).rjust(width)
            style = GOOD if kind == "+" else BAD if kind == "-" else DIM
            text.append("    {} {} {}\n".format(number, kind, line), style=style)

        text.rstrip()
        self.console.print(text, soft_wrap=True)

    # -- messages ---------------------------------------------

    def note(self, text):
        self.console.print(Text(text, style=DIM))

    def say(self, renderable):
        self.console.print(renderable)

    def warn(self, text):
        self.console.print(Text("● ", style=WARN).append(text, style=WARN))

    def error(self, text):
        self.console.print(Text("● ", style=BAD).append(text, style=BAD))

    # -- questions --------------------------------------------

    def read(self, history_file, completer=None, toolbar=None, on_cycle=None):
        """One line of input from the user.

        Enter sends it; esc then enter adds a line. Shift+tab calls on_cycle
        (the permission mode). Raises KeyboardInterrupt or EOFError as the
        terminal does.
        """

        from prompt_toolkit.formatted_text import HTML

        if self._session is None:
            self._session = self._make_session(history_file)

        self._on_cycle = on_cycle

        return self._session.prompt(
            HTML("<style fg='{}'><b>&gt;</b></style> ".format(ACCENT)),
            completer=completer,
            bottom_toolbar=(lambda: toolbar()) if toolbar is not None else None,
        )

    def _make_session(self, history_file):
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory, InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.styles import Style

        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _(event):
            event.current_buffer.insert_text("\n")

        @bindings.add("s-tab")
        def _(event):
            if getattr(self, "_on_cycle", None) is not None:
                self._on_cycle()
                event.app.invalidate()

        try:
            history_file.parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(history_file))
        except OSError:
            history = InMemoryHistory()

        return PromptSession(
            history=history,
            key_bindings=bindings,
            complete_while_typing=True,
            style=Style.from_dict({"bottom-toolbar": "noreverse fg:#808080"}),
        )

    def confirm(self, question, always=None):
        """Ask before a change. Returns (answer, feedback).

        answer is "yes", "always" or "no". Anything typed that is not one of
        the choices counts as "no" with that text as what to do instead.
        """

        from prompt_toolkit import prompt
        from prompt_toolkit.formatted_text import HTML

        choices = "<b>y</b> yes"

        if always:
            choices += " · <b>a</b> {}".format(always)

        choices += " · <b>n</b> no · or say what to do instead"

        self.console.print()

        try:
            answer = prompt(HTML(
                "<style fg='{}'>?</style> {}  <style fg='gray'>{}</style>\n<style fg='{}'>&gt;</style> ".format(
                    WARN, _escape(question), choices, WARN)
            )).strip()
        except (KeyboardInterrupt, EOFError):
            return "no", ""

        lowered = answer.lower()

        if lowered in ("y", "yes", ""):
            return "yes", ""

        if always and lowered in ("a", "always"):
            return "always", ""

        if lowered in ("n", "no"):
            return "no", ""

        return "no", answer


def _escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ============================================================
# PIPED  (zy -p)
# ============================================================

class _PlainStream:

    def __init__(self):
        self.shown = 0

    def status(self, label):
        pass

    def prose(self, text, final=False):
        if len(text) > self.shown:
            sys.stdout.write(text[self.shown:])
            sys.stdout.flush()
            self.shown = len(text)

        if final and self.shown:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def calling(self, label):
        pass

    def end(self):
        pass


class Plain:
    """The answer on stdout, the working on stderr."""

    interactive = False

    def _err(self, text):
        sys.stderr.write(text + "\n")
        sys.stderr.flush()

    def banner(self, title, lines, tips):
        pass

    def stream(self, label="Thinking"):
        return _PlainStream()

    def gap(self):
        pass

    def call(self, label):
        self._err("* " + label)

    def outcome(self, summary, ok=True):
        self._err("  - " + summary)

    def output(self, line):
        self._err("  | " + line.rstrip("\n"))

    def output_end(self):
        pass

    def diff(self, old, new):
        for kind, before, after, line in diff_lines(old, new):
            if kind in "+-":
                self._err("    {} {}".format(kind, line))

    def note(self, text):
        self._err(text)

    def say(self, renderable):
        self._err(str(renderable))

    def warn(self, text):
        self._err("warning: " + text)

    def error(self, text):
        self._err("error: " + text)

    def confirm(self, question, always=None):
        # Nobody to ask. Changes need --yes.
        return "no", ""
