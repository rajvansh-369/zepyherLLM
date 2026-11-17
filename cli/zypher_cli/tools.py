"""
What the tool calls do: read, list, search, write, edit, run.

Nothing here asks permission or prints -- the agent does both. Writes and
edits come in two halves, plan and commit, so the change can be shown as a
diff and approved before anything touches the disk.
"""

import codecs
import difflib
import fnmatch
import html as html_lib
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", ".idea", ".vscode", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", "target", ".gradle", ".memory", ".logs",
}

MAX_FILE_BYTES = 1000000
MAX_LIST_ENTRIES = 120
MAX_MATCHES = 40
MAX_LINE_CHARS = 240
SEARCH_SECONDS = 8.0


OUTLINE_PATTERNS = {
    "python": re.compile(r"^\s*(?:async\s+def|def|class)\s+\w+"),
    "js": re.compile(
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
        r"(?:function\b|class\s+\w|interface\s+\w|type\s+\w+\s*=|enum\s+\w"
        r"|(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>"
        r"|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?function\b"
        r"|(?:async\s+)?\w+\s*\([^)]*\)\s*\{\s*$)"
    ),
    "css": re.compile(r"^[^\s{}/*][^{}]*\{"),
    "markdown": re.compile(r"^#{1,6}\s"),
    "php": re.compile(r"^\s*(?:(?:public|private|protected|static|abstract|final)\s+)*(?:function|class|interface|trait)\s+\w+"),
}

OUTLINE_KINDS = {
    ".py": "python",
    ".js": "js", ".mjs": "js", ".cjs": "js", ".jsx": "js", ".ts": "js", ".tsx": "js",
    ".css": "css", ".scss": "css", ".less": "css",
    ".md": "markdown", ".markdown": "markdown",
    ".php": "php",
    ".html": "html", ".htm": "html", ".vue": "html", ".svelte": "html",
}

_TAG = re.compile(r"<[^>]*>")
_INLINE_HIDDEN = re.compile(r"<!--.*?-->|<(script|style)\b.*?</\1>", re.I)
_HIDDEN_OPEN = re.compile(r"<!--|<(script|style)\b", re.I)
_LANDMARK = re.compile(r"<(h[1-6]|section|header|footer|nav|main|form|title)\b[^>]*>", re.I)
_ID = re.compile(r"""\bid\s*=\s*["']([^"']+)""", re.I)


def _page_line(row):
    """What a line of a page shows: its text, or the landmark it opens."""

    text = " ".join(html_lib.unescape(_TAG.sub(" ", row)).split())

    if sum(char.isalpha() for char in text) >= 3:
        return text

    landmark = _LANDMARK.search(row)

    if landmark:
        found = _ID.search(row)
        return "<{}{}>".format(landmark.group(1).lower(), " id=" + found.group(1) if found else "")

    return ""


_DEFINITION = {
    ".py": re.compile(r"^([ \t]*)(?:async[ \t]+)?(?:def|class)[ \t]+(\w+)", re.M),
    ".js": re.compile(r"^([ \t]*)(?:export[ \t]+)?(?:async[ \t]+)?function[ \t]+(\w+)", re.M),
}
_DEFINITION[".mjs"] = _DEFINITION[".cjs"] = _DEFINITION[".ts"] = _DEFINITION[".js"]


def _defined_twice(text, suffix):
    """A name defined twice at the same level -- the later one silently
    wins, which is how an edit that added a new version next to the old one
    shows -- or ""."""

    pattern = _DEFINITION.get(suffix)

    if pattern is None:
        return ""

    seen = {}
    scope = 0   # counts top-level definitions, so methods of two classes differ

    for found in pattern.finditer(text):
        indent, name = found.group(1), found.group(2)

        if not indent:
            scope += 1

        key = (indent, name, scope if indent else 0)
        line = text.count("\n", 0, found.start()) + 1

        if key in seen:
            return "{} is defined twice, at lines {} and {}; only the later one is used. " \
                   "Remove the one that is not wanted.".format(name, seen[key], line)

        seen[key] = line

    return ""


def _node_check(suffix, text):
    handle, name = tempfile.mkstemp(suffix=suffix)

    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(text)

        done = subprocess.run(
            ["node", "--check", name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass

    if done.returncode == 0:
        return ""

    # "<file>:12\n<code>\n   ^\n\nSyntaxError: ..." -> "line 12: SyntaxError: ..."
    output = done.stdout.decode("utf-8", "replace").replace(name, "").replace("\r", "")
    line = re.search(r"^:(\d+)", output, re.M)
    error = re.search(r"^\w*Error:.*$", output, re.M)

    return "{}{}".format(
        "line {}: ".format(line.group(1)) if line else "",
        error.group(0) if error else output.strip().split("\n")[-1],
    )


class ToolError(Exception):
    """A tool call that could not be carried out. The text goes to the model."""


class Result:

    def __init__(self, body, summary, ok=True):
        self.body = body          # what the model reads
        self.summary = summary    # one line for the terminal
        self.ok = ok


# ============================================================
# SHELL
# ============================================================

def _split_chain(command):
    """Split on && outside quotes."""

    parts = []
    current = []
    quote = None
    index = 0

    while index < len(command):
        char = command[index]

        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif command.startswith("&&", index):
            parts.append("".join(current).strip())
            current = []
            index += 2
            continue

        current.append(char)
        index += 1

    parts.append("".join(current).strip())

    return [part for part in parts if part]


class Shell:
    """How a command line is run on this system."""

    def __init__(self, choice=""):
        self.kind = "sh"
        self.program = None

        if os.name != "nt":
            self.program = os.environ.get("SHELL") or "/bin/sh"
            self.name = os.path.basename(self.program)
            return

        if choice == "cmd":
            self.kind = "cmd"
            self.name = "cmd.exe"
            return

        if choice == "bash":
            for candidate in (
                os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "bin", "bash.exe"),
                os.path.join(os.environ.get("LocalAppData", ""), "Programs", "Git", "bin", "bash.exe"),
            ):
                if os.path.isfile(candidate):
                    self.program = candidate
                    self.name = "bash (Git Bash)"
                    return

        # PowerShell 7 understands &&; Windows PowerShell 5.1 does not, and
        # models write it all the time, so it is rewritten for 5.1.
        modern = shutil.which("pwsh")
        self.kind = "pwsh" if modern else "powershell"
        self.program = modern or shutil.which("powershell") or "powershell"
        self.name = "PowerShell"

    def command(self, line):
        """What to hand subprocess.Popen."""

        if self.kind == "cmd":
            return 'cmd.exe /d /s /c "{}"'.format(line)

        if self.kind == "sh":
            return [self.program, "-c", line]

        if self.kind == "powershell":
            parts = _split_chain(line)
            line = parts[-1]

            for part in reversed(parts[:-1]):
                line = "{}; if ($?) {{ {} }}".format(part, line)

        # UTF-8 out, and UTF-8 in for Get-Content and Select-String, which
        # Windows PowerShell otherwise read in the ANSI code page.
        script = (
            "[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
            "$PSDefaultParameterValues['Get-Content:Encoding'] = 'UTF8'; "
            "$PSDefaultParameterValues['Select-String:Encoding'] = 'UTF8'; " + line
        )

        return [self.program, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script]


def _kill_tree(process):
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            import signal

            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


# ============================================================
# EDITING
# ============================================================

def _indent(line):
    return line[:len(line) - len(line.lstrip())]


def apply_edit(text, search, replace):
    """text with the one place matching search replaced.

    An exact match is tried first. Failing that, lines are compared ignoring
    trailing whitespace, then ignoring indentation as well -- a model copies
    code faithfully but not its leading spaces -- and in that case the
    replacement is shifted to the indentation the file really has.
    """

    if not search.strip():
        raise ToolError(
            "the SEARCH part is empty. Put the existing lines to replace in it; "
            "to make a new file use <write>."
        )

    count = text.count(search)

    if count == 1:
        return text.replace(search, replace, 1)

    if count > 1:
        raise ToolError(
            "the SEARCH text appears {} times in the file. Add the lines around "
            "it so it matches exactly one place.".format(count)
        )

    lines = text.split("\n")
    wanted = search.split("\n")

    while wanted and not wanted[0].strip():
        wanted.pop(0)

    while wanted and not wanted[-1].strip():
        wanted.pop()

    size = len(wanted)

    for loose in (False, True):
        norm = (lambda s: s.strip()) if loose else (lambda s: s.rstrip())
        target = [norm(line) for line in wanted]
        hits = [
            start for start in range(len(lines) - size + 1)
            if [norm(line) for line in lines[start:start + size]] == target
        ]

        if len(hits) > 1:
            raise ToolError(
                "the SEARCH text matches {} places in the file. Add the lines "
                "around it so it matches exactly one.".format(len(hits))
            )

        if hits:
            start = hits[0]
            new = replace.split("\n")

            if loose:
                have, given = _indent(lines[start]), _indent(wanted[0])

                if have.startswith(given) and have != given:
                    shift = have[len(given):]
                    new = [shift + line if line.strip() else line for line in new]

            return "\n".join(lines[:start] + new + lines[start + size:])

    raise ToolError("the SEARCH text is not in the file")


def _anchor(line):
    """A line distinctive enough to place a change by: not blank, not just
    brackets or a bare keyword."""

    stripped = line.strip()

    return len(stripped) > 3 and any(char.isalnum() for char in stripped) and stripped not in (
        "else:", "try:", "pass", "return", "break", "continue", "</div>", "<div>", "end",
    )


def merge_loose(old, body):
    """old with body worked into it, for an edit sent without SEARCH/REPLACE.

    A small model asked for a change often sends the new version of the
    part it changed -- the whole file, or one function with a few lines
    around it -- instead of a SEARCH/REPLACE block. body is lined up against
    old: what it leaves out at the start and end is the rest of the file and
    stays as it is; inside, its version wins.
    """

    before = old.split("\n")
    after = _unfence(body).split("\n")

    if not any(_anchor(line) for line in after):
        raise ToolError("the edit has no code in it")

    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    anchored = [
        b for b in matcher.get_matching_blocks()
        if b.size and any(_anchor(line) for line in before[b.a:b.a + b.size])
    ]

    if not anchored:
        # One definition sent alone with its header changed: no line is in
        # the file as it is, but the header is close to one that is.
        return _replace_block(before, after)

    first, last = anchored[0], anchored[-1]
    old_span = last.a + last.size - first.a
    new_span = last.b + last.size - first.b

    # Lines that happen to match far apart would make the body stand in for
    # a stretch of the file it never showed, and delete it.
    if old_span > 2 * new_span + 10:
        raise ToolError(
            "it is not clear where in the file this goes: its lines match places "
            "{} lines apart".format(old_span)
        )

    # Before the first and after the last line it shares with the file, the
    # body either rewrites the lines next to it (a changed function header,
    # an edited last method) or adds new ones. Whichever reads closer.
    head_new = after[:first.b]
    head_old = before[:first.a]
    keep_head = len(head_old) - _closest(head_old[::-1], head_new[::-1])

    tail_new = after[last.b + last.size:]
    tail_old = before[last.a + last.size:]
    keep_tail = _closest(tail_old, tail_new)

    return "\n".join(
        before[:keep_head] + after + before[last.a + last.size + keep_tail:]
    )


_DEFINES = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:def|class|function|interface|(?:const|let|var))\s+(\w+)"
)


def _defined(lines):
    """The name the first non-blank line defines, or None."""

    for line in lines:
        if line.strip():
            found = _DEFINES.match(line)
            return found.group(1) if found else None

    return None


def _closest(old, new):
    """How many lines from the start of old the lines new are a version of:
    0 when new reads as something added rather than something changed.

    Only a whole block can be replaced: the cut falls at a blank line or at
    the end, never part way through the next function.
    """

    if not old or not [line for line in new if line.strip()]:
        return 0

    mine, theirs = _defined(new), _defined(old)

    if mine and theirs and mine != theirs:
        return 0   # a new definition next to an old one

    text = "\n".join(new)
    best, best_ratio = 0, 0.0
    limit = len(new) * 2 + 6
    named = {found.group(1) for found in map(_DEFINES.match, new) if found}

    for count in range(1, min(len(old), limit) + 1):
        if count < len(old) and old[count].strip():
            continue

        # Never swallow a definition the new text does not have a version of.
        if any(found and found.group(1) not in named for found in map(_DEFINES.match, old[:count])):
            break

        ratio = difflib.SequenceMatcher(None, "\n".join(old[:count]), text, autojunk=False).ratio()

        if ratio > best_ratio:
            best, best_ratio = count, ratio

    return best if best_ratio >= 0.5 else 0


def _indent_of(line):
    return len(line) - len(line.lstrip())


def _replace_block(before, after):
    """after put in place of the block in before whose first line is closest
    to after's first line."""

    while after and not after[-1].strip():
        after = after[:-1]

    header = next((line for line in after if line.strip()), "")
    scores = sorted(
        ((difflib.SequenceMatcher(None, line.strip(), header.strip()).ratio(), index)
         for index, line in enumerate(before) if _anchor(line)),
        reverse=True,
    )

    if not scores or scores[0][0] < 0.7 or (len(scores) > 1 and scores[1][0] > scores[0][0] - 0.05):
        raise ToolError(
            "it is not clear where in the file this goes: none of its lines are in the file"
        )

    if _defined([header]) != _defined([before[scores[0][1]]]):
        raise ToolError(
            "it is not clear where in the file this goes: none of its lines are in the file"
        )

    start = scores[0][1]
    level = _indent_of(before[start])
    end = start + 1

    # The block: everything indented deeper, then a closing line at the
    # header's own level (a "}" or an end tag).
    while end < len(before) and (not before[end].strip() or _indent_of(before[end]) > level):
        end += 1

    if end < len(before) and _indent_of(before[end]) == level \
            and before[end].strip().startswith(("}", ")", "]", "</")):
        end += 1

    while end > start + 1 and not before[end - 1].strip():
        end -= 1

    return "\n".join(before[:start] + after + before[end:])


def _unfence(body):
    lines = body.strip("\n").split("\n")

    if lines and re.match(r"^\s*(?:```|~~~)[\w+-]*\s*$", lines[0]):
        lines = lines[1:]

    if lines and lines[-1].strip() in ("```", "~~~"):
        lines = lines[:-1]

    return "\n".join(lines)


def diff_stats(old, new):
    """(added, removed) line counts."""

    added = removed = 0

    for line in difflib.ndiff((old or "").splitlines(), new.splitlines()):
        if line.startswith("+ "):
            added += 1
        elif line.startswith("- "):
            removed += 1

    return added, removed


# ============================================================
# WORKSPACE
# ============================================================

class Workspace:
    """The project folder, and the tools that work inside it."""

    def __init__(self, root, shell_choice="", result_chars=6000):
        self.root = Path(os.path.abspath(root))
        self.shell = Shell(shell_choice)
        self.result_chars = result_chars
        self.backup_dir = None      # where originals go before a change; None: nowhere
        self.backed_up = set()

    # -- paths ------------------------------------------------

    def resolve(self, raw):
        raw = (raw or "").strip().strip("\"'`")

        if not raw:
            raise ToolError("no path given")

        path = Path(os.path.expanduser(raw))

        if not path.is_absolute():
            path = self.root / path

        return Path(os.path.normpath(str(path)))

    def inside(self, path):
        root = os.path.normcase(str(self.root))
        target = os.path.normcase(str(path))

        try:
            return os.path.commonpath([root, target]) == root
        except ValueError:  # another drive
            return False

    def show(self, path):
        """A path as it is shown: relative to the project when inside it."""

        if self.inside(path):
            relative = os.path.relpath(str(path), str(self.root))
            return relative.replace("\\", "/")

        return str(path)

    # -- files ------------------------------------------------

    def load(self, path):
        """(text with \\n line ends, original newline, had BOM)."""

        if path.is_dir():
            raise ToolError("{} is a folder. Use <list path=\"{}\"/>.".format(
                self.show(path), self.show(path)))

        if not path.is_file():
            raise ToolError(self._missing(path))

        if path.stat().st_size > MAX_FILE_BYTES:
            raise ToolError("{} is larger than 1 MB, too big to read here".format(self.show(path)))

        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ToolError("cannot read {}: {}".format(self.show(path), error.strerror or error))

        if b"\x00" in raw[:4096]:
            raise ToolError("{} is a binary file".format(self.show(path)))

        bom = raw.startswith(codecs.BOM_UTF8)
        text = raw[3:].decode("utf-8", "replace") if bom else raw.decode("utf-8", "replace")
        newline = "\r\n" if "\r\n" in text else "\n"

        return text.replace("\r\n", "\n"), newline, bom

    def _missing(self, path):
        """'No such file', with the nearest names in that folder if any."""

        message = "there is no file {}".format(self.show(path))

        try:
            names = os.listdir(str(path.parent))
        except OSError:
            return message + " (and no folder {})".format(self.show(path.parent))

        close = difflib.get_close_matches(path.name, names, n=3, cutoff=0.6)

        if close:
            message += ". Did you mean: {}?".format(", ".join(close))

        return message

    def read(self, path, lines=None):
        text, _, _ = self.load(path)
        shown = self.show(path)
        rows = text.split("\n")

        if rows and rows[-1] == "":
            rows.pop()

        total = len(rows)

        if total == 0:
            return Result("(the file is empty)", "empty file")

        first, last = 1, total
        match = re.match(r"\s*(\d+)\s*(?:[-:,]\s*(\d+))?\s*$", lines or "")

        if match:
            first = max(1, int(match.group(1)))
            last = min(total, int(match.group(2))) if match.group(2) else total

            if first > total:
                raise ToolError("{} has only {} lines".format(shown, total))

        budget = self.result_chars
        outline = ""

        # A whole file that will not fit: a map of it first, so the next
        # read can go straight to the part that matters.
        if not match and len(text) > budget:
            # A page's map is its text, which is most of what gets changed.
            share = 2 if OUTLINE_KINDS.get(path.suffix.lower()) == "html" else 3
            outline = self.outline(path, rows, budget // share)
            budget -= len(outline)

        picked = []
        used = 0

        for row in rows[first - 1:last]:
            if len(row) > MAX_LINE_CHARS * 4:
                row = row[:MAX_LINE_CHARS * 4] + " [line cut]"

            used += len(row) + 1

            if used > budget and picked:
                break

            picked.append(row)

        end = first - 1 + len(picked)
        body = "\n".join(picked)

        if first == 1 and end == total:
            return Result(body, "{} line{}".format(total, "" if total == 1 else "s"))

        body = "[lines {}-{} of {}]\n{}".format(first, end, total, body)

        if end < total:
            body += '\n[more below: <read path="{}" lines="{}-{}"/>]'.format(
                shown, end + 1, min(total, end + (end - first + 1))
            )

        if outline:
            body = (
                "[map of {}, {} lines: line number and what is there. Read the "
                "part you need with <read path=\"{}\" lines=\"FROM-TO\"/>]\n{}\n\n{}".format(
                    shown, total, shown, outline, body)
            )

        return Result(body, "lines {}-{} of {}{}".format(
            first, end, total, " + map" if outline else ""))

    # -- outline ----------------------------------------------

    def outline(self, path, rows, limit):
        """Numbered lines that show how a file is laid out: definitions in
        code, headings in Markdown, visible text in a page. "" if unknown."""

        kind = OUTLINE_KINDS.get(path.suffix.lower())

        if kind is None:
            return ""

        picked = []
        used = 0
        hidden = None    # the end marker of a <script>, <style> or comment we are in

        for number, row in enumerate(rows, 1):
            if kind == "html":
                lowered = row.lower()

                if hidden:
                    if hidden in lowered:
                        hidden = None

                    continue

                row = _INLINE_HIDDEN.sub(" ", row)
                lowered = row.lower()
                opened = _HIDDEN_OPEN.search(lowered)

                if opened:
                    hidden = "-->" if opened.group(0) == "<!--" else "</" + opened.group(1) + ">"
                    row = row[:opened.start()]

                entry = _page_line(row)
            else:
                entry = row.strip() if OUTLINE_PATTERNS[kind].match(row) else ""

            if not entry:
                continue

            line = "{}: {}".format(number, entry[:110])
            used += len(line) + 1

            if used > limit:
                picked.append("[... map cut short ...]")
                break

            picked.append(line)

        return "\n".join(picked)

    # -- checks -----------------------------------------------

    def check(self, path, text):
        """A syntax error in text as the file at path, or "". Only for kinds
        of file that can be checked quickly without running anything."""

        suffix = path.suffix.lower()
        twice = _defined_twice(text, suffix)

        if twice:
            return twice

        if suffix == ".py":
            try:
                compile(text, str(path), "exec", dont_inherit=True)
            except SyntaxError as error:
                rows = text.split("\n")
                number = error.lineno or 0
                shown = rows[number - 1].strip()[:160] if 0 < number <= len(rows) else ""

                return "line {}: {}{}".format(
                    number, error.msg, "\n    {}: {}".format(number, shown) if shown else "")
            except ValueError as error:
                return str(error)

        elif suffix == ".json":
            try:
                json.loads(text)
            except ValueError as error:
                return str(error)

        elif suffix in (".js", ".mjs", ".cjs") and shutil.which("node"):
            return _node_check(suffix, text)

        return ""

    def list(self, path):
        raw = str(path)

        # <list path="src/*.py"/>
        if any(char in raw for char in "*?["):
            return self._glob(path)

        if path.is_file():
            raise ToolError("{} is a file. Use <read path=\"{}\"/>.".format(
                self.show(path), self.show(path)))

        if not path.is_dir():
            raise ToolError("there is no folder {}".format(self.show(path)))

        rows = []
        truncated = [False]

        def walk(folder, depth):
            try:
                entries = sorted(os.scandir(str(folder)), key=lambda e: (not e.is_dir(), e.name.lower()))
            except OSError:
                return

            for entry in entries:
                if len(rows) >= MAX_LIST_ENTRIES:
                    truncated[0] = True
                    return

                is_dir = entry.is_dir()

                if is_dir and entry.name in IGNORED_DIRS:
                    continue

                rows.append("  " * depth + entry.name + ("/" if is_dir else ""))

                if is_dir and depth < 1:
                    walk(entry.path, depth + 1)

        walk(path, 0)

        if not rows:
            return Result("(the folder is empty)", "empty folder")

        body = "\n".join(rows)

        if truncated[0]:
            body += "\n[more entries not shown: list a sub-folder]"

        return Result(body, "{} entr{}".format(len(rows), "y" if len(rows) == 1 else "ies"))

    def _glob(self, pattern):
        relative = self.show(pattern)
        matches = []

        for folder, dirs, files in os.walk(str(self.root)):
            dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)

            for name in sorted(files):
                shown = self.show(Path(folder) / name)

                if fnmatch.fnmatch(shown, relative) or fnmatch.fnmatch(name, relative):
                    matches.append(shown)

                    if len(matches) >= MAX_LIST_ENTRIES:
                        break

            if len(matches) >= MAX_LIST_ENTRIES:
                break

        if not matches:
            return Result("(no file matches {})".format(relative), "no matches")

        return Result("\n".join(matches), "{} file{}".format(
            len(matches), "" if len(matches) == 1 else "s"))

    def search(self, pattern, path):
        try:
            regex = re.compile(pattern, re.I)
        except re.error:
            regex = re.compile(re.escape(pattern), re.I)

        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = self._files(path)
        else:
            raise ToolError("there is no file or folder {}".format(self.show(path)))

        matches = []
        hit_files = set()
        deadline = time.monotonic() + SEARCH_SECONDS
        more = False

        for file in files:
            if time.monotonic() > deadline:
                more = True
                break

            try:
                if file.stat().st_size > MAX_FILE_BYTES:
                    continue

                raw = file.read_bytes()
            except OSError:
                continue

            if b"\x00" in raw[:4096]:
                continue

            for number, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
                if regex.search(line):
                    if len(matches) >= MAX_MATCHES:
                        more = True
                        break

                    line = line.strip()

                    if len(line) > MAX_LINE_CHARS:
                        line = line[:MAX_LINE_CHARS] + "..."

                    matches.append("{}:{}: {}".format(self.show(file), number, line))
                    hit_files.add(file)

            if more:
                break

        if not matches:
            return Result("(no matches for {})".format(pattern), "no matches")

        body = "\n".join(matches)

        if more:
            body += "\n[more matches not shown: narrow the pattern or the path]"

        return Result(body, "{} match{} in {} file{}".format(
            len(matches), "" if len(matches) == 1 else "es",
            len(hit_files), "" if len(hit_files) == 1 else "s",
        ))

    def _files(self, folder):
        for current, dirs, files in os.walk(str(folder)):
            dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)

            for name in sorted(files):
                yield Path(current) / name

    # -- changes ----------------------------------------------

    def plan_write(self, path, content):
        """(old text or None, new text) for replacing a whole file."""

        if path.is_dir():
            raise ToolError("{} is a folder".format(self.show(path)))

        old = self.load(path)[0] if path.is_file() else None
        new = content if content.endswith("\n") or not content else content + "\n"

        return old, new

    def plan_edit(self, path, blocks):
        """(old text, new text) after applying each (search, replace)."""

        old = self.load(path)[0]
        new = old

        for number, (search, replace) in enumerate(blocks, 1):
            try:
                new = apply_edit(new, search, replace)
            except ToolError as error:
                where = " (block {} of {})".format(number, len(blocks)) if len(blocks) > 1 else ""
                raise ToolError("{}{}".format(error, where))

        if new == old:
            raise ToolError("that edit changes nothing: the new lines equal the old ones")

        return old, new

    def plan_replace(self, path, find, put):
        """(old text, new text, places) for replacing find everywhere."""

        old = self.load(path)[0]

        if not find:
            raise ToolError("the old text is empty")

        if find == put:
            raise ToolError("old and new are the same text")

        # "--" or "&" is in a page's markup as much as in its text: replacing
        # it everywhere breaks comments, class names and entities.
        if sum(char.isalnum() for char in find) < 3:
            raise ToolError(
                "the old text {!r} is too short to replace everywhere: it also "
                "matches markup and code. Use <edit> on the lines you mean".format(find)
            )

        places = old.count(find)

        if not places:
            message = "the text {!r} is not in {}".format(find, self.show(path))
            other = sorted(set(re.findall(re.escape(find), old, re.I)))

            if other:
                message += ". It does contain: {}".format(", ".join(repr(o) for o in other[:5]))

            raise ToolError(message)

        return old, old.replace(find, put), places

    def near(self, path, search, context=2, limit=3):
        """The parts of a file that most resemble a SEARCH text that did not
        match, numbered, or "" when nothing comes close."""

        try:
            rows = self.load(path)[0].split("\n")
        except ToolError:
            return ""

        wanted = [line.strip() for line in search.split("\n") if len(line.strip()) > 3][:4]
        stripped = [row.strip() for row in rows]
        found = []

        for line in wanted:
            for close in difflib.get_close_matches(line, set(stripped), n=2, cutoff=0.6):
                index = stripped.index(close)

                if index not in found:
                    found.append(index)

        parts = []

        for index in sorted(found[:limit]):
            first, last = max(0, index - context), min(len(rows), index + context + 1)
            parts.append("\n".join(
                "{}: {}".format(number + 1, rows[number][:MAX_LINE_CHARS])
                for number in range(first, last)
            ))

        return "\n...\n".join(parts)

    def summary(self, path):
        """A big file's map alone, for attaching to a request: a whole first
        page of it as well would crowd out the request itself. None when the
        file is small enough to attach whole."""

        text = self.load(path)[0]

        if len(text) <= self.result_chars:
            return None

        rows = text.split("\n")
        outline = self.outline(path, rows, self.result_chars // 2)
        shown = self.show(path)

        return Result(
            "[{} has {} lines, too long to show here. Its map -- line number and "
            "what is there:]\n{}\n[Read the lines you need with <read path=\"{}\" "
            "lines=\"FROM-TO\"/>, or find text with <search pattern=\"TEXT\" "
            "path=\"{}\"/>.]".format(shown, len(rows), outline or "(no map for this kind of file)", shown, shown),
            "map",
        )

    def backup(self, path):
        """Keep the file as it was before its first change in this run."""

        if self.backup_dir is None or not path.is_file():
            return

        key = os.path.normcase(str(path))

        if key in self.backed_up:
            return

        self.backed_up.add(key)
        relative = self.show(path) if self.inside(path) else path.name
        target = self.backup_dir / relative.replace("..", "_")

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(target))
        except OSError:
            pass

    def commit(self, path, text):
        """Write text, keeping the file's own line ends and BOM."""

        newline, bom = "\n", False
        self.backup(path)

        if path.is_file():
            try:
                _, newline, bom = self.load(path)
            except ToolError:
                pass

        data = text.replace("\n", newline).encode("utf-8")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((codecs.BOM_UTF8 if bom else b"") + data)
        except OSError as error:
            raise ToolError("cannot write {}: {}".format(self.show(path), error.strerror or error))

    def restore(self, path, text):
        """Undo: put the old text back, or remove a file that did not exist."""

        if text is None:
            try:
                path.unlink()
            except OSError:
                pass
        else:
            self.commit(path, text)

    # -- commands ---------------------------------------------

    def run(self, command, timeout=120, on_output=None):
        """Run a command line. Returns (output, exit code or None, note).

        note is "" normally, "timed out" or "interrupted". Ctrl+C stops the
        command -- and everything it started -- not the CLI.
        """

        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        options = {}

        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True

        try:
            process = subprocess.Popen(
                self.shell.command(command), cwd=str(self.root), env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                **options
            )
        except OSError as error:
            raise ToolError("could not start the shell: {}".format(error))

        lines = queue.Queue()

        def pump():
            for raw in iter(process.stdout.readline, b""):
                lines.put(raw.decode("utf-8", "replace").replace("\r\n", "\n"))

            lines.put(None)

        threading.Thread(target=pump, daemon=True).start()

        output = []
        size = 0
        note = ""
        deadline = time.monotonic() + timeout

        try:
            while True:
                try:
                    line = lines.get(timeout=0.1)
                except queue.Empty:
                    if time.monotonic() > deadline:
                        note = "timed out"
                        _kill_tree(process)
                        break

                    continue

                if line is None:
                    break

                if size < 400000:
                    output.append(line)
                    size += len(line)

                if on_output is not None:
                    on_output(line)

        except KeyboardInterrupt:
            note = "interrupted"
            _kill_tree(process)

        try:
            code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            code = None

        return "".join(output), (None if note else code), note

    def clip(self, text):
        """Long output cut to its head and tail, where the errors are."""

        limit = self.result_chars

        if len(text) <= limit:
            return text

        head = text[:limit // 3]
        tail = text[-(limit * 2 // 3):]
        dropped = text[len(head):len(text) - len(tail)].count("\n")

        return "{}\n[... {} lines cut ...]\n{}".format(head.rstrip("\n"), dropped, tail.lstrip("\n"))
