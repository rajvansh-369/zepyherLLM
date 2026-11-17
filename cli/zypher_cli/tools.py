"""
What the tool calls do: read, list, search, write, edit, run.

Nothing here asks permission or prints -- the agent does both. Writes and
edits come in two halves, plan and commit, so the change can be shown as a
diff and approved before anything touches the disk.
"""

import codecs
import difflib
import fnmatch
import os
import queue
import re
import shutil
import subprocess
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

        script = "[Console]::OutputEncoding = [Text.Encoding]::UTF8; " + line

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

        picked = []
        used = 0

        for row in rows[first - 1:last]:
            if len(row) > MAX_LINE_CHARS * 4:
                row = row[:MAX_LINE_CHARS * 4] + " [line cut]"

            used += len(row) + 1

            if used > self.result_chars and picked:
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

        return Result(body, "lines {}-{} of {}".format(first, end, total))

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

    def commit(self, path, text):
        """Write text, keeping the file's own line ends and BOM."""

        newline, bom = "\n", False

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
