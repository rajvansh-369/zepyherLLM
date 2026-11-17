"""
The `zy` command: arguments, the interactive loop and its slash commands.

    zy                       interactive session in the current folder
    zy "fix the failing test"    start with a request
    zy -c                    carry on this folder's last conversation
    zy -p "explain main.py"  one request, answer on stdout, then exit
    git diff | zy -p "review this"

Everything that needs the model goes through Agent; this file only decides
what to ask it, and handles what never reaches it: /commands, !commands, and
getting the API up.
"""

import argparse
import os
import sys
import threading
import time

from . import __version__, server, session
from .agent import MODES, RUN_TIMEOUT, Agent
from .api import Api, ApiError, Offline
from .settings import Settings
from .tools import IGNORED_DIRS, Workspace

COMMANDS = [
    ("/help", "this list"),
    ("/clear", "start a new conversation"),
    ("/compact", "trim the conversation to free context"),
    ("/continue", "carry on an answer that was cut short"),
    ("/undo", "undo the last file change"),
    ("/mode [ask|edits|all]", "what may happen without asking (shift+tab cycles)"),
    ("/chat", "switch between the coding agent and plain chat"),
    ("/web [on|off|auto|QUESTION]", "web lookup, or search for one question"),
    ("/tokens [N|auto]", "longest reply, in tokens"),
    ("/remember TEXT", "keep a fact about you in the model's memory"),
    ("/forget [last|all|TEXT]", "drop memories"),
    ("/good, /bad", "rate the last plain-chat answer"),
    ("/memory", "what the model remembers about you"),
    ("/status", "server, model and context"),
    ("/exit", "leave (also ctrl+d)"),
]

TIPS = "/help for commands · @file attaches a file · !cmd runs a command · ctrl+d to exit"


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="zy",
        description="zypherLL in the terminal: a coding agent on your local model.",
    )
    parser.add_argument("prompt", nargs="*", help="a request to start with")
    parser.add_argument("-p", "--print", dest="once", action="store_true",
                        help="answer one request on stdout and exit")
    parser.add_argument("-c", "--continue", dest="resume", action="store_true",
                        help="carry on this folder's last conversation")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="allow file changes and commands without asking")
    parser.add_argument("--mode", choices=sorted(MODES), help="ask (default), edits or all")
    parser.add_argument("--chat", action="store_true", help="plain chat, no tools")
    parser.add_argument("--web", action="store_true", help="search the web for the first request")
    parser.add_argument("-C", "--cwd", metavar="DIR", help="project folder (default: here)")
    parser.add_argument("--url", help="API address (default: from .env, http://127.0.0.1:8000)")
    parser.add_argument("--max-tokens", type=int, metavar="N", help="longest reply, in tokens")
    parser.add_argument("--no-start", action="store_true",
                        help="do not start the API when it is not running")
    parser.add_argument("--version", action="version", version="zy " + __version__)

    return parser.parse_args(argv)


# ============================================================
# THE API
# ============================================================

def connect(api, settings, ui, autostart=True):
    """Make sure the API is up and the model loaded. Returns /health, or None
    after an error that has been reported."""

    def report(error):
        if isinstance(error, Offline):
            ui.error("{}. Start it with: python main.py serve".format(error.message))
        else:
            ui.error(error.message)

    try:
        try:
            health = api.health()
        except Offline as error:
            if not autostart or not server.can_start(settings):
                report(error)
                return None

            log = server.start(settings)
            ui.note("  Starting the zypherLL API (log: {})".format(log))
            health = None

        if health is None or health.get("status") != "ready":
            health = _wait(api, ui)

        # /health needs no key; this does, so a wrong one is caught here
        # rather than as an empty first answer.
        api.settings()

    except ApiError as error:
        if error.status == 401:
            ui.error("The API rejected the key. ZYPHER_API_KEY must match the server's .env.")
        else:
            report(error)

        return None

    except KeyboardInterrupt:
        ui.note("  Interrupted.")
        return None

    return health


def _wait(api, ui):
    label = "Loading the model (the first start downloads it)"

    if not ui.interactive:
        ui.note(label + "...")
        return server.wait_ready(api)

    with ui.console.status(label + "…  ctrl+c to stop waiting", spinner="dots"):
        return server.wait_ready(api)


# ============================================================
# SLASH COMMANDS
# ============================================================

class Quit(Exception):
    pass


class Session:
    """One run of the interactive loop: the agent and what surrounds it."""

    def __init__(self, settings, api, agent, ui, health):
        self.settings = settings
        self.api = api
        self.agent = agent
        self.ui = ui
        self.health = health or {}

    # -- helpers ----------------------------------------------

    def save(self):
        session.save(self.settings, self.agent.ws.root, self.agent.export())

    def ask(self, text, web=None):
        text, names = self.agent.attach(text)

        for name in names:
            self.ui.note("  attached {}".format(name))

        try:
            self.agent.ask(text, web=web)
        except KeyboardInterrupt:
            self.agent._close_turn("(interrupted)")
            self.ui.note("  Interrupted.")
        finally:
            self.save()

    def call(self, function, *args):
        """An API call whose failure is reported, not raised."""

        try:
            return function(*args)
        except ApiError as error:
            self.agent.report(error)
            return None

    def cycle_mode(self):
        order = list(MODES)
        self.agent.auto = order[(order.index(self.agent.auto) + 1) % len(order)]

    def toolbar(self):
        agent = self.agent

        return "  {} · {} · context {:.0f}%".format(
            "agent" if agent.tools else "chat",
            MODES[agent.auto] if agent.tools else "no tools",
            agent.usage() * 100,
        )

    # -- input ------------------------------------------------

    def handle(self, line):
        line = line.strip()

        if not line:
            return

        if line.startswith("!"):
            self.shell(line[1:].strip())
        elif line.startswith("/") and not line.startswith("//"):
            self.command(line)
        else:
            self.ask(line)

    def shell(self, command):
        """Run a command for the user; the model is told what it printed."""

        if not command:
            return

        agent, ui = self.agent, self.ui
        ui.call("Run({})".format(command))
        output, code, note = agent.ws.run(command, RUN_TIMEOUT, ui.output)
        ui.output_end()

        status = note or "exit {}".format(code)
        ui.outcome(status, ok=code == 0)

        agent.pending += "[The user ran `{}` themselves. Output:\n{}\n[{}]]\n\n".format(
            command, agent.ws.clip(output).rstrip("\n") or "(no output)", status
        )

    def command(self, line):
        name, _, rest = line.partition(" ")
        rest = rest.strip()
        method = getattr(self, "do_" + name[1:].lower().replace("-", "_"), None)

        if method is None:
            self.ui.error("Unknown command {}. /help lists them.".format(name))
            return

        method(rest)

    # -- commands ---------------------------------------------

    def do_help(self, rest):
        width = max(len(usage) for usage, _ in COMMANDS)

        self.ui.gap()

        for usage, what in COMMANDS:
            self.ui.note("  {}  {}".format(usage.ljust(width), what))

        self.ui.note("")
        self.ui.note("  @path in a request attaches that file. !command runs a command yourself.")
        self.ui.note("  esc then enter adds a line. A ZYPHER.md in the project is given to the model.")

    def do_exit(self, rest):
        raise Quit()

    do_quit = do_exit
    do_q = do_exit

    def do_clear(self, rest):
        self.agent.clear()
        self.agent.rebuild_system()
        self.ui.note("  New conversation.")

    do_new = do_clear

    def do_compact(self, rest):
        before = self.agent.usage()
        cut = self.agent.compact(force=True)
        self.save()

        if cut:
            self.ui.note("  Context {:.0f}% -> {:.0f}%.".format(before * 100, self.agent.usage() * 100))
        else:
            self.ui.note("  Nothing to trim.")

    def do_continue(self, rest):
        try:
            self.agent.resume()
        except KeyboardInterrupt:
            self.agent._close_turn("(interrupted)")
            self.ui.note("  Interrupted.")
        finally:
            self.save()

    def do_undo(self, rest):
        shown = self.agent.revert()
        self.ui.note("  Restored {}.".format(shown) if shown else "  Nothing to undo.")

    def do_mode(self, rest):
        if rest:
            choice = {"edit": "edits", "yes": "all", "auto": "all"}.get(rest.lower(), rest.lower())

            if choice not in MODES:
                self.ui.error("Modes: {}.".format(", ".join(MODES)))
                return

            self.agent.auto = choice

        self.ui.note("  Mode: {} ({}).".format(self.agent.auto, MODES[self.agent.auto]))

    def do_chat(self, rest):
        agent = self.agent
        agent.tools = not agent.tools
        agent.clear()
        self.ui.note(
            "  Coding agent: it can read and change this folder. New conversation."
            if agent.tools else
            "  Plain chat: no tools, memory and web lookup as the server has them. New conversation."
        )

    do_agent = do_chat

    def do_web(self, rest):
        choice = rest.lower()

        if choice in ("on", "off", "auto", ""):
            if choice:
                self.agent.web = {"on": True, "off": False, "auto": None}[choice]

            state = {True: "on", False: "off", None: "auto"}[self.agent.web]
            self.ui.note("  Web lookup: {}{}.".format(
                state, " (off for agent turns, automatic in plain chat)" if state == "auto" else ""
            ))
            return

        self.ask(rest, web=True)

    def do_tokens(self, rest):
        if rest.lower() in ("auto", "default", "off"):
            self.agent.max_tokens = None
        elif rest:
            if not rest.isdigit() or int(rest) < 1:
                self.ui.error("Usage: /tokens N, or /tokens auto.")
                return

            self.agent.max_tokens = int(rest)

        self.ui.note("  Longest reply: {}.".format(
            "{} tokens".format(self.agent.max_tokens) if self.agent.max_tokens else "the server's default"
        ))

    def do_remember(self, rest):
        if not rest:
            self.ui.error("Usage: /remember TEXT")
            return

        if self.call(self.api.note, rest) is not None:
            self.agent.refresh()
            self.ui.note("  Remembered.")

    def do_forget(self, rest):
        reply = self.call(self.api.forget, rest or "last")

        if reply is not None:
            self.agent.refresh()
            self.ui.note("  Removed {} memor{}.".format(
                reply.get("removed"), "y" if reply.get("removed") == 1 else "ies"
            ))

    def _rate(self, good):
        record = ((self.agent.last or {}).get("meta") or {}).get("memory_id")

        if record is None:
            self.ui.note("  Nothing to rate: only plain-chat answers are kept in memory (/chat).")
            return

        if self.call(self.api.rate, record, good) is not None:
            self.ui.note("  Rated {}.".format("good" if good else "bad"))

    def do_good(self, rest):
        self._rate(True)

    def do_bad(self, rest):
        self._rate(False)

    def do_memory(self, rest):
        reply = self.call(self.api.memory)

        if reply is None:
            return

        stats = reply.get("stats") or {}
        profile = reply.get("profile") or []

        self.ui.gap()

        if stats:
            self.ui.note("  " + " · ".join(
                "{} {}".format(stats[key], key)
                for key in ("total", "notes", "exchanges", "rated") if key in stats
            ) + ("" if stats.get("enabled", True) else " · memory is off"))

        for note in profile:
            self.ui.note("  - {}".format(note))

        if not profile:
            self.ui.note("  No notes about you yet. /remember TEXT adds one.")

    def do_status(self, rest):
        agent, ui = self.agent, self.ui
        health = self.call(self.api.health) or {}
        self.health = health or self.health

        ui.gap()
        ui.note("  API       {} ({})".format(self.api.url, health.get("status", "not answering")))

        engine = _engine(self.health)

        if engine:
            ui.note("  Model     {}".format(engine))

        ui.note("  Folder    {}".format(agent.ws.root))
        ui.note("  Shell     {}".format(agent.ws.shell.name))
        ui.note("  Session   {}".format("coding agent" if agent.tools else "plain chat"))
        ui.note("  Mode      {} ({})".format(agent.auto, MODES[agent.auto]))
        ui.note("  Context   {:.0f}% of {} tokens, {} messages".format(
            agent.usage() * 100, agent.budget, len(agent.messages)
        ))


def _engine(health):
    parts = [str(health[key]) for key in ("backend", "gpu") if health.get(key)]

    if not health.get("gpu") and health.get("device"):
        parts.append(str(health["device"]))

    if health.get("context"):
        parts.append("{} token context".format(health["context"]))

    return " · ".join(parts)


# ============================================================
# COMPLETION
# ============================================================

def completer(workspace):
    """Slash commands at the start of the line, files after an @."""

    from prompt_toolkit.completion import Completer, Completion

    names = sorted({usage.split()[0].rstrip(",") for usage, _ in COMMANDS} | {"/good", "/bad"})

    class _Completer(Completer):

        def get_completions(self, document, event):
            text = document.text_before_cursor

            if text.startswith("/") and " " not in text:
                for name in names:
                    if name.startswith(text):
                        yield Completion(name, start_position=-len(text))

                return

            word = text.split()[-1] if text and not text[-1].isspace() else ""

            if not word.startswith("@"):
                return

            partial = word[1:]
            folder, _, prefix = partial.rpartition("/")

            try:
                entries = sorted(
                    os.scandir(str(workspace.root / folder)),
                    key=lambda e: (not e.is_dir(), e.name.lower()),
                )
            except OSError:
                return

            for entry in entries:
                is_dir = entry.is_dir()

                if is_dir and entry.name in IGNORED_DIRS:
                    continue

                if entry.name.lower().startswith(prefix.lower()):
                    yield Completion(
                        entry.name + ("/" if is_dir else ""), start_position=-len(prefix)
                    )

    return _Completer()


# ============================================================
# RUNNING
# ============================================================

STDIN_WAIT = 3.0

# The thread reading stdin, when it was given up on and may still be blocked.
_stdin_reader = None


def _stdin_text(have_prompt):
    """What was piped in, if anything.

    Stdin that is not a terminal is not always a pipe with something in it:
    started from a script or a scheduler it can be a handle that stays open
    and never sends a byte. So when there is already a request on the command
    line, input gets a few seconds to start arriving; once it has, all of it
    is read however long that takes.
    """

    try:
        if sys.stdin is None or sys.stdin.isatty():
            return ""
    except (OSError, ValueError):
        return ""

    started = threading.Event()
    done = threading.Event()
    chunks = []

    def pump():
        try:
            source = getattr(sys.stdin, "buffer", None)

            while True:
                chunk = source.read1(65536) if source is not None else sys.stdin.read()

                if not chunk:
                    break

                chunks.append(chunk)
                started.set()

                if source is None:
                    break
        except (OSError, ValueError):
            pass
        finally:
            started.set()
            done.set()

    global _stdin_reader

    _stdin_reader = threading.Thread(target=pump, daemon=True)
    _stdin_reader.start()

    if not started.wait(STDIN_WAIT if have_prompt else None):
        return ""

    done.wait()

    if chunks and isinstance(chunks[0], bytes):
        return b"".join(chunks).decode("utf-8", "replace").strip()

    return "".join(chunks).strip()


def _utf8(stream):
    """Piped output on Windows defaults to a legacy code page."""

    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def main(argv=None):
    code = _main(argv)

    if _stdin_reader is not None and _stdin_reader.is_alive():
        # Still blocked on a stdin that never sent anything. The interpreter
        # cannot shut down cleanly around that read, so leave directly.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (OSError, ValueError):
                pass

        os._exit(code or 0)

    return code


def _main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    prompt = " ".join(args.prompt).strip()

    try:
        interactive = not args.once and sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        interactive = False

    if not interactive:
        piped = _stdin_text(bool(prompt))

        if piped:
            prompt = (prompt + "\n\n" + piped) if prompt else piped

        if not prompt:
            sys.stderr.write("zy: nothing to do -- give a request, or pipe one in.\n")
            return 2

    folder = os.path.abspath(args.cwd or os.getcwd())

    if not os.path.isdir(folder):
        sys.stderr.write("zy: no such folder: {}\n".format(folder))
        return 2

    _utf8(sys.stdout)
    _utf8(sys.stderr)

    # Imported here so --help and argument errors do not wait for rich.
    from .ui import Plain, Terminal

    ui = Terminal() if interactive else Plain()
    settings = Settings(args.url)
    settings.ensure_home()
    api = Api(settings.url, settings.key)

    try:
        health = connect(api, settings, ui, autostart=not args.no_start)

        if health is None:
            return 1

        workspace = Workspace(folder, settings.shell)
        workspace.backup_dir = settings.home / "backups" / time.strftime("%Y%m%d-%H%M%S")
        mode = "all" if args.yes else (args.mode or "ask")
        agent = Agent(api, workspace, ui, auto=mode, tools=not args.chat)
        agent.max_tokens = args.max_tokens
        agent.refresh()

        resumed = 0

        if args.resume:
            data = session.load(settings, workspace.root)

            if data:
                if not args.chat:
                    agent.tools = bool(data.get("tools", True))

                agent.restore(data)
                resumed = len(agent.messages)

        run = Session(settings, api, agent, ui, health)

        if not interactive:
            return _once(run, prompt, args.web)

        return _loop(run, prompt, args.web, resumed, args.resume)

    finally:
        api.close()


def _once(run, prompt, web):
    agent = run.agent
    text, _ = agent.attach(prompt)

    try:
        finished = agent.ask(text, web=True if web else None)
    except KeyboardInterrupt:
        agent._close_turn("(interrupted)")
        finished = False

    run.save()
    _backups(run)

    return 0 if finished else 1


def _backups(run):
    ws = run.agent.ws

    if ws.backed_up:
        run.ui.note("  Originals of the {} changed file{} are in {}".format(
            len(ws.backed_up), "" if len(ws.backed_up) == 1 else "s", ws.backup_dir))


def _loop(run, prompt, web, resumed, wanted_resume):
    agent, ui = run.agent, run.ui

    lines = [str(agent.ws.root)]
    engine = _engine(run.health)

    if engine:
        lines.append(engine)

    lines.append("{} · {}".format(
        "coding agent" if agent.tools else "plain chat",
        MODES[agent.auto] if agent.tools else "no tools",
    ))

    ui.banner("zypher " + __version__, lines, TIPS)

    if resumed:
        ui.note("  Carrying on the last conversation here ({} messages). /clear starts fresh.".format(resumed))
        ui.gap()
    elif wanted_resume:
        ui.note("  No saved conversation for this folder.")
        ui.gap()

    complete = completer(agent.ws)
    last_interrupt = 0.0

    if prompt:
        ui.say("> " + prompt)
        run.ask(prompt, web=True if web else None)

    while True:
        try:
            line = ui.read(
                run.settings.history_file, completer=complete,
                toolbar=run.toolbar, on_cycle=run.cycle_mode,
            )
        except KeyboardInterrupt:
            # Once clears the line; twice in a row leaves.
            if time.monotonic() - last_interrupt < 2.0:
                break

            last_interrupt = time.monotonic()
            ui.note("  (ctrl+c again, or ctrl+d, to exit)")
            continue
        except EOFError:
            break

        try:
            run.handle(line)
        except Quit:
            break
        except KeyboardInterrupt:
            ui.note("  Interrupted.")

        ui.gap()

    run.save()
    _backups(run)

    if agent.messages:
        ui.note("  Saved. `zy -c` carries on from here.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
