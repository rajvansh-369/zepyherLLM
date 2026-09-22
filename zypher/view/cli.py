"""
The console chat.

Reads a line, runs commands itself, and hands everything else to the
Assistant. All it keeps is the bare conversation and what the last answer
left behind for 'continue' and '/good'.
"""

import sys
import time

import torch

from zypher import config
from zypher.model import retrieval
from zypher.presenter import Assistant

HELP = """\
Type 'exit' to quit.
Type 'clear' to clear conversation (memory survives it).
Type 'continue' to extend a reply that hit the budget.
Type '/tokens N' to change the answer budget.
Type '/web' to toggle live web lookup (now: {web}).
Type '/web <question>' to force a lookup for one question.
Type '/good' or '/bad' to rate the last answer -- this is how the
     runner learns which of its own replies to lean on again.
Type '/remember <fact>' to keep something about you for good.
Type '/forget last|all|<text>' to drop memories.
Type '/memory' for what has been learned, '/memory off' to stop
     recalling it (now: {memory})."""

COMMANDS = ("/tokens", "/web", "/memory", "/good", "/bad", "/remember", "/forget")


class ConsoleStream:
    """Writes streamed text to stdout, flushing at most every STREAM_FLUSH_SECONDS."""

    def __init__(self, interval=config.STREAM_FLUSH_SECONDS):
        self.interval = interval
        self.pending = []
        self.last_flush = time.monotonic()

    def __call__(self, text):
        self.pending.append(text)

        now = time.monotonic()

        if now - self.last_flush >= self.interval:
            self.flush()
            self.last_flush = now

    def flush(self):
        if self.pending:
            sys.stdout.write("".join(self.pending))
            del self.pending[:]

        sys.stdout.flush()


def on_off(flag):
    return "on" if flag else "off"


def print_memory_stats(assistant):
    stats = assistant.memory_stats()

    if stats is None or not stats["ready"]:
        print("Memory: loading its encoder in the background...")
        return

    print("Memory: {} stored ({} exchanges, {} notes, {} rated), encoder {}.".format(
        stats["total"], stats["exchanges"], stats["notes"], stats["rated"], stats["encoder"],
    ))
    print("        {}".format(stats["dir"]))

    if stats["error"] is not None:
        print("        last error: {}".format(stats["error"]))


def print_reply_stats(result):
    usage, meta = result["usage"], result["meta"]

    if meta["seconds"] > 0 and usage["completion_tokens"] > 0:
        print("\n[{} tokens, {:.1f} tok/s{}]".format(
            usage["completion_tokens"],
            usage["completion_tokens"] / meta["seconds"],
            ", {} rounds".format(meta["rounds"]) if meta["rounds"] > 1 else "",
        ))
    else:
        print()

    if result["finish_reason"] == "length":
        print(
            "[stopped at the {}-token answer budget -- type 'continue' for "
            "more, or '/tokens N' to raise it]".format(config.MAX_TOTAL_NEW_TOKENS)
        )


def print_sources(meta):
    """List the web sources behind an answer, so its [n] can be checked."""

    if not meta["sources"]:
        return

    print("Sources:" if meta["cited"] else "Searched:")

    for source in meta["sources"]:
        print("  [{}] {} -- {}".format(source["n"], source["title"], source["url"]))


class Console:

    def __init__(self, assistant):
        self.assistant = assistant
        self.messages = []
        self.truncated = False
        self.last_memory_id = None
        self.last_live = False

    # -- commands ---------------------------------------------

    def clear(self):
        self.messages = []
        self.truncated = False
        self.last_memory_id = None
        self.last_live = False
        self.assistant.reset()
        print("Conversation cleared.")

    def cmd_tokens(self, argument):
        if not argument.isdigit() or int(argument) < 1:
            print("Usage: /tokens N   (current: {})".format(config.MAX_TOTAL_NEW_TOKENS))
            return

        config.MAX_TOTAL_NEW_TOKENS = int(argument)
        print("Answer budget is now {} tokens.".format(config.MAX_TOTAL_NEW_TOKENS))
        print(
            "Each context token costs ~128 KB of VRAM, so if generation "
            "starts running out of memory, lower it again or 'clear' first."
        )

    def cmd_web(self, argument):
        """'/web' toggles; '/web <question>' forces a lookup for that one."""

        if argument:
            self.ask(argument, web=True)
            return

        config.RETRIEVAL_ENABLED = not config.RETRIEVAL_ENABLED

        # Switching it back on is also how a user says "the network is back"
        # after repeated failures paused lookups.
        if config.RETRIEVAL_ENABLED:
            retrieval.reset_failures()

        print("Live web lookup is now {}.".format(on_off(config.RETRIEVAL_ENABLED)))

    def cmd_memory(self, argument):
        # '/memory' alone reads as a question about the state of things, so it
        # reports and changes nothing.
        if argument.lower() in ("on", "off"):
            config.MEMORY_ENABLED = argument.lower() == "on"
            print("Memory is now {}.".format(on_off(config.MEMORY_ENABLED)))
        elif argument:
            print("Usage: /memory [on|off]")
            return
        else:
            print("Memory is {}.".format(on_off(config.MEMORY_ENABLED)))

        print_memory_stats(self.assistant)

    def cmd_rate(self, good):
        if self.last_memory_id is None:
            print(
                "Answers about the present are not kept in memory -- they go "
                "stale -- so there is nothing to rate."
                if self.last_live else "Nothing to rate yet -- ask something first."
            )
            return

        record = self.assistant.rate(self.last_memory_id, good)

        if record is None:
            print("That answer is no longer in memory.")
        elif good:
            print("Noted -- that answer now ranks higher when a similar question "
                  "comes up (score {}).".format(record["score"]))
        else:
            # A thumbs-down excludes rather than nudges: the model should stop
            # being shown that answer, not be shown it slightly less often.
            print("Noted -- that answer is now excluded from recall (score {}). "
                  "Ask again for a fresh attempt.".format(record["score"]))

    def cmd_remember(self, argument):
        if not argument:
            print("Usage: /remember <something worth keeping>")
            return

        record = self.assistant.note(argument)

        if record is None:
            print("Memory is off or unavailable -- nothing stored.")
        else:
            self.last_memory_id = record["id"]
            print("Stored.")

    def cmd_forget(self, argument):
        removed = self.assistant.forget(argument or "last")

        print("Forgot {} memor{}.".format(removed, "y" if removed == 1 else "ies"))

        if removed:
            self.last_memory_id = None

    # -- answering --------------------------------------------

    def ask(self, question, web="auto"):
        self.messages.append({"role": "user", "content": question})

        if not self.generate(web=web):
            self.messages.pop()

    def resume(self):
        if not self.truncated or not self.messages or self.messages[-1]["role"] != "assistant":
            print("Nothing to continue -- the last reply finished on its own.")
            return

        self.generate(continuing=True)

    def generate(self, web="auto", continuing=False):
        """Run one answer. Returns False if it failed and nothing was added."""

        stream = ConsoleStream()
        started = [False]

        def on_event(kind, payload):
            if kind == "status":
                if payload["stage"] == "searching":
                    print("[searching the web...]", flush=True)
                elif payload["stage"] == "recalled":
                    count = payload["count"]
                    print("[recalled {} memor{}]".format(count, "y" if count == 1 else "ies"))
                return

            if not started[0]:
                print("\nAI: ", end="", flush=True)
                started[0] = True

            stream(payload)

        try:
            try:
                result = self.assistant.answer(self.messages, web=web, on_event=on_event)
            finally:
                stream.flush()

        except KeyboardInterrupt:
            print("\n[interrupted]")
            return False

        except torch.cuda.OutOfMemoryError:
            print(
                "\nOut of GPU memory. Lower the budget with '/tokens N', or "
                "type 'clear' to reset the conversation."
            )
            return False

        except Exception as error:
            print("\nGeneration error: {}: {}".format(type(error).__name__, error))
            return False

        if not started[0]:
            print("\nAI: ", end="")

        meta = result["meta"]

        if continuing:
            self.messages[-1]["content"] += result["text"]
        else:
            self.messages.append({"role": "assistant", "content": result["text"]})

        # Turns that no longer fit the budget will never be sent again, and
        # re-encoding them every turn only to drop them costs time.
        kept = meta["history_kept"] + (0 if continuing else 1)
        self.messages = self.messages[-kept:]

        self.truncated = result["finish_reason"] == "length"
        self.last_live = meta["live"]
        self.last_memory_id = meta["memory_id"]

        print_reply_stats(result)
        print_sources(meta)

        for note in meta["notes_captured"]:
            print("[noted: {}]".format(note))

        return True

    # -- loop -------------------------------------------------

    def run(self):
        print("\n" + "=" * 60)
        print("MODEL READY")
        print("=" * 60)
        print("Answer budget: {} tokens (prompt budget: {}).".format(
            config.MAX_TOTAL_NEW_TOKENS, config.MAX_PROMPT_TOKENS
        ))
        print_memory_stats(self.assistant)
        print("-" * 60)
        print(HELP.format(web=on_off(config.RETRIEVAL_ENABLED),
                          memory=on_off(config.MEMORY_ENABLED)))
        print("=" * 60)

        while True:
            try:
                line = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                return

            if not line:
                continue

            lowered = line.lower()

            if lowered in ("exit", "quit"):
                print("Exiting...")
                return

            if lowered == "clear":
                self.clear()
                continue

            if lowered == "continue":
                self.resume()
                continue

            command, _, argument = line.partition(" ")
            command = command.lower()
            argument = argument.strip()

            if command == "/tokens":
                self.cmd_tokens(argument)
            elif command == "/web":
                self.cmd_web(argument)
            elif command == "/memory":
                self.cmd_memory(argument)
            elif command in ("/good", "/bad"):
                self.cmd_rate(command == "/good")
            elif command == "/remember":
                self.cmd_remember(argument)
            elif command == "/forget":
                self.cmd_forget(argument)
            elif command.startswith("/"):
                # Rejected rather than sent to the model, so a typo'd /good
                # does not silently become a question.
                print("Unknown command: {}. Known: {}".format(command, " ".join(COMMANDS)))
            else:
                self.ask(line)


def main():
    assistant = Assistant()

    try:
        assistant.load()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1
    except Exception as error:
        print("\nCould not start: {}: {}".format(type(error).__name__, error))
        return 1

    Console(assistant).run()

    return 0
