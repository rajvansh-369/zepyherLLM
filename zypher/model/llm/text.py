"""What every backend shares: the system prompt, reply clean-up, and errors.

Nothing here imports an inference library, so both engines -- and anything
that only wants to tidy a reply -- can use it without loading one.
"""

import datetime
import re

from zypher.config import SYSTEM_PROMPT


class ClientGone(Exception):
    """Raised from an on_text callback to stop a generation nobody is reading."""


class OutOfMemory(Exception):
    """The GPU ran out of memory mid-answer."""


class PromptTooLong(ValueError):
    """One message is larger than the context window can hold."""


def system_prompt(notes=()):
    """SYSTEM_PROMPT plus today's date and what is known about the user.

    The date is stated on every turn, not only on the ones that searched the
    web: without it the model reasons from its training year and gets ages,
    durations and "how long ago" wrong.

    The notes go here rather than into the user turn. A preference applies to
    every question, including the many it does not resemble, and text at the
    head of the prompt stays in the KV cache instead of being read again on
    each turn. Its wording must therefore stay stable from turn to turn.
    """

    parts = [
        SYSTEM_PROMPT,
        "",
        "Today's date is {}.".format(datetime.date.today().strftime("%d %B %Y")),
    ]

    if notes:
        parts.append("")
        parts.append(
            "What you know about the user -- follow their stated preferences, "
            "and use the facts only where they are relevant:"
        )
        parts.extend("- {}".format(note) for note in notes)

    return "\n".join(parts)


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
