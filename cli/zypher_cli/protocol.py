"""
The tool protocol: what the model is told, and how its replies are read.

Zephyr 7B has no native function calling, and told about tools in the system
prompt alone it ignores them and answers as a chat model would. What does work
is showing it: a short worked example ahead of the real conversation, in the
same shape real turns take. From that it reliably opens its reply with a tool
call -- and then tends to carry on and invent the result. So the reader here
takes the first call in a reply and the caller stops the stream there.

Calls are XML-like tags rather than JSON: a file's content goes between the
tags as it is, with no escaping for a small model to get wrong.

    <read path="FILE"/>                     <list path="DIR"/>
    <search pattern="TEXT" path="DIR"/>     <run>COMMAND</run>
    <replace path="FILE" old="TEXT" new="TEXT"/>
    <write path="FILE">CONTENT</write>
    <edit path="FILE">
    <<<<<<< SEARCH
    old lines
    =======
    new lines
    >>>>>>> REPLACE
    </edit>

The reader is forgiving about what a 7B gets slightly wrong: a call wrapped in
a code fence, <read>FILE</read>, a missing closing tag at the end of a reply,
file content wrapped in ```.
"""

import re

TOOLS = ("read", "list", "search", "run", "replace", "write", "edit")

# Appended to every message the model reads, so the rule is always the most
# recent thing it saw. Kept constant: text that never changes stays in the
# server's prompt cache.
USER_REMINDER = (
    "\n\n(Use a tool call when you need to see or change something. "
    "Do not ask me to paste files: read them.)"
)
RESULT_REMINDER = "\n\n(Continue: one more tool call, or the final answer if the task is done.)"

SYSTEM = """\
In this session you are "zypher", a coding agent working in the user's terminal, inside their project folder. This overrides the format rules above: keep replies short and plain, with no headings and no long explanations unless the user asks for them.

Environment:
{environment}

You cannot see files or run anything by yourself, and the user will not paste files for you. To look at or change the project, write ONE tool call and stop. Its result arrives in the next message.

Tool calls:
<read path="FILE"/>
    Show a file. For part of a long file: <read path="FILE" lines="1-120"/>
<list path="DIR"/>
    List a folder.
<search pattern="TEXT" path="DIR"/>
    Find text in the project's files.
<run>COMMAND</run>
    Run a shell command in the project folder.
<write path="FILE">
COMPLETE FILE CONTENT
</write>
    Create a file, or replace all of it.
<edit path="FILE">
<<<<<<< SEARCH
existing lines, copied exactly
=======
the lines to put in their place
>>>>>>> REPLACE
</edit>
    Change part of a file.
<replace path="FILE" old="TEXT" new="TEXT"/>
    Replace TEXT everywhere it appears in a file. old is the exact text in the file now, new is what it becomes. Use this to rename something that is in many places.

Rules:
- One tool call per reply, and nothing after it. Never write a result yourself.
- Never guess what a file contains. Read a file before you edit it, and copy the old text from what you read.
- A file the user names is attached to their message as [file NAME]: use it, do not read it again. A long file comes with a map: line numbers and what is on them.
- To change a name or a value, first read the file to see what it is now, then replace that current text. Replace the value itself (the name written in the file), never the label next to it.
- New feature or logic: find where it belongs (search, list), read those files, then add to them with edit. Put new code in a new file with write only when it does not belong in an existing one. Match the style the project already uses.
- Changing the text of a page: edit the exact lines that hold it. Keep the tags and attributes around it as they are.
- Change only what the task asks for. Never remove code you were not asked to change.
- Do not put tool calls in code fences.
- After changing program code, run its tests or the program when that is possible. Do not run pages, styles or text files: there is nothing to run. A syntax error in a change is reported in its result: fix it first.
- When the task is done, or the question needs no tool, reply in plain text without a tool call, in a few sentences."""


def system_prompt(folder, system, shell, listing, branch=None, notes="", profile=()):
    """The system text for an agent session. Stable from turn to turn."""

    environment = [
        "- Project folder: {}".format(folder),
        "- System: {}. Commands run in {}.".format(system, shell),
    ]

    if branch:
        environment.append("- Git branch: {}".format(branch))

    environment.append("- In the folder: {}".format(listing or "(empty)"))

    text = SYSTEM.format(environment="\n".join(environment))

    if notes:
        text += "\n\nProject notes from ZYPHER.md:\n" + notes

    if profile:
        text += "\n\nAbout the user:\n" + "\n".join("- " + note for note in profile)

    return text


def result_message(title, body):
    return "[result of {}]\n{}{}".format(title, body.rstrip("\n"), RESULT_REMINDER)


# The worked example. It is sent ahead of every agent conversation and is
# never trimmed: without it the model stops using tools.
SHOTS = [
    ("user", "the greeting in hello.py is wrong, it should say Hello" + USER_REMINDER),
    ("assistant", '<read path="hello.py"/>'),
    ("user", result_message("read hello.py", 'print("Helo")')),
    ("assistant",
     '<edit path="hello.py">\n<<<<<<< SEARCH\nprint("Helo")\n=======\n'
     'print("Hello")\n>>>>>>> REPLACE\n</edit>'),
    ("user", result_message("edit hello.py", "Edited hello.py (1 line changed).")),
    ("assistant", "<run>python hello.py</run>"),
    ("user", result_message("run python hello.py", "Hello\n[exit code 0]")),
    ("assistant", "Fixed the typo in hello.py. It now prints Hello."),
]


def shots():
    return [{"role": role, "content": content} for role, content in SHOTS]


# ============================================================
# READING A REPLY
# ============================================================

_NAMES = "|".join(TOOLS)

# A call starts a line. Tolerated in front of it: indentation, a role label
# the model sometimes writes, and the backticks of an inline code span.
OPEN = re.compile(
    r"^[ \t]*(?:Assistant:[ \t]*)?`{0,3}[ \t]*<(" + _NAMES + r")(?=[\s/>])",
    re.I | re.M,
)

ATTRIBUTE = re.compile(r"""([\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>/]+))""")

# Where a reply stops being the model's own turn and becomes it playing the
# user, or inventing a tool result.
RUNAWAY = re.compile(r"^[ \t]*(?:User:|\[result of |\(Continue: )", re.M)

ROLE_LABEL = re.compile(r"^[ \t]*Assistant:[ \t]*", re.M)

BLOCK = re.compile(
    r"^<{5,9} ?SEARCH[^\n]*\n(.*?)^={5,9}[ \t]*\n(.*?)^>{5,9} ?REPLACE[^\n]*$",
    re.S | re.M,
)
XML_BLOCK = re.compile(
    r"<(?:search|old)>\n?(.*?)\n?</(?:search|old)>\s*<(?:replace|new)>\n?(.*?)\n?</(?:replace|new)>",
    re.S | re.I,
)
MARKER_LINE = re.compile(r"^(?:<{5,9}|={5,9}|>{5,9})", re.M)
FENCE_LINE = re.compile(r"^[ \t]*(?:```|~~~)[\w+-]*[ \t]*$")

PATH_KEYS = ("path", "file", "filename", "name", "dir", "folder")
OLD_KEYS = ("old", "from", "find", "search")
NEW_KEYS = ("new", "to", "with", "replace")
PATTERN_KEYS = ("pattern", "text", "query", "q", "regex")

# What old tool calls are cut down to when the conversation is compacted. A
# model that copies this from its own history into a new call is told off.
ELIDED = "[... {} lines, already written ...]"
ELIDED_PATTERN = re.compile(r"\[\.\.\. \d+ lines, already written \.\.\.\]")


class Call:
    """One tool call read out of a reply."""

    def __init__(self, name):
        self.name = name
        self.args = {}
        self.body = ""
        self.complete = False
        self.error = None

    @property
    def path(self):
        return self.args.get("path", "")

    def blocks(self):
        """[(search, replace)] of an edit."""

        found = BLOCK.findall(self.body) or XML_BLOCK.findall(self.body)

        return [(_chomp(search), _chomp(replace)) for search, replace in found]

    def label(self):
        """How the call is shown: Read(calc.py), Run(pytest -q)."""

        if self.name == "run":
            subject = " ".join(self.body.split())
        elif self.name == "search":
            subject = self.args.get("pattern", "")

            if self.path not in ("", "."):
                subject += " in " + self.path
        elif self.name == "replace":
            subject = "{}: {} -> {}".format(
                self.path, self.args.get("old", ""), self.args.get("new", "")
            )
        else:
            subject = self.path or "."

            if self.name == "read" and self.args.get("lines"):
                subject += ":" + self.args["lines"]

        if len(subject) > 90:
            subject = subject[:87] + "..."

        return "{}({})".format(self.name.capitalize(), subject)

    def title(self):
        """How its result is headed in the conversation: read calc.py."""

        if self.name == "run":
            subject = " ".join(self.body.split())[:120]
        elif self.name == "search":
            subject = self.args.get("pattern", "")
        elif self.name == "replace":
            subject = "{} in {}".format(self.args.get("old", ""), self.path)[:120]
        else:
            subject = self.path or "."

        return "{} {}".format(self.name, subject).strip()

    def render(self, elide=False):
        """The call as the model should have written it.

        Stored in the conversation in this form whatever the model actually
        typed, so its own history only ever shows it well-formed calls.
        """

        name = self.name

        if name in ("read", "list"):
            extra = ' lines="{}"'.format(self.args["lines"]) if self.args.get("lines") else ""
            return '<{} path="{}"{}/>'.format(name, self.path or ".", extra)

        if name == "search":
            return '<search pattern="{}" path="{}"/>'.format(
                self.args.get("pattern", ""), self.path or "."
            )

        if name == "run":
            return "<run>{}</run>".format(self.body)

        if name == "replace":
            return "<replace path={} old={} new={}/>".format(
                _quoted(self.path), _quoted(self.args.get("old", "")),
                _quoted(self.args.get("new", "")),
            )

        body = self.body

        if elide:
            body = ELIDED.format(body.count("\n") + 1)

        return '<{} path="{}">\n{}\n</{}>'.format(name, self.path, body.strip("\n"), name)


class Parsed:

    def __init__(self, prose, call):
        self.prose = prose
        self.call = call


def _quoted(value):
    """An attribute value in whichever quotes it does not contain."""

    return "'{}'".format(value) if '"' in value else '"{}"'.format(value)


def replace_call(path, old, new):
    """A <replace> call written out, for telling the model what to send."""

    return "<replace path={} old={} new={}/>".format(_quoted(path), _quoted(old), _quoted(new))


def _chomp(text):
    return text[:-1] if text.endswith("\n") else text


def _tag_end(text, start):
    """Scan an opening tag from just after its name.

    Returns (index after the tag, attribute text, self-closing), or None if
    the tag has not ended yet. A line break outside quotes ends it too: a
    forgotten '>' should not swallow the rest of the reply.
    """

    quote = None

    for index in range(start, len(text)):
        char = text[index]

        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == ">":
            inner = text[start:index]
            return index + 1, inner.rstrip("/"), inner.endswith("/")
        elif char == "\n":
            return index + 1, text[start:index], True

    return None


def _attributes(text):
    args = {}

    for key, double, single, bare in ATTRIBUTE.findall(text):
        key = key.lower()
        value = double or single or bare

        if key in PATH_KEYS:
            key = "path"
        elif key in PATTERN_KEYS:
            key = "pattern"

        args.setdefault(key, value)

    return args


def _unfence(body):
    """File content without the code fence a model likes to wrap it in."""

    lines = body.split("\n")

    while lines and not lines[0].strip():
        lines.pop(0)

    while lines and not lines[-1].strip():
        lines.pop()

    # Either end on its own: a reply cut short loses the closing fence, and
    # a model sometimes forgets the opening one.
    if lines and FENCE_LINE.match(lines[0]):
        lines = lines[1:]

    if lines and lines[-1].strip() in ("```", "~~~"):
        lines = lines[:-1]

    return "\n".join(lines)


def _strip_tail(text):
    """Drop what follows an unterminated call: invented results, stray fences."""

    cut = RUNAWAY.search(text)

    if cut:
        text = text[:cut.start()]

    lines = text.rstrip().split("\n")

    while lines and (FENCE_LINE.match(lines[-1]) or not lines[-1].strip()):
        lines.pop()

    return "\n".join(lines)


def parse(text, final=False):
    """Split a reply into its prose and its first tool call.

    While a reply is still streaming (final=False) a call that has started
    but not ended comes back with complete=False. With final=True whatever is
    there is taken as the whole call.
    """

    match = OPEN.search(text)

    if match is None:
        return Parsed(text, None)

    call = Call(match.group(1).lower())
    prose = text[:match.start()]
    tag = _tag_end(text, match.end())

    if tag is None:
        if not final:
            return Parsed(prose, call)

        tag = (len(text), text[match.end():], True)

    after, attribute_text, self_closing = tag
    call.args = _attributes(attribute_text)
    name = call.name
    closing = "</" + name

    # -- replace: everything is in the tag ---------------------

    if name == "replace":
        raw = {}

        for key, double, single, bare in ATTRIBUTE.findall(attribute_text):
            raw.setdefault(key.lower(), double or single or bare)

        for target, keys in (("old", OLD_KEYS), ("new", NEW_KEYS)):
            for key in keys:
                if key in raw:
                    call.args[target] = raw[key]
                    break

        if not call.path:
            call.error = 'replace needs a path: <replace path="FILE" old="TEXT" new="TEXT"/>'
        elif not call.args.get("old") or "new" not in call.args:
            call.error = (
                'replace needs old and new: <replace path="FILE" old="TEXT" new="TEXT"/>. '
                "For a change of several lines use <edit>."
            )

        call.complete = True
        return Parsed(prose, call)

    # -- read, list, search: everything is in the tag ----------

    if name in ("read", "list", "search"):
        key = "pattern" if name == "search" else "path"

        if key not in call.args and not self_closing:
            # <read>calc.py</read>
            ends = [i for i in (text.find(closing, after), text.find("\n", after)) if i != -1]

            if ends:
                call.args[key] = text[after:min(ends)].strip().strip("\"'`")
            elif final:
                call.args[key] = text[after:].strip().strip("\"'`")
            else:
                return Parsed(prose, call)

        if key not in call.args:
            # <read calc.py/>
            bare = attribute_text.strip().strip("\"'`")

            if bare and "=" not in bare:
                call.args[key] = bare

        if name == "search" and not call.args.get("pattern"):
            call.error = "search needs a pattern: <search pattern=\"TEXT\" path=\"DIR\"/>"
        elif name == "read" and not call.args.get("path"):
            call.error = "read needs a path: <read path=\"FILE\"/>"

        call.complete = True
        return Parsed(prose, call)

    # -- run, write, edit: a body up to the closing tag --------

    if name == "run" and self_closing and call.args.get("command"):
        call.body = call.args["command"]
        call.complete = True
        return Parsed(prose, call)

    end = text.lower().find(closing, after)

    if end != -1:
        body = text[after:end]
    elif final:
        body = _strip_tail(text[after:])
    else:
        call.body = text[after:]
        return Parsed(prose, call)

    if name == "run":
        call.body = _unfence(body).strip().strip("`").strip()

        if not call.body:
            call.error = "run needs a command: <run>COMMAND</run>"

    else:
        if not call.path:
            bare = attribute_text.strip().strip("\"'`")

            if bare and "=" not in bare:
                call.args["path"] = bare

        if name == "write":
            call.body = _unfence(body)
        else:
            call.body = body.strip("\n")

        if not call.path:
            call.error = '{0} needs a path: <{0} path="FILE">'.format(name)
        elif ELIDED_PATTERN.search(call.body):
            call.error = (
                "that placeholder is not real content. Write the complete, real "
                "text between the tags."
            )
        elif name == "edit" and not call.blocks() and MARKER_LINE.search(call.body):
            # A block that is there but broken. Without any markers at all
            # the body is taken as the new code itself (see merge_loose).
            call.error = (
                "edit needs a block of this shape between the tags:\n"
                "<<<<<<< SEARCH\nexisting lines\n=======\nnew lines\n>>>>>>> REPLACE"
            )

    call.complete = True
    return Parsed(prose, call)


def clean_prose(text):
    """Prose as it should be shown and stored: no role labels, and cut where
    the model starts speaking for the user."""

    cut = RUNAWAY.search(text)

    if cut:
        text = text[:cut.start()]

    return ROLE_LABEL.sub("", text).strip()


def ran_away(text):
    """True once a reply has started inventing the other side's turn."""

    return RUNAWAY.search(text) is not None


_HOLD = ("<", "`", "[result of ", "User:", "Assistant:", "(Continue: ")


def displayable(prose):
    """The part of still-streaming prose that is safe to show.

    The last line is held back while it could yet turn out to be the start of
    a tool call or of a runaway, so neither flashes on screen.
    """

    head, newline, last = prose.rpartition("\n")
    probe = last.strip()

    if probe and any(mark.startswith(probe) or probe.startswith(mark) for mark in _HOLD):
        prose = head

    return clean_prose(prose)
