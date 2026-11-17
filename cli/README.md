# zy -- zypherLL in the terminal

A coding agent that runs on your local zypherLL model. It works in the folder
you start it in: reads files, searches, edits, runs commands, and shows each
change as a diff before it is made.

It talks to zypherLL only through the HTTP API (`python main.py serve`). If the
API is not running, `zy` starts it in the background and waits for the model.

## Install

```bash
cd D:\GIT\zypherLL
pip install -e cli
```

This adds the `zy` command. Keep `-e`: it is how `zy` finds the zypherLL
checkout, to read the API key and port from its `.env` and to start the API.
Without installing, `python -m zypher_cli` from inside `cli/` does the same.

## Use

```bash
zy                              # interactive, in the current folder
zy "fix the failing test"       # start with a request
zy -c                           # carry on this folder's last conversation
zy -p "explain main.py"         # one answer on stdout, then exit
git diff | zy -p "review this"  # piped input is added to the request
zy -p -y "add type hints to calc.py"   # -y: change files without asking
```

| Option | |
|---|---|
| `-p`, `--print` | answer one request and exit; the answer goes to stdout, the working to stderr |
| `-c`, `--continue` | resume the last conversation of this folder |
| `-y`, `--yes` | allow edits and commands without asking |
| `--mode ask\|edits\|all` | what may happen without asking (default `ask`) |
| `--chat` | plain chat: no tools, with the server's memory and web lookup |
| `--web` | search the web for the first request |
| `-C DIR` | work in another folder |
| `--url URL` | API address |
| `--max-tokens N` | longest reply |
| `--no-start` | do not start the API if it is down |

Without `-y`, `zy -p` can read and search but declines edits and commands.

## In a session

- A file you name in a request (`fix cart.py`, `src/app.js`, `@notes.txt`) is
  attached to it, so the model starts from what is really in the file. A big
  file comes as a map -- line numbers with the functions, classes or visible
  page text on them -- plus its first part. Tab completes paths after `@`.
- `!command` runs a command yourself; the model is told what it printed.
- Esc then Enter adds a line. Shift+Tab cycles the permission mode.
- Ctrl+C interrupts the answer or the running command; twice at the prompt exits.
- Before a change you are asked: `y` yes, `a` always, `n` no -- or type what
  to do instead.

| Command | |
|---|---|
| `/help` | list commands |
| `/clear` | new conversation |
| `/compact` | trim the conversation to free context |
| `/continue` | carry on an answer cut short by the token budget |
| `/undo` | undo the last file change |
| `/mode [ask\|edits\|all]` | permission mode |
| `/chat` | switch between coding agent and plain chat |
| `/web [on\|off\|auto\|QUESTION]` | web lookup, or search for one question |
| `/tokens [N\|auto]` | longest reply |
| `/remember TEXT`, `/forget [last\|all\|TEXT]`, `/memory` | the model's notes about you |
| `/good`, `/bad` | rate the last plain-chat answer |
| `/status` | server, model, context |
| `/exit` | leave |

### What it checks for you

- After every change to a `.py`, `.json` or `.js` file it checks the syntax,
  and whether a function is now defined twice. A problem goes back to the
  model with the line, to fix before anything else.
- An edit sent as plain new code (no SEARCH/REPLACE block) is lined up
  against the file: the parts left out stay as they are.
- Files are only changed through its own tools. Shell commands that rewrite
  files (`Set-Content`, `Out-File`, `sed -i`, ...) are refused: Windows
  PowerShell writes them in the ANSI code page and corrupts non-English text.
- Installing packages (`pip install`, `npm install`, ...) always asks first,
  even with `-y`, and is refused in `zy -p`.
- If nothing was changed in the end, it says so, whatever the model claims.

### Asking for features

Name the file and the function, and say how to check it:
`in cart.py add a remove(name) method to Cart, add a test in test_cart.py, run pytest`.
One feature per request works better than several.

### Asking for renames

The model is a 7B: say what the text is now, not what it stands for.
`replace Dhruv with Snehal and Aishwarya with Shreya everywhere in index1.html`
works. `update the bride and groom names` makes it guess which text that is,
and it guesses badly. Replacement is exact and case-sensitive, so `dhruv` in a
URL needs its own request. `/undo` reverts the last change.

A `ZYPHER.md` in the project folder is given to the model as project notes.

### Which model

zy works with whatever GGUF the API runs. For coding, Qwen2.5-Coder 7B is far
stronger than the default Zephyr 7B and fits the same 6 GB card -- set this in
the zypherLL `.env` and restart the API (it downloads 4.7 GB once):

```
ZYPHER_MODEL_PATH=D:\AI\Models\qwen2.5-coder-7b-instruct
ZYPHER_GGUF_REPO=bartowski/Qwen2.5-Coder-7B-Instruct-GGUF
ZYPHER_GGUF_FILE=Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf
ZYPHER_MODEL_ID=qwen2.5-coder-7b
```

Remove those lines to go back to Zephyr.

## Configuration

The shell environment wins, then `cli/.env`, then `~/.zypher/.env`, then the
zypherLL `.env`. On the machine that runs the model nothing needs setting.

| Variable | |
|---|---|
| `ZYPHER_API_URL` | API address; default `http://ZYPHER_HOST:ZYPHER_PORT` |
| `ZYPHER_API_KEY` | must match the server's |
| `ZYPHER_HOME` | the zypherLL checkout, if not the parent of `cli/` |
| `ZYPHER_PYTHON` | the Python that starts the API; default: the one running `zy` |
| `ZYPHER_SHELL` | on Windows: `cmd` or `bash` (Git Bash); default PowerShell |
| `ZYPHER_CLI_HOME` | history and saved conversations; default `~/.zypher` |

A server started by `zy` logs to `.logs/api.log` and keeps running after `zy`
exits; it releases its VRAM when idle.

## Files

```
zypher_cli/
    app.py        the zy command: arguments, the loop, /commands
    agent.py      one request worked through: generate, tool call, result
    protocol.py   what the model is told about tools; reading its replies
    tools.py      read, list, search, write, edit, run
    ui.py         the terminal (rich) and the piped output
    api.py        HTTP to zypherLL
    server.py     starting the API when it is down
    session.py    saved conversations, per folder
    settings.py   address, key, folders
```
