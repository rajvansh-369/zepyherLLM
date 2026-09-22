"""
zypherLL entry point.

    python main.py                  console chat
    python main.py serve            HTTP API on ZYPHER_HOST:ZYPHER_PORT
    python main.py serve --port 9000
"""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local Zephyr 7B assistant.")
    commands = parser.add_subparsers(dest="command")

    commands.add_parser("chat", help="console chat (the default)")

    serve = commands.add_parser("serve", help="OpenAI-compatible HTTP API")
    serve.add_argument("--host", help="default: ZYPHER_HOST or 127.0.0.1")
    serve.add_argument("--port", type=int, help="default: ZYPHER_PORT or 8000")

    args = parser.parse_args(argv)

    # Imported here: loading torch takes seconds, and --help should not.
    if args.command == "serve":
        from zypher.view.api.app import serve as run_server

        run_server(args.host, args.port)
        return 0

    from zypher.view.cli import main as run_console

    return run_console()


if __name__ == "__main__":
    sys.exit(main())
