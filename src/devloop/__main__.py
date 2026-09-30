"""The `devloop` command.

The tick loop is not built yet, so this exposes the one piece that is: `devloop run` drives a
single prompt through a `ClaudeDriver` and prints its events as they arrive. That is enough
to measure what a cycle costs, which is the number the driver choice is waiting on.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from devloop.drivers import ClaudeDriver
from devloop.drivers.cli import CliDriver, DriverError
from devloop.events import Event, Result, SessionStarted, Text, ToolCall, Unknown


def _version() -> str:
    try:
        return version("devloop")
    except PackageNotFoundError:
        return "unknown"


def format_event(event: Event) -> str:
    match event:
        case SessionStarted(session_id=session_id):
            return f"session {session_id or '(no id)'}"
        case ToolCall(name=name, detail=detail):
            return f"  {name}: {detail}" if detail else f"  {name}"
        case Text(text=text):
            return f"  > {text}"
        case Result():
            # A missing cost is printed as missing: the same rule the parser keeps.
            cost = "cost not reported" if event.cost_usd is None else f"${event.cost_usd:.4f}"
            usage = event.usage
            return (
                f"result {event.subtype}: {event.turns} turns, {event.duration_ms / 1000:.1f}s, "
                f"{cost}, {usage.input_tokens} in / {usage.output_tokens} out / "
                f"{usage.cache_read_tokens} cache read"
            )
        case Unknown(raw=raw):
            return f"  ? {raw}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devloop",
        description="Reviews open PRs and develops ready issues, one repository at a time.",
        epilog="The tick loop is not built yet; `run` drives a single prompt.",
    )
    parser.add_argument("--version", action="version", version=f"devloop {_version()}")
    commands = parser.add_subparsers(dest="command")

    run = commands.add_parser("run", help="run one prompt through Claude and stream the events")
    run.add_argument("prompt")
    run.add_argument("--cwd", type=Path, default=Path.cwd(), help="repository to run in")
    run.add_argument("--permission-mode", default="acceptEdits")
    return parser


def _run(prompt: str, cwd: Path, driver: ClaudeDriver) -> int:
    if not cwd.is_dir():
        print(f"devloop: {cwd} is not a directory", file=sys.stderr)
        return 1

    print(f"driver {driver.name} in {cwd}", flush=True)
    result: Result | None = None
    try:
        for event in driver.run(prompt, cwd):
            print(format_event(event), flush=True)
            if isinstance(event, Result):
                result = event
    except DriverError as exc:
        print(f"devloop: {exc}", file=sys.stderr)
        return 1

    # No Result means the run did not finish, whatever the process exit code said.
    return 0 if result is not None and result.ok else 1


def main(argv: Sequence[str] | None = None, *, driver: ClaudeDriver | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    match args.command:
        case "run":
            chosen = driver or CliDriver(permission_mode=args.permission_mode)
            return _run(args.prompt, args.cwd, chosen)
        case _:
            parser.print_help()
            return 0


if __name__ == "__main__":
    sys.exit(main())
