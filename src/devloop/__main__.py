"""The `devloop` command.

`run` drives one prompt through a `ClaudeDriver` and prints its events, which is enough to
measure what a cycle costs. `status` reports what the loop would do. `tick` does it: reviews
and development until a budget or the work runs out, a dry run unless `--execute` is given.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from devloop.decide import actionable_prs, classify_pr, next_issue, rank_issues
from devloop.drivers import ClaudeDriver
from devloop.drivers.cli import CliDriver, DriverError
from devloop.events import Event, Result, SessionStarted, Text, ToolCall, Unknown
from devloop.github import GhRepository, GitHubError, RecordedWrites, Repository, Writes
from devloop.loop import StopReason, Tick, TickOutcome
from devloop.models import Budget
from devloop.store import Store, default_path, open_store


def _use_utf8_stdout() -> None:
    """Print UTF-8 regardless of the console's code page.

    A Windows console defaults to cp1252, which turns any non-ASCII character in a GitHub
    title into a replacement glyph. `errors="replace"` keeps a stream that cannot be
    reconfigured from taking the process down with it.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            # A stream that refuses is not worth failing over.
            with contextlib.suppress(OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")


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
        epilog="`tick` needs --execute to act; without it, it reports and changes nothing.",
    )
    parser.add_argument("--version", action="version", version=f"devloop {_version()}")
    commands = parser.add_subparsers(dest="command")

    run = commands.add_parser("run", help="run one prompt through Claude and stream the events")
    run.add_argument("prompt")
    run.add_argument("--cwd", type=Path, default=Path.cwd(), help="repository to run in")
    run.add_argument("--permission-mode", default="acceptEdits")

    status = commands.add_parser("status", help="report what the loop would do, changing nothing")
    status.add_argument("repo", help="owner/name, e.g. oscarsoares/altrus")
    status.add_argument("--state", type=Path, default=None, help="state database to read")

    tick = commands.add_parser("tick", help="run one tick of the loop")
    tick.add_argument("repo", help="owner/name, e.g. oscarsoares/altrus")
    tick.add_argument("--cwd", type=Path, default=Path.cwd(), help="local checkout to work in")
    tick.add_argument("--state", type=Path, default=None)
    tick.add_argument("--permission-mode", default="acceptEdits")
    tick.add_argument("--max-cycles", type=int, default=Budget().max_cycles)
    tick.add_argument("--max-minutes", type=int, default=Budget().max_minutes)
    tick.add_argument(
        "--max-cost", type=float, default=None, help="stop the tick once this many USD are spent"
    )
    # Opt in, not out. A loop that edits labels and drives Claude by default is the wrong
    # default for a tool whose whole selling point is that you can inspect it first.
    tick.add_argument(
        "--execute",
        action="store_true",
        help="actually drive Claude and write labels; omit for a dry run",
    )
    return parser


def _status(repo: Repository, budget: Budget, store: Store) -> int:
    """What the loop sees and what it would pick. Read-only by construction.

    This is deliberately the first command after `run`: a decision you can inspect without it
    acting is what makes the rest trustworthy.
    """
    fetched = repo.open_pull_requests()
    issues = (
        repo.open_issues_given(fetched) if isinstance(repo, GhRepository) else repo.open_issues()
    )
    # Rounds live only here. Without this the budget reads as zero for every PR, and a PR
    # holding its merge gate would be reviewed on every tick forever.
    prs = store.hydrate_rounds(repo.slug, fetched)

    print(f"{repo.slug}: {len(prs)} open PR(s), {len(issues)} open issue(s)")

    if prs:
        print("\nPull requests")
        for verdict in (classify_pr(pr, budget) for pr in prs):
            if verdict.reason:
                state = verdict.reason.value
            else:
                state = f"waiting - {verdict.waiting.value if verdict.waiting else '?'}"
            mark = "*" if verdict.actionable else " "
            spent = f"{verdict.pr.rounds}/{budget.max_review_rounds}"
            print(
                f" {mark} #{verdict.pr.number:<5} {spent:<5} {state:<32} "
                f"{verdict.pr.title[:52]}"
            )

    actionable = actionable_prs(prs, budget)
    ready = [i for i in issues if "ready-to-dev" in i.labels]
    candidate, from_triage = next_issue(ready, issues, budget)

    escalations = store.open_escalations(repo.slug)
    if escalations:
        # Printed before the next move on purpose: a blocked issue nobody notices is work
        # that disappeared, and it should not sit below the thing the loop is about to do.
        print("\nWaiting on you")
        for item in escalations:
            print(f"   {item.subject} #{item.number:<5} {item.reason[:70]}")

    spend = store.spend(repo.slug)
    if spend.cycles:
        per = spend.per_cycle()
        floor = " at least," if spend.partial else ""
        average = f", ${per:.4f}/cycle" if per is not None else ""
        print(
            f"\nSpend over {spend.cycles} recorded cycle(s):{floor} "
            f"${spend.cost_usd:.4f}{average}"
        )

    print("\nNext move")
    if actionable:
        first = actionable[0]
        print(
            f"  review PR #{first.pr.number} - {first.reason.value} "
            f"(iteration {first.pr.rounds + 1} of {budget.max_review_rounds})"
        )
    elif candidate is not None:
        source = "triage" if from_triage else "the ready-to-dev queue"
        print(f"  develop issue #{candidate.issue.number} from {source} - {candidate.why}")
        print(f"    {candidate.issue.title[:70]}")
        runners_up = [c for c in rank_issues(issues, budget) if c is not candidate][:3]
        if runners_up:
            trailing = ", ".join(f"#{c.issue.number} ({c.why})" for c in runners_up)
            print(f"    runners-up: {trailing}")
    else:
        print("  nothing — no actionable PR and no selectable issue")

    return 0


def _tick_summary(outcome: TickOutcome, *, dry_run: bool) -> str:
    """What one tick did, in the terms the user asks about: reviews, developments, cost."""
    if dry_run:
        # The cycles of a dry run are decisions, not runs: nothing was reviewed or developed,
        # and no cost exists to report, so neither count nor cost is shown as though it did.
        reviews, developments = "PRs that would be reviewed:", "Issues that would be developed:"
        cost = "n/a (nothing ran)"
    else:
        reviews, developments = "PRs reviewed:", "Issues developed:"
        unreported = any(c.cost_usd is None for c in outcome.cycles)
        spent = sum(c.cost_usd or 0.0 for c in outcome.cycles)
        cost = f"{'at least ' if unreported else ''}${spent:.4f}"
    lines = [
        "",
        "Dry run: nothing was written and Claude was not run." if dry_run else "Tick finished.",
        f"  {reviews:<32} {outcome.reviews}",
        f"  {developments:<32} {outcome.developments}",
        f"  {'Cost this tick:':<32} {cost}",
        f"  {'Cost, all recorded:':<32} {'at least ' if outcome.spend.partial else ''}"
        f"${outcome.spend.cost_usd:.4f} over {outcome.spend.cycles} cycle(s)",
        f"  {'Stopped:':<32} {outcome.stop_reason.value}",
    ]
    return "\n".join(lines)


def _tick(
    args: argparse.Namespace,
    *,
    driver: ClaudeDriver | None,
    repo: Repository | None,
    writes: Writes | None,
) -> int:
    if not args.cwd.is_dir():
        print(f"devloop: {args.cwd} is not a directory", file=sys.stderr)
        return 1

    dry_run = not args.execute
    chosen_repo = repo or GhRepository(slug=args.repo)
    # A dry run is handed a writer that only records, so it cannot write by construction.
    chosen_writes: Writes = (
        RecordedWrites()
        if dry_run
        else writes or (chosen_repo if isinstance(chosen_repo, GhRepository) else RecordedWrites())
    )
    budget = Budget(
        max_cycles=args.max_cycles, max_minutes=args.max_minutes, max_cost_usd=args.max_cost
    )
    chosen_driver = driver or CliDriver(permission_mode=args.permission_mode)

    mode = "dry run" if dry_run else "EXECUTING"
    print(f"{chosen_repo.slug} in {args.cwd}: {mode}, driver {chosen_driver.name}", flush=True)
    try:
        with open_store(args.state or default_path()) as store:
            tick = Tick(
                repo=chosen_repo,
                writes=chosen_writes,
                driver=chosen_driver,
                store=store,
                cwd=args.cwd,
                budget=budget,
                dry_run=dry_run,
                on_event=lambda line: print(line, flush=True),
            )
            outcome = tick.run()
    except (GitHubError, DriverError) as exc:
        print(f"devloop: {exc}", file=sys.stderr)
        return 1

    print(_tick_summary(outcome, dry_run=dry_run))
    return 1 if outcome.stop_reason is StopReason.CYCLE_FAILED else 0


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


def main(
    argv: Sequence[str] | None = None,
    *,
    driver: ClaudeDriver | None = None,
    repo: Repository | None = None,
    writes: Writes | None = None,
) -> int:
    _use_utf8_stdout()
    parser = _parser()
    args = parser.parse_args(argv)

    match args.command:
        case "run":
            chosen = driver or CliDriver(permission_mode=args.permission_mode)
            return _run(args.prompt, args.cwd, chosen)
        case "status":
            try:
                with open_store(args.state or default_path()) as store:
                    return _status(repo or GhRepository(slug=args.repo), Budget(), store)
            except GitHubError as exc:
                print(f"devloop: {exc}", file=sys.stderr)
                return 1
        case "tick":
            return _tick(args, driver=driver, repo=repo, writes=writes)
        case _:
            parser.print_help()
            return 0


if __name__ == "__main__":
    sys.exit(main())
