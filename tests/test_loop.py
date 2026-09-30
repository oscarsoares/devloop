"""A whole tick, with no network, no Claude, and no real time passing.

Budgets are only trustworthy if they are tested, and they cannot be tested against a real
clock — which is why the clock is injected rather than called.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from devloop.events import Event, Result, Usage
from devloop.github import RecordedWrites
from devloop.loop import StopReason, Tick
from devloop.models import Budget, CheckState, Issue, PullRequest
from devloop.store import Store, open_store

REPO = "oscarsoares/altrus"
START = datetime(2026, 9, 30, 9, 0, tzinfo=UTC)


@dataclass
class FakeRepo:
    """A repository whose state the test controls, including how it reacts to work."""

    slug: str = REPO
    prs: list[PullRequest] = field(default_factory=list[PullRequest])
    issues: list[Issue] = field(default_factory=list[Issue])
    pr_calls: int = 0

    def open_pull_requests(self) -> list[PullRequest]:
        self.pr_calls += 1
        return list(self.prs)

    def open_issues(self) -> list[Issue]:
        return list(self.issues)


@dataclass
class FakeDriver:
    """Yields a canned Result per call, so a tick's control flow is what is under test."""

    results: list[Result | None] = field(default_factory=list[Result | None])
    prompts: list[str] = field(default_factory=list[str])
    name: str = "fake"

    def run(self, prompt: str, cwd: Path) -> Iterator[Event]:
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.results) - 1)
        result = self.results[index] if self.results else None
        if result is not None:
            yield result


def ok_result(cost: float | None = 0.25) -> Result:
    return Result(subtype="success", turns=3, duration_ms=1000, cost_usd=cost, usage=Usage(1, 2))


def failed_result() -> Result:
    return Result(subtype="error_during_execution", is_error=True)


@pytest.fixture
def store() -> Iterator[Store]:
    with open_store(":memory:") as opened:
        yield opened


class Clock:
    """A clock the test advances, so time budgets are testable."""

    def __init__(self, start: datetime = START, step: timedelta = timedelta(minutes=1)) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> datetime:
        self.now += self.step
        return self.now


def build(
    store: Store,
    repo: FakeRepo,
    driver: FakeDriver,
    *,
    budget: Budget | None = None,
    dry_run: bool = False,
    clock: Clock | None = None,
    writes: RecordedWrites | None = None,
) -> tuple[Tick, RecordedWrites]:
    recorded = writes or RecordedWrites()
    tick = Tick(
        repo=repo,
        writes=recorded,
        driver=driver,
        store=store,
        cwd=Path.cwd(),
        budget=budget or Budget(),
        dry_run=dry_run,
        clock=clock or Clock(),
        on_event=lambda _: None,
    )
    return tick, recorded


class TestNothingToDo:
    def test_an_empty_repository_stops_idle(self, store: Store) -> None:
        tick, _ = build(store, FakeRepo(), FakeDriver())
        outcome = tick.run()
        assert outcome.cycles == []
        assert outcome.stop_reason is StopReason.IDLE

    def test_a_pr_waiting_on_a_human_is_not_work(self, store: Store) -> None:
        repo = FakeRepo(prs=[PullRequest(number=77, labels=frozenset({"needs-review"}))])
        tick, _ = build(store, repo, FakeDriver())
        assert tick.run().stop_reason is StopReason.IDLE


class TestReviewsFirst:
    def test_a_never_reviewed_pr_is_reviewed_before_any_issue(self, store: Store) -> None:
        repo = FakeRepo(
            prs=[PullRequest(number=94)],
            issues=[Issue(number=13, labels=frozenset({"ready-to-dev", "P0-critical"}))],
        )
        driver = FakeDriver(results=[ok_result()])
        tick, _ = build(store, repo, driver, budget=Budget(max_cycles=1))
        outcome = tick.run()
        assert driver.prompts == ["/mvp-review 94"]
        assert outcome.reviews == 1
        assert outcome.developments == 0

    def test_a_completed_round_is_counted(self, store: Store) -> None:
        repo = FakeRepo(prs=[PullRequest(number=94)])
        tick, _ = build(store, repo, FakeDriver(results=[ok_result()]), budget=Budget(max_cycles=1))
        tick.run()
        assert store.rounds_for(REPO, 94) == 1

    def test_a_failed_round_is_not_counted_and_stops_the_tick(self, store: Store) -> None:
        """A run that died halfway corrected nothing; charging it would retire the PR."""
        repo = FakeRepo(prs=[PullRequest(number=94)])
        tick, _ = build(store, repo, FakeDriver(results=[failed_result()]))
        outcome = tick.run()
        assert outcome.stop_reason is StopReason.CYCLE_FAILED
        assert store.rounds_for(REPO, 94) == 0

    def test_a_run_with_no_result_is_a_failure(self, store: Store) -> None:
        repo = FakeRepo(prs=[PullRequest(number=94)])
        tick, _ = build(store, repo, FakeDriver(results=[None]))
        assert tick.run().stop_reason is StopReason.CYCLE_FAILED

    def test_a_spent_budget_takes_the_pr_out_of_the_loop(self, store: Store) -> None:
        """The reason the rounds table exists: without it this PR is reviewed forever."""
        for _ in range(3):
            store.record_review_round(REPO, 94)
        repo = FakeRepo(prs=[PullRequest(number=94, labels=frozenset({"review-blocking"}))])
        tick, _ = build(store, repo, FakeDriver(results=[ok_result()]))
        outcome = tick.run()
        assert outcome.cycles == []
        assert outcome.stop_reason is StopReason.IDLE


class TestOpeningAPrIsNotAnExitCondition:
    def test_the_loop_returns_to_review_what_it_just_produced(self, store: Store) -> None:
        """The predecessor's bug: it announced the PR and left its own output unreviewed."""
        repo = FakeRepo(issues=[Issue(number=13, labels=frozenset({"ready-to-dev"}))])

        class Reacting(FakeRepo):
            """Publishes a PR for the issue once development has run, as a real run would."""

            def open_pull_requests(self) -> list[PullRequest]:
                self.pr_calls += 1
                if any(p.startswith("/mvp-next") for p in driver.prompts):
                    return [PullRequest(number=200, head_ref="agent/issue-13-feed")]
                return []

        driver = FakeDriver(results=[ok_result(), ok_result()])
        reacting = Reacting(issues=repo.issues)
        tick, _ = build(store, reacting, driver, budget=Budget(max_cycles=2))
        outcome = tick.run()
        assert driver.prompts == ["/mvp-next 13", "/mvp-review 200"]
        assert outcome.developments == 1
        assert outcome.reviews == 1


class TestBudgets:
    def test_the_cycle_budget_stops_the_tick(self, store: Store) -> None:
        repo = FakeRepo(prs=[PullRequest(number=94)])
        driver = FakeDriver(results=[ok_result()] * 10)
        tick, _ = build(store, repo, driver, budget=Budget(max_cycles=2))
        outcome = tick.run()
        assert len(outcome.cycles) == 2
        assert outcome.stop_reason is StopReason.CYCLE_BUDGET

    def test_the_time_budget_stops_the_tick(self, store: Store) -> None:
        repo = FakeRepo(prs=[PullRequest(number=94)])
        driver = FakeDriver(results=[ok_result()] * 10)
        # Each clock read advances an hour, so a 90-minute budget allows one cycle.
        clock = Clock(step=timedelta(hours=1))
        tick, _ = build(store, repo, driver, budget=Budget(max_minutes=90), clock=clock)
        outcome = tick.run()
        assert outcome.stop_reason is StopReason.TIME_BUDGET
        assert len(outcome.cycles) <= 1

    def test_a_cycle_in_flight_is_never_cut_off(self, store: Store) -> None:
        """The budget is checked between cycles, so a half-applied review cannot happen."""
        repo = FakeRepo(prs=[PullRequest(number=94)])
        clock = Clock(step=timedelta(days=1))
        tick, _ = build(store, repo, FakeDriver(results=[ok_result()]), clock=clock)
        outcome = tick.run()
        assert outcome.stop_reason is StopReason.TIME_BUDGET
        assert all(c.ok for c in outcome.cycles)


class TestLabelLifecycle:
    def test_development_claims_the_issue(self, store: Store) -> None:
        repo = FakeRepo(issues=[Issue(number=13, labels=frozenset({"ready-to-dev"}))])
        tick, writes = build(
            store, repo, FakeDriver(results=[ok_result()]), budget=Budget(max_cycles=1)
        )
        tick.run()
        assert ("issue", 13, ("in-progress",), ("ready-to-dev",)) in writes.relabels

    def test_the_claim_is_handed_over_once_a_pr_owns_the_issue(self, store: Store) -> None:
        """Closes the gap that swallowed work: in-progress was never removed on success."""
        repo = FakeRepo(
            issues=[
                Issue(number=13, labels=frozenset({"ready-to-dev"}), has_open_pr=False),
            ]
        )

        def after_development() -> list[Issue]:
            return [Issue(number=13, labels=frozenset({"in-progress"}), has_open_pr=True)]

        tick, writes = build(
            store, repo, FakeDriver(results=[ok_result()]), budget=Budget(max_cycles=1)
        )
        repo.issues = [Issue(number=13, labels=frozenset({"ready-to-dev"}))]
        original = repo.open_issues
        calls = {"n": 0}

        def sequenced() -> list[Issue]:
            calls["n"] += 1
            return original() if calls["n"] == 1 else after_development()

        repo.open_issues = sequenced  # pyright: ignore[reportAttributeAccessIssue]
        tick.run()
        assert ("issue", 13, ("needs-review",), ("in-progress",)) in writes.relabels

    def test_the_claim_is_kept_when_no_pr_took_the_work(self, store: Store) -> None:
        """Clearing in-progress with no PR would put the issue back in the queue unfinished."""
        repo = FakeRepo(issues=[Issue(number=13, labels=frozenset({"ready-to-dev"}))])
        tick, writes = build(
            store, repo, FakeDriver(results=[ok_result()]), budget=Budget(max_cycles=1)
        )
        tick.run()
        handovers = [r for r in writes.relabels if "needs-review" in r[2]]
        assert handovers == []

    def test_a_failed_development_escalates_visibly(self, store: Store) -> None:
        repo = FakeRepo(issues=[Issue(number=13, labels=frozenset({"ready-to-dev"}))])
        tick, writes = build(store, repo, FakeDriver(results=[failed_result()]))
        outcome = tick.run()
        assert outcome.stop_reason is StopReason.CYCLE_FAILED
        assert ("issue", 13, ("blocked",), ("in-progress",)) in writes.relabels
        assert writes.comments
        assert [e.number for e in store.open_escalations(REPO)] == [13]


class TestDryRun:
    def test_it_writes_nothing_anywhere(self, store: Store) -> None:
        repo = FakeRepo(
            prs=[PullRequest(number=94, checks=(CheckState.PASSED,))],
            issues=[Issue(number=13, labels=frozenset({"ready-to-dev"}))],
        )
        driver = FakeDriver(results=[ok_result()])
        tick, writes = build(store, repo, driver, dry_run=True)
        outcome = tick.run()

        assert driver.prompts == []
        assert writes.relabels == []
        assert writes.comments == []
        assert store.spend(REPO).cycles == 0
        assert store.rounds_for(REPO, 94) == 0
        assert outcome.stop_reason is StopReason.DRY_RUN

    def test_it_still_reports_the_decision(self, store: Store) -> None:
        repo = FakeRepo(prs=[PullRequest(number=94)])
        decisions: list[str] = []
        tick = Tick(
            repo=repo,
            writes=RecordedWrites(),
            driver=FakeDriver(),
            store=store,
            cwd=Path.cwd(),
            dry_run=True,
            clock=Clock(),
            on_event=decisions.append,
        )
        tick.run()
        assert any("review PR #94" in line for line in decisions)


def test_spend_is_reported_from_the_store(store: Store) -> None:
    repo = FakeRepo(prs=[PullRequest(number=94)])
    driver = FakeDriver(results=[ok_result(cost=0.5)])
    tick, _ = build(store, repo, driver, budget=Budget(max_cycles=1))
    outcome = tick.run()
    assert outcome.spend.cycles == 1
    assert outcome.spend.cost_usd == 0.5
