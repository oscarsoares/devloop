"""A whole tick, with no network, no Claude, and no real time passing.

Budgets are only trustworthy if they are tested, and they cannot be tested against a real
clock — which is why the clock is injected rather than called.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
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
    context_calls: list[int] = field(default_factory=list[int])

    def open_pull_requests(self) -> list[PullRequest]:
        self.pr_calls += 1
        return list(self.prs)

    def open_issues(self) -> list[Issue]:
        return list(self.issues)

    def with_review_context(self, pr: PullRequest) -> PullRequest:
        self.context_calls.append(pr.number)
        return replace(pr, diff="+the diff", comments=("an earlier comment",))


@dataclass
class FakeDriver:
    """Yields a canned Result per call, so a tick's control flow is what is under test."""

    results: list[Result | None] = field(default_factory=list[Result | None])
    prompts: list[str] = field(default_factory=list[str])
    cwds: list[Path] = field(default_factory=list[Path])
    name: str = "fake"

    def run(self, prompt: str, cwd: Path) -> Iterator[Event]:
        self.prompts.append(prompt)
        self.cwds.append(cwd)
        index = min(len(self.prompts) - 1, len(self.results) - 1)
        result = self.results[index] if self.results else None
        if result is not None:
            yield result


def ok_result(cost: float | None = 0.25) -> Result:
    return Result(subtype="success", turns=3, duration_ms=1000, cost_usd=cost, usage=Usage(1, 2))


def review_result(decision: str = "approve", cost: float | None = 0.25) -> Result:
    text = f"Looks fine apart from one nit.\n\nDECISION: {decision}"
    return Result(
        subtype="success", turns=3, duration_ms=1000, cost_usd=cost, usage=Usage(1, 2), text=text
    )


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
        driver = FakeDriver(results=[review_result()])
        tick, _ = build(store, repo, driver, budget=Budget(max_cycles=1))
        outcome = tick.run()
        assert len(driver.prompts) == 1
        assert driver.prompts[0].startswith("Review pull request #94")
        assert outcome.reviews == 1
        assert outcome.developments == 0

    def test_a_completed_round_is_counted(self, store: Store) -> None:
        repo = FakeRepo(prs=[PullRequest(number=94)])
        tick, _ = build(
            store, repo, FakeDriver(results=[review_result()]), budget=Budget(max_cycles=1)
        )
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
        tick, _ = build(store, repo, FakeDriver(results=[review_result()]))
        outcome = tick.run()
        assert outcome.cycles == []
        assert outcome.stop_reason is StopReason.IDLE


class TestReviewCycle:
    """A review is prompt -> run -> DECISION -> comment, labels and store."""

    def run_review(
        self, store: Store, result: Result, *, budget: Budget | None = None
    ) -> tuple[FakeRepo, FakeDriver, RecordedWrites]:
        repo = FakeRepo(prs=[PullRequest(number=94, title="Fix the widget", body="Widgets leak")])
        driver = FakeDriver(results=[result])
        tick, writes = build(store, repo, driver, budget=budget or Budget(max_cycles=1))
        tick.run()
        return repo, driver, writes

    def test_the_prompt_carries_the_pr_with_its_diff_and_comments(self, store: Store) -> None:
        repo, driver, _ = self.run_review(store, review_result())
        assert repo.context_calls == [94]
        prompt = driver.prompts[0]
        for expected in ("Fix the widget", "Widgets leak", "+the diff", "an earlier comment"):
            assert expected in prompt
        assert "DECISION: approve | request_changes | block" in prompt

    def test_the_reply_is_posted_as_a_comment(self, store: Store) -> None:
        _, _, writes = self.run_review(store, review_result("approve"))
        assert writes.comments == [
            ("pr", 94, "Looks fine apart from one nit.\n\nDECISION: approve")
        ]

    def test_the_review_is_recorded_with_cost_and_decision(self, store: Store) -> None:
        self.run_review(store, review_result("request_changes", cost=0.4))
        history = store.review_history(REPO, 94)
        assert history is not None
        assert (history.rounds, history.cost_usd, history.last_decision) == (
            1,
            0.4,
            "request_changes",
        )

    def test_an_unreported_cost_is_zero_in_the_history_but_partial_in_spend(
        self, store: Store
    ) -> None:
        self.run_review(store, review_result(cost=None))
        history = store.review_history(REPO, 94)
        assert history is not None
        assert history.cost_usd == 0.0
        assert store.spend(REPO).partial

    def test_approval_hands_the_pr_to_a_human(self, store: Store) -> None:
        _, _, writes = self.run_review(store, review_result("approve"))
        assert writes.relabels == [("pr", 94, ("needs-review",), ("review-blocking",))]

    @pytest.mark.parametrize("decision", ["request_changes", "block"])
    def test_findings_hold_the_merge_gate(self, store: Store, decision: str) -> None:
        _, _, writes = self.run_review(store, review_result(decision))
        assert writes.relabels == [("pr", 94, ("review-blocking",), ("needs-review",))]

    def test_only_a_block_raises_an_escalation(self, store: Store) -> None:
        self.run_review(store, review_result("request_changes"))
        assert store.open_escalations(REPO) == []
        self.run_review(store, review_result("block"))
        assert [e.number for e in store.open_escalations(REPO)] == [94]

    def test_a_reply_without_a_decision_is_a_failed_round(self, store: Store) -> None:
        """It cost money, so the cycle is recorded; it corrected nothing, so no round is."""
        result = Result(subtype="success", cost_usd=0.3, text="Looks fine to me.")
        repo = FakeRepo(prs=[PullRequest(number=94)])
        tick, writes = build(store, repo, FakeDriver(results=[result]))
        outcome = tick.run()
        assert outcome.stop_reason is StopReason.CYCLE_FAILED
        assert writes.comments == []
        assert writes.relabels == []
        assert store.review_history(REPO, 94) is None
        assert store.rounds_for(REPO, 94) == 0
        assert store.spend(REPO).cost_usd == 0.3

    def test_a_pr_at_its_round_limit_is_not_reviewed_again(self, store: Store) -> None:
        for _ in range(3):
            store.record_review(REPO, 94, 0.1, "request_changes")
        repo = FakeRepo(prs=[PullRequest(number=94, labels=frozenset({"review-blocking"}))])
        driver = FakeDriver(results=[review_result()])
        tick, _ = build(store, repo, driver)
        assert tick.run().stop_reason is StopReason.IDLE
        assert driver.prompts == []
        assert repo.context_calls == []

    def test_the_driver_runs_in_the_checkout(self, store: Store) -> None:
        _, driver, _ = self.run_review(store, review_result())
        assert driver.cwds == [Path.cwd()]


class TestOpeningAPrIsNotAnExitCondition:
    def test_the_loop_returns_to_review_what_it_just_produced(self, store: Store) -> None:
        """The predecessor's bug: it announced the PR and left its own output unreviewed."""
        repo = FakeRepo(issues=[Issue(number=13, labels=frozenset({"ready-to-dev"}))])

        class Reacting(FakeRepo):
            """Publishes a PR for the issue once development has run, as a real run would."""

            def open_pull_requests(self) -> list[PullRequest]:
                self.pr_calls += 1
                if any(p.startswith("Implement issue #13") for p in driver.prompts):
                    return [PullRequest(number=200, head_ref="agent/issue-13-feed")]
                return []

        driver = FakeDriver(results=[ok_result(), review_result()])
        reacting = Reacting(issues=repo.issues)
        tick, _ = build(store, reacting, driver, budget=Budget(max_cycles=2))
        outcome = tick.run()
        assert [p.split(":")[0] for p in driver.prompts] == [
            "Implement issue #13",
            "Review pull request #200",
        ]
        assert outcome.developments == 1
        assert outcome.reviews == 1


class TestBudgets:
    def test_the_cycle_budget_stops_the_tick(self, store: Store) -> None:
        repo = FakeRepo(prs=[PullRequest(number=94)])
        driver = FakeDriver(results=[review_result()] * 10)
        tick, _ = build(store, repo, driver, budget=Budget(max_cycles=2))
        outcome = tick.run()
        assert len(outcome.cycles) == 2
        assert outcome.stop_reason is StopReason.CYCLE_BUDGET

    def test_the_time_budget_stops_the_tick(self, store: Store) -> None:
        repo = FakeRepo(prs=[PullRequest(number=94)])
        driver = FakeDriver(results=[review_result()] * 10)
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
        tick, _ = build(store, repo, FakeDriver(results=[review_result()]), clock=clock)
        outcome = tick.run()
        assert outcome.stop_reason is StopReason.TIME_BUDGET
        assert all(c.ok for c in outcome.cycles)


class TestCostBudget:
    def test_the_tick_stops_once_the_cap_is_reached(self, store: Store) -> None:
        repo = FakeRepo(prs=[PullRequest(number=94)])
        driver = FakeDriver(results=[review_result(cost=0.6)] * 10)
        tick, _ = build(store, repo, driver, budget=Budget(max_cost_usd=1.0))
        outcome = tick.run()
        assert len(outcome.cycles) == 2
        assert outcome.stop_reason is StopReason.COST_BUDGET

    def test_no_cap_means_no_cost_stop(self, store: Store) -> None:
        repo = FakeRepo(prs=[PullRequest(number=94)])
        driver = FakeDriver(results=[review_result(cost=50.0)] * 10)
        tick, _ = build(store, repo, driver, budget=Budget(max_cycles=3))
        assert tick.run().stop_reason is StopReason.CYCLE_BUDGET


class TestDevelopmentCycle:
    ISSUE = Issue(
        number=13,
        title="Add export",
        body="Export as CSV",
        labels=frozenset({"ready-to-dev", "feature"}),
    )

    def test_the_prompt_carries_the_issue_and_asks_for_a_pr(self, store: Store) -> None:
        repo = FakeRepo(issues=[self.ISSUE])
        driver = FakeDriver(results=[ok_result()])
        tick, _ = build(store, repo, driver, budget=Budget(max_cycles=1))
        tick.run()
        prompt = driver.prompts[0]
        for expected in ("Add export", "Export as CSV", "feature", "Closes #13", "agent/issue-13"):
            assert expected in prompt
        assert driver.cwds == [Path.cwd()]

    def test_a_finished_attempt_is_recorded_as_done(self, store: Store) -> None:
        repo = FakeRepo(issues=[self.ISSUE])
        driver = FakeDriver(results=[ok_result(cost=0.7)])
        tick, _ = build(store, repo, driver, budget=Budget(max_cycles=1))
        tick.run()
        history = store.development_history(REPO, 13)
        assert history is not None
        assert (history.attempts, history.cost_usd, history.status) == (1, 0.7, "done")

    def test_a_failed_attempt_is_recorded_as_abandoned(self, store: Store) -> None:
        repo = FakeRepo(issues=[self.ISSUE])
        tick, _ = build(store, repo, FakeDriver(results=[failed_result()]))
        tick.run()
        history = store.development_history(REPO, 13)
        assert history is not None
        assert history.status == "abandoned"


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
        assert repo.context_calls == []
        assert store.development_history(REPO, 13) is None
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
    driver = FakeDriver(results=[review_result(cost=0.5)])
    tick, _ = build(store, repo, driver, budget=Budget(max_cycles=1))
    outcome = tick.run()
    assert outcome.spend.cycles == 1
    assert outcome.spend.cost_usd == 0.5
