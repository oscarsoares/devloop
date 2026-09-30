"""One tick.

    any actionable PR?    -> review it    -> back to the top
    else any ready issue? -> develop it   -> back to the top
    else                  -> nothing to do, stop

Opening a PR is not an exit condition: the PR the loop just produced has not been reviewed,
so on the next pass it is the most actionable thing there is. The predecessor stopped there,
announced the PR, and left its own output unreviewed.

Everything this needs is injected — the repository, the writer, the driver, the store and the
clock — so a tick is exercised end to end in tests with no network, no Claude, and no real
time passing. The budgets in particular are only trustworthy if they are tested, and they
cannot be tested against a real clock.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from devloop.decide import actionable_prs, next_issue, rank_issues
from devloop.drivers import ClaudeDriver
from devloop.events import Result
from devloop.github import Repository, Writes
from devloop.models import Budget, Issue, PullRequest, Spend
from devloop.prompts import issue_development_prompt, parse_decision, pr_review_prompt
from devloop.store import CycleRecord, Store


class StopReason(StrEnum):
    IDLE = "no actionable PR and no selectable issue"
    CYCLE_BUDGET = "cycle budget spent; the rest carries to the next tick"
    TIME_BUDGET = "time budget spent; the rest carries to the next tick"
    COST_BUDGET = "cost budget spent; the rest carries to the next tick"
    CYCLE_FAILED = "a cycle failed"
    DRY_RUN = "dry run: stopped after one decision"


@dataclass(frozen=True, slots=True)
class Cycle:
    kind: str
    target: int
    ok: bool
    cost_usd: float | None


@dataclass(slots=True)
class TickOutcome:
    cycles: list[Cycle] = field(default_factory=list[Cycle])
    stop_reason: StopReason = StopReason.IDLE
    spend: Spend = field(default_factory=Spend)

    @property
    def reviews(self) -> int:
        return sum(1 for c in self.cycles if c.kind == "review")

    @property
    def developments(self) -> int:
        return sum(1 for c in self.cycles if c.kind == "develop")

    def summary(self) -> str:
        return (
            f"{len(self.cycles)} cycle(s) - {self.reviews} review(s), "
            f"{self.developments} development(s). Stopped: {self.stop_reason.value}"
        )


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Tick:
    repo: Repository
    writes: Writes
    driver: ClaudeDriver
    store: Store
    cwd: Path
    budget: Budget = field(default_factory=Budget)
    dry_run: bool = False
    clock: Callable[[], datetime] = _utcnow
    on_event: Callable[[str], None] = print

    def _drive(self, prompt: str) -> Result | None:
        """Run one prompt, reporting progress, and return its Result if it produced one."""
        result: Result | None = None
        for event in self.driver.run(prompt, self.cwd):
            self.on_event(str(event))
            if isinstance(event, Result):
                result = event
        return result

    def _record(
        self,
        kind: str,
        target: int,
        started: datetime,
        result: Result | None,
        *,
        usable: bool = True,
    ) -> Cycle:
        # No Result means the run did not finish, whatever the process said on the way out.
        # `usable` is for a run that finished but produced nothing the loop can act on, such
        # as a review with no readable DECISION: it cost money, so it is recorded, but it is
        # not a completed round.
        ok = result is not None and result.ok and usable
        cost = result.cost_usd if result else None
        if not self.dry_run:
            usage = result.usage if result else None
            self.store.record_cycle(
                self.repo.slug,
                CycleRecord(
                    kind="review" if kind == "review" else "develop",
                    target=target,
                    started_at=started.isoformat(timespec="seconds"),
                    ok=ok,
                    cost_usd=cost,
                    input_tokens=usage.input_tokens if usage else 0,
                    output_tokens=usage.output_tokens if usage else 0,
                    cache_read=usage.cache_read_tokens if usage else 0,
                    cache_write=usage.cache_write_tokens if usage else 0,
                ),
            )
        return Cycle(kind=kind, target=target, ok=ok, cost_usd=cost)

    def _selectable_issues(self) -> tuple[list[Issue], list[Issue]]:
        prs = self.repo.open_pull_requests()
        issues = self.repo.open_issues()
        ready = [issue for issue in issues if "ready-to-dev" in issue.labels]
        del prs
        return ready, issues

    def _finish_development(self, issue: Issue) -> None:
        """Hand the issue over once its PR exists.

        This is the gap that swallowed work in the predecessor: `in-progress` was set at the
        start and never removed on success, so an issue whose PR was closed without merging
        stayed excluded from selection forever, with no signal. The label is only moved once a
        PR actually owns the issue — if none does, `in-progress` is wrong to clear, because
        nothing took the work over.
        """
        owned = any(
            issue.number == candidate.number
            for candidate in self.repo.open_issues()
            if candidate.has_open_pr
        )
        if owned:
            self.writes.relabel("issue", issue.number, add=["needs-review"], remove=["in-progress"])

    def _review(self, pr: PullRequest, started: datetime) -> Cycle:
        """Review one PR: prompt, run, read the verdict, then say it on GitHub and in the store.

        The store is written before GitHub is: a comment that fails to post loses text, while
        a round that was not counted lets the same PR be reviewed past its budget.
        """
        detailed = self.repo.with_review_context(pr)
        result = self._drive(pr_review_prompt(detailed))
        reply = result.text if result and result.ok else ""
        decision = parse_decision(reply)
        cycle = self._record("review", pr.number, started, result, usable=decision is not None)
        if decision is None:
            self.on_event(f"review of PR #{pr.number} ended without a readable DECISION")
            return cycle

        self.store.record_review(self.repo.slug, pr.number, cycle.cost_usd or 0.0, decision)
        self.writes.comment("pr", pr.number, reply)
        if decision == "approve":
            self.writes.relabel("pr", pr.number, add=["needs-review"], remove=["review-blocking"])
        else:
            self.writes.relabel("pr", pr.number, add=["review-blocking"], remove=["needs-review"])
        if decision == "block":
            self.store.raise_escalation(
                self.repo.slug, "pr", pr.number, "The review blocked this PR; a human must decide."
            )
        return cycle

    def _escalate(self, issue: Issue, reason: str) -> None:
        self.writes.relabel("issue", issue.number, add=["blocked"], remove=["in-progress"])
        self.writes.comment("issue", issue.number, reason)
        if not self.dry_run:
            self.store.raise_escalation(self.repo.slug, "issue", issue.number, reason)

    def run(self) -> TickOutcome:
        outcome = TickOutcome()
        deadline = self.clock() + timedelta(minutes=self.budget.max_minutes)

        while True:
            if len(outcome.cycles) >= self.budget.max_cycles:
                outcome.stop_reason = StopReason.CYCLE_BUDGET
                break
            # Checked between cycles only: a cycle in flight is never cut off, because a
            # half-applied review is worse than a late one.
            if self.clock() >= deadline:
                outcome.stop_reason = StopReason.TIME_BUDGET
                break
            # A cycle that reported no cost counts as zero here, so this is a floor: it can
            # stop a tick late, never early.
            cap = self.budget.max_cost_usd
            if cap is not None and sum(c.cost_usd or 0.0 for c in outcome.cycles) >= cap:
                outcome.stop_reason = StopReason.COST_BUDGET
                break

            fetched = self.repo.open_pull_requests()
            prs = self.store.hydrate_rounds(self.repo.slug, fetched)
            actionable = actionable_prs(prs, self.budget)

            if actionable:
                target = actionable[0]
                started = self.clock()
                self.on_event(
                    f"review PR #{target.pr.number} - {target.reason.value} "
                    f"(iteration {target.pr.rounds + 1} of {self.budget.max_review_rounds})"
                )
                if self.dry_run:
                    outcome.cycles.append(self._record("review", target.pr.number, started, None))
                    outcome.stop_reason = StopReason.DRY_RUN
                    break

                # A round is counted only once it produced a verdict: a run that died halfway
                # corrected nothing, and charging it would retire a PR never actually reviewed.
                cycle = self._review(target.pr, started)
                outcome.cycles.append(cycle)
                if not cycle.ok:
                    outcome.stop_reason = StopReason.CYCLE_FAILED
                    break
                continue

            ready, backlog = self._selectable_issues()
            candidate, from_triage = next_issue(ready, backlog, self.budget)
            if candidate is None:
                outcome.stop_reason = StopReason.IDLE
                break

            issue = candidate.issue
            source = "triage" if from_triage else "the ready-to-dev queue"
            self.on_event(f"develop issue #{issue.number} from {source} - {candidate.why}")
            runners_up = [c for c in rank_issues(backlog, self.budget) if c is not candidate][:3]
            if runners_up:
                trailing = ", ".join(f"#{c.issue.number} ({c.why})" for c in runners_up)
                self.on_event(f"  runners-up: {trailing}")

            self.writes.relabel("issue", issue.number, add=["in-progress"], remove=["ready-to-dev"])
            started = self.clock()
            result = None if self.dry_run else self._drive(issue_development_prompt(issue))
            cycle = self._record("develop", issue.number, started, result)
            outcome.cycles.append(cycle)

            if self.dry_run:
                outcome.stop_reason = StopReason.DRY_RUN
                break

            self.store.record_development(
                self.repo.slug,
                issue.number,
                cycle.cost_usd or 0.0,
                "done" if cycle.ok else "abandoned",
            )
            if cycle.ok:
                self._finish_development(issue)
            else:
                self._escalate(
                    issue,
                    "The local executor failed while developing this issue, so it is marked "
                    "`blocked` for a human. See the loop's log on the development machine.",
                )
                outcome.stop_reason = StopReason.CYCLE_FAILED
                break

        outcome.spend = self.store.spend(self.repo.slug)
        return outcome
