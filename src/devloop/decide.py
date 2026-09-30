"""What to do next: pure functions over repository facts.

Reviews come before new development, because the next task may depend on an open PR
landing. Within reviews, the distinction that keeps the loop from spinning is *actionable*
versus *waiting*: a PR awaiting CI, or awaiting a human after its gate cleared, is waiting,
and treating waiting as work is how an executor burns tokens making no progress.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from devloop.models import Budget, Issue, PullRequest


class Reason(StrEnum):
    NOT_REVIEWED = "not reviewed yet"
    GATE_HELD = "L1/L2 findings open"
    CI_FAILED = "CI failed"
    CHANGES_REQUESTED = "changes requested"


class Waiting(StrEnum):
    BUDGET_SPENT = "review budget spent, escalated to a human"
    CI_RUNNING = "waiting on CI"
    HUMAN = "waiting on a human"


@dataclass(frozen=True, slots=True)
class PrVerdict:
    pr: PullRequest
    reason: Reason | None
    waiting: Waiting | None

    @property
    def actionable(self) -> bool:
        return self.reason is not None


@dataclass(frozen=True, slots=True)
class Actionable:
    """A PR the loop can move forward now, with the reason it can.

    A separate type rather than a filtered `PrVerdict`, so `reason` is non-optional by
    construction: callers cannot forget that the filtered list always has one.
    """

    pr: PullRequest
    reason: Reason


def classify_pr(pr: PullRequest, budget: Budget) -> PrVerdict:
    """Decide whether the loop can move this PR forward now.

    The order of the first two cases is load-bearing and not obvious.

    `never_reviewed` is checked *before* pending CI on purpose. A PR the loop just opened
    always has CI running, and reading a diff does not depend on CI — so deferring here
    would make the loop ignore the very work it just produced and go start something else.
    Once a PR has been reviewed, pending CI genuinely is waiting, because the CI result is
    then the next signal. Both cases are covered by tests, because a tidy-up of this
    function that collapses them would break the loop silently.
    """
    if pr.rounds >= budget.max_review_rounds:
        return PrVerdict(pr, None, Waiting.BUDGET_SPENT)

    if pr.never_reviewed:
        return PrVerdict(pr, Reason.NOT_REVIEWED, None)

    if pr.has_pending_checks:
        return PrVerdict(pr, None, Waiting.CI_RUNNING)

    if pr.gate_held:
        return PrVerdict(pr, Reason.GATE_HELD, None)

    # The merge-gate check fails by design while the gate is held. That is the loop's own
    # signal, not independent CI breakage, so it must not be counted twice.
    gate_only_failure = pr.failed_check_count == 1 and pr.gate_held
    if pr.failed_check_count > 0 and not gate_only_failure:
        return PrVerdict(pr, Reason.CI_FAILED, None)

    if pr.review_decision == "CHANGES_REQUESTED":
        return PrVerdict(pr, Reason.CHANGES_REQUESTED, None)

    return PrVerdict(pr, None, Waiting.HUMAN)


def actionable_prs(prs: list[PullRequest], budget: Budget) -> list[Actionable]:
    """Actionable PRs, oldest first: the one closest to landing unblocks the most."""
    found = [
        Actionable(pr=verdict.pr, reason=verdict.reason)
        for verdict in (classify_pr(pr, budget) for pr in prs)
        if verdict.reason is not None
    ]
    return sorted(found, key=lambda a: a.pr.number)


_MONTH = re.compile(r"month\s*(\d+)", re.IGNORECASE)
_POST_MVP = re.compile(r"post-?mvp", re.IGNORECASE)

_TIER_BUG = 0
_TIER_MILESTONE = 1
_TIER_UNPLANNED = 2


@dataclass(frozen=True, slots=True)
class Candidate:
    issue: Issue
    tier: int
    month: int
    why: str

    def sort_key(self) -> tuple[int, int, int, int]:
        return (self.tier, self.month, int(self.issue.priority), self.issue.number)


def rank_issues(issues: list[Issue], budget: Budget) -> list[Candidate]:
    """Order the backlog by rules the documentation already defines.

    The loop must never invent a priority. The order comes from the project's own MVP
    timeline (mirrored by the GitHub milestones) and its priority labels:

    1. a bug at P0 or P1, whatever its milestone — a tenant leak or a broken runtime is
       not something to build features on top of;
    2. the MVP timeline, earliest month first;
    3. no milestone, which means not planned.

    Severity alone cannot tell a defect from a feature, so tier 1 requires the `bug` label.
    """
    candidates: list[Candidate] = []

    for issue in issues:
        if issue.labels & budget.labels_excluding_selection:
            continue
        if issue.has_open_pr:
            continue
        if issue.milestone and _POST_MVP.search(issue.milestone):
            continue

        if issue.is_bug and issue.priority <= 1:
            candidates.append(
                Candidate(issue, _TIER_BUG, 0, f"bug at {issue.priority.name}")
            )
            continue

        match = _MONTH.search(issue.milestone) if issue.milestone else None
        if match:
            month = int(match.group(1))
            candidates.append(
                Candidate(issue, _TIER_MILESTONE, month, f"milestone month {month}")
            )
        else:
            candidates.append(Candidate(issue, _TIER_UNPLANNED, 0, "no milestone"))

    return sorted(candidates, key=lambda c: c.sort_key())


def next_issue(
    ready: list[Issue], backlog: list[Issue], budget: Budget
) -> tuple[Candidate | None, bool]:
    """The next issue to develop, and whether it came from triage rather than the queue.

    An explicit `ready-to-dev` label wins: when a human chooses, that choice stands. With
    nothing queued, the backlog is triaged rather than idled over — an executor that only
    runs when someone remembers to label something is not an executor.
    """
    queued = rank_issues(ready, budget)
    if queued:
        return queued[0], False

    triaged = rank_issues(backlog, budget)
    return (triaged[0] if triaged else None), True
