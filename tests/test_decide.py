"""The decisions, with no network and no Claude.

Every case here was a `-SelfTest` fixture in the PowerShell version this replaces. They
survive because they encode judgements that are easy to break while tidying up, not because
the code is fragile.
"""

from __future__ import annotations

import pytest

from devloop.decide import Reason, Waiting, actionable_prs, classify_pr, next_issue, rank_issues
from devloop.models import Budget, CheckState, Issue, Priority, PullRequest

BUDGET = Budget()

PASSED = (CheckState.PASSED,)
PENDING = (CheckState.PENDING,)
FAILED = (CheckState.FAILED,)


def pr(**kwargs: object) -> PullRequest:
    defaults: dict[str, object] = {"number": 1, "checks": PASSED}
    return PullRequest(**{**defaults, **kwargs})  # pyright: ignore[reportArgumentType]


class TestActionable:
    def test_never_reviewed_is_actionable(self) -> None:
        assert classify_pr(pr(), BUDGET).reason is Reason.NOT_REVIEWED

    def test_gate_held_is_actionable(self) -> None:
        # One failing check here is the merge gate itself, not independent CI breakage.
        verdict = classify_pr(pr(labels=frozenset({"review-blocking"}), checks=FAILED), BUDGET)
        assert verdict.reason is Reason.GATE_HELD

    def test_real_ci_failure_is_actionable(self) -> None:
        verdict = classify_pr(pr(labels=frozenset({"needs-review"}), checks=FAILED), BUDGET)
        assert verdict.reason is Reason.CI_FAILED

    def test_changes_requested_is_actionable(self) -> None:
        verdict = classify_pr(
            pr(labels=frozenset({"needs-review"}), review_decision="CHANGES_REQUESTED"), BUDGET
        )
        assert verdict.reason is Reason.CHANGES_REQUESTED

    def test_no_checks_configured_is_still_reviewable(self) -> None:
        assert classify_pr(pr(checks=()), BUDGET).reason is Reason.NOT_REVIEWED


class TestWaiting:
    def test_reviewed_and_clean_waits_on_a_human(self) -> None:
        verdict = classify_pr(pr(labels=frozenset({"needs-review"})), BUDGET)
        assert verdict.reason is None
        assert verdict.waiting is Waiting.HUMAN

    def test_spent_budget_escalates_instead_of_looping(self) -> None:
        verdict = classify_pr(pr(labels=frozenset({"review-blocking"}), rounds=3), BUDGET)
        assert verdict.reason is None
        assert verdict.waiting is Waiting.BUDGET_SPENT


class TestPendingCiCutsBothWays:
    """The pair that the ordering exists for.

    Same CI state, opposite verdicts. A freshly opened PR must be reviewed while CI runs,
    because reading a diff does not depend on CI and the loop would otherwise ignore what it
    just produced. Once reviewed, pending CI is the next signal, so it waits.
    """

    def test_never_reviewed_beats_pending_ci(self) -> None:
        assert classify_pr(pr(checks=PENDING), BUDGET).reason is Reason.NOT_REVIEWED

    def test_already_reviewed_waits_for_pending_ci(self) -> None:
        verdict = classify_pr(pr(labels=frozenset({"needs-review"}), checks=PENDING), BUDGET)
        assert verdict.waiting is Waiting.CI_RUNNING


def test_actionable_prs_are_oldest_first() -> None:
    prs = [pr(number=94), pr(number=77), pr(number=89)]
    assert [a.pr.number for a in actionable_prs(prs, BUDGET)] == [77, 89, 94]


def issue(**kwargs: object) -> Issue:
    defaults: dict[str, object] = {"number": 1}
    return Issue(**{**defaults, **kwargs})  # pyright: ignore[reportArgumentType]


P0 = frozenset({"P0-critical"})
P1 = frozenset({"P1-high"})
P2 = frozenset({"P2-medium"})
P3 = frozenset({"P3-low"})
BUG = frozenset({"bug"})


class TestTriageOrder:
    def test_full_order(self) -> None:
        issues = [
            issue(number=2, labels=P0, milestone="MVP Month 2 - Application Layer"),
            issue(number=13, labels=P0, milestone="MVP Month 3 - UI & Core Features"),
            issue(number=28, labels=P2, milestone="MVP Month 4 - Polish & Launch"),
            issue(number=74, labels=P1 | BUG),
            issue(number=87, labels=P1 | BUG),
            issue(number=50, labels=P0 | BUG),
            issue(number=60, labels=P3 | BUG),
        ]
        ranked = [c.issue.number for c in rank_issues(issues, BUDGET)]
        assert ranked == [50, 74, 87, 2, 13, 28, 60]

    def test_a_low_severity_bug_does_not_jump_the_queue(self) -> None:
        issues = [
            issue(number=60, labels=P3 | BUG),
            issue(number=2, labels=P0, milestone="MVP Month 2"),
        ]
        assert [c.issue.number for c in rank_issues(issues, BUDGET)] == [2, 60]

    def test_severity_alone_does_not_make_a_bug(self) -> None:
        """Without the `bug` label a P1 is a feature, and sorts after the MVP months."""
        issues = [
            issue(number=99, labels=P1),
            issue(number=2, labels=P2, milestone="MVP Month 4"),
        ]
        assert [c.issue.number for c in rank_issues(issues, BUDGET)] == [2, 99]

    @pytest.mark.parametrize(
        ("labels", "milestone", "has_open_pr"),
        [
            (P0 | frozenset({"in-progress"}), "MVP Month 2", False),
            (P0 | frozenset({"blocked"}), "MVP Month 2", False),
            (P0, "Post-MVP", False),
            (P0, "MVP Month 2", True),
        ],
        ids=["in-progress", "blocked", "post-mvp", "owned by an open PR"],
    )
    def test_exclusions(
        self, labels: frozenset[str], milestone: str, has_open_pr: bool
    ) -> None:
        excluded = issue(number=98, labels=labels, milestone=milestone, has_open_pr=has_open_pr)
        assert rank_issues([excluded], BUDGET) == []


class TestNextIssue:
    def test_an_explicit_queue_wins_over_triage(self) -> None:
        queued = [issue(number=500, labels=P3)]
        backlog = [issue(number=50, labels=P0 | BUG)]
        candidate, from_triage = next_issue(queued, backlog, BUDGET)
        assert candidate is not None
        assert candidate.issue.number == 500
        assert from_triage is False

    def test_an_empty_queue_falls_back_to_triage(self) -> None:
        backlog = [issue(number=13, labels=P0, milestone="MVP Month 3")]
        candidate, from_triage = next_issue([], backlog, BUDGET)
        assert candidate is not None
        assert candidate.issue.number == 13
        assert from_triage is True

    def test_nothing_anywhere_is_not_an_error(self) -> None:
        candidate, from_triage = next_issue([], [], BUDGET)
        assert candidate is None
        assert from_triage is True


def test_priority_none_sorts_after_p3() -> None:
    assert Priority.from_labels(frozenset()) > Priority.P3
