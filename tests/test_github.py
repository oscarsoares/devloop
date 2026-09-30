"""Turning gh payloads into the values the decisions read.

No process is spawned here. The mapping is the part that was wrong in the predecessor — the
two check shapes differ by one field name and reading the wrong one silently misreads CI — so
it is tested against payloads rather than against a live repository.
"""

from __future__ import annotations

import pytest

from devloop.github import (
    check_state,
    issue_from,
    issues_owned_by,
    pull_request_from,
)
from devloop.models import CheckState, Priority


class TestCheckShapes:
    """A CheckRun carries status+conclusion; a StatusContext carries only state."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ({"status": "COMPLETED", "conclusion": "SUCCESS"}, CheckState.PASSED),
            ({"status": "IN_PROGRESS"}, CheckState.PENDING),
            ({"status": "QUEUED"}, CheckState.PENDING),
            ({"status": "COMPLETED", "conclusion": "FAILURE"}, CheckState.FAILED),
            ({"status": "COMPLETED", "conclusion": "TIMED_OUT"}, CheckState.FAILED),
            ({"state": "SUCCESS", "context": "legacy/build"}, CheckState.PASSED),
            ({"state": "PENDING", "context": "legacy/build"}, CheckState.PENDING),
            ({"state": "FAILURE", "context": "legacy/build"}, CheckState.FAILED),
            ({"state": "ERROR", "context": "legacy/build"}, CheckState.FAILED),
        ],
        ids=[
            "run passed", "run in progress", "run queued", "run failed", "run timed out",
            "context passed", "context pending", "context failed", "context errored",
        ],
    )
    def test_both_shapes(self, raw: dict[str, object], expected: CheckState) -> None:
        assert check_state(raw) is expected

    def test_an_unrecognised_shape_is_not_a_failure(self) -> None:
        """Absent fields must not read as broken CI, which would send the loop fixing nothing."""
        assert check_state({}) is CheckState.PASSED
        assert check_state({"status": "SOMETHING_NEW"}) is CheckState.PASSED


class TestPullRequestMapping:
    def test_full_payload(self) -> None:
        pr = pull_request_from(
            {
                "number": 94,
                "title": "feat: publish a service request",
                "labels": [{"name": "needs-review"}, {"name": "P0-critical"}],
                "reviewDecision": "CHANGES_REQUESTED",
                "statusCheckRollup": [
                    {"status": "COMPLETED", "conclusion": "SUCCESS"},
                    {"state": "FAILURE"},
                ],
                "headRefName": "agent/issue-68-publish-ui",
                "body": "Closes #68",
            }
        )
        assert pr.number == 94
        assert pr.labels == frozenset({"needs-review", "P0-critical"})
        assert pr.checks == (CheckState.PASSED, CheckState.FAILED)
        assert pr.failed_check_count == 1
        assert not pr.never_reviewed

    def test_missing_fields_default_rather_than_raise(self) -> None:
        pr = pull_request_from({"number": 7})
        assert pr.checks == ()
        assert pr.labels == frozenset()
        assert pr.never_reviewed

    def test_rounds_is_not_read_from_github(self) -> None:
        """Iterations spent are local state; GitHub has no such field.

        Until durable state exists every PR maps as round zero, which means the 3-iteration
        budget cannot yet be enforced across ticks.
        """
        assert pull_request_from({"number": 7, "rounds": 2}).rounds == 0


class TestIssueMapping:
    def test_full_payload(self) -> None:
        issue = issue_from(
            {
                "number": 13,
                "title": "[US-P-010] Ver Pedidos Disponíveis",
                "labels": [{"name": "P0-critical"}, {"name": "feature"}],
                "milestone": {"title": "MVP Month 3 - UI & Core Features"},
            },
            owned_numbers=frozenset(),
        )
        assert issue.priority is Priority.P0
        assert issue.milestone == "MVP Month 3 - UI & Core Features"
        assert not issue.is_bug
        assert not issue.has_open_pr

    def test_a_null_milestone_is_none_not_a_string(self) -> None:
        issue = issue_from({"number": 74, "milestone": None}, owned_numbers=frozenset())
        assert issue.milestone is None

    def test_ownership_comes_from_the_pr_list(self) -> None:
        issue = issue_from({"number": 68}, owned_numbers=frozenset({68}))
        assert issue.has_open_pr


class TestOwnership:
    def test_a_branch_name_claims_its_issue(self) -> None:
        prs = [pull_request_from({"number": 94, "headRefName": "agent/issue-68-publish-ui"})]
        assert issues_owned_by(prs) == frozenset({68})

    def test_a_closes_reference_claims_its_issue(self) -> None:
        prs = [pull_request_from({"number": 94, "body": "Closes #68\nRelated to #5"})]
        assert issues_owned_by(prs) == frozenset({68, 5})

    def test_nothing_claimed_is_an_empty_set(self) -> None:
        prs = [pull_request_from({"number": 94, "headRefName": "main", "body": "no refs"})]
        assert issues_owned_by(prs) == frozenset()

    def test_a_branch_number_is_not_confused_with_a_longer_one(self) -> None:
        prs = [pull_request_from({"number": 1, "headRefName": "agent/issue-680-x"})]
        assert issues_owned_by(prs) == frozenset({680})
