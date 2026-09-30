"""Turning gh payloads into the values the decisions read.

No real `gh` is spawned: `subprocess.run` is replaced. The mapping is the part that was wrong
in the predecessor — the two check shapes differ by one field name and reading the wrong one
silently misreads CI — so it is tested against payloads, not a live repository.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from devloop.github import (
    GhRepository,
    GitHubError,
    RecordedWrites,
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
            "run passed",
            "run in progress",
            "run queued",
            "run failed",
            "run timed out",
            "context passed",
            "context pending",
            "context failed",
            "context errored",
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


class FakeGh:
    """Stands in for `subprocess.run`, so no test reaches a real `gh` or the network."""

    def __init__(
        self,
        stdout: str = "[]",
        returncode: int = 0,
        stderr: str = "",
        raises: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._result = subprocess.CompletedProcess([], returncode, stdout, stderr)
        self._raises = raises

    def __call__(self, argv: tuple[str, ...], **_: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        if self._raises:
            raise self._raises
        return self._result


def patched(monkeypatch: pytest.MonkeyPatch, fake: FakeGh) -> GhRepository:
    monkeypatch.setattr(subprocess, "run", fake)
    return GhRepository("o/r")


class TestGhReads:
    def test_pull_requests_are_listed_scoped_to_the_repo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeGh(json.dumps([{"number": 7, "title": "t", "labels": [{"name": "bug"}]}]))
        prs = patched(monkeypatch, fake).open_pull_requests()
        assert [(p.number, p.labels) for p in prs] == [(7, frozenset({"bug"}))]
        assert fake.calls[0][:3] == ("gh", "pr", "list")
        assert fake.calls[0][-2:] == ("--repo", "o/r")

    def test_issue_ownership_is_resolved_from_the_pr_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr = {"number": 1, "headRefName": "agent/issue-5-x"}
        issues = [{"number": 5, "title": "a"}, {"number": 6, "title": "b"}]
        replies = iter([json.dumps([pr]), json.dumps(issues)])

        def run(argv: tuple[str, ...], **_: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess([], 0, next(replies), "")

        monkeypatch.setattr(subprocess, "run", run)
        found = GhRepository("o/r").open_issues()
        assert {i.number: i.has_open_pr for i in found} == {5: True, 6: False}

    def test_a_nonzero_exit_raises_with_the_stderr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = patched(monkeypatch, FakeGh(returncode=1, stderr="auth required"))
        with pytest.raises(GitHubError, match="auth required"):
            repo.open_pull_requests()

    def test_non_json_output_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(GitHubError, match="did not return JSON"):
            patched(monkeypatch, FakeGh("not json")).open_pull_requests()

    @pytest.mark.parametrize("exc", [FileNotFoundError("gh"), subprocess.TimeoutExpired("gh", 1)])
    def test_a_missing_or_hung_gh_is_a_github_error(
        self, monkeypatch: pytest.MonkeyPatch, exc: Exception
    ) -> None:
        with pytest.raises(GitHubError):
            patched(monkeypatch, FakeGh(raises=exc)).open_pull_requests()


class TestGhWrites:
    def test_adds_are_one_call_and_removes_are_one_call_each(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeGh()
        patched(monkeypatch, fake).relabel("issue", 3, add=("a", "b"), remove=("c", "d"))
        assert [c[:4] for c in fake.calls] == [
            ("gh", "issue", "edit", "3"),
            ("gh", "issue", "edit", "3"),
            ("gh", "issue", "edit", "3"),
        ]
        assert "--add-label" in fake.calls[0]
        assert fake.calls[0].count("--add-label") == 2
        assert [c[4:6] for c in fake.calls[1:]] == [
            ("--remove-label", "c"),
            ("--remove-label", "d"),
        ]

    def test_removing_a_label_the_item_lacks_is_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = patched(monkeypatch, FakeGh(returncode=1, stderr="label not found"))
        repo.relabel("issue", 3, remove=("ready-to-dev",))

    def test_a_failed_add_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = patched(monkeypatch, FakeGh(returncode=1, stderr="denied"))
        with pytest.raises(GitHubError):
            repo.relabel("pr", 3, add=("needs-review",))

    def test_comment_passes_the_body_as_one_argument(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeGh()
        patched(monkeypatch, fake).comment("pr", 9, "two words; `rm -rf`")
        assert fake.calls[0][:6] == ("gh", "pr", "comment", "9", "--body", "two words; `rm -rf`")


class TestRecordedWrites:
    def test_records_and_describes_without_acting(self) -> None:
        writes = RecordedWrites()
        writes.relabel("issue", 1, add=("in-progress",), remove=("ready-to-dev",))
        writes.comment("pr", 2, "hi")
        assert writes.describe() == [
            "label issue #1 +in-progress -ready-to-dev",
            "comment on pr #2",
        ]
