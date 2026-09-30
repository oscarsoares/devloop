"""Reading the repository state the decisions are made from.

`gh` is the transport because it already holds the credentials, and because the alternative —
a token in this project's own configuration — is a secret to manage for no gain today. It sits
behind `Repository` for the same reason the Claude driver does: swapping to the REST API later
must not ripple into the decisions.

Nothing here decides anything. It turns payloads into `PullRequest` and `Issue` values, and
that is the whole contract.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Protocol

from devloop._json import as_dict, as_int, as_str, dicts_in, strings_in
from devloop.models import CheckState, Issue, PullRequest


class GitHubError(RuntimeError):
    @classmethod
    def command_failed(cls, args: tuple[str, ...], output: str) -> GitHubError:
        return cls(f"`{' '.join(args)}` failed: {output.strip()[:400]}")

    @classmethod
    def not_json(cls, args: tuple[str, ...]) -> GitHubError:
        return cls(f"`{' '.join(args)}` did not return JSON")


class Repository(Protocol):
    """One repository's open work, as the decisions need to see it."""

    @property
    def slug(self) -> str: ...

    def open_pull_requests(self) -> list[PullRequest]: ...

    def open_issues(self) -> list[Issue]: ...


# gh's statusCheckRollup mixes two shapes: a CheckRun carries `status` and `conclusion`, a
# StatusContext carries only `state`. Reading the wrong one is a silent misread of CI, so both
# are handled explicitly and both are covered by tests.
_PENDING_STATUS = frozenset({"QUEUED", "IN_PROGRESS", "WAITING", "PENDING"})
_FAILED_CONCLUSION = frozenset(
    {"FAILURE", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED", "STARTUP_FAILURE"}
)
_FAILED_STATE = frozenset({"FAILURE", "ERROR"})


def check_state(raw: dict[str, object]) -> CheckState:
    status = as_str(raw.get("status"))
    conclusion = as_str(raw.get("conclusion"))
    state = as_str(raw.get("state"))

    if (status and status in _PENDING_STATUS) or state == "PENDING":
        return CheckState.PENDING
    if (conclusion and conclusion in _FAILED_CONCLUSION) or (state and state in _FAILED_STATE):
        return CheckState.FAILED
    return CheckState.PASSED


def pull_request_from(raw: dict[str, object]) -> PullRequest:
    return PullRequest(
        number=as_int(raw.get("number")),
        title=as_str(raw.get("title")) or "",
        labels=strings_in(raw.get("labels"), key="name"),
        review_decision=as_str(raw.get("reviewDecision")) or "",
        checks=tuple(check_state(c) for c in dicts_in(raw.get("statusCheckRollup"))),
        head_ref=as_str(raw.get("headRefName")) or "",
        body=as_str(raw.get("body")) or "",
    )


def issue_from(raw: dict[str, object], *, owned_numbers: frozenset[int]) -> Issue:
    milestone = as_dict(raw.get("milestone"))
    number = as_int(raw.get("number"))
    return Issue(
        number=number,
        title=as_str(raw.get("title")) or "",
        labels=strings_in(raw.get("labels"), key="name"),
        milestone=as_str(milestone.get("title")) if milestone else None,
        has_open_pr=number in owned_numbers,
    )


def issues_owned_by(prs: list[PullRequest]) -> frozenset[int]:
    """Issue numbers an open PR already covers.

    A branch named `agent/issue-68-...` or a body saying `Closes #68` both count. Such an issue
    belongs to that PR, not to a new branch — starting a second branch for it is how the same
    work gets done twice.
    """
    owned: set[int] = set()
    for pr in prs:
        branch = re.search(r"issue-(\d+)\b", pr.head_ref)
        if branch:
            owned.add(int(branch.group(1)))
        owned.update(int(m) for m in re.findall(r"#(\d+)\b", pr.body))
    return frozenset(owned)


_PR_FIELDS = "number,title,labels,reviewDecision,statusCheckRollup,headRefName,body"
_ISSUE_FIELDS = "number,title,labels,milestone"


@dataclass(frozen=True, slots=True)
class GhRepository:
    """A repository read through the `gh` CLI."""

    slug: str
    limit: int = 100
    executable: str = "gh"
    timeout_seconds: int = 120

    def _json(self, *args: str) -> list[dict[str, object]]:
        argv = (self.executable, *args, "--repo", self.slug)
        try:
            completed = subprocess.run(  # noqa: S603 - argv is built here, never shell-parsed
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except OSError as exc:
            raise GitHubError.command_failed(argv, str(exc)) from exc

        if completed.returncode != 0:
            raise GitHubError.command_failed(argv, completed.stderr or completed.stdout)

        try:
            payload: object = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise GitHubError.not_json(argv) from exc
        return dicts_in(payload)

    def open_pull_requests(self) -> list[PullRequest]:
        rows = self._json("pr", "list", "--state", "open", "--limit", str(self.limit),
                          "--json", _PR_FIELDS)
        return [pull_request_from(row) for row in rows]

    def open_issues(self) -> list[Issue]:
        prs = self.open_pull_requests()
        return self.open_issues_given(prs)

    def open_issues_given(self, prs: list[PullRequest]) -> list[Issue]:
        """Issues, with PR ownership resolved against an already-fetched PR list.

        Exposed separately so a tick that already listed the PRs does not list them twice.
        """
        owned = issues_owned_by(prs)
        rows = self._json("issue", "list", "--state", "open", "--limit", str(self.limit),
                          "--json", _ISSUE_FIELDS)
        return [issue_from(row, owned_numbers=owned) for row in rows]
