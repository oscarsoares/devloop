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
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
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

    def with_review_context(self, pr: PullRequest) -> PullRequest: ...


class Writes(Protocol):
    """The only way the loop changes anything outside itself.

    A separate protocol from `Repository` so a dry run can be handed something that cannot
    write, rather than a writer it is trusted to not call. The difference matters: one is
    checked by the type system, the other by remembering an `if`.
    """

    def relabel(
        self, subject: str, number: int, *, add: Sequence[str] = (), remove: Sequence[str] = ()
    ) -> None:
        """Adjust labels. Removing a label an item does not carry is not an error."""
        ...

    def comment(self, subject: str, number: int, body: str) -> None: ...


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
        body=as_str(raw.get("body")) or "",
    )


def comment_bodies_from(raw: dict[str, object]) -> tuple[str, ...]:
    """The text of each comment in a `gh pr view --json comments` payload, oldest first."""
    found = (as_str(comment.get("body")) for comment in dicts_in(raw.get("comments")))
    return tuple(body for body in found if body)


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
_ISSUE_FIELDS = "number,title,labels,milestone,body"


def _label_args(flag: str, labels: Sequence[str]) -> tuple[str, ...]:
    return tuple(part for label in labels for part in (flag, label))


@dataclass(slots=True)
class RecordedWrites:
    """A `Writes` that records instead of acting, for a dry run.

    The dry-run path is the same code as the real one, with a different collaborator. Nothing
    is skipped, so what a dry run reports is what a real run would do — which is the only way
    the report is worth reading.
    """

    relabels: list[tuple[str, int, tuple[str, ...], tuple[str, ...]]] = field(
        default_factory=list[tuple[str, int, tuple[str, ...], tuple[str, ...]]]
    )
    comments: list[tuple[str, int, str]] = field(default_factory=list[tuple[str, int, str]])

    def relabel(
        self, subject: str, number: int, *, add: Sequence[str] = (), remove: Sequence[str] = ()
    ) -> None:
        self.relabels.append((subject, number, tuple(add), tuple(remove)))

    def comment(self, subject: str, number: int, body: str) -> None:
        self.comments.append((subject, number, body))

    def describe(self) -> list[str]:
        lines = [
            f"label {subject} #{number}"
            + (f" +{'/'.join(add)}" if add else "")
            + (f" -{'/'.join(remove)}" if remove else "")
            for subject, number, add, remove in self.relabels
        ]
        lines += [f"comment on {subject} #{number}" for subject, number, _ in self.comments]
        return lines


@dataclass(frozen=True, slots=True)
class GhRepository:
    """A repository read through the `gh` CLI."""

    slug: str
    limit: int = 100
    executable: str = "gh"
    timeout_seconds: int = 120

    def _exec(self, *args: str) -> str:
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
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GitHubError.command_failed(argv, str(exc)) from exc
        if completed.returncode != 0:
            raise GitHubError.command_failed(argv, completed.stderr or completed.stdout)
        return completed.stdout

    def _payload(self, *args: str) -> object:
        try:
            return json.loads(self._exec(*args))
        except json.JSONDecodeError as exc:
            raise GitHubError.not_json((self.executable, *args)) from exc

    def _json(self, *args: str) -> list[dict[str, object]]:
        return dicts_in(self._payload(*args))

    def with_review_context(self, pr: PullRequest) -> PullRequest:
        """`pr` with its diff and comments, for the one PR about to be reviewed.

        Separate from `open_pull_requests` because that runs every tick over every open PR,
        and a diff plus a comment fetch each would multiply the calls for PRs nobody reviews.
        A diff that cannot be fetched (too large, or a race with a merge) leaves `None` so the
        prompt says so; comments are not swallowed, since a review that silently ignores what
        was already said is worse than a failed one.
        """
        number = str(pr.number)
        try:
            diff: str | None = self._exec("pr", "diff", number)
        except GitHubError:
            diff = None
        payload = as_dict(self._payload("pr", "view", number, "--json", "comments")) or {}
        return replace(pr, diff=diff, comments=comment_bodies_from(payload))

    def open_pull_requests(self) -> list[PullRequest]:
        rows = self._json(
            "pr", "list", "--state", "open", "--limit", str(self.limit), "--json", _PR_FIELDS
        )
        return [pull_request_from(row) for row in rows]

    def open_issues(self) -> list[Issue]:
        prs = self.open_pull_requests()
        return self.open_issues_given(prs)

    def relabel(
        self, subject: str, number: int, *, add: Sequence[str] = (), remove: Sequence[str] = ()
    ) -> None:
        # Adds and removes are separate calls because `gh` fails the whole command when asked
        # to remove a label the item does not carry — and a triaged issue never carries
        # `ready-to-dev`. Losing a tick to label bookkeeping is not a trade worth making.
        if add:
            self._exec(subject, "edit", str(number), *_label_args("--add-label", add))
        for label in remove:
            try:
                self._exec(subject, "edit", str(number), "--remove-label", label)
            except GitHubError:  # noqa: PERF203 - one call per label is the point
                continue

    def comment(self, subject: str, number: int, body: str) -> None:
        self._exec(subject, "comment", str(number), "--body", body)

    def open_issues_given(self, prs: list[PullRequest]) -> list[Issue]:
        """Issues, with PR ownership resolved against an already-fetched PR list.

        Exposed separately so a tick that already listed the PRs does not list them twice.
        """
        owned = issues_owned_by(prs)
        rows = self._json(
            "issue", "list", "--state", "open", "--limit", str(self.limit), "--json", _ISSUE_FIELDS
        )
        return [issue_from(row, owned_numbers=owned) for row in rows]
