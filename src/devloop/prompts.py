"""The prompts sent to Claude.

Pure string builders over the models, so what Claude is asked can be asserted without a
subprocess. The `DECISION:` line is the contract the parser of the reply relies on.
"""

from __future__ import annotations

from devloop.models import Issue, PullRequest

DECISIONS = ("approve", "request_changes", "block")


def _section(heading: str, content: str) -> str:
    return f"## {heading}\n\n{content}"


def pr_review_prompt(pr: PullRequest) -> str:
    body = pr.body.strip() or "(no description)"
    if pr.diff is None:
        diff = "(diff not available; inspect the changes yourself, e.g. with `gh pr diff`)"
    else:
        diff = f"```diff\n{pr.diff}\n```"
    comments = "\n\n".join(f"- {c}" for c in pr.comments) or "(no comments yet)"

    sections = [
        f"Review pull request #{pr.number}: {pr.title}",
        _section("Description", body),
        _section("Diff", diff),
        _section("Existing comments", comments),
        _section(
            "Task",
            "Review this change. Identify bugs, regressions, missing tests and unclear "
            "code, and suggest concrete improvements. Take the existing comments into "
            "account and do not repeat points already raised.",
        ),
        _section(
            "Decision",
            "End your reply with exactly one final line in this format:\n\n"
            f"DECISION: {' | '.join(DECISIONS)}\n\n"
            "Use `approve` when it can merge as is, `request_changes` when it needs "
            "fixes, and `block` when it must not proceed without a human.",
        ),
    ]
    return "\n\n".join(sections) + "\n"


def issue_development_prompt(issue: Issue) -> str:
    body = issue.body.strip() or "(no description)"
    labels = ", ".join(sorted(issue.labels)) or "(none)"

    sections = [
        f"Implement issue #{issue.number}: {issue.title}",
        _section("Description", body),
        _section("Labels", labels),
        _section(
            "Task",
            "Implement what the issue asks for:\n\n"
            "1. Create or change the files needed.\n"
            "2. Write tests that cover the new behaviour.\n"
            "3. Run the tests and linters, and fix what they report.\n"
            "4. Commit the work when it is finished, with a conventional commit message.",
        ),
    ]
    return "\n\n".join(sections) + "\n"
