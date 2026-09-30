"""The prompts, as strings: no subprocess, no I/O."""

from __future__ import annotations

from devloop.models import Issue, PullRequest
from devloop.prompts import issue_development_prompt, pr_review_prompt


class TestPrReview:
    def test_includes_title_body_diff_and_comments(self) -> None:
        prompt = pr_review_prompt(
            PullRequest(
                number=7,
                title="Fix the widget",
                body="Widgets were leaking.",
                diff="+def widget(): ...",
                comments=("Looks odd on line 3",),
            )
        )
        assert "#7" in prompt
        assert "Fix the widget" in prompt
        assert "Widgets were leaking." in prompt
        assert "+def widget(): ..." in prompt
        assert "Looks odd on line 3" in prompt

    def test_asks_for_a_parseable_decision(self) -> None:
        prompt = pr_review_prompt(PullRequest(number=1))
        assert "DECISION: approve | request_changes | block" in prompt

    def test_missing_diff_is_stated_not_rendered_as_none(self) -> None:
        prompt = pr_review_prompt(PullRequest(number=1, title="t", diff=None))
        assert "diff not available" in prompt
        assert "None" not in prompt
        assert "```diff" not in prompt

    def test_missing_body_and_comments_get_placeholders(self) -> None:
        prompt = pr_review_prompt(PullRequest(number=1))
        assert "(no description)" in prompt
        assert "(no comments yet)" in prompt


class TestIssueDevelopment:
    def test_includes_title_body_and_labels(self) -> None:
        prompt = issue_development_prompt(
            Issue(
                number=3,
                title="Add export",
                body="Export the report as CSV.",
                labels=frozenset({"bug", "P1-high"}),
            )
        )
        assert "#3" in prompt
        assert "Add export" in prompt
        assert "Export the report as CSV." in prompt
        assert "P1-high" in prompt
        assert "bug" in prompt

    def test_instructs_files_tests_and_commit(self) -> None:
        prompt = issue_development_prompt(Issue(number=1)).lower()
        assert "create" in prompt
        assert "tests" in prompt
        assert "commit" in prompt

    def test_missing_body_and_labels_get_placeholders(self) -> None:
        prompt = issue_development_prompt(Issue(number=1))
        assert "(no description)" in prompt
        assert "(none)" in prompt
