"""The entry point, with a scripted driver in place of Claude."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from devloop.__main__ import format_event, main
from devloop.drivers.cli import DriverError
from devloop.events import Event, Result, SessionStarted, Text, ToolCall, Unknown, Usage
from devloop.github import GitHubError, RecordedWrites
from devloop.models import Issue, PullRequest
from devloop.store import open_store


@dataclass
class ScriptedDriver:
    events: list[Event]
    calls: list[tuple[str, Path]] = field(default_factory=lambda: [])

    @property
    def name(self) -> str:
        return "scripted"

    def run(self, prompt: str, cwd: Path) -> Iterator[Event]:
        self.calls.append((prompt, cwd))
        yield from self.events


class FailingDriver:
    @property
    def name(self) -> str:
        return "failing"

    def run(self, prompt: str, cwd: Path) -> Iterator[Event]:
        raise DriverError.could_not_start("claude", FileNotFoundError("not found"))
        yield  # pragma: no cover - makes this a generator, like the real driver


OK = Result(subtype="success", turns=3, duration_ms=4200, cost_usd=0.1234)


class TestBare:
    def test_no_command_prints_help_and_succeeds(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main([]) == 0
        assert "usage: devloop" in capsys.readouterr().out

    def test_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0
        assert capsys.readouterr().out.startswith("devloop ")


class TestRun:
    def test_passes_prompt_and_directory(self, tmp_path: Path) -> None:
        driver = ScriptedDriver([OK])
        assert main(["run", "review #7", "--cwd", str(tmp_path)], driver=driver) == 0
        assert driver.calls == [("review #7", tmp_path)]

    def test_streams_each_event(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        driver = ScriptedDriver([SessionStarted("s1"), ToolCall("Bash", "make"), Text("Done."), OK])
        main(["run", "p", "--cwd", str(tmp_path)], driver=driver)
        out = capsys.readouterr().out
        assert out.index("s1") < out.index("Bash: make") < out.index("Done.") < out.index("0.1234")

    def test_failed_result_exits_non_zero(self, tmp_path: Path) -> None:
        driver = ScriptedDriver([Result(subtype="error_during_execution", is_error=True)])
        assert main(["run", "p", "--cwd", str(tmp_path)], driver=driver) == 1

    def test_no_result_at_all_exits_non_zero(self, tmp_path: Path) -> None:
        """A stream that ends without a Result did not finish, whatever the exit code said."""
        driver = ScriptedDriver([SessionStarted("s1"), Unknown("exit code 1")])
        assert main(["run", "p", "--cwd", str(tmp_path)], driver=driver) == 1

    def test_driver_error_is_a_message_not_a_traceback(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        assert main(["run", "p", "--cwd", str(tmp_path)], driver=FailingDriver()) == 1
        assert "could not start 'claude'" in capsys.readouterr().err

    def test_missing_directory_is_rejected_before_spawning(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        driver = ScriptedDriver([OK])
        assert main(["run", "p", "--cwd", str(tmp_path / "nope")], driver=driver) == 1
        assert driver.calls == []
        assert "not a directory" in capsys.readouterr().err


class TestFormat:
    def test_missing_cost_is_not_printed_as_zero(self) -> None:
        """Absent must not read as free — the same rule the parser keeps."""
        line = format_event(Result(subtype="success", usage=Usage(output_tokens=10)))
        assert "$0" not in line
        assert "cost not reported" in line

    def test_result_with_cost(self) -> None:
        assert "$0.1234" in format_event(OK)


@dataclass
class FakeRepo:
    slug: str = "o/r"
    prs: list[PullRequest] = field(default_factory=lambda: [])
    issues: list[Issue] = field(default_factory=lambda: [])
    context_calls: list[int] = field(default_factory=lambda: [])
    fails: bool = False

    def open_pull_requests(self) -> list[PullRequest]:
        if self.fails:
            raise GitHubError.command_failed(("gh", "pr", "list"), "auth required")
        return list(self.prs)

    def open_issues(self) -> list[Issue]:
        return list(self.issues)

    def with_review_context(self, pr: PullRequest) -> PullRequest:
        self.context_calls.append(pr.number)
        return replace(pr, diff="+x")


def review(decision: str = "approve", cost: float | None = 0.5) -> Result:
    return Result(subtype="success", cost_usd=cost, text=f"Fine.\n\nDECISION: {decision}", turns=1)


class TestTick:
    def tick(
        self,
        tmp_path: Path,
        repo: FakeRepo,
        driver: ScriptedDriver,
        *extra: str,
        writes: RecordedWrites | None = None,
    ) -> int:
        argv = ["tick", "o/r", "--cwd", str(tmp_path), "--state", str(tmp_path / "s.db"), *extra]
        return main(argv, driver=driver, repo=repo, writes=writes)

    def test_without_execute_it_is_a_dry_run(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        repo = FakeRepo(prs=[PullRequest(number=94)])
        driver = ScriptedDriver([review()])
        assert self.tick(tmp_path, repo, driver) == 0
        out = capsys.readouterr().out
        assert driver.calls == []
        assert repo.context_calls == []
        assert "dry run" in out.lower()
        assert "review PR #94" in out
        with open_store(tmp_path / "s.db") as store:
            assert store.review_history("o/r", 94) is None

    def test_execute_reviews_writes_records_and_summarises(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        repo = FakeRepo(prs=[PullRequest(number=94)])
        driver = ScriptedDriver([review("approve", cost=0.5)])
        writes = RecordedWrites()
        code = self.tick(tmp_path, repo, driver, "--execute", "--max-cycles", "1", writes=writes)
        out = capsys.readouterr().out
        assert code == 0
        assert len(driver.calls) == 1
        assert driver.calls[0][1] == tmp_path
        assert [c[:2] for c in writes.comments] == [("pr", 94)]
        assert "PRs reviewed:        1" in out
        assert "Issues developed:    0" in out
        assert "Cost this tick:      $0.5000" in out
        with open_store(tmp_path / "s.db") as store:
            history = store.review_history("o/r", 94)
        assert history is not None
        assert (history.rounds, history.last_decision) == (1, "approve")

    def test_execute_develops_a_ready_issue(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        repo = FakeRepo(
            issues=[Issue(number=13, title="Add export", labels=frozenset({"ready-to-dev"}))]
        )
        driver = ScriptedDriver([OK])
        code = self.tick(
            tmp_path, repo, driver, "--execute", "--max-cycles", "1", writes=RecordedWrites()
        )
        assert code == 0
        assert "Implement issue #13" in driver.calls[0][0]
        assert "Issues developed:    1" in capsys.readouterr().out

    def test_max_cost_ends_the_tick(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        repo = FakeRepo(prs=[PullRequest(number=94)])
        driver = ScriptedDriver([review(cost=0.6)])
        code = self.tick(
            tmp_path, repo, driver, "--execute", "--max-cost", "0.5", writes=RecordedWrites()
        )
        assert code == 0
        assert len(driver.calls) == 1
        assert "cost budget spent" in capsys.readouterr().out

    def test_an_unreported_cost_is_a_floor_not_zero(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        repo = FakeRepo(prs=[PullRequest(number=94)])
        driver = ScriptedDriver([review(cost=None)])
        self.tick(tmp_path, repo, driver, "--execute", "--max-cycles", "1", writes=RecordedWrites())
        assert "at least $0.0000" in capsys.readouterr().out

    def test_a_failed_cycle_exits_non_zero(self, tmp_path: Path) -> None:
        repo = FakeRepo(prs=[PullRequest(number=94)])
        driver = ScriptedDriver([Result(subtype="error_during_execution", is_error=True)])
        assert self.tick(tmp_path, repo, driver, "--execute", writes=RecordedWrites()) == 1

    def test_a_github_error_is_a_message_not_a_traceback(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        repo = FakeRepo(fails=True)
        assert self.tick(tmp_path, repo, ScriptedDriver([review()])) == 1
        assert "auth required" in capsys.readouterr().err

    def test_a_missing_directory_is_rejected_before_anything_runs(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        driver = ScriptedDriver([review()])
        argv = ["tick", "o/r", "--cwd", str(tmp_path / "nope"), "--execute"]
        assert main(argv, driver=driver, repo=FakeRepo()) == 1
        assert "not a directory" in capsys.readouterr().err
