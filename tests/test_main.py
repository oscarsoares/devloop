"""The entry point, with a scripted driver in place of Claude."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from devloop.__main__ import format_event, main
from devloop.drivers.cli import DriverError
from devloop.events import Event, Result, SessionStarted, Text, ToolCall, Unknown, Usage


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
