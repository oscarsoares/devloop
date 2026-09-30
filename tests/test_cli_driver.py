"""The CLI driver, with the process replaced and the contract kept.

`Popen` is swapped for a fake so these run without `claude` installed. What is under test is
what the driver promises the loop: the argv it builds, that it streams rather than buffers,
that merged stderr survives, and that a bad exit is reported without masking the stream.
"""

from __future__ import annotations

import json
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import cast

import pytest

from devloop.drivers import ClaudeDriver
from devloop.drivers import cli as cli_module
from devloop.drivers.cli import CliDriver, DriverError
from devloop.events import Event, Result, SessionStarted, Text, Unknown

INIT = json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}) + "\n"
TEXT = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Hi"}]}})
RESULT = json.dumps({"type": "result", "subtype": "success", "total_cost_usd": 0.5}) + "\n"


def as_generator(run: Iterator[Event]) -> Generator[Event]:
    """The protocol promises an iterator; closing one early needs the generator underneath."""
    assert isinstance(run, Generator)
    return cast("Generator[Event]", run)


class FakeStdout:
    """Hands out lines one at a time and records how many were read, and whether closed."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.read = 0
        self.closed = False

    def __iter__(self) -> Iterator[str]:
        for line in self._lines:
            self.read += 1
            yield line

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(self, lines: list[str], code: int) -> None:
        self.stdout = FakeStdout(lines)
        self._code = code
        self.waited_with: float | None = None

    def wait(self, timeout: float | None = None) -> int:
        self.waited_with = timeout
        return self._code


class FakePopen:
    """Stands in for `subprocess.Popen` and remembers how it was called."""

    def __init__(self, lines: list[str], code: int = 0) -> None:
        self.process = FakeProcess(lines, code)
        self.argv: list[str] = []
        self.kwargs: dict[str, object] = {}

    def __call__(self, argv: list[str], **kwargs: object) -> FakeProcess:
        self.argv = argv
        self.kwargs = kwargs
        return self.process


@pytest.fixture
def popen(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakePopen]:
    fake = FakePopen([INIT, TEXT + "\n", RESULT])
    monkeypatch.setattr(cli_module.subprocess, "Popen", fake)
    yield fake


def test_satisfies_the_driver_protocol() -> None:
    driver: ClaudeDriver = CliDriver()
    assert driver.name == "cli(acceptEdits)"


class TestArgv:
    def test_stream_json_needs_verbose(self, popen: FakePopen, tmp_path: Path) -> None:
        """Without --verbose, stream-json does not emit the full stream."""
        list(CliDriver().run("review #7", tmp_path))
        assert popen.argv[:3] == ["claude", "-p", "review #7"]
        assert popen.argv[popen.argv.index("--output-format") + 1] == "stream-json"
        assert "--verbose" in popen.argv

    def test_permission_mode_and_extra_args(self, popen: FakePopen, tmp_path: Path) -> None:
        driver = CliDriver(
            executable="claude-dev", permission_mode="plan", extra_args=("--model", "x")
        )
        list(driver.run("p", tmp_path))
        assert popen.argv[0] == "claude-dev"
        assert popen.argv[popen.argv.index("--permission-mode") + 1] == "plan"
        assert popen.argv[-2:] == ["--model", "x"]

    def test_runs_in_the_given_directory_with_stderr_merged(
        self, popen: FakePopen, tmp_path: Path
    ) -> None:
        list(CliDriver().run("p", tmp_path))
        assert popen.kwargs["cwd"] == tmp_path
        assert popen.kwargs["stderr"] is cli_module.subprocess.STDOUT

    def test_a_prompt_is_one_argument_never_shell_parsed(
        self, popen: FakePopen, tmp_path: Path
    ) -> None:
        prompt = 'fix "this" && rm -rf / ; echo $HOME'
        list(CliDriver().run(prompt, tmp_path))
        assert popen.argv[2] == prompt
        assert popen.kwargs.get("shell") in (None, False)


class TestStream:
    def test_events_in_order(self, popen: FakePopen, tmp_path: Path) -> None:
        events = list(CliDriver().run("p", tmp_path))
        assert events[0] == SessionStarted(session_id="s1")
        assert events[1] == Text(text="Hi")
        assert isinstance(events[2], Result)
        assert events[2].cost_usd == 0.5
        assert len(events) == 3

    def test_streams_rather_than_buffers(self, popen: FakePopen, tmp_path: Path) -> None:
        """A run that prints nothing until it ends looks exactly like one that has hung."""
        run = as_generator(CliDriver().run("p", tmp_path))
        first = next(run)
        assert first == SessionStarted(session_id="s1")
        assert popen.process.stdout.read == 1
        run.close()

    def test_stdout_closed_and_process_reaped(self, popen: FakePopen, tmp_path: Path) -> None:
        list(CliDriver(timeout_seconds=42).run("p", tmp_path))
        assert popen.process.stdout.closed
        assert popen.process.waited_with == 42

    def test_abandoning_the_stream_still_reaps(self, popen: FakePopen, tmp_path: Path) -> None:
        run = as_generator(CliDriver().run("p", tmp_path))
        next(run)
        run.close()
        assert popen.process.stdout.closed
        assert popen.process.waited_with is not None

    def test_merged_stderr_is_kept(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Often the only explanation a failed run gives."""
        fake = FakePopen(["Failed to authenticate: OAuth session expired\n"], code=1)
        monkeypatch.setattr(cli_module.subprocess, "Popen", fake)
        events = list(CliDriver().run("p", tmp_path))
        assert events[0] == Unknown(raw="Failed to authenticate: OAuth session expired")


class TestExit:
    def test_zero_exit_adds_nothing(self, popen: FakePopen, tmp_path: Path) -> None:
        events = list(CliDriver().run("p", tmp_path))
        assert not any(isinstance(e, Unknown) for e in events)

    def test_non_zero_exit_is_reported_after_the_stream(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Not raised: the events already streamed must not be masked by the exit code."""
        fake = FakePopen([INIT, RESULT], code=3)
        monkeypatch.setattr(cli_module.subprocess, "Popen", fake)
        events = list(CliDriver().run("p", tmp_path))
        assert isinstance(events[1], Result)
        assert events[-1] == Unknown(raw="exit code 3")

    def test_a_missing_executable_raises_driver_error(self, tmp_path: Path) -> None:
        """A real spawn, not the fake: this is the one failure that must raise."""
        run = CliDriver(executable="devloop-no-such-binary-4f2a").run("p", tmp_path)
        with pytest.raises(DriverError, match="devloop-no-such-binary-4f2a"):
            next(run)
