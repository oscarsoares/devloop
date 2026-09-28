"""Drives Claude by spawning the `claude` CLI.

This is the driver that runs on a Claude Code subscription. It is the default today because
it costs nothing beyond the subscription; see the package docstring for why the choice is
behind an interface at all.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from devloop.events import Event, Unknown, parse_line


class DriverError(RuntimeError):
    """The process could not be started, or ended in a way the caller must know about."""

    @classmethod
    def could_not_start(cls, executable: str, cause: OSError) -> DriverError:
        return cls(f"could not start {executable!r}: {cause}")


@dataclass(frozen=True, slots=True)
class CliDriver:
    executable: str = "claude"
    permission_mode: str = "acceptEdits"
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    timeout_seconds: int | None = 3 * 60 * 60

    @property
    def name(self) -> str:
        return f"cli({self.permission_mode})"

    def _argv(self, prompt: str) -> list[str]:
        # stream-json, not the default text format: with `text`, nothing is printed until
        # the run finishes. --verbose is required for stream-json to emit the full stream.
        return [
            self.executable,
            "-p",
            prompt,
            "--permission-mode",
            self.permission_mode,
            "--output-format",
            "stream-json",
            "--verbose",
            *self.extra_args,
        ]

    def run(self, prompt: str, cwd: Path) -> Iterator[Event]:
        try:
            process = subprocess.Popen(  # noqa: S603 - argv is built here, never shell-parsed
                self._argv(prompt),
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise DriverError.could_not_start(self.executable, exc) from exc

        stdout = process.stdout
        if stdout is None:  # pragma: no cover - Popen with PIPE always provides one
            raise DriverError.could_not_start(self.executable, OSError("no stdout"))

        try:
            # stderr is merged into stdout on purpose. The CLI reports authentication and
            # startup failures there, and those lines are often the only explanation a
            # failed run ever gives; parse_line keeps them as Unknown rather than dropping
            # them.
            for line in stdout:
                yield from parse_line(line)
        finally:
            stdout.close()
            code = process.wait(timeout=self.timeout_seconds)

        if code != 0:
            # Not raised: a non-zero exit after a stream that already explained itself should
            # not mask those events. The caller decides, having seen the Result (or its
            # absence).
            yield Unknown(raw=f"exit code {code}")
