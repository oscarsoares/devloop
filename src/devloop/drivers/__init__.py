"""The seam between the orchestrator and Claude.

Everything above this package works in terms of `Event`. Nothing above it knows whether
those events came from a spawned CLI process or from the Agent SDK — and that is the point.

The two are not interchangeable in cost. The CLI authenticates with a Claude Code
subscription and is free under its usage limits; the Agent SDK's documented auth methods are
API key and the cloud providers, so it bills per token. Which one is right depends on a
measured cost per cycle that we do not have yet, and on `--bare` eventually becoming the
default for `claude -p`, which will end the subscription path on its own schedule.

Keeping the choice behind one interface means that decision costs a new driver, not a
rewrite. It is deferred by design, not by indecision.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from devloop.events import Event


class ClaudeDriver(Protocol):
    """Runs one prompt to completion, streaming events as they arrive.

    Implementations must stream rather than buffer. A review takes minutes, and a run that
    prints nothing until it finishes is indistinguishable from one that has hung — which is
    exactly how the first real tick of the predecessor was misread as blocked.
    """

    @property
    def name(self) -> str:
        """Identifies the driver in logs, so a tick's transcript says how it was produced."""
        ...

    def run(self, prompt: str, cwd: Path) -> Iterator[Event]:
        """Yield events until the run ends. The last event is normally a `Result`."""
        ...
