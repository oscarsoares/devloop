"""The repository facts the decisions are made from.

These are deliberately plain: whatever fetches them — `gh`, the REST API, a fixture in a
test — builds these, and the decision functions take nothing else. That is what makes the
decisions testable without a network, which is the property the previous implementation
lacked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Priority(IntEnum):
    """Lower sorts first. `NONE` is last, so an unprioritised issue never jumps a P3."""

    P0 = 0
    P1 = 1
    P2 = 2
    P3 = 3
    NONE = 9

    @classmethod
    def from_labels(cls, labels: frozenset[str]) -> Priority:
        by_label = {
            "P0-critical": cls.P0,
            "P1-high": cls.P1,
            "P2-medium": cls.P2,
            "P3-low": cls.P3,
        }
        found = [value for label, value in by_label.items() if label in labels]
        return min(found) if found else cls.NONE


class CheckState(IntEnum):
    PASSED = 0
    PENDING = 1
    FAILED = 2


@dataclass(frozen=True, slots=True)
class PullRequest:
    number: int
    title: str = ""
    labels: frozenset[str] = frozenset()
    review_decision: str = ""
    checks: tuple[CheckState, ...] = ()
    rounds: int = 0
    """Review iterations already spent. The budget is per PR."""
    head_ref: str = ""
    body: str = ""
    """Kept only to work out which issues this PR already owns; no decision reads them."""

    @property
    def has_pending_checks(self) -> bool:
        return CheckState.PENDING in self.checks

    @property
    def failed_check_count(self) -> int:
        return sum(1 for c in self.checks if c is CheckState.FAILED)

    @property
    def gate_held(self) -> bool:
        return "review-blocking" in self.labels

    @property
    def never_reviewed(self) -> bool:
        """True when the loop has not reviewed it: a review always leaves one of its labels."""
        return not (self.labels & {"needs-review", "review-blocking"})


@dataclass(frozen=True, slots=True)
class Issue:
    number: int
    title: str = ""
    labels: frozenset[str] = frozenset()
    milestone: str | None = None
    has_open_pr: bool = False

    @property
    def priority(self) -> Priority:
        return Priority.from_labels(self.labels)

    @property
    def is_bug(self) -> bool:
        return "bug" in self.labels


@dataclass(frozen=True, slots=True)
class Spend:
    """Accumulated cost for a tick. Client-side estimates, not a bill."""

    cost_usd: float = 0.0
    partial: bool = False
    """True when at least one cycle reported no cost, so the total is a floor, not a figure."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cycles: int = 0

    def per_cycle(self) -> float | None:
        return self.cost_usd / self.cycles if self.cycles else None


@dataclass(frozen=True, slots=True)
class Budget:
    """Runaway guards for one tick, not limits on intent. Leftovers carry to the next tick."""

    max_cycles: int = 6
    max_minutes: int = 150
    max_review_rounds: int = 3
    """Iterations one PR may receive before it escalates to a human."""
    labels_excluding_selection: frozenset[str] = field(
        default_factory=lambda: frozenset({"in-progress", "blocked"})
    )
