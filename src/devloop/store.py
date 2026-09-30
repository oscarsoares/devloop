"""Durable state.

Three things must survive a tick ending, and none of them live on GitHub:

- **Review rounds per PR.** The iteration budget is the only thing standing between a PR
  whose merge gate stays held and an executor that reviews it forever. GitHub has no such
  field, so without this table the budget is not enforced at all — which was true of the
  predecessor and is the reason this table exists first.
- **Escalations.** A blocked issue that nobody notices is work that disappeared. Recording
  them makes "what is waiting on me" answerable.
- **Cycle history and spend.** The measured cost per cycle is what the CLI-versus-SDK
  decision waits on, and one tick's figure is not evidence; a series is.

SQLite from the standard library, no ORM. This is a single-writer local process, so the
schema is small and the queries are plain.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from devloop.models import PullRequest, Spend

SCHEMA_VERSION = 1

CycleKind = Literal["review", "develop"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pr_review (
    repo          TEXT    NOT NULL,
    pr_number     INTEGER NOT NULL,
    rounds        INTEGER NOT NULL DEFAULT 0,
    last_round_at TEXT,
    PRIMARY KEY (repo, pr_number)
);

CREATE TABLE IF NOT EXISTS cycle (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    repo          TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    target        INTEGER NOT NULL,
    started_at    TEXT    NOT NULL,
    ok            INTEGER NOT NULL,
    -- NULL means the run reported no cost. Distinct from 0.0, which means it reported free:
    -- collapsing the two understates a total and leads to the wrong billing decision.
    cost_usd      REAL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read    INTEGER NOT NULL DEFAULT 0,
    cache_write   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS cycle_by_repo ON cycle (repo, started_at);

CREATE TABLE IF NOT EXISTS escalation (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    repo        TEXT    NOT NULL,
    subject     TEXT    NOT NULL,
    number      INTEGER NOT NULL,
    reason      TEXT    NOT NULL,
    raised_at   TEXT    NOT NULL,
    resolved_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS escalation_open
    ON escalation (repo, subject, number) WHERE resolved_at IS NULL;
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class Escalation:
    subject: str
    """"issue" or "pr" — what the number refers to."""
    number: int
    reason: str
    raised_at: str


@dataclass(frozen=True, slots=True)
class CycleRecord:
    kind: CycleKind
    target: int
    started_at: str
    ok: bool
    cost_usd: float | None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0


class Store:
    """The state of one machine's loop, across every repository it drives."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection
        self._migrate()

    def _migrate(self) -> None:
        # user_version keeps the check to one integer read. The schema is CREATE IF NOT
        # EXISTS throughout, so applying it to an existing database is a no-op and opening
        # an older one is safe.
        self._db.executescript(_SCHEMA)
        self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._db.commit()

    @property
    def schema_version(self) -> int:
        row = self._db.execute("PRAGMA user_version").fetchone()
        return _int(row, 0)

    def close(self) -> None:
        self._db.close()

    # --- review rounds -------------------------------------------------------

    def rounds_for(self, repo: str, pr_number: int) -> int:
        row = self._db.execute(
            "SELECT rounds FROM pr_review WHERE repo = ? AND pr_number = ?",
            (repo, pr_number),
        ).fetchone()
        return _int(row, 0)

    def record_review_round(self, repo: str, pr_number: int) -> int:
        """Count one completed review iteration and return the new total.

        Called when a round finished, not when one started: a run that died halfway
        corrected nothing, and charging it against the budget would retire a PR the loop
        never actually reviewed.
        """
        self._db.execute(
            """
            INSERT INTO pr_review (repo, pr_number, rounds, last_round_at)
                 VALUES (?, ?, 1, ?)
            ON CONFLICT (repo, pr_number)
              DO UPDATE SET rounds = rounds + 1, last_round_at = excluded.last_round_at
            """,
            (repo, pr_number, _now()),
        )
        self._db.commit()
        return self.rounds_for(repo, pr_number)

    def hydrate_rounds(self, repo: str, prs: Sequence[PullRequest]) -> list[PullRequest]:
        """Fill in each PR's spent rounds from the store.

        The bridge between GitHub and the decisions: `github.py` cannot know this number and
        `decide.py` must not guess it.
        """
        from dataclasses import replace

        return [replace(pr, rounds=self.rounds_for(repo, pr.number)) for pr in prs]

    # --- cycles and spend ----------------------------------------------------

    def record_cycle(self, repo: str, record: CycleRecord) -> None:
        self._db.execute(
            """
            INSERT INTO cycle (repo, kind, target, started_at, ok, cost_usd,
                               input_tokens, output_tokens, cache_read, cache_write)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                repo,
                record.kind,
                record.target,
                record.started_at,
                int(record.ok),
                record.cost_usd,
                record.input_tokens,
                record.output_tokens,
                record.cache_read,
                record.cache_write,
            ),
        )
        self._db.commit()

    def spend(self, repo: str | None = None) -> Spend:
        """Totals across recorded cycles.

        `partial` is true when any cycle reported no cost, so the total is a floor rather
        than a figure. A total that quietly understates itself is worse than none, because it
        is acted on with confidence.
        """
        where, params = ("WHERE repo = ?", (repo,)) if repo else ("", ())
        row = self._db.execute(
            f"""
            SELECT COALESCE(SUM(cost_usd), 0.0),
                   SUM(CASE WHEN cost_usd IS NULL THEN 1 ELSE 0 END),
                   COALESCE(SUM(input_tokens), 0),
                   COALESCE(SUM(output_tokens), 0),
                   COALESCE(SUM(cache_read), 0),
                   COALESCE(SUM(cache_write), 0),
                   COUNT(*)
              FROM cycle {where}
            """,  # noqa: S608 - `where` is a literal chosen above, never user input
            params,
        ).fetchone()
        values = _row(row, 7)
        return Spend(
            cost_usd=_as_float(values[0]),
            partial=_as_int(values[1]) > 0,
            input_tokens=_as_int(values[2]),
            output_tokens=_as_int(values[3]),
            cache_read_tokens=_as_int(values[4]),
            cache_write_tokens=_as_int(values[5]),
            cycles=_as_int(values[6]),
        )

    # --- escalations ---------------------------------------------------------

    def raise_escalation(self, repo: str, subject: str, number: int, reason: str) -> None:
        """Record something waiting on a human. Re-raising the same thing updates the reason."""
        self._db.execute(
            """
            INSERT INTO escalation (repo, subject, number, reason, raised_at)
                 VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (repo, subject, number) WHERE resolved_at IS NULL
              DO UPDATE SET reason = excluded.reason
            """,
            (repo, subject, number, reason, _now()),
        )
        self._db.commit()

    def resolve_escalation(self, repo: str, subject: str, number: int) -> None:
        self._db.execute(
            """
            UPDATE escalation SET resolved_at = ?
             WHERE repo = ? AND subject = ? AND number = ? AND resolved_at IS NULL
            """,
            (_now(), repo, subject, number),
        )
        self._db.commit()

    def open_escalations(self, repo: str) -> list[Escalation]:
        rows = self._db.execute(
            """
            SELECT subject, number, reason, raised_at
              FROM escalation
             WHERE repo = ? AND resolved_at IS NULL
             ORDER BY raised_at, number
            """,
            (repo,),
        ).fetchall()
        return [
            Escalation(
                subject=_as_str(values[0]),
                number=_as_int(values[1]),
                reason=_as_str(values[2]),
                raised_at=_as_str(values[3]),
            )
            for values in (_row(row, 4) for row in cast("list[object]", rows))
        ]


def _row(row: object, width: int) -> tuple[object, ...]:
    """A result row as a plain tuple. sqlite3 returns `Any`; strict typing needs a boundary."""
    if not isinstance(row, tuple):
        return (None,) * width
    values = cast("tuple[object, ...]", row)
    return values if len(values) >= width else values + (None,) * (width - len(values))


def _int(row: object, default: int) -> int:
    value = _row(row, 1)[0]
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _as_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    return float(value) if isinstance(value, (int, float)) else 0.0


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def connect(path: Path | Literal[":memory:"]) -> sqlite3.Connection:
    target = path if path == ":memory:" else str(path)
    if isinstance(path, Path):
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target, timeout=30.0, isolation_level=None)
    # WAL so a reader (a `status` run) never blocks the writer (a tick in flight).
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def open_store(path: Path | Literal[":memory:"]) -> Generator[Store]:
    store = Store(connect(path))
    try:
        yield store
    finally:
        store.close()


def default_path() -> Path:
    """Where state lives when nothing says otherwise: beside the user's other tool state."""
    return Path.home() / ".devloop" / "state.db"
