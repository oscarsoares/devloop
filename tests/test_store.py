"""Durable state, against a real SQLite database in memory.

No mocks: an in-memory database exercises the same SQL the real one runs, including the
partial unique index that makes re-raising an escalation an update rather than a duplicate.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from devloop.models import PullRequest
from devloop.store import (
    SCHEMA_VERSION,
    CycleRecord,
    InvalidValueError,
    Store,
    connect,
    open_store,
)

REPO = "oscarsoares/altrus"


@pytest.fixture
def store() -> Iterator[Store]:
    with open_store(":memory:") as opened:
        yield opened


class TestSchema:
    def test_version_is_recorded(self, store: Store) -> None:
        assert store.schema_version == SCHEMA_VERSION

    def test_opening_twice_is_a_no_op(self) -> None:
        """Migration must be idempotent: every process start reapplies it."""
        connection = connect(":memory:")
        first = Store(connection)
        first.record_review_round(REPO, 94)
        second = Store(connection)  # same connection, migrated again
        assert second.rounds_for(REPO, 94) == 1


class TestReviewRounds:
    def test_an_unseen_pr_has_no_rounds(self, store: Store) -> None:
        assert store.rounds_for(REPO, 94) == 0

    def test_rounds_accumulate(self, store: Store) -> None:
        assert store.record_review_round(REPO, 94) == 1
        assert store.record_review_round(REPO, 94) == 2
        assert store.record_review_round(REPO, 94) == 3
        assert store.rounds_for(REPO, 94) == 3

    def test_rounds_are_per_pr_and_per_repo(self, store: Store) -> None:
        store.record_review_round(REPO, 94)
        store.record_review_round(REPO, 94)
        store.record_review_round(REPO, 96)
        store.record_review_round("oscarsoares/chronos", 94)
        assert store.rounds_for(REPO, 94) == 2
        assert store.rounds_for(REPO, 96) == 1
        assert store.rounds_for("oscarsoares/chronos", 94) == 1

    def test_hydrate_fills_the_gap_github_cannot(self, store: Store) -> None:
        """The number no API carries, which is why the budget needed this table."""
        store.record_review_round(REPO, 94)
        store.record_review_round(REPO, 94)
        prs = [PullRequest(number=94), PullRequest(number=96)]
        hydrated = {pr.number: pr.rounds for pr in store.hydrate_rounds(REPO, prs)}
        assert hydrated == {94: 2, 96: 0}

    def test_hydrate_changes_nothing_else(self, store: Store) -> None:
        original = PullRequest(number=94, title="keep me", labels=frozenset({"needs-review"}))
        (hydrated,) = store.hydrate_rounds(REPO, [original])
        assert hydrated.title == "keep me"
        assert hydrated.labels == frozenset({"needs-review"})


def cycle(**kwargs: object) -> CycleRecord:
    defaults: dict[str, object] = {
        "kind": "review",
        "target": 94,
        "started_at": "2026-09-30T10:00:00+00:00",
        "ok": True,
        "cost_usd": 0.5,
    }
    return CycleRecord(**{**defaults, **kwargs})  # pyright: ignore[reportArgumentType]


class TestSpend:
    def test_nothing_recorded_is_zero_not_an_error(self, store: Store) -> None:
        spend = store.spend(REPO)
        assert spend.cycles == 0
        assert spend.cost_usd == 0.0
        assert not spend.partial
        assert spend.per_cycle() is None

    def test_costs_and_tokens_accumulate(self, store: Store) -> None:
        store.record_cycle(REPO, cycle(cost_usd=0.5, output_tokens=100))
        store.record_cycle(REPO, cycle(cost_usd=1.5, output_tokens=200, cache_read=9))
        spend = store.spend(REPO)
        assert spend.cost_usd == 2.0
        assert spend.output_tokens == 300
        assert spend.cache_read_tokens == 9
        assert spend.cycles == 2
        assert spend.per_cycle() == 1.0

    def test_a_cycle_reporting_no_cost_marks_the_total_partial(self, store: Store) -> None:
        """A floor, not a figure. Understating a total silently is worse than reporting none."""
        store.record_cycle(REPO, cycle(cost_usd=1.0))
        store.record_cycle(REPO, cycle(cost_usd=None))
        spend = store.spend(REPO)
        assert spend.cost_usd == 1.0
        assert spend.partial
        assert spend.cycles == 2

    def test_a_reported_zero_is_not_partial(self, store: Store) -> None:
        """Free and unreported are different facts, and the schema keeps them apart."""
        store.record_cycle(REPO, cycle(cost_usd=0.0))
        assert not store.spend(REPO).partial

    def test_spend_is_scoped_per_repo(self, store: Store) -> None:
        store.record_cycle(REPO, cycle(cost_usd=1.0))
        store.record_cycle("oscarsoares/chronos", cycle(cost_usd=4.0))
        assert store.spend(REPO).cost_usd == 1.0
        assert store.spend("oscarsoares/chronos").cost_usd == 4.0
        assert store.spend().cost_usd == 5.0

    def test_failed_cycles_still_count_toward_spend(self, store: Store) -> None:
        """A run that failed still consumed tokens. Excluding it would flatter the figure."""
        store.record_cycle(REPO, cycle(ok=False, cost_usd=0.25))
        assert store.spend(REPO).cost_usd == 0.25


class TestEscalations:
    def test_none_by_default(self, store: Store) -> None:
        assert store.open_escalations(REPO) == []

    def test_raising_makes_it_answerable(self, store: Store) -> None:
        store.raise_escalation(REPO, "issue", 74, "ProviderProfile is not IMultiTenant")
        (found,) = store.open_escalations(REPO)
        assert found.number == 74
        assert found.subject == "issue"
        assert "IMultiTenant" in found.reason

    def test_raising_the_same_thing_twice_updates_rather_than_duplicates(
        self, store: Store
    ) -> None:
        store.raise_escalation(REPO, "issue", 74, "first reading")
        store.raise_escalation(REPO, "issue", 74, "sharper reading")
        (found,) = store.open_escalations(REPO)
        assert found.reason == "sharper reading"

    def test_resolving_removes_it_from_the_open_list(self, store: Store) -> None:
        store.raise_escalation(REPO, "issue", 74, "why")
        store.resolve_escalation(REPO, "issue", 74)
        assert store.open_escalations(REPO) == []

    def test_the_same_subject_can_be_raised_again_after_resolution(self, store: Store) -> None:
        """The partial unique index covers open rows only, so history is kept."""
        store.raise_escalation(REPO, "issue", 74, "first time")
        store.resolve_escalation(REPO, "issue", 74)
        store.raise_escalation(REPO, "issue", 74, "came back")
        (found,) = store.open_escalations(REPO)
        assert found.reason == "came back"

    def test_an_issue_and_a_pr_with_the_same_number_are_distinct(self, store: Store) -> None:
        store.raise_escalation(REPO, "issue", 74, "issue reason")
        store.raise_escalation(REPO, "pr", 74, "pr reason")
        assert len(store.open_escalations(REPO)) == 2

    def test_resolving_something_never_raised_is_not_an_error(self, store: Store) -> None:
        store.resolve_escalation(REPO, "issue", 999)
        assert store.open_escalations(REPO) == []


class TestReviewHistory:
    def test_an_unseen_pr_has_no_history(self, store: Store) -> None:
        assert store.review_history(REPO, 94) is None

    def test_a_review_records_round_cost_decision_and_time(self, store: Store) -> None:
        assert store.record_review(REPO, 94, 0.25, "request_changes") == 1
        history = store.review_history(REPO, 94)
        assert history is not None
        assert (history.rounds, history.cost_usd, history.last_decision) == (
            1,
            0.25,
            "request_changes",
        )
        assert history.last_action_at

    def test_rounds_and_cost_accumulate_and_the_last_decision_wins(self, store: Store) -> None:
        store.record_review(REPO, 94, 0.25, "request_changes")
        assert store.record_review(REPO, 94, 0.5, "approve") == 2
        history = store.review_history(REPO, 94)
        assert history is not None
        assert (history.rounds, history.cost_usd, history.last_decision) == (2, 0.75, "approve")

    def test_it_feeds_the_same_budget_as_record_review_round(self, store: Store) -> None:
        store.record_review_round(REPO, 94)
        store.record_review(REPO, 94, 0.1, "block")
        assert store.rounds_for(REPO, 94) == 2

    def test_a_round_without_a_review_leaves_no_decision(self, store: Store) -> None:
        store.record_review_round(REPO, 94)
        history = store.review_history(REPO, 94)
        assert history is not None
        assert history.last_decision is None
        assert history.cost_usd == 0.0

    def test_a_decision_outside_the_contract_is_rejected_and_not_stored(self, store: Store) -> None:
        with pytest.raises(InvalidValueError, match="decision"):
            store.record_review(REPO, 94, 0.1, "lgtm")  # pyright: ignore[reportArgumentType]
        assert store.review_history(REPO, 94) is None


class TestDevelopmentHistory:
    def test_an_unseen_issue_has_no_history(self, store: Store) -> None:
        assert store.development_history(REPO, 5) is None

    def test_attempts_and_cost_accumulate_and_status_follows_the_last(self, store: Store) -> None:
        assert store.record_development(REPO, 5, 1.0, "in_progress") == 1
        assert store.record_development(REPO, 5, 0.5, "abandoned") == 2
        assert store.record_development(REPO, 5, 2.0, "done") == 3
        history = store.development_history(REPO, 5)
        assert history is not None
        assert (history.attempts, history.cost_usd, history.status) == (3, 3.5, "done")
        assert history.updated_at

    def test_issues_are_tracked_per_repo(self, store: Store) -> None:
        store.record_development(REPO, 5, 1.0, "done")
        store.record_development("oscarsoares/chronos", 5, 2.0, "abandoned")
        history = store.development_history(REPO, 5)
        assert history is not None
        assert history.status == "done"

    def test_a_status_outside_the_contract_is_rejected(self, store: Store) -> None:
        with pytest.raises(InvalidValueError, match="status"):
            store.record_development(REPO, 5, 1.0, "finished")  # pyright: ignore[reportArgumentType]
        assert store.development_history(REPO, 5) is None


class TestTotalCost:
    def test_nothing_recorded_is_zero(self, store: Store) -> None:
        assert store.total_cost() == 0.0

    def test_sums_reviews_and_development(self, store: Store) -> None:
        store.record_review(REPO, 94, 0.25, "approve")
        store.record_review(REPO, 96, 0.5, "block")
        store.record_development(REPO, 5, 1.0, "done")
        assert store.total_cost() == 1.75

    def test_can_be_scoped_to_a_repo(self, store: Store) -> None:
        store.record_review(REPO, 94, 0.25, "approve")
        store.record_development("oscarsoares/chronos", 5, 1.0, "done")
        assert store.total_cost(REPO) == 0.25
        assert store.total_cost("oscarsoares/chronos") == 1.0


class TestMigration:
    def test_a_version_1_database_gains_the_new_columns_and_keeps_its_rounds(self) -> None:
        connection = connect(":memory:")
        connection.executescript(
            """
            CREATE TABLE pr_review (
                repo TEXT NOT NULL, pr_number INTEGER NOT NULL,
                rounds INTEGER NOT NULL DEFAULT 0, last_round_at TEXT,
                PRIMARY KEY (repo, pr_number)
            );
            INSERT INTO pr_review VALUES ('r', 1, 2, '2026-01-01T00:00:00+00:00');
            PRAGMA user_version = 1;
            """
        )
        store = Store(connection)
        assert store.schema_version == SCHEMA_VERSION
        assert store.rounds_for("r", 1) == 2
        store.record_review("r", 1, 0.5, "approve")
        history = store.review_history("r", 1)
        assert history is not None
        assert (history.rounds, history.cost_usd) == (3, 0.5)
