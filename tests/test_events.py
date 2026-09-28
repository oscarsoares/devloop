"""Stream parsing, including the shapes that must not raise.

A parser that raises kills the run that produced the line. Every malformed case here is a
test rather than a comment for that reason.
"""

from __future__ import annotations

import json

from devloop.events import Result, SessionStarted, Text, ToolCall, Unknown, parse_line


def one(line: str) -> object:
    events = parse_line(line)
    assert len(events) == 1, f"expected exactly one event, got {events!r}"
    return events[0]


class TestRecognised:
    def test_session_init(self) -> None:
        event = one('{"type":"system","subtype":"init","session_id":"abc"}')
        assert event == SessionStarted(session_id="abc")

    def test_tool_call_prefers_the_command(self) -> None:
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Bash", "input": {"command": "dotnet build"}}
                    ]
                },
            }
        )
        assert one(line) == ToolCall(name="Bash", detail="dotnet build")

    def test_tool_call_falls_back_to_the_file(self) -> None:
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Edit", "input": {"file_path": "src/A.cs"}}
                    ]
                },
            }
        )
        assert one(line) == ToolCall(name="Edit", detail="src/A.cs")

    def test_tool_call_with_no_recognised_field(self) -> None:
        line = json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "name": "TodoWrite", "input": {}}]},
            }
        )
        assert one(line) == ToolCall(name="TodoWrite", detail="")

    def test_text_keeps_only_the_first_line(self) -> None:
        line = json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Found an L1.\nDetail follows."}]},
            }
        )
        assert one(line) == Text(text="Found an L1.")

    def test_one_message_can_yield_several_events(self) -> None:
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Building."},
                        {"type": "tool_use", "name": "Bash", "input": {"command": "make"}},
                    ]
                },
            }
        )
        assert parse_line(line) == [Text(text="Building."), ToolCall(name="Bash", detail="make")]


class TestResult:
    def test_cost_and_usage(self) -> None:
        line = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "num_turns": 12,
                "duration_ms": 94_000,
                "total_cost_usd": 0.8421,
                "usage": {
                    "input_tokens": 1500,
                    "output_tokens": 9800,
                    "cache_read_input_tokens": 410_000,
                    "cache_creation_input_tokens": 22_000,
                },
            }
        )
        event = one(line)
        assert isinstance(event, Result)
        assert event.ok
        assert event.cost_usd == 0.8421
        assert event.usage.cache_read_tokens == 410_000

    def test_a_missing_cost_is_none_not_zero(self) -> None:
        """The distinction the billing decision rests on: absent must not read as free."""
        event = one('{"type":"result","subtype":"success","usage":{"output_tokens":120}}')
        assert isinstance(event, Result)
        assert event.cost_usd is None
        assert event.usage.output_tokens == 120

    def test_error_result_is_not_ok(self) -> None:
        event = one('{"type":"result","subtype":"error_during_execution","is_error":true}')
        assert isinstance(event, Result)
        assert not event.ok


class TestNeverRaises:
    def test_blank_lines_yield_nothing(self) -> None:
        assert parse_line("   ") == []
        assert parse_line("") == []

    def test_unknown_event_type_yields_nothing(self) -> None:
        assert parse_line('{"type":"stream_event","event":{"foo":1}}') == []

    def test_malformed_json_is_kept_verbatim(self) -> None:
        raw = '{"type":"assistant","message":{'
        assert one(raw) == Unknown(raw=raw)

    def test_a_plain_cli_message_is_kept(self) -> None:
        """Often the only explanation a failed run gives."""
        raw = "Failed to authenticate: OAuth session expired"
        assert one(raw) == Unknown(raw=raw)

    def test_json_that_is_not_an_object(self) -> None:
        assert one("[1,2,3]") == Unknown(raw="[1,2,3]")

    def test_wrong_types_where_fields_are_expected(self) -> None:
        """A shape mismatch must degrade, not raise. This class of bug killed the predecessor."""
        line = json.dumps(
            {"type": "result", "subtype": "success", "num_turns": "twelve", "usage": "nope"}
        )
        event = one(line)
        assert isinstance(event, Result)
        assert event.turns == 0
        assert event.usage.output_tokens == 0

    def test_a_boolean_cost_is_rejected(self) -> None:
        """`True` is an int in Python. Without a guard it would become a cost of $1.00."""
        event = one('{"type":"result","subtype":"success","total_cost_usd":true}')
        assert isinstance(event, Result)
        assert event.cost_usd is None

    def test_assistant_message_with_a_broken_content_shape(self) -> None:
        assert parse_line('{"type":"assistant","message":{"content":"not a list"}}') == []
        assert parse_line('{"type":"assistant"}') == []
