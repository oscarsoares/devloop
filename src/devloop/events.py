"""Typed events from a Claude run.

The CLI emits newline-delimited JSON; the Agent SDK emits typed objects. Both are reduced
to the events below so the rest of the orchestrator never sees either shape. That is the
whole point of the seam: swapping the driver must not ripple past this module.

Parsing is deliberately tolerant. A stream is a live process's stdout, so a malformed or
unrecognised line must never end a run that is otherwise fine — it becomes `Unknown` and
the loop keeps going. That is a lesson from the version this replaces, where a formatter
that raised would have killed the tick that produced the line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from devloop._json import as_dict, as_float, as_int, as_list, as_str


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts for one run. Absent fields read as zero, never as an error."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class SessionStarted:
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Text:
    text: str


@dataclass(frozen=True, slots=True)
class Result:
    """The terminal event.

    `cost_usd` is None when the run reported none. That distinction matters: a missing cost
    must never be read as zero, because a total that silently understates itself leads to
    the wrong billing decision with full confidence.
    """

    subtype: str
    turns: int = 0
    duration_ms: int = 0
    cost_usd: float | None = None
    usage: Usage = field(default_factory=Usage)
    is_error: bool = False

    @property
    def ok(self) -> bool:
        return not self.is_error and self.subtype == "success"


@dataclass(frozen=True, slots=True)
class Unknown:
    """A line the parser did not recognise, kept verbatim so nothing is lost silently."""

    raw: str


Event = SessionStarted | ToolCall | Text | Result | Unknown

# The field a tool call is most usefully identified by, per tool. First match wins.
_TOOL_DETAIL_FIELDS = ("command", "file_path", "pattern", "path", "prompt")

_MAX_DETAIL = 120


def _parse_usage(raw: object) -> Usage:
    usage = as_dict(raw)
    if usage is None:
        return Usage()
    return Usage(
        input_tokens=as_int(usage.get("input_tokens")),
        output_tokens=as_int(usage.get("output_tokens")),
        cache_read_tokens=as_int(usage.get("cache_read_input_tokens")),
        cache_write_tokens=as_int(usage.get("cache_creation_input_tokens")),
    )


def _tool_detail(raw_input: object) -> str:
    tool_input = as_dict(raw_input)
    if tool_input is None:
        return ""
    for name in _TOOL_DETAIL_FIELDS:
        value = as_str(tool_input.get(name))
        if value:
            detail = " ".join(value.split())
            if len(detail) > _MAX_DETAIL:
                detail = detail[: _MAX_DETAIL - 3] + "..."
            return detail
    return ""


def _parse_assistant(message: object) -> list[Event]:
    envelope = as_dict(message)
    if envelope is None:
        return []
    content = as_list(envelope.get("content"))
    if content is None:
        return []

    events: list[Event] = []
    for raw_block in content:
        block = as_dict(raw_block)
        if block is None:
            continue
        match as_str(block.get("type")):
            case "tool_use":
                events.append(
                    ToolCall(
                        name=as_str(block.get("name")) or "tool",
                        detail=_tool_detail(block.get("input")),
                    )
                )
            case "text":
                text = as_str(block.get("text")) or ""
                first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
                if first:
                    events.append(Text(text=first))
            case _:
                continue
    return events


def parse_line(line: str) -> list[Event]:
    """Reduce one stream-json line to zero or more events.

    Returns a list because a single assistant message can carry several content blocks, and
    because a blank line legitimately carries nothing.
    """
    stripped = line.strip()
    if not stripped:
        return []
    if not stripped.startswith("{"):
        # A plain CLI message, e.g. an authentication failure. Keep it: these lines are
        # often the only explanation a failed run ever gives.
        return [Unknown(raw=stripped)]

    try:
        payload: object = json.loads(stripped)
    except json.JSONDecodeError:
        return [Unknown(raw=stripped)]

    event = as_dict(payload)
    if event is None:
        return [Unknown(raw=stripped)]

    match as_str(event.get("type")):
        case "system":
            if as_str(event.get("subtype")) == "init":
                return [SessionStarted(session_id=as_str(event.get("session_id")))]
            return []
        case "assistant":
            return _parse_assistant(event.get("message"))
        case "result":
            return [
                Result(
                    subtype=as_str(event.get("subtype")) or "unknown",
                    turns=as_int(event.get("num_turns")),
                    duration_ms=as_int(event.get("duration_ms")),
                    cost_usd=as_float(event.get("total_cost_usd")),
                    usage=_parse_usage(event.get("usage")),
                    is_error=event.get("is_error") is True,
                )
            ]
        case _:
            return []
