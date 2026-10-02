"""Common interface for provider stream accumulators."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agent_perf.streams.sse import SSEDecoder, SSEEvent


@dataclass
class Usage:
    prompt_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass
class StreamResult:
    usage: Usage = field(default_factory=Usage)
    model: str | None = None
    # Offsets (seconds since request start) of output-content delta events.
    delta_offsets_s: list[float] = field(default_factory=list)
    error: str | None = None


class StreamAccumulator(ABC):
    """Consumes a streamed (SSE) response body chunk by chunk."""

    def __init__(self) -> None:
        self._decoder = SSEDecoder()
        self.result = StreamResult()

    def feed(self, chunk: bytes, offset_s: float) -> None:
        for ev in self._decoder.feed(chunk):
            self._dispatch(ev, offset_s)

    def finish(self, offset_s: float) -> StreamResult:
        for ev in self._decoder.flush():
            self._dispatch(ev, offset_s)
        return self.result

    def _dispatch(self, ev: SSEEvent, offset_s: float) -> None:
        if not ev.data or ev.data == "[DONE]":
            return
        try:
            payload = json.loads(ev.data)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            self.on_event(ev.event or payload.get("type"), payload, offset_s)

    @abstractmethod
    def on_event(self, name: str | None, payload: dict[str, Any], offset_s: float) -> None: ...

    @abstractmethod
    def parse_body(self, body: dict[str, Any]) -> StreamResult:
        """Parse a non-streamed JSON response body."""


def as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def format_error(err: Any, code_key: str, default: str) -> str:
    if not isinstance(err, dict):
        return default
    return f"{err.get(code_key) or default}: {err.get('message', '')}".strip()
