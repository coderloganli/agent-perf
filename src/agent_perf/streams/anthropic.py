"""Anthropic Messages API stream parser.

Event flow (platform.claude.com streaming docs): ``message_start`` carries a
Message with initial ``usage``; ``content_block_delta`` events carry
``text_delta`` / ``input_json_delta`` / ``thinking_delta``; ``message_delta``
carries ``usage`` whose counts are cumulative; ``ping`` and ``error`` events
may appear anywhere. Unknown event types are ignored.
"""

from __future__ import annotations

from typing import Any

from agent_perf.streams.base import (
    StreamAccumulator,
    StreamResult,
    Usage,
    as_int,
    format_error,
)

OUTPUT_DELTA_TYPES = {"text_delta", "input_json_delta", "thinking_delta"}


def apply_usage(target: Usage, usage: dict[str, Any]) -> None:
    """Merge an Anthropic ``usage`` object; later (cumulative) values win.

    ``input_tokens`` counts only tokens after the last cache breakpoint, so the
    normalized ``prompt_tokens`` adds cache reads and writes back in.
    """
    read = as_int(usage.get("cache_read_input_tokens"))
    write = as_int(usage.get("cache_creation_input_tokens"))
    if read is not None:
        target.cache_read_tokens = read
    if write is not None:
        target.cache_write_tokens = write
    uncached = as_int(usage.get("input_tokens"))
    if uncached is not None:
        target.prompt_tokens = (
            uncached + (target.cache_read_tokens or 0) + (target.cache_write_tokens or 0)
        )
    out = as_int(usage.get("output_tokens"))
    if out is not None:
        target.output_tokens = out


class AnthropicStreamAccumulator(StreamAccumulator):
    def on_event(self, name: str | None, payload: dict[str, Any], offset_s: float) -> None:
        r = self.result
        if name == "message_start":
            msg = payload.get("message") or {}
            r.model = msg.get("model") or r.model
            apply_usage(r.usage, msg.get("usage") or {})
        elif name == "content_block_delta":
            if (payload.get("delta") or {}).get("type") in OUTPUT_DELTA_TYPES:
                r.delta_offsets_s.append(offset_s)
        elif name == "message_delta":
            apply_usage(r.usage, payload.get("usage") or {})
        elif name == "error":
            r.error = format_error(payload.get("error"), "type", "error")

    def parse_body(self, body: dict[str, Any]) -> StreamResult:
        r = StreamResult(model=body.get("model"))
        apply_usage(r.usage, body.get("usage") or {})
        if body.get("type") == "error":
            r.error = format_error(body.get("error"), "type", "error")
        return r
