"""OpenAI Responses API stream parser (the wire API Codex uses).

Output deltas arrive as ``response.output_text.delta``,
``response.function_call_arguments.delta`` and reasoning delta events. The
terminal ``response.completed`` / ``response.incomplete`` / ``response.failed``
event carries ``response.usage`` with ``input_tokens`` (inclusive of
``input_tokens_details.cached_tokens`` and ``cache_write_tokens``),
``output_tokens`` and ``output_tokens_details.reasoning_tokens``.
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

TERMINAL_EVENTS = {"response.completed", "response.incomplete", "response.failed"}


def is_output_delta(name: str) -> bool:
    return name.startswith("response.") and name.endswith(".delta")


def apply_usage(target: Usage, usage: dict[str, Any]) -> None:
    in_details = usage.get("input_tokens_details") or {}
    out_details = usage.get("output_tokens_details") or {}
    target.prompt_tokens = as_int(usage.get("input_tokens"))
    target.cache_read_tokens = as_int(in_details.get("cached_tokens"))
    target.cache_write_tokens = as_int(in_details.get("cache_write_tokens"))
    target.output_tokens = as_int(usage.get("output_tokens"))
    target.reasoning_tokens = as_int(out_details.get("reasoning_tokens"))


class OpenAIResponsesStreamAccumulator(StreamAccumulator):
    def on_event(self, name: str | None, payload: dict[str, Any], offset_s: float) -> None:
        if not name:
            return
        r = self.result
        if name == "response.created":
            r.model = (payload.get("response") or {}).get("model") or r.model
        elif is_output_delta(name):
            r.delta_offsets_s.append(offset_s)
        elif name in TERMINAL_EVENTS:
            resp = payload.get("response") or {}
            r.model = resp.get("model") or r.model
            if resp.get("usage"):
                apply_usage(r.usage, resp["usage"])
            if name == "response.failed":
                r.error = format_error(resp.get("error"), "code", "failed")
        elif name == "error":
            r.error = format_error(payload, "code", "error")

    def parse_body(self, body: dict[str, Any]) -> StreamResult:
        r = StreamResult(model=body.get("model"))
        if body.get("usage"):
            apply_usage(r.usage, body["usage"])
        if body.get("error"):
            r.error = format_error(body["error"], "code", "error")
        return r
