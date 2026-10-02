"""Attribute a model request to the harness component that issued it.

Two sources are used, and nothing else is inferred:

1. Request headers the harness sends.

   Claude Code (documented in the Claude Code gateway compatibility guide):
   ``x-claude-code-session-id``, ``x-claude-code-agent-id`` (only on requests
   from a spawned subagent) and ``x-claude-code-parent-agent-id`` (nested
   agents) are always sent. The "gateway hint" headers
   ``x-claude-code-request-class``, ``x-claude-code-agent-type``,
   ``x-claude-code-prompt-id`` and ``x-claude-code-prev-tool-durations`` are
   off by default for a custom base URL and require
   ``CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`` in the harness environment.

   Codex: ``session-id``, ``thread-id``, ``x-codex-parent-thread-id`` and
   ``x-openai-subagent`` (value ``collab_spawn`` for spawned agents, or
   ``review`` / ``compact`` / ``memory_consolidation`` / other labels).
   TODO: these names come from reading the openai/codex source
   (codex-rs/codex-api/src/requests/headers.rs, codex-rs/core/src/client.rs),
   not from documentation, and may change between Codex releases. Verify
   against a captured Codex request before relying on them.

2. The request body. The tool results carried in the trailing turn of a
   request identify the tool calls whose completion triggered it; the tool
   names are resolved from the matching tool-call blocks earlier in the same
   body. This works identically for both harnesses because it only relies on
   the public Messages / Responses API schemas.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote


@dataclass
class Attribution:
    session_id: str | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    request_class: str | None = None
    agent_type: str | None = None
    prompt_id: str | None = None
    prev_tool_durations_ms: dict[str, int] = field(default_factory=dict)


def _lower(headers: Mapping[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items()}


def parse_prev_tool_durations(value: str | None) -> dict[str, int]:
    """Parse ``x-claude-code-prev-tool-durations`` (``Bash=742;Read=9``).

    Names are percent-encoded. Repeated tool names are summed, since the
    header carries one entry per tool call.
    """
    out: dict[str, int] = {}
    if not value:
        return out
    for entry in value.split(";"):
        name, sep, ms = entry.partition("=")
        if not sep:
            continue
        try:
            dur = int(ms)
        except ValueError:
            continue
        key = unquote(name)
        out[key] = out.get(key, 0) + dur
    return out


def attribution_from_headers(headers: Mapping[str, str]) -> Attribution:
    h = _lower(headers)
    if "x-claude-code-session-id" in h:
        agent_id = h.get("x-claude-code-agent-id")
        request_class = h.get("x-claude-code-request-class")
        if request_class is None:
            # Without hint headers, the agent-id header alone tells main from subagent.
            request_class = "subagent" if agent_id else "main"
        return Attribution(
            session_id=h.get("x-claude-code-session-id"),
            agent_id=agent_id,
            parent_agent_id=h.get("x-claude-code-parent-agent-id"),
            request_class=request_class,
            agent_type=h.get("x-claude-code-agent-type"),
            prompt_id=h.get("x-claude-code-prompt-id"),
            prev_tool_durations_ms=parse_prev_tool_durations(
                h.get("x-claude-code-prev-tool-durations")
            ),
        )
    if "session-id" in h or "x-openai-subagent" in h or "thread-id" in h:
        label = h.get("x-openai-subagent")
        if label is None:
            request_class = "main"
        elif label == "collab_spawn":
            request_class = "subagent"
        elif label == "compact":
            request_class = "compaction"
        else:
            request_class = "auxiliary"
        return Attribution(
            session_id=h.get("session-id"),
            # TODO: assumes the main thread sends no x-openai-subagent header and
            # spawned threads are distinguished by thread-id. Unverified.
            agent_id=h.get("thread-id") if label else None,
            parent_agent_id=h.get("x-codex-parent-thread-id"),
            request_class=request_class,
            agent_type=label,
        )
    return Attribution()


# --- body analysis ---------------------------------------------------------

_CALL_TYPES = {"function_call", "custom_tool_call", "local_shell_call"}
_OUTPUT_TYPES = {"function_call_output", "custom_tool_call_output", "local_shell_call_output"}


@dataclass
class BodyShape:
    model: str | None
    stream: bool
    n_input_items: int
    prefix_hashes: list[str]
    prefix_chars: list[int]
    triggering_tools: list[str]


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def prefix_fingerprint(head: Any, items: list[Any]) -> tuple[list[str], list[int]]:
    """Cumulative hashes and character lengths of ``head`` followed by ``items``.

    Element ``k`` of each list describes the prefix made of the head (model,
    system prompt, tools) plus the first ``k`` input items. Two requests share
    a prefix of length ``k`` exactly when their ``k``-th hashes match.
    """
    hasher = hashlib.sha256()
    hashes: list[str] = []
    chars: list[int] = []
    total = 0
    for part in [head, *items]:
        text = _canonical(part)
        total += len(text)
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\x1e")
        hashes.append(hasher.copy().hexdigest()[:16])
        chars.append(total)
    return hashes, chars


def _anthropic_triggering_tools(messages: list[Any]) -> list[str]:
    if not messages:
        return []
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return []
    content = last.get("content")
    if not isinstance(content, list):
        return []
    ids = [
        b.get("tool_use_id")
        for b in content
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    if not ids:
        return []
    names: dict[str, str] = {}
    for msg in messages[:-1]:
        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
            for b in msg["content"]:
                if isinstance(b, dict) and b.get("type") in ("tool_use", "server_tool_use"):
                    names[b.get("id")] = b.get("name", "unknown")
    return [names.get(i, "unknown") for i in ids]


def _responses_triggering_tools(items: list[Any]) -> list[str]:
    trailing: list[str] = []
    for item in reversed(items):
        if isinstance(item, dict) and item.get("type") in _OUTPUT_TYPES:
            trailing.append(item.get("call_id"))
        else:
            break
    if not trailing:
        return []
    names = {
        it.get("call_id"): it.get("name") or it.get("type")
        for it in items
        if isinstance(it, dict) and it.get("type") in _CALL_TYPES
    }
    return [names.get(c, "unknown") for c in reversed(trailing)]


def analyze_body(provider: str, body: dict[str, Any]) -> BodyShape:
    model = body.get("model")
    stream = bool(body.get("stream", False))
    if provider == "anthropic":
        items = body.get("messages") or []
        head = {"model": model, "system": body.get("system"), "tools": body.get("tools")}
        triggering = _anthropic_triggering_tools(items)
    elif provider == "openai":
        raw = body.get("input")
        items = [{"role": "user", "content": raw}] if isinstance(raw, str) else (raw or [])
        head = {
            "model": model,
            "instructions": body.get("instructions"),
            "tools": body.get("tools"),
        }
        # TODO: when a request uses previous_response_id, ``input`` holds only the
        # new items and the prefix fingerprint covers the delta, not the full context.
        triggering = _responses_triggering_tools(items)
    else:
        raise ValueError(f"unknown provider: {provider!r}")
    hashes, chars = prefix_fingerprint(head, list(items))
    return BodyShape(
        model=model,
        stream=stream,
        n_input_items=len(items),
        prefix_hashes=hashes,
        prefix_chars=chars,
        triggering_tools=triggering,
    )
