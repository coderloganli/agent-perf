"""Trace schema: one record per model request issued by a harness.

Records are written as JSON Lines by the capture proxy and read back by
``agent_perf.analysis.load``. Token fields are normalized across providers:

* ``prompt_tokens`` is the total number of input tokens the model processed,
  including tokens served from or written to the prompt cache.
* ``cache_read_tokens`` / ``cache_write_tokens`` are the cached subsets.

Anthropic reports ``input_tokens`` excluding cache reads and writes
(total = input_tokens + cache_creation_input_tokens + cache_read_input_tokens),
while the OpenAI Responses API reports ``input_tokens`` inclusive of
``input_tokens_details.cached_tokens``. The stream parsers perform that
normalization.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class RequestRecord:
    # Identity
    record_id: str
    run_id: str
    provider: str  # "anthropic" | "openai"
    method: str
    path: str
    status_code: int | None = None
    model: str | None = None
    stream: bool = False

    # Origin attribution (see agent_perf.attribution for where each comes from)
    session_id: str | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    request_class: str | None = None
    agent_type: str | None = None
    prompt_id: str | None = None
    triggering_tools: list[str] = field(default_factory=list)
    prev_tool_durations_ms: dict[str, int] = field(default_factory=dict)

    # Token usage (normalized, see module docstring)
    prompt_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None

    # Timing: wall-clock epoch seconds for the request start, offsets in seconds
    t_start: float = 0.0
    first_byte_s: float | None = None
    first_token_s: float | None = None
    end_s: float | None = None
    # Arrival offsets of every output-content delta event (text, tool input,
    # reasoning). Deltas may carry several tokens each.
    delta_offsets_s: list[float] = field(default_factory=list)

    # Request shape, used for prefix-reuse analysis without storing prompts.
    n_input_items: int = 0
    prefix_hashes: list[str] = field(default_factory=list)
    prefix_chars: list[int] = field(default_factory=list)

    error: str | None = None
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequestRecord:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def write_records(path: str | Path, records: Iterable[RequestRecord]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")


def read_records(path: str | Path) -> Iterator[RequestRecord]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield RequestRecord.from_dict(json.loads(line))
