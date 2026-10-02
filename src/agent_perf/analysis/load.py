"""Load trace files into pandas DataFrames."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from agent_perf.metrics import record_latency
from agent_perf.schema import RequestRecord, read_records


def records_to_frame(records: Iterable[RequestRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = asdict(rec)
        lat = record_latency(rec)
        row.update(
            ttft_s=lat.ttft_s,
            decode_s=lat.decode_s,
            itl_s=lat.itl_s,
            decode_tok_per_s=lat.decode_tok_per_s,
            e2e_tok_per_s=lat.e2e_tok_per_s,
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["t_end"] = df["t_start"] + df["end_s"].fillna(0.0)
    df["is_subagent"] = df["request_class"].eq("subagent") | df["agent_id"].notna()
    # One "stream" is one agent's conversation: the main thread or one subagent.
    df["stream_id"] = (
        df["run_id"].astype(str)
        + "/"
        + df["session_id"].fillna("unknown-session").astype(str)
        + "/"
        + df["agent_id"].fillna("main").astype(str)
    )
    return df.sort_values("t_start", kind="stable").reset_index(drop=True)


def load_traces(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """Load one or more JSONL trace files (or directories of ``*.jsonl``)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files: list[Path] = []
    for p in map(Path, paths):
        files.extend(sorted(p.glob("*.jsonl")) if p.is_dir() else [p])
    records: list[RequestRecord] = []
    for f in files:
        records.extend(read_records(f))
    return records_to_frame(records)
