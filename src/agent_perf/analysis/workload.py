"""Workload-shape characterization over a trace DataFrame.

All functions take the frame produced by :func:`agent_perf.analysis.load.load_traces`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LATENCY_COLUMNS = ["ttft_s", "itl_s", "decode_tok_per_s", "e2e_tok_per_s"]


# --- context growth --------------------------------------------------------


def context_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Prompt size per turn for every agent stream (main thread or subagent)."""
    cols = ["run_id", "stream_id", "is_subagent", "t_start", "prompt_tokens", "n_input_items"]
    out = df[cols].sort_values(["stream_id", "t_start"], kind="stable").copy()
    out["turn"] = out.groupby("stream_id").cumcount()
    return out.reset_index(drop=True)


def context_growth_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per stream: turns, first/max prompt size and least-squares growth per turn."""
    growth = context_growth(df)

    def _summ(g: pd.DataFrame) -> pd.Series:
        tok = g["prompt_tokens"].astype(float)
        valid = tok.notna()
        slope = np.nan
        if valid.sum() >= 2:
            slope = float(np.polyfit(g.loc[valid, "turn"], tok[valid], 1)[0])
        return pd.Series(
            {
                "run_id": g["run_id"].iloc[0],
                "is_subagent": bool(g["is_subagent"].iloc[0]),
                "turns": len(g),
                "first_prompt_tokens": tok.iloc[0],
                "max_prompt_tokens": tok.max(),
                "tokens_per_turn": slope,
            }
        )

    return growth.groupby("stream_id").apply(_summ, include_groups=False).reset_index()


# --- fan-out ---------------------------------------------------------------


def concurrency_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Step function of in-flight requests over time, per run.

    Returns rows ``(run_id, t, in_flight)`` where ``in_flight`` holds from ``t``
    until the next row of the same run. Ends are processed before starts at
    the same timestamp.
    """
    frames = []
    for run_id, g in df.groupby("run_id"):
        events = pd.concat(
            [
                pd.DataFrame({"t": g["t_start"], "delta": 1}),
                pd.DataFrame({"t": g["t_end"], "delta": -1}),
            ]
        ).sort_values(["t", "delta"], kind="stable")
        events["in_flight"] = events["delta"].cumsum()
        events["run_id"] = run_id
        frames.append(events[["run_id", "t", "in_flight"]])
    if not frames:
        return pd.DataFrame(columns=["run_id", "t", "in_flight"])
    return pd.concat(frames).reset_index(drop=True)


def peak_concurrency(df: pd.DataFrame) -> int:
    tl = concurrency_timeline(df)
    return int(tl["in_flight"].max()) if not tl.empty else 0


def fanout(df: pd.DataFrame) -> pd.DataFrame:
    """Per session: subagents spawned, their share of requests, peak concurrency."""
    rows = []
    for (run_id, session_id), g in df.groupby(["run_id", "session_id"], dropna=False):
        rows.append(
            {
                "run_id": run_id,
                "session_id": session_id,
                "requests": len(g),
                "subagents": g.loc[g["is_subagent"], "agent_id"].nunique(),
                "subagent_request_share": float(g["is_subagent"].mean()),
                "subagent_prompt_token_share": _share(g, "prompt_tokens"),
                "peak_in_flight": peak_concurrency(g),
            }
        )
    return pd.DataFrame(rows)


def _share(g: pd.DataFrame, col: str) -> float:
    total = g[col].sum()
    return float(g.loc[g["is_subagent"], col].sum() / total) if total else float("nan")


# --- prefix reuse ----------------------------------------------------------


def prefix_reuse(df: pd.DataFrame) -> pd.DataFrame:
    """How much of each prompt repeats a prefix already sent in the same run.

    ``reused_items`` is the longest prefix (head + k input items) whose
    fingerprint appeared in any earlier request of the run, and
    ``reused_char_frac`` its share of the serialized request. This is a
    client-side upper bound on what a prefix cache could serve;
    ``cache_hit_frac`` is what the provider actually reported as cache reads.
    """
    out = []
    for _run_id, g in df.sort_values("t_start", kind="stable").groupby("run_id", sort=False):
        seen: set[str] = set()
        for idx, row in g.iterrows():
            hashes = list(row["prefix_hashes"] or [])
            chars = list(row["prefix_chars"] or [])
            k = -1
            for i, h in enumerate(hashes):
                if h in seen:
                    k = i
                else:
                    break
            seen.update(hashes)
            frac = chars[k] / chars[-1] if k >= 0 and chars and chars[-1] else 0.0
            out.append(
                {
                    "index": idx,
                    "reused_items": k,  # -1: not even the head (system + tools) was seen
                    "reused_char_frac": frac,
                }
            )
    reuse = pd.DataFrame(out).set_index("index")
    res = df[["run_id", "stream_id", "is_subagent", "t_start"]].join(reuse)
    prompt = df["prompt_tokens"].astype(float)
    res["cache_hit_frac"] = df["cache_read_tokens"].astype(float) / prompt.where(prompt > 0)
    return res


# --- latency ---------------------------------------------------------------


def latency_summary(
    df: pd.DataFrame, by: str | list[str] = "run_id", quantiles=(0.5, 0.9, 0.99)
) -> pd.DataFrame:
    """Quantiles of TTFT, ITL and throughput per group."""
    q = df.groupby(by)[LATENCY_COLUMNS].quantile(list(quantiles)).unstack()
    q.columns = [f"{col}_p{int(round(p * 100))}" for col, p in q.columns]
    q.insert(0, "requests", df.groupby(by).size())
    return q
