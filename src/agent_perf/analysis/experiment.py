"""Compare inference load against task pass rate across experiment variants."""

from __future__ import annotations

import pandas as pd

from agent_perf.analysis.workload import peak_concurrency


def inference_load(traces: pd.DataFrame) -> pd.DataFrame:
    """Aggregate inference load per run (one run per variant)."""
    rows = []
    for run_id, g in traces.groupby("run_id"):
        prompt = g["prompt_tokens"].sum()
        rows.append(
            {
                "run_id": run_id,
                "requests": len(g),
                "sessions": g["session_id"].nunique(),
                "subagent_request_share": float(g["is_subagent"].mean()),
                "prompt_tokens": prompt,
                "cache_read_tokens": g["cache_read_tokens"].sum(),
                "output_tokens": g["output_tokens"].sum(),
                "cache_read_share": (g["cache_read_tokens"].sum() / prompt) if prompt else None,
                "peak_in_flight": peak_concurrency(g),
            }
        )
    return pd.DataFrame(rows).set_index("run_id") if rows else pd.DataFrame()


def pass_rate(results: pd.DataFrame) -> pd.DataFrame:
    g = results.groupby("run_id")
    return pd.DataFrame(
        {
            "trials": g.size(),
            "passed": g["passed"].sum(),
            "pass_rate": g["passed"].mean(),
            "errored": g["exception_type"].count(),
        }
    )


def compare_variants(traces: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Join per-run inference load with per-run pass rate.

    Adds per-trial normalizations so variants with different trial counts
    remain comparable.
    """
    load = inference_load(traces)
    rate = pass_rate(results)
    out = load.join(rate, how="outer")
    for col in ("requests", "prompt_tokens", "output_tokens"):
        out[f"{col}_per_trial"] = out[col] / out["trials"]
    return out
