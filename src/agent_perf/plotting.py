"""Matplotlib figures for the workload analyses. Each function returns a Figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from agent_perf.analysis import (  # noqa: E402
    concurrency_timeline,
    context_growth,
    prefix_reuse,
)


def plot_context_growth(df: pd.DataFrame) -> Figure:
    growth = context_growth(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    for _stream_id, g in growth.groupby("stream_id"):
        sub = bool(g["is_subagent"].iloc[0])
        ax.plot(
            g["turn"],
            g["prompt_tokens"],
            color="tab:orange" if sub else "tab:blue",
            alpha=0.6,
            linewidth=1,
            label=None,
        )
    ax.plot([], [], color="tab:blue", label="main thread")
    ax.plot([], [], color="tab:orange", label="subagent")
    ax.set_xlabel("turn within agent stream")
    ax.set_ylabel("prompt tokens")
    ax.set_title("Context growth")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_concurrency(df: pd.DataFrame) -> Figure:
    tl = concurrency_timeline(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    for run_id, g in tl.groupby("run_id"):
        t0 = g["t"].min()
        ax.step(g["t"] - t0, g["in_flight"], where="post", label=str(run_id))
    ax.set_xlabel("seconds since first request")
    ax.set_ylabel("in-flight requests")
    ax.set_title("Request fan-out")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_prefix_reuse(df: pd.DataFrame) -> Figure:
    reuse = prefix_reuse(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    for run_id, g in reuse.groupby("run_id"):
        ax.hist(g["reused_char_frac"].dropna(), bins=20, range=(0, 1), alpha=0.5, label=str(run_id))
    ax.set_xlabel("fraction of request that repeats an earlier prefix")
    ax.set_ylabel("requests")
    ax.set_title("Prefix reuse")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_latency_cdf(df: pd.DataFrame, column: str = "ttft_s") -> Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    for run_id, g in df.groupby("run_id"):
        vals = g[column].dropna().sort_values()
        if vals.empty:
            continue
        ax.plot(
            vals.to_numpy(),
            (pd.RangeIndex(1, len(vals) + 1) / len(vals)).to_numpy(),
            label=str(run_id),
        )
    ax.set_xlabel(column)
    ax.set_ylabel("CDF")
    ax.set_title(f"{column} distribution")
    ax.legend()
    fig.tight_layout()
    return fig


PLOTS = {
    "context_growth": plot_context_growth,
    "concurrency": plot_concurrency,
    "prefix_reuse": plot_prefix_reuse,
    "ttft_cdf": lambda df: plot_latency_cdf(df, "ttft_s"),
    "itl_cdf": lambda df: plot_latency_cdf(df, "itl_s"),
}


def save_all(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in PLOTS.items():
        fig = fn(df)
        path = out / f"{name}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written
