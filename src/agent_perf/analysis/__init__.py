"""Trace analysis with pandas."""

from agent_perf.analysis.experiment import compare_variants, inference_load, pass_rate
from agent_perf.analysis.load import load_traces, records_to_frame
from agent_perf.analysis.workload import (
    concurrency_timeline,
    context_growth,
    context_growth_summary,
    fanout,
    latency_summary,
    peak_concurrency,
    prefix_reuse,
)

__all__ = [
    "compare_variants",
    "concurrency_timeline",
    "context_growth",
    "context_growth_summary",
    "fanout",
    "inference_load",
    "latency_summary",
    "load_traces",
    "pass_rate",
    "peak_concurrency",
    "prefix_reuse",
    "records_to_frame",
]
