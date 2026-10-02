"""Per-request latency metrics derived from a recorded stream.

Definitions (all offsets are seconds since the proxy forwarded the request):

* TTFT: arrival of the first output-content delta event (text, tool-call
  arguments or reasoning). Keep-alive pings and metadata events do not count.
* Decode time: last delta arrival minus first delta arrival.
* Inter-token latency (ITL): decode time divided by ``output_tokens - 1``.
  Providers batch several tokens into one delta event, so per-token gaps are
  not observable on the wire; this is the mean ITL implied by the usage
  count. The raw inter-chunk gaps are exposed separately.
* Output throughput: ``(output_tokens - 1) / decode time`` (tokens/s during
  decode), and end-to-end ``output_tokens / total time``.

Reasoning tokens that are generated but not streamed (for example encrypted
reasoning) are included in ``output_tokens`` but produce no deltas, which
inflates ITL for such requests. ITL is therefore reported alongside
``reasoning_tokens`` so the analysis can filter or split on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent_perf.schema import RequestRecord


@dataclass(frozen=True)
class LatencyMetrics:
    ttft_s: float | None
    decode_s: float | None
    itl_s: float | None
    decode_tok_per_s: float | None
    e2e_tok_per_s: float | None
    total_s: float | None


def inter_chunk_gaps(offsets: list[float]) -> np.ndarray:
    if len(offsets) < 2:
        return np.empty(0)
    return np.diff(np.asarray(offsets, dtype=float))


def latency_metrics(
    delta_offsets_s: list[float],
    output_tokens: int | None,
    end_s: float | None,
) -> LatencyMetrics:
    ttft = delta_offsets_s[0] if delta_offsets_s else None
    decode = delta_offsets_s[-1] - delta_offsets_s[0] if len(delta_offsets_s) >= 2 else None
    itl = None
    decode_tps = None
    if decode is not None and decode > 0 and output_tokens is not None and output_tokens > 1:
        itl = decode / (output_tokens - 1)
        decode_tps = (output_tokens - 1) / decode
    e2e = None
    if end_s is not None and end_s > 0 and output_tokens is not None:
        e2e = output_tokens / end_s
    return LatencyMetrics(ttft, decode, itl, decode_tps, e2e, end_s)


def record_latency(rec: RequestRecord) -> LatencyMetrics:
    return latency_metrics(rec.delta_offsets_s, rec.output_tokens, rec.end_s)
