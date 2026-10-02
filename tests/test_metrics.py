from __future__ import annotations

import numpy as np
import pytest

from agent_perf.metrics import inter_chunk_gaps, latency_metrics


def test_latency_metrics_basic():
    m = latency_metrics([0.5, 0.7, 1.5], output_tokens=11, end_s=2.0)
    assert m.ttft_s == pytest.approx(0.5)
    assert m.decode_s == pytest.approx(1.0)
    assert m.itl_s == pytest.approx(0.1)  # 1.0 s over 10 gaps
    assert m.decode_tok_per_s == pytest.approx(10.0)
    assert m.e2e_tok_per_s == pytest.approx(5.5)
    assert m.total_s == 2.0


def test_latency_metrics_single_delta_has_ttft_only():
    m = latency_metrics([0.3], output_tokens=1, end_s=0.4)
    assert m.ttft_s == pytest.approx(0.3)
    assert m.decode_s is None and m.itl_s is None and m.decode_tok_per_s is None


def test_latency_metrics_no_stream():
    m = latency_metrics([], output_tokens=None, end_s=None)
    assert m == latency_metrics([], None, None)
    assert m.ttft_s is None and m.e2e_tok_per_s is None


def test_inter_chunk_gaps():
    np.testing.assert_allclose(inter_chunk_gaps([0.1, 0.3, 0.6]), [0.2, 0.3])
    assert inter_chunk_gaps([1.0]).size == 0
