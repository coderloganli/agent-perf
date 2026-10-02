from __future__ import annotations

import json

import pandas as pd
import pytest
from conftest import make_record

from agent_perf.analysis import (
    compare_variants,
    concurrency_timeline,
    context_growth,
    context_growth_summary,
    fanout,
    latency_summary,
    load_traces,
    peak_concurrency,
    prefix_reuse,
    records_to_frame,
)
from agent_perf.attribution import prefix_fingerprint
from agent_perf.schema import write_records


def _session(run_id: str, t0: float = 0.0):
    """Synthetic session: a main thread with 3 turns, two overlapping subagents."""
    head = {"system": "sys"}
    msgs = ["u1", "a1", "u2", "a2", "u3"]
    recs = []
    for turn in range(3):
        hashes, chars = prefix_fingerprint(head, msgs[: 2 * turn + 1])
        recs.append(
            make_record(
                record_id=f"{run_id}-main-{turn}",
                run_id=run_id,
                session_id="s1",
                request_class="main",
                t_start=t0 + turn * 10.0,
                end_s=2.0,
                prompt_tokens=1000 + 500 * turn,
                cache_read_tokens=0 if turn == 0 else 1000 + 500 * (turn - 1),
                output_tokens=11,
                delta_offsets_s=[0.5, 1.5],
                prefix_hashes=hashes,
                prefix_chars=chars,
            )
        )
    for i, agent in enumerate(["a", "b"]):
        hashes, chars = prefix_fingerprint({"system": f"sub-{agent}"}, ["task"])
        recs.append(
            make_record(
                record_id=f"{run_id}-{agent}",
                run_id=run_id,
                session_id="s1",
                agent_id=agent,
                request_class="subagent",
                t_start=t0 + 12.0 + i * 0.5,
                end_s=3.0,
                prompt_tokens=400,
                cache_read_tokens=0,
                output_tokens=21,
                delta_offsets_s=[1.0, 3.0],
                prefix_hashes=hashes,
                prefix_chars=chars,
            )
        )
    return recs


@pytest.fixture
def df():
    return records_to_frame(_session("run"))


def test_records_to_frame_derives_latency_and_streams(df):
    main = df[df["record_id"] == "run-main-0"].iloc[0]
    assert main["ttft_s"] == pytest.approx(0.5)
    assert main["itl_s"] == pytest.approx(0.1)
    assert main["t_end"] == pytest.approx(2.0)
    assert set(df["stream_id"]) == {"run/s1/main", "run/s1/a", "run/s1/b"}
    assert df["is_subagent"].sum() == 2


def test_load_traces_roundtrip(tmp_path):
    path = tmp_path / "t.jsonl"
    write_records(path, _session("run"))
    loaded = load_traces(tmp_path)
    assert len(loaded) == 5
    assert loaded["t_start"].is_monotonic_increasing


def test_context_growth(df):
    growth = context_growth(df)
    main = growth[growth["stream_id"] == "run/s1/main"]
    assert main["turn"].tolist() == [0, 1, 2]
    summary = context_growth_summary(df).set_index("stream_id")
    assert summary.loc["run/s1/main", "tokens_per_turn"] == pytest.approx(500.0)
    assert summary.loc["run/s1/main", "max_prompt_tokens"] == 2000
    assert summary.loc["run/s1/a", "turns"] == 1


def test_concurrency_and_fanout(df):
    # main turn 1 runs 10-12; subagents run 12-15 and 12.5-15.5; ends sort before starts.
    assert peak_concurrency(df) == 2
    tl = concurrency_timeline(df)
    assert tl["in_flight"].iloc[-1] == 0
    fo = fanout(df).iloc[0]
    assert fo["subagents"] == 2
    assert fo["subagent_request_share"] == pytest.approx(2 / 5)
    assert fo["subagent_prompt_token_share"] == pytest.approx(800 / (1000 + 1500 + 2000 + 800))
    assert fo["peak_in_flight"] == 2


def test_prefix_reuse(df):
    reuse = prefix_reuse(df).set_index(df["record_id"])
    assert reuse.loc["run-main-0", "reused_items"] == -1
    assert reuse.loc["run-main-0", "reused_char_frac"] == 0.0
    # Turn 1 repeats head + first user message from turn 0.
    assert reuse.loc["run-main-1", "reused_items"] == 1
    assert 0 < reuse.loc["run-main-1", "reused_char_frac"] < 1
    assert reuse.loc["run-main-2", "reused_items"] == 3
    # Subagents have their own head: nothing reused.
    assert reuse.loc["run-a", "reused_items"] == -1
    assert reuse.loc["run-main-1", "cache_hit_frac"] == pytest.approx(1000 / 1500)


def test_latency_summary(df):
    s = latency_summary(df)
    assert s.loc["run", "requests"] == 5
    assert s.loc["run", "ttft_s_p50"] == pytest.approx(0.5)
    assert "itl_s_p99" in s.columns


def test_compare_variants():
    traces = records_to_frame(_session("exp-subagents_on") + _session("exp-subagents_off", 100))
    results = pd.DataFrame(
        {
            "run_id": ["exp-subagents_on"] * 2 + ["exp-subagents_off"] * 2,
            "passed": [True, False, True, True],
            "exception_type": [None, None, None, "TimeoutError"],
        }
    )
    out = compare_variants(traces, results)
    assert out.loc["exp-subagents_on", "pass_rate"] == pytest.approx(0.5)
    assert out.loc["exp-subagents_off", "pass_rate"] == pytest.approx(1.0)
    assert out.loc["exp-subagents_off", "errored"] == 1
    assert out.loc["exp-subagents_on", "requests_per_trial"] == pytest.approx(2.5)
    assert out.loc["exp-subagents_on", "prompt_tokens"] == 5300


def test_plots_render(df, tmp_path):
    from agent_perf.plotting import save_all

    paths = save_all(df, tmp_path)
    assert all(p.stat().st_size > 0 for p in paths)


def test_reader_ignores_unknown_fields(tmp_path):
    rec = json.loads(make_record().to_json())
    rec["future_field"] = 1
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(rec) + "\n\n", encoding="utf-8")
    assert len(load_traces(path)) == 1
