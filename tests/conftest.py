from __future__ import annotations

from pathlib import Path

import pytest

from agent_perf.schema import RequestRecord

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def anthropic_sse() -> bytes:
    return (FIXTURES / "anthropic_stream.sse").read_bytes()


@pytest.fixture
def openai_sse() -> bytes:
    return (FIXTURES / "openai_responses_stream.sse").read_bytes()


def make_record(**kw) -> RequestRecord:
    base = dict(
        record_id=kw.pop("record_id", "r"),
        run_id=kw.pop("run_id", "run"),
        provider=kw.pop("provider", "anthropic"),
        method="POST",
        path="/v1/messages",
    )
    base.update(kw)
    return RequestRecord(**base)


def chunks(data: bytes, size: int) -> list[bytes]:
    return [data[i : i + size] for i in range(0, len(data), size)]
