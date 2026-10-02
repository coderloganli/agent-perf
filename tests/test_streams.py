from __future__ import annotations

import pytest
from conftest import chunks

from agent_perf.streams import (
    AnthropicStreamAccumulator,
    OpenAIResponsesStreamAccumulator,
    SSEDecoder,
)


def test_sse_decoder_handles_split_lines_comments_and_multiline_data():
    raw = ': keep-alive\r\nevent: a\r\ndata: one\r\ndata: two\r\n\r\ndata: {"x": "\xc3\xa9"}\n\n'
    dec = SSEDecoder()
    events = []
    for c in chunks(raw.encode("latin-1"), 3):
        events.extend(dec.feed(c))
    events.extend(dec.flush())
    assert [(e.event, e.data) for e in events] == [("a", "one\ntwo"), (None, '{"x": "é"}')]


def test_sse_decoder_flushes_unterminated_event():
    dec = SSEDecoder()
    assert dec.feed(b"event: x\ndata: 1") == []
    assert [(e.event, e.data) for e in dec.flush()] == [("x", "1")]


@pytest.mark.parametrize("chunk_size", [1, 7, 64, 100_000])
def test_anthropic_stream_usage_and_deltas(anthropic_sse, chunk_size):
    acc = AnthropicStreamAccumulator()
    for i, c in enumerate(chunks(anthropic_sse, chunk_size)):
        acc.feed(c, offset_s=float(i))
    r = acc.finish(offset_s=1e6)
    assert r.model == "test-model"
    # input_tokens excludes cache reads/writes; normalized prompt_tokens includes them.
    assert r.usage.prompt_tokens == 50 + 200 + 1000
    assert r.usage.cache_read_tokens == 1000
    assert r.usage.cache_write_tokens == 200
    # message_delta usage is cumulative and overrides message_start.
    assert r.usage.output_tokens == 11
    # two text deltas + one tool-input delta; ping and block start/stop ignored
    assert len(r.delta_offsets_s) == 3
    assert r.delta_offsets_s == sorted(r.delta_offsets_s)
    assert r.error is None


def test_anthropic_stream_error_event():
    acc = AnthropicStreamAccumulator()
    acc.feed(
        b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error",'
        b'"message":"Overloaded"}}\n\n',
        0.1,
    )
    assert acc.finish(0.2).error == "overloaded_error: Overloaded"


def test_anthropic_non_streamed_body():
    body = {
        "type": "message",
        "model": "m",
        "usage": {"input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 3},
    }
    r = AnthropicStreamAccumulator().parse_body(body)
    assert (r.usage.prompt_tokens, r.usage.cache_read_tokens, r.usage.output_tokens) == (15, 5, 3)


@pytest.mark.parametrize("chunk_size", [1, 13, 100_000])
def test_openai_responses_stream_usage_and_deltas(openai_sse, chunk_size):
    acc = OpenAIResponsesStreamAccumulator()
    for i, c in enumerate(chunks(openai_sse, chunk_size)):
        acc.feed(c, offset_s=float(i))
    r = acc.finish(1e6)
    assert r.model == "test-model"
    # Responses API input_tokens already includes cached tokens.
    assert r.usage.prompt_tokens == 1500
    assert r.usage.cache_read_tokens == 1200
    assert r.usage.cache_write_tokens is None
    assert r.usage.output_tokens == 21
    assert r.usage.reasoning_tokens == 5
    assert len(r.delta_offsets_s) == 3


def test_openai_failed_response_records_error():
    acc = OpenAIResponsesStreamAccumulator()
    acc.feed(
        b'event: response.failed\ndata: {"type":"response.failed","response":'
        b'{"status":"failed","error":{"code":"server_error","message":"boom"}}}\n\n',
        0.5,
    )
    assert acc.finish(0.6).error == "server_error: boom"
