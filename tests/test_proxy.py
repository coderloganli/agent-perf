"""End-to-end proxy tests against a local fake upstream (no real API calls)."""

from __future__ import annotations

import asyncio
import json

from aiohttp import web
from conftest import chunks

from agent_perf.proxy import ProxyConfig, RecordingProxy, split_run_prefix
from agent_perf.schema import read_records


def _fake_upstream(sse: bytes, content_type: str = "text/event-stream") -> web.Application:
    seen: dict = {}

    async def stream(request: web.Request) -> web.StreamResponse:
        seen["path"] = request.path_qs
        seen["headers"] = dict(request.headers)
        seen["body"] = await request.read()
        resp = web.StreamResponse(headers={"Content-Type": content_type, "request-id": "req_1"})
        await resp.prepare(request)
        for c in chunks(sse, 200):
            await resp.write(c)
            await asyncio.sleep(0.005)
        await resp.write_eof()
        return resp

    async def hello(request: web.Request) -> web.Response:
        return web.Response(text="hi")

    app = web.Application()
    app.router.add_post("/v1/messages", stream)
    app.router.add_post("/v1/responses", stream)
    app.router.add_get("/api/hello", hello)
    app["seen"] = seen
    return app


async def _proxy_client(aiohttp_client, tmp_path, provider, sse, content_type="text/event-stream"):
    upstream_app = _fake_upstream(sse, content_type)
    upstream = await aiohttp_client(upstream_app)
    trace = tmp_path / "trace.jsonl"
    proxy = RecordingProxy(
        ProxyConfig(provider=provider, upstream=str(upstream.make_url("")), trace_path=trace)
    )
    client = await aiohttp_client(proxy.build_app())
    return client, upstream_app["seen"], trace


def test_split_run_prefix():
    assert split_run_prefix("/r/exp-on/v1/messages", "d") == ("exp-on", "/v1/messages")
    assert split_run_prefix("/v1/messages", "d") == ("d", "/v1/messages")
    assert split_run_prefix("/r/x", "d") == ("x", "/")


async def test_anthropic_stream_is_relayed_and_recorded(aiohttp_client, tmp_path, anthropic_sse):
    client, seen, trace = await _proxy_client(aiohttp_client, tmp_path, "anthropic", anthropic_sse)
    body = {
        "model": "test-model",
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    resp = await client.post(
        "/r/exp-subagents_on/v1/messages?beta=true",
        data=json.dumps(body),
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "some-beta",
            "x-claude-code-session-id": "sess",
            "x-claude-code-agent-id": "agent-1",
        },
    )
    relayed = await resp.read()
    assert resp.status == 200
    assert relayed == anthropic_sse
    assert resp.headers["request-id"] == "req_1"

    # Forwarded unchanged: path (minus run prefix), query, required headers, body.
    assert seen["path"] == "/v1/messages?beta=true"
    assert seen["headers"]["anthropic-beta"] == "some-beta"
    assert seen["headers"]["x-api-key"] == "test-key"
    assert json.loads(seen["body"]) == body

    [rec] = list(read_records(trace))
    assert rec.run_id == "exp-subagents_on"
    assert rec.provider == "anthropic" and rec.status_code == 200 and rec.stream
    assert (rec.session_id, rec.agent_id, rec.request_class) == ("sess", "agent-1", "subagent")
    assert rec.prompt_tokens == 1250 and rec.cache_read_tokens == 1000
    assert rec.output_tokens == 11
    assert len(rec.delta_offsets_s) == 3
    assert 0 < rec.first_byte_s <= rec.first_token_s <= rec.end_s
    assert rec.n_input_items == 1 and len(rec.prefix_hashes) == 2


async def test_openai_responses_recorded(aiohttp_client, tmp_path, openai_sse):
    client, _, trace = await _proxy_client(aiohttp_client, tmp_path, "openai", openai_sse)
    resp = await client.post(
        "/v1/responses",
        data=json.dumps({"model": "m", "stream": True, "input": "hi"}),
        headers={"session-id": "cs", "authorization": "Bearer test"},
    )
    assert await resp.read() == openai_sse
    [rec] = list(read_records(trace))
    assert rec.run_id == "default"
    assert rec.session_id == "cs" and rec.request_class == "main"
    assert (rec.prompt_tokens, rec.cache_read_tokens, rec.output_tokens) == (1500, 1200, 21)


async def test_non_streamed_json_response(aiohttp_client, tmp_path):
    payload = json.dumps(
        {"type": "message", "model": "m", "usage": {"input_tokens": 7, "output_tokens": 2}}
    ).encode()
    client, _, trace = await _proxy_client(
        aiohttp_client, tmp_path, "anthropic", payload, "application/json"
    )
    resp = await client.post("/v1/messages", data=json.dumps({"model": "m", "messages": []}))
    assert await resp.read() == payload
    [rec] = list(read_records(trace))
    assert (rec.prompt_tokens, rec.output_tokens) == (7, 2)
    assert rec.first_token_s is None


async def test_non_inference_paths_pass_through_unrecorded(aiohttp_client, tmp_path):
    client, _, trace = await _proxy_client(aiohttp_client, tmp_path, "anthropic", b"")
    resp = await client.get("/api/hello")
    assert await resp.text() == "hi"
    assert not trace.exists()
