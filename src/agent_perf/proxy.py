"""Recording reverse proxy for model APIs.

The harness is pointed at the proxy instead of the provider:

* Claude Code: ``ANTHROPIC_BASE_URL=http://<host>:<port>`` (Anthropic Messages
  format; inference posts to ``/v1/messages?beta=true``).
* Codex: ``OPENAI_BASE_URL=http://<host>:<port>/v1`` (Responses API; inference
  posts to ``/v1/responses``). Harbor writes this into Codex's
  ``config.toml`` as ``openai_base_url``.

Every request is forwarded byte-for-byte to the upstream, and responses are
relayed chunk by chunk as they arrive (the Claude Code gateway guide requires
streaming without buffering, including ``ping`` events). Inference requests
additionally produce one :class:`~agent_perf.schema.RequestRecord` each.

An optional ``/r/<run_id>/`` path prefix tags requests with a run identifier
and is stripped before forwarding, so one proxy can serve several runs.

TODO: Codex can use a Responses API WebSocket transport for providers that
advertise ``supports_websockets``. This proxy only handles HTTP; confirm that
Codex falls back to HTTP streaming when pointed at a custom base URL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from aiohttp import web

from agent_perf.attribution import analyze_body, attribution_from_headers
from agent_perf.schema import RequestRecord
from agent_perf.streams import accumulator_for

log = logging.getLogger(__name__)

DEFAULT_UPSTREAMS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
}

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
}
STRIP_RESPONSE = HOP_BY_HOP | {"content-encoding"}


def is_inference_path(provider: str, path: str) -> bool:
    if provider == "anthropic":
        return path.rstrip("/").endswith("/messages")
    if provider == "openai":
        return path.rstrip("/").endswith("/responses")
    return False


def split_run_prefix(path: str, default_run: str) -> tuple[str, str]:
    """``/r/<run>/v1/messages`` -> (``<run>``, ``/v1/messages``)."""
    if path.startswith("/r/"):
        rest = path[3:]
        run, sep, tail = rest.partition("/")
        if run:
            return run, "/" + tail if sep else "/"
    return default_run, path


class TraceWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def write(self, record: RequestRecord) -> None:
        line = record.to_json() + "\n"
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)


@dataclass
class ProxyConfig:
    provider: str
    upstream: str
    trace_path: Path
    default_run: str = "default"


class RecordingProxy:
    def __init__(self, config: ProxyConfig) -> None:
        if config.provider not in DEFAULT_UPSTREAMS:
            raise ValueError(f"unknown provider: {config.provider!r}")
        self.config = config
        self.writer = TraceWriter(config.trace_path)
        self._session: aiohttp.ClientSession | None = None

    def build_app(self) -> web.Application:
        app = web.Application(client_max_size=1024**3)
        app.router.add_route("*", "/{tail:.*}", self.handle)
        app.on_startup.append(self._start)
        app.on_cleanup.append(self._stop)
        return app

    async def _start(self, _app: web.Application) -> None:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
        self._session = aiohttp.ClientSession(timeout=timeout, auto_decompress=False)

    async def _stop(self, _app: web.Application) -> None:
        if self._session is not None:
            await self._session.close()

    async def handle(self, request: web.Request) -> web.StreamResponse:
        assert self._session is not None
        run_id, path = split_run_prefix(request.path, self.config.default_run)
        url = self.config.upstream.rstrip("/") + path
        if request.query_string:
            url += "?" + request.query_string
        body = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        # Ask for an uncompressed body so the stream can be parsed while relayed.
        headers["Accept-Encoding"] = "identity"

        record: RequestRecord | None = None
        if request.method == "POST" and is_inference_path(self.config.provider, path):
            record = self._new_record(run_id, request, path, body)

        t0 = time.perf_counter()
        if record is not None:
            record.t_start = time.time()
        try:
            upstream = await self._session.request(
                request.method, url, headers=headers, data=body or None, allow_redirects=False
            )
        except aiohttp.ClientError as exc:
            if record is not None:
                record.error = f"upstream_error: {exc}"
                record.end_s = time.perf_counter() - t0
                await self.writer.write(record)
            return web.Response(status=502, text=f"agent-perf proxy: upstream error: {exc}")

        async with upstream:
            resp = web.StreamResponse(status=upstream.status, reason=upstream.reason)
            for k, v in upstream.headers.items():
                if k.lower() not in STRIP_RESPONSE:
                    resp.headers.add(k, v)
            await resp.prepare(request)

            is_sse = upstream.headers.get("content-type", "").startswith("text/event-stream")
            acc = accumulator_for(self.config.provider) if record is not None else None
            buffered = bytearray()
            try:
                async for chunk in upstream.content.iter_any():
                    offset = time.perf_counter() - t0
                    if record is not None and record.first_byte_s is None:
                        record.first_byte_s = offset
                    if acc is not None:
                        if is_sse:
                            acc.feed(chunk, offset)
                        else:
                            buffered.extend(chunk)
                    await resp.write(chunk)
                await resp.write_eof()
            except (ConnectionResetError, aiohttp.ClientError, asyncio.CancelledError) as exc:
                if record is not None:
                    record.error = f"stream_interrupted: {type(exc).__name__}"
                if isinstance(exc, asyncio.CancelledError):
                    if record is not None:
                        await self._finish(record, acc, is_sse, buffered, upstream.status, t0)
                    raise
            if record is not None:
                await self._finish(record, acc, is_sse, buffered, upstream.status, t0)
            return resp

    def _new_record(
        self, run_id: str, request: web.Request, path: str, body: bytes
    ) -> RequestRecord:
        attr = attribution_from_headers(request.headers)
        record = RequestRecord(
            record_id=uuid.uuid4().hex,
            run_id=run_id,
            provider=self.config.provider,
            method=request.method,
            path=path,
            session_id=attr.session_id,
            agent_id=attr.agent_id,
            parent_agent_id=attr.parent_agent_id,
            request_class=attr.request_class,
            agent_type=attr.agent_type,
            prompt_id=attr.prompt_id,
            prev_tool_durations_ms=attr.prev_tool_durations_ms,
        )
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            shape = analyze_body(self.config.provider, payload)
            record.model = shape.model
            record.stream = shape.stream
            record.n_input_items = shape.n_input_items
            record.prefix_hashes = shape.prefix_hashes
            record.prefix_chars = shape.prefix_chars
            record.triggering_tools = shape.triggering_tools
        return record

    async def _finish(self, record, acc, is_sse, buffered, status, t0) -> None:
        record.status_code = status
        record.end_s = time.perf_counter() - t0
        if acc is not None:
            if is_sse:
                result = acc.finish(record.end_s)
            else:
                try:
                    payload = json.loads(bytes(buffered)) if buffered else {}
                except json.JSONDecodeError:
                    payload = {}
                result = acc.parse_body(payload if isinstance(payload, dict) else {})
            u = result.usage
            record.model = result.model or record.model
            record.prompt_tokens = u.prompt_tokens
            record.cache_read_tokens = u.cache_read_tokens
            record.cache_write_tokens = u.cache_write_tokens
            record.output_tokens = u.output_tokens
            record.reasoning_tokens = u.reasoning_tokens
            record.delta_offsets_s = [round(x, 6) for x in result.delta_offsets_s]
            if result.delta_offsets_s:
                record.first_token_s = result.delta_offsets_s[0]
            record.error = record.error or result.error
        await self.writer.write(record)


async def serve(config: ProxyConfig, host: str, port: int) -> web.AppRunner:
    """Start the proxy in the running event loop; caller must ``cleanup()``."""
    runner = web.AppRunner(RecordingProxy(config).build_app())
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    log.info("recording %s proxy on %s:%d -> %s", config.provider, host, port, config.upstream)
    return runner
