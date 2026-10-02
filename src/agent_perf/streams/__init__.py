"""Incremental parsers for streamed model responses."""

from agent_perf.streams.anthropic import AnthropicStreamAccumulator
from agent_perf.streams.base import StreamAccumulator, StreamResult, Usage
from agent_perf.streams.openai import OpenAIResponsesStreamAccumulator
from agent_perf.streams.sse import SSEDecoder, SSEEvent


def accumulator_for(provider: str) -> StreamAccumulator:
    if provider == "anthropic":
        return AnthropicStreamAccumulator()
    if provider == "openai":
        return OpenAIResponsesStreamAccumulator()
    raise ValueError(f"unknown provider: {provider!r}")


__all__ = [
    "AnthropicStreamAccumulator",
    "OpenAIResponsesStreamAccumulator",
    "SSEDecoder",
    "SSEEvent",
    "StreamAccumulator",
    "StreamResult",
    "Usage",
    "accumulator_for",
]
