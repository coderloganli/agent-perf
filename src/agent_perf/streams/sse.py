"""Incremental server-sent events decoder.

Bytes are fed as they arrive from the network; complete events are returned
as soon as their terminating blank line is seen, so callers can timestamp
them at arrival.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass


@dataclass(frozen=True)
class SSEEvent:
    event: str | None
    data: str


class SSEDecoder:
    def __init__(self) -> None:
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buf = ""
        self._event: str | None = None
        self._data: list[str] = []

    def feed(self, chunk: bytes | str) -> list[SSEEvent]:
        if isinstance(chunk, bytes):
            chunk = self._utf8.decode(chunk)
        self._buf += chunk.replace("\r\n", "\n").replace("\r", "\n")
        events: list[SSEEvent] = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            ev = self._line(line)
            if ev is not None:
                events.append(ev)
        return events

    def flush(self) -> list[SSEEvent]:
        events: list[SSEEvent] = []
        self._buf += self._utf8.decode(b"", final=True)
        if self._buf:
            ev = self._line(self._buf)
            self._buf = ""
            if ev is not None:
                events.append(ev)
        ev = self._line("")
        if ev is not None:
            events.append(ev)
        return events

    def _line(self, line: str) -> SSEEvent | None:
        if line == "":
            if not self._data and self._event is None:
                return None
            ev = SSEEvent(self._event, "\n".join(self._data))
            self._event, self._data = None, []
            return ev
        if line.startswith(":"):
            return None
        name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if name == "event":
            self._event = value
        elif name == "data":
            self._data.append(value)
        return None
