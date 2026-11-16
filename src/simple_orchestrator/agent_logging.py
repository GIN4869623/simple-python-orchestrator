"""Append complete Knowledge/DeviceManager messages without truncating history."""

from __future__ import annotations

import base64
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from .models import AgentResult


_WRITE_LOCK = Lock()
_LOGGED_AGENTS = {"knowledge", "device_manager"}


@dataclass(frozen=True)
class AgentMessageLog:
    path: str | Path

    def record(
        self, agent_id: str, event: str, metadata: Mapping[str, Any], *,
        payload: Any = None, body: bytes | None = None, error: str | None = None,
    ) -> None:
        if agent_id not in _LOGGED_AGENTS:
            return
        entry = {
            **metadata,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "event": event,
        }
        if event in {"request", "response"}:
            entry["payload"] = payload
        if body is not None:
            try:
                entry["raw_body"] = body.decode("utf-8")
            except UnicodeDecodeError:
                entry["raw_body_base64"] = base64.b64encode(body).decode("ascii")
            try:
                entry["payload"] = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                pass
        if error is not None:
            entry["error"] = error
        try:
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            with _WRITE_LOCK:
                path = Path(self.path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(line)
        except (OSError, ValueError, TypeError) as exc:
            warnings.warn(f"无法写入智能体报文日志 {self.path}：{exc}", RuntimeWarning)


class LoggedFakeAgent:
    """Record actual local Port calls, clearly distinguished from network traffic."""

    def __init__(self, agent: Any, agent_id: str, log: AgentMessageLog) -> None:
        self.agent = agent
        self.agent_id = agent_id
        self.log = log

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)

    def invoke(
        self, request: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None,
    ) -> AgentResult:
        log_metadata = {**(metadata or {}), "transport": "fake"}
        self.log.record(self.agent_id, "request", log_metadata, payload=request)
        try:
            result = self.agent.invoke(request, metadata=metadata)
        except Exception as exc:
            self.log.record(self.agent_id, "error", log_metadata, error=f"{type(exc).__name__}: {exc}")
            raise
        self.log.record(self.agent_id, "response", log_metadata, payload=asdict(result))
        return result
