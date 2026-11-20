"""Read-only network supervision via the agent's HTTP chat endpoint."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from http.client import HTTPException
from typing import Any, Iterator, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from .agent_logging import AgentMessageLog
from .models import AgentResult, ResultStatus
from .remote import RemoteCallError

_MAX_REPLY_BYTES = 4 * 1024 * 1024
_MAX_LINE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class NetworkChatAdapter:
    endpoint: str
    timeout: float = 120.0
    sender: str = "编排智能体"
    message_log: AgentMessageLog | None = None

    def __post_init__(self) -> None:
        url = urlsplit(self.endpoint)
        if (url.scheme not in {"http", "https"} or not url.hostname
                or url.username or url.password or url.fragment
                or not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("网络监管接口地址或超时配置无效")

    def invoke(
        self, request: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None,
    ) -> AgentResult:
        message = request.get("message")
        if not isinstance(message, str) or not message.strip():
            return AgentResult(ResultStatus.REJECTED, "网络监管查询缺少内容", error="empty_network_query")
        if (metadata or {}).get("capability") not in (None, "", "device.list", "device.inspect"):
            return AgentResult(ResultStatus.REJECTED, "网络监管接口只接收只读查询", error="invalid_network_capability")

        if urlsplit(self.endpoint).path.rstrip("/") == "/api/agent/messages":
            payload = {"message": message, "agentId": "query", "sessionId": f"network-{uuid4().hex}"}
        else:
            payload = {"message": message, "stream": True}
        context = {**(metadata or {}), "sender": self.sender, "transport": "http_chat", "endpoint": self.endpoint}
        if self.message_log:
            self.message_log.record("device_manager", "request", context, payload=payload)
        req = Request(
            self.endpoint, method="POST", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream, application/json, application/x-ndjson"},
        )
        try:
            deadline = time.monotonic() + self.timeout
            opener = build_opener(ProxyHandler({}))
            with opener.open(req, timeout=self.timeout) as response:
                media_type = response.headers.get_content_type()
                if media_type == "application/json":
                    body = response.read(_MAX_REPLY_BYTES + 1)
                    if len(body) > _MAX_REPLY_BYTES:
                        raise RemoteCallError("回复超过接收限制")
                    reply = _network_result(json.loads(body))
                elif media_type in {"text/event-stream", "application/x-ndjson", "application/ndjson"}:
                    frames = _network_frames(response, media_type == "text/event-stream", deadline)
                    reply = _read_network_result(frames)
                else:
                    raise RemoteCallError(f"未识别的响应类型（{media_type}）")
                response_context = {**context, "http_status": response.status}
            if self.message_log:
                self.message_log.record("device_manager", "response", response_context, payload=reply.data)
            return reply
        except HTTPError as exc:
            detail = f"HTTP {exc.code}"
            exc.close()
        except URLError as exc:
            detail = "连接被拒绝" if isinstance(exc.reason, ConnectionRefusedError) else "无法建立连接"
        except TimeoutError:
            detail = "等待最终回复超时"
        except RemoteCallError as exc:
            detail = str(exc)
        except (HTTPException, OSError, ValueError):
            detail = "连接中断或回复格式无效"
        if self.message_log:
            self.message_log.record("device_manager", "error", context, error=detail)
        return AgentResult(ResultStatus.FAILED, f"网络监管查询失败：{detail}", error="network_chat_failed")


def _network_frames(
    response: Any, sse: bool, deadline: float,
) -> Iterator[tuple[str, str]]:
    """Yield complete SSE events or NDJSON records without exposing control events as answers."""
    received = 0
    event = ""
    data_lines: list[str] = []
    while True:
        raw_line = response.readline(_MAX_LINE_BYTES + 1)
        if not raw_line:
            if data_lines:
                yield event, "\n".join(data_lines)
            return
        received += len(raw_line)
        if received > _MAX_REPLY_BYTES or len(raw_line) > _MAX_LINE_BYTES:
            raise RemoteCallError("回复超过接收限制")
        if time.monotonic() > deadline:
            raise TimeoutError
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not sse:
            if line.strip():
                yield "", line
        elif not line:
            if data_lines:
                yield event, "\n".join(data_lines)
            event = ""
            data_lines.clear()
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].removeprefix(" "))


def _read_network_result(frames: Iterator[tuple[str, str]]) -> AgentResult:
    fragments: list[str] = []
    run_id = None
    for event, data in frames:
        if data.strip() == "[DONE]":
            return _network_result({}, "".join(fragments), run_id=run_id)
        try:
            frame = json.loads(data)
        except json.JSONDecodeError:
            if event in {"delta", "token"}:
                fragments.append(data)
                continue
            raise
        if not isinstance(frame, dict):
            raise RemoteCallError("流式事件必须为对象")
        if isinstance(frame.get("runId"), str) and frame["runId"]:
            run_id = frame["runId"]
        kind = frame.get("type") or event
        if frame.get("cancelled") is True:
            return AgentResult(ResultStatus.REJECTED, "网络监管查询已取消", error="network_chat_cancelled")
        if kind == "error" or frame.get("error") or frame.get("success") is False:
            return _network_result(frame, failed=True)
        if frame.get("done") is True or kind in {"done", "final", "completed"}:
            return _network_result(frame, "".join(fragments), run_id=run_id)
        if "delta" in frame or kind in {"delta", "token"} or frame.get("done") is False:
            part = _reply_text(frame, ("delta", "text", "content", "response"))
            if part:
                fragments.append(part)
    raise RemoteCallError("连接已结束，未收到最终回复")


def _reply_text(frame: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    return next((frame[key] for key in keys if isinstance(frame.get(key), str) and frame[key]), "")


def _network_result(
    frame: Any, streamed: str = "", *, failed: bool = False, run_id: str | None = None,
) -> AgentResult:
    if not isinstance(frame, dict):
        raise RemoteCallError("回复必须为 JSON 对象")
    if (failed or frame.get("error") or frame.get("success") is False
            or frame.get("status") in ("failed", "error", "rejected")):
        detail = _reply_text(frame, ("error", "message", "detail")) or "网络监管服务返回错误"
        return AgentResult(ResultStatus.FAILED, detail, error="network_chat_error")
    answer = (_reply_text(frame, ("answer", "reply", "response", "text", "content")) or streamed).strip()
    data: dict[str, Any] = {"response_received": True}
    if run_id:
        data["run_id"] = run_id
    verification = frame.get("verification")
    if isinstance(verification, dict) and verification.get("status") in ("partial", "failed"):
        data["verification_status"] = verification["status"]
        return AgentResult(ResultStatus.FAILED, answer or "网络监管查询未通过数据核查",
                           data, error="network_query_unverified")
    devices = frame.get("network_devices")
    if not isinstance(devices, list) and isinstance(frame.get("data"), dict):
        devices = frame["data"].get("network_devices")
    if isinstance(devices, list) and all(isinstance(device, dict) for device in devices):
        data["network_devices"] = devices
    if not answer and "network_devices" not in data:
        raise RemoteCallError("最终回复缺少查询结果")
    if answer:
        data["answer"] = answer
    return AgentResult(ResultStatus.SUCCEEDED, answer or "已收到网络设备清单", data)
