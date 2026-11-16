"""Adapter for the knowledge service's plain HTTP JSON answer endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from .agent_logging import AgentMessageLog
from .models import AgentResult, ResultStatus
from .remote import RemoteCallError


DEFAULT_KNOWLEDGE_API_URL = "http://192.168.1.103:18001/api/knowledge/answer"


@dataclass(frozen=True)
class KnowledgeHttpAdapter:
    endpoint: str = DEFAULT_KNOWLEDGE_API_URL
    timeout: float = 30.0
    sender: str = "编排智能体"
    top_k: int = 3
    model_profile: str = "local"
    api_key: str | None = field(default=None, repr=False)
    message_log: AgentMessageLog | None = None

    def __post_init__(self) -> None:
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("知识接口地址必须是 HTTP(S) URL")
        if self.timeout <= 0:
            raise ValueError("知识接口超时必须大于 0")
        if not isinstance(self.sender, str) or not 1 <= len(self.sender.strip()) <= 128:
            raise ValueError("sender 去除首尾空白后必须为 1～128 字符")
        if type(self.top_k) is not int or not 1 <= self.top_k <= 20:
            raise ValueError("top_k 必须是 1～20 的整数")
        if self.model_profile not in {"local", "cloud"}:
            raise ValueError("model_profile 必须是 local 或 cloud")

    def invoke(
        self, request: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None,
    ) -> AgentResult:
        question = request.get("message")
        if set(request) != {"message"} or not isinstance(question, str):
            raise ValueError("知识任务请求必须只包含自然语言 message")
        question = question.strip()
        if not 1 <= len(question) <= 2000:
            raise ValueError("知识问题去除首尾空白后必须为 1～2000 字符")
        # Each attempt gets a fresh, bounded ID; workflow IDs stay in log metadata.
        knowledge_request_id = str(uuid4())
        payload = {
            "request_id": knowledge_request_id,
            "sender": self.sender.strip(),
            "question": question,
            "knowledge_domain_ids": [],
            "top_k": self.top_k,
            "model_profile": self.model_profile,
        }
        log_metadata = {
            **(metadata or {}),
            "knowledge_request_id": knowledge_request_id,
            "transport": self.endpoint.split(":", 1)[0],
            "endpoint": self.endpoint,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._log("request", log_metadata, body=body)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Knowledge-API-Key"] = self.api_key
        http_request = Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            try:
                with urlopen(http_request, timeout=self.timeout) as response:
                    response_body = response.read()
                    self._log("response", {**log_metadata, "http_status": response.status}, body=response_body)
            except HTTPError as exc:
                self._log("response", {**log_metadata, "http_status": exc.code}, body=exc.read())
                raise RemoteCallError(f"知识接口 HTTP 错误：{exc.code}") from exc
            except (URLError, TimeoutError, OSError) as exc:
                raise RemoteCallError(f"知识接口连接失败或超时：{exc}") from exc
            try:
                envelope = json.loads(response_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RemoteCallError("知识接口返回无法解析的 JSON 响应") from exc
            return self._parse_response(envelope)
        except Exception as exc:
            self._log("error", log_metadata, error=f"{type(exc).__name__}: {exc}")
            raise

    def _log(
        self, event: str, metadata: Mapping[str, Any], *,
        body: bytes | None = None, error: str | None = None,
    ) -> None:
        if self.message_log is not None:
            self.message_log.record("knowledge", event, metadata, body=body, error=error)

    @staticmethod
    def _parse_response(envelope: Any) -> AgentResult:
        if not isinstance(envelope, dict):
            raise RemoteCallError("知识接口响应必须是对象")
        raw_status = envelope.get("status")
        try:
            status = {
                "success": ResultStatus.SUCCEEDED,
                "failed": ResultStatus.FAILED,
                "rejected": ResultStatus.REJECTED,
                "unknown": ResultStatus.UNKNOWN,
            }[raw_status]
        except (TypeError, KeyError) as exc:
            raise RemoteCallError("知识接口返回未知业务状态") from exc
        summary = envelope.get("summary")
        data = envelope.get("data", {})
        error = envelope.get("error")
        if not isinstance(summary, str) or not isinstance(data, dict):
            raise RemoteCallError("知识接口 summary 必须是字符串，data 必须是对象")
        if error is not None and not isinstance(error, str):
            raise RemoteCallError("知识接口 error 必须是字符串或 null")
        if status is ResultStatus.SUCCEEDED and not isinstance(data.get("knowledge"), str):
            raise RemoteCallError("知识接口成功响应缺少字符串 data.knowledge")
        # Keep the answer verbatim. Additional envelope fields are not consumed.
        return AgentResult(status, summary, data, error)
