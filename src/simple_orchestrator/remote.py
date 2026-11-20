"""HTTP + JSON-RPC 2.0 adapter for delegating business tasks to agents.

The main graph selects an agent ID, never an arbitrary URL. Each adapter is
constructed from trusted static configuration and exposes the same Port shape
as today's local Fake.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .agent_logging import AgentMessageLog
from .models import AgentResult, ResultStatus
from .task_messages import invocation_metadata, make_agent_request, make_knowledge_request


class RemoteCallError(RuntimeError):
    """The remote call outcome is not known (e.g. connection lost or timeout)."""


@dataclass(frozen=True)
class JsonRpcAgentAdapter:
    endpoint: str
    agent_id: str
    method: str = "tasks.send"
    timeout: float = 30.0
    auth_token: str | None = None
    message_log: AgentMessageLog | None = None

    def __post_init__(self) -> None:
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("远程智能体地址必须是 HTTP(S) URL")
        if self.timeout <= 0:
            raise ValueError("远程智能体超时必须大于 0")

    def invoke(
        self, request: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None,
    ) -> AgentResult:
        return self._call(request, metadata=metadata)

    def execute(self, context: Mapping[str, Any]) -> AgentResult:
        return self._call(context)

    def _make_payload(
        self, context: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if metadata is not None and self.agent_id in {"knowledge", "device_manager"}:
            message = context.get("message")
            if set(context) != {"message"} or not isinstance(message, str) or not message.strip():
                raise ValueError("专业智能体任务请求必须只包含自然语言 message")
            request_id, task_id, attempt = (metadata.get(key) for key in ("request_id", "task_id", "attempt"))
            if not isinstance(request_id, str) or not isinstance(task_id, str) or not isinstance(attempt, int):
                raise ValueError("远程调用缺少有效请求/任务/尝试标识")
            return {
                "jsonrpc": "2.0", "id": f"{request_id}:{task_id}:{attempt}",
                "method": self.method, "params": {"message": message},
            }
        current_task = context.get("current_task")
        if not isinstance(current_task, Mapping):
            raise ValueError("远程调用缺少 current_task")
        request_id = context.get("request_id")
        task_id = current_task.get("task_id")
        attempt = current_task.get("attempt")
        if not isinstance(request_id, str) or not isinstance(task_id, str) or not isinstance(attempt, int):
            raise ValueError("远程调用缺少有效请求/任务/尝试标识")
        attempt_id = f"{request_id}:{task_id}:{attempt}"
        task = {
            "goal": current_task.get("description", ""),
            "inputs": current_task.get("inputs", {}),
            "expected_output": current_task.get("expected_output", []),
            "completion_criteria": current_task.get("completion_criteria", []),
            # Business capability for matching and policy checks, not a tool call.
            "routing_capability": current_task.get("capability", ""),
        }
        task_context = {
            "user_request": context.get("user_request", ""),
            "dependency_results": context.get("previous_results", {}),
        }
        if "skill_ref" in context:
            task_context["skill_ref"] = context["skill_ref"]
        if "selected_device" in context:
            task_context["selected_device"] = context["selected_device"]
        params = {
            "contract_version": "1.0",
            "request_id": request_id,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "source": "orchestrator",
            "target": self.agent_id,
        }
        if self.agent_id == "knowledge":
            params = make_knowledge_request(context)
        elif self.agent_id == "device_manager":
            params = make_agent_request(context)
        else:
            params["task"] = task
            params["context"] = task_context
        return {"jsonrpc": "2.0", "id": attempt_id, "method": self.method, "params": params}

    def _call(
        self, context: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None,
    ) -> AgentResult:
        payload = self._make_payload(context, metadata=metadata)
        metadata = metadata if metadata is not None else invocation_metadata(context)
        self._log_message(
            "request", payload, metadata, message=payload,
            body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )
        try:
            envelope = self._exchange(payload, metadata)
            return self._parse_response(envelope, payload, task_id=metadata["task_id"])
        except Exception as exc:
            self._log_message("error", payload, metadata, error=f"{type(exc).__name__}: {exc}")
            raise

    def _log_message(
        self, event: str, request: Mapping[str, Any], metadata: Mapping[str, Any], *, message: Any = None,
        body: bytes | None = None, error: str | None = None, http_status: int | None = None,
    ) -> None:
        if self.message_log is None:
            return
        log_metadata = {
            "request_id": metadata["request_id"],
            "task_id": metadata["task_id"],
            "attempt_id": request["id"],
            "transport": self.endpoint.split(":", 1)[0],
            "endpoint": self.endpoint,
        }
        if http_status is not None:
            log_metadata["http_status"] = http_status
        self.message_log.record(
            self.agent_id, event, log_metadata, payload=message, body=body, error=error,
        )

    def _exchange(self, payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                self._log_message("response", payload, metadata, body=body, http_status=response.status)
                envelope = json.loads(body.decode("utf-8"))
        except HTTPError as exc:
            self._log_message("response", payload, metadata, body=exc.read(), http_status=exc.code)
            raise RemoteCallError(f"{self.agent_id} 远程调用结果未知：{exc}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RemoteCallError(f"{self.agent_id} 远程调用结果未知：{exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteCallError(f"{self.agent_id} 返回无法解析的响应") from exc
        return envelope

    def _parse_response(self, envelope: Any, payload: Mapping[str, Any], *, task_id: str) -> AgentResult:
        attempt_id = payload["id"]
        if not isinstance(envelope, dict) or envelope.get("jsonrpc") != "2.0" or envelope.get("id") != attempt_id:
            raise RemoteCallError(f"{self.agent_id} JSON-RPC 响应标识不匹配")
        if "error" in envelope:
            error = envelope["error"]
            if not isinstance(error, dict):
                raise RemoteCallError(f"{self.agent_id} JSON-RPC error 格式错误")
            return AgentResult(
                ResultStatus.FAILED,
                str(error.get("message", "远程服务错误")),
                error=f"rpc:{error.get('code', 'unknown')}",
            )
        result = envelope.get("result")
        if not isinstance(result, dict):
            raise RemoteCallError(f"{self.agent_id} 缺少业务结果")
        if result.get("task_id") != task_id:
            raise RemoteCallError(f"{self.agent_id} 返回了不匹配的 task_id")
        try:
            status = ResultStatus(result.get("status"))
        except (ValueError, TypeError) as exc:
            raise RemoteCallError(f"{self.agent_id} 返回未知业务状态") from exc
        data = result.get("data", {})
        if not isinstance(data, dict):
            raise RemoteCallError(f"{self.agent_id} 的 data 必须是对象")
        return AgentResult(
            status,
            str(result.get("summary", "")),
            data,
            str(result["error"]) if result.get("error") is not None else None,
        )


def call_ris_chat(context: Mapping[str, Any], *, endpoint: str, timeout: float = 180.0,
                  message_log: AgentMessageLog | None = None) -> AgentResult:
    """Consume the RIS chat stream and return its final text to the caller."""
    import time
    from urllib.parse import urlsplit

    if urlsplit(endpoint).scheme not in {"http", "https"} or timeout <= 0:
        raise ValueError("RIS 接口地址或超时配置无效")
    metadata = invocation_metadata(context)
    instruction = (context.get("user_request", "") if context.get("single_task")
                   else make_agent_request(context)["message"])
    payload = {
        "instruction": instruction,
        "session_id": f"orchestrator-{metadata['request_id']}-{metadata['task_id']}",
        "runId": metadata["attempt_id"], "task": context["current_task"]["description"],
        "mode": "think",
    }
    log_metadata = {**metadata, "endpoint": endpoint, "transport": "sse"}
    def log(event: str, **values: Any) -> None:
        if message_log:
            message_log.record("ris", event, log_metadata, **values)
    log("request", payload=payload)
    received = 0
    deadline = time.monotonic() + timeout
    data_lines: list[str] = []
    def consume() -> AgentResult | None:
        if not data_lines:
            return None
        event = json.loads("\n".join(data_lines))
        data_lines.clear()
        if not isinstance(event, dict):
            raise RemoteCallError("RIS 流式事件必须为对象")
        log("response", payload=event)
        if event.get("type") == "error":
            return AgentResult(ResultStatus.FAILED,
                str(event.get("message") or event.get("content") or event.get("error") or "RIS 返回错误"),
                error="ris_error")
        if event.get("type") != "done":
            return None
        reply = event.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            raise RemoteCallError("RIS 最终回复缺少文字内容")
        return AgentResult(ResultStatus.SUCCEEDED, reply, {
            "answer": reply, "response_received": True,
            "run_id": payload["runId"], "session_id": payload["session_id"],
            "tools_total": event.get("tools_total"), "timing": event.get("timing", {}),
            "completion_kind": "text_reply",
        })
    request = Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      headers={"Content-Type": "application/json", "Accept": "text/event-stream"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.headers.get_content_type() != "text/event-stream":
                raise RemoteCallError("RIS 未返回 SSE 流式响应")
            while True:
                line = response.readline(1024 * 1024 + 1)
                if not line:
                    result = consume()
                    if result:
                        return result
                    break
                received += len(line)
                if received > 4 * 1024 * 1024 or len(line) > 1024 * 1024:
                    raise RemoteCallError("RIS 回复超过接收限制")
                if time.monotonic() > deadline:
                    raise RemoteCallError("RIS 等待最终回复超时")
                text = line.decode("utf-8").rstrip("\r\n")
                if not text:
                    result = consume()
                    if result:
                        return result
                elif text.startswith("data:"):
                    data_lines.append(text[5:].removeprefix(" "))
        raise RemoteCallError("RIS 连接已结束，但未收到最终回复")
    except (HTTPError, URLError, OSError, UnicodeError, ValueError, RemoteCallError) as exc:
        log("error", error=f"{type(exc).__name__}: {exc}")
        # The task may have been accepted; never trigger an automatic duplicate.
        return AgentResult(ResultStatus.UNKNOWN, f"RIS 回复未完整收到：{exc}", error="ris_response_unknown")
