"""HTTP + JSON-RPC 2.0 adapter for independently deployed agents.

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

from .models import AgentResult, ResultStatus


class RemoteCallError(RuntimeError):
    """The remote call outcome is not known (e.g. connection lost or timeout)."""


@dataclass(frozen=True)
class JsonRpcAgentAdapter:
    endpoint: str
    agent_id: str
    method: str = "tasks.send"
    timeout: float = 30.0
    auth_token: str | None = None

    def __post_init__(self) -> None:
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("远程智能体地址必须是 HTTP(S) URL")
        if self.timeout <= 0:
            raise ValueError("远程智能体超时必须大于 0")

    def invoke(self, context: Mapping[str, Any]) -> AgentResult:
        return self._call(context)

    def execute(self, context: Mapping[str, Any]) -> AgentResult:
        return self._call(context)

    def _call(self, context: Mapping[str, Any]) -> AgentResult:
        current_task = context.get("current_task")
        if not isinstance(current_task, Mapping):
            raise ValueError("远程调用缺少 current_task")
        request_id = context.get("request_id")
        task_id = current_task.get("task_id")
        attempt = current_task.get("attempt")
        if not isinstance(request_id, str) or not isinstance(task_id, str) or not isinstance(attempt, int):
            raise ValueError("远程调用缺少有效请求/任务/尝试标识")
        attempt_id = f"{request_id}:{task_id}:{attempt}"
        params = {
            "request_id": request_id,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "source": "orchestrator",
            "target": self.agent_id,
            "task": dict(current_task),
            "user_request": context.get("user_request", ""),
            "previous_results": context.get("previous_results", {}),
        }
        if "skill_ref" in context:
            params["skill_ref"] = context["skill_ref"]
        if "selected_device" in context:
            params["selected_device"] = context["selected_device"]
        payload = {"jsonrpc": "2.0", "id": attempt_id, "method": self.method, "params": params}
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
                envelope = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RemoteCallError(f"{self.agent_id} 远程调用结果未知：{exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteCallError(f"{self.agent_id} 返回无法解析的响应") from exc
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
