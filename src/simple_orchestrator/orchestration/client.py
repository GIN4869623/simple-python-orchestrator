"""Submit precomputed complex intents to the independent orchestration service."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from ..errors import PlanningError
from ..models import HandlingMode, IntentAnalysis


@dataclass
class OrchestrationHttpClient:
    base_url: str
    auth_token: str = field(repr=False)
    timeout: float = 600.0
    poll_interval: float = 0.2
    on_submitted: Callable[[str], None] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            url = urlsplit(self.base_url)
            port = url.port
        except ValueError as exc:
            raise PlanningError("编排服务地址无效") from exc
        if (url.scheme not in {"http", "https"} or not url.hostname
                or url.username is not None or url.password is not None
                or url.path not in {"", "/"} or url.query or url.fragment
                or port is not None and not 1 <= port <= 65535):
            raise PlanningError("编排服务地址必须是 HTTP(S) 服务根地址")
        if not self.auth_token.strip() or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise PlanningError("编排服务必须配置密钥和有效超时")
        if not math.isfinite(self.poll_interval) or self.poll_interval <= 0:
            raise PlanningError("编排查询间隔必须大于 0")

    def _request(self, path: str, deadline: float, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PlanningError("等待编排服务超时；请先检查该请求的执行状态，勿自动重试")
        req = Request(
            self.base_url.rstrip("/") + path,
            data=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.auth_token}"},
        )
        try:
            with urlopen(req, timeout=min(10.0, remaining)) as response:
                data = json.loads(response.read())
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read(65536)).get("detail")
            except (ValueError, AttributeError):
                detail = None
            reason = detail if isinstance(detail, str) else "请按请求编号检查执行状态"
            raise PlanningError(f"编排服务请求失败（HTTP {exc.code}）：{reason}") from exc
        except (OSError, URLError, ValueError) as exc:
            raise PlanningError(f"无法调用编排服务：{exc}；请先检查执行状态，勿自动重试") from exc
        if not isinstance(data, dict):
            raise PlanningError("编排服务返回的响应不是对象")
        return data

    def run(self, user_request: str, *, request_id: str | None = None, intent: IntentAnalysis | None = None) -> dict[str, Any]:
        if intent is None or intent.handling_mode is not HandlingMode.ORCHESTRATED:
            raise PlanningError("只有已识别的动态编排请求可以转交编排服务")
        request_id = request_id or f"req-{uuid4().hex}"
        deadline = time.monotonic() + self.timeout
        data = self._request("/api/orchestrator/dispatch", deadline, {
            "request_id": request_id, "request": user_request, "intent": asdict(intent),
        })
        if data.get("request_id") != request_id or data.get("status") != "running":
            raise PlanningError("编排服务未返回有效的受理记录")
        if self.on_submitted:
            self.on_submitted(request_id)
        while data.get("status") == "running":
            time.sleep(min(self.poll_interval, max(0, deadline - time.monotonic())))
            data = self._request(f"/api/orchestrator/runs/{request_id}", deadline)
            if data.get("request_id") != request_id:
                raise PlanningError("编排结果的 request_id 不匹配")
        if data.get("status") not in {"succeeded", "failed", "rejected", "unknown", "needs_clarification", "unsupported"}:
            raise PlanningError("编排服务返回未知执行状态")
        if not isinstance(data.get("final_answer"), str) or not isinstance(data.get("plan_history"), list):
            raise PlanningError("编排服务返回的执行结果不完整")
        return data
