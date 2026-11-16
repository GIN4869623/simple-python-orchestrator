"""HTTP implementation of IntentAgentPort; failures never fall back locally."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from ..errors import PlanningError
from ..http_server import MAX_REQUEST_CHARS
from ..models import IntentAnalysis, ParameterPolicy
from .agent import _apply_parameter_policy, _parse_intent_analysis


@dataclass
class IntentHttpClient:
    base_url: str = "http://127.0.0.1:18003"
    timeout: float = 180.0
    auth_token: str = field(default="", repr=False)
    parameter_policy: ParameterPolicy = ParameterPolicy.STRICT

    def __post_init__(self) -> None:
        try:
            url = urlsplit(self.base_url)
            port = url.port
        except ValueError as exc:
            raise PlanningError("意图服务地址无效") from exc
        if (
            url.scheme not in {"http", "https"} or not url.hostname
            or url.username is not None or url.password is not None
            or url.query or url.fragment or url.path not in {"", "/"}
            or port is not None and not 1 <= port <= 65535
        ):
            raise PlanningError("意图服务地址必须是 HTTP(S) 服务根地址")
        if not self.auth_token.strip():
            raise PlanningError("请配置 INTENT_API_TOKEN 或 ORCHESTRATOR_API_TOKEN")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise PlanningError("意图服务超时必须是大于 0 的有限秒数")

    def analyze_intent(self, user_request: str) -> IntentAnalysis:
        if not 1 <= len(user_request.strip()) <= MAX_REQUEST_CHARS:
            raise PlanningError("意图服务 request 必须是 1～2000 字符的任务文本")
        request_id = f"intent-{uuid4().hex}"
        request = Request(
            f"{self.base_url.rstrip('/')}/api/intent/analyze",
            data=json.dumps({"request_id": request_id, "request": user_request}, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.auth_token}"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise PlanningError(f"意图识别服务请求失败（HTTP {exc.code}）") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise PlanningError(f"无法调用意图识别服务 {self.base_url}：{exc}") from exc
        except (ValueError, UnicodeDecodeError) as exc:
            raise PlanningError("意图识别服务返回的响应不是有效 JSON") from exc
        if not isinstance(payload, dict) or payload.get("request_id") != request_id:
            raise PlanningError("意图识别服务返回的 request_id 不匹配")
        if payload.get("parameter_policy") != ParameterPolicy.STRICT.value:
            raise PlanningError("意图识别服务必须返回未使用演示默认值的严格分析结果")
        raw_intent = payload.get("intent")
        if not isinstance(raw_intent, dict) or "handling_mode" not in raw_intent:
            raise PlanningError("意图识别服务缺少 intent 或 handling_mode")
        intent = _parse_intent_analysis(raw_intent)
        if intent.provided_parameters.get("parameter_policy") == "use_backend_defaults":
            raise PlanningError("意图识别服务不能预先应用演示默认参数")
        return _apply_parameter_policy(intent, self.parameter_policy)
