"""Authenticated HTTP entry point for the LAN command-center frontend."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from typing import Any, Callable
from uuid import uuid4

from .graph import GraphOrchestrator
from .execution import MainState
from .router import RequestRouter
from .main import build_orchestrator, build_parser
from .planner import PlanningError
from ..intent.agent import _parse_intent_analysis
from ..models import ExecutionMode, HandlingMode, IntentAnalysis, IntentType

from ..model_client import ModelSettings
from ..http_server import MAX_BODY_BYTES, MAX_REQUEST_CHARS, JsonRequestHandler
STATE_FIELDS = (
    "request_id", "user_request", "status", "final_answer", "error", "intent",
    "plan_history", "task_statuses", "results", "reviews",
    "handling_mode", "clarification_questions",
)


def serialize_state(state: MainState) -> dict[str, Any]:
    """Keep complete answers, attempts and plan history in the web response."""
    def encode(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    return json.loads(json.dumps(
        {key: state[key] for key in STATE_FIELDS if key in state},
        default=encode, ensure_ascii=False,
    ))


class OrchestratorHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        factory: Callable[[], RequestRouter | GraphOrchestrator],
        *,
        token: str,
        runtime_info: dict[str, Any],
        model_settings: ModelSettings | None = None,
    ) -> None:
        if not token:
            raise ValueError("HTTP 接口必须配置 ORCHESTRATOR_API_TOKEN")
        self.factory = factory
        self.token = token
        self.runtime_info = runtime_info
        self.model_settings = model_settings
        self.run_lock = threading.Lock()
        self.runs_lock = threading.Lock()
        self.runs: dict[str, tuple[int, dict[str, Any]]] = {}
        super().__init__(address, OrchestratorRequestHandler)

    def execute(
        self, message: str, request_id: str | None = None, intent: IntentAnalysis | None = None,
        *, on_progress: Callable[[MainState], None] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        try:
            runtime = self.factory()
            if isinstance(runtime, RequestRouter) and intent is None:
                state = runtime.run(message, request_id=request_id)
            else:
                graph = runtime.orchestrator if isinstance(runtime, RequestRouter) else runtime
                options = {"on_progress": on_progress} if on_progress is not None and isinstance(graph, GraphOrchestrator) else {}
                state = graph.run(message, request_id=request_id, intent=intent, **options)
                if intent is not None:
                    state["handling_mode"] = HandlingMode.ORCHESTRATED
            return HTTPStatus.OK, {**serialize_state(state), "runtime": self.runtime_info}
        except PlanningError as exc:
            return HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": f"规划失败：{exc}", "request_id": request_id}
        except Exception:
            logging.exception("Orchestrator HTTP request failed")
            return HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": "编排执行异常，请查看后端日志", "request_id": request_id}

    def start_run(self, message: str, *, intent: IntentAnalysis | None = None, request_id: str | None = None) -> dict[str, Any] | None:
        if not self.run_lock.acquire(blocking=False):
            return None
        request_id = request_id or f"req-{uuid4().hex}"
        pending = {"request_id": request_id, "status": "running"}
        with self.runs_lock:
            if request_id in self.runs:
                self.run_lock.release()
                raise ValueError("请求编号已存在，请查询原请求，勿重复执行")
            # Only one run can be active. Retain the latest 20 completed/running runs.
            while len(self.runs) >= 20:
                self.runs.pop(next(iter(self.runs)))
            self.runs[request_id] = (HTTPStatus.OK, pending)

        def publish_progress(state: MainState) -> None:
            snapshot = {**serialize_state(state), "runtime": self.runtime_info}
            if intent is not None:
                snapshot["handling_mode"] = HandlingMode.ORCHESTRATED
            with self.runs_lock:
                self.runs[request_id] = (HTTPStatus.OK, snapshot)

        def work() -> None:
            try:
                result = self.execute(message, request_id, intent, on_progress=publish_progress)
                with self.runs_lock:
                    self.runs[request_id] = result
            finally:
                self.run_lock.release()

        threading.Thread(target=work, name=request_id, daemon=True).start()
        return pending


class OrchestratorRequestHandler(JsonRequestHandler):
    server: OrchestratorHttpServer
    authentication_error = "编排接口认证失败"

    def do_GET(self) -> None:
        if not self._authorized():
            return
        if self.path == "/api/model":
            self._model_get()
            return
        if self.path.startswith("/api/orchestrator/runs/"):
            request_id = self.path.removeprefix("/api/orchestrator/runs/")
            with self.server.runs_lock:
                run = self.server.runs.get(request_id)
            if run is None:
                self._json(HTTPStatus.NOT_FOUND, {"detail": "任务记录不存在或服务已重启，请检查后端日志确认执行状态"})
            else:
                self._json(*run)
            return
        if self.path != "/health":
            self._json(HTTPStatus.NOT_FOUND, {"detail": "接口不存在"})
            return
        self._json(HTTPStatus.OK, {
            "status": "ok", "service": "orchestration", "pid": os.getpid(),
            "busy": self.server.run_lock.locked(),
            "runtime": self.server.runtime_info,
        })

    def do_POST(self) -> None:
        if not self._authorized():
            return
        if self.path == "/api/model":
            self._model_post([self.server.run_lock])
            return
        if self.path not in {"/api/orchestrator/run", "/api/orchestrator/runs", "/api/orchestrator/dispatch"}:
            self._json(HTTPStatus.NOT_FOUND, {"detail": "接口不存在"})
            return
        payload = self._read_json()
        if payload is None:
            return
        message = self._read_message(payload)
        if message is None:
            return
        intent = None
        request_id = None
        if self.path == "/api/orchestrator/dispatch":
            try:
                request_id = payload.get("request_id")
                if not isinstance(request_id, str) or not re.fullmatch(r"req-[a-f0-9]{32}", request_id):
                    raise PlanningError("动态编排请求编号无效")
                raw_intent = payload.get("intent")
                if not isinstance(raw_intent, dict) or raw_intent.get("handling_mode") != HandlingMode.ORCHESTRATED.value:
                    raise PlanningError("必须提供明确的动态编排意图")
                intent = _parse_intent_analysis(raw_intent)
                if intent.handling_mode is not HandlingMode.ORCHESTRATED or intent.needs_clarification or intent.intent_type is IntentType.UNSUPPORTED:
                    raise PlanningError("编排服务只接收已就绪的动态编排意图")
                if intent.provided_parameters.get("parameter_policy") == "use_backend_defaults":
                    raise PlanningError("外部意图不得使用演示默认参数")
                if intent.intent_type is IntentType.HARDWARE_EXECUTION and (
                    intent.target_type not in {"usrp", "ris"} or intent.execution_mode is not ExecutionMode.PHYSICAL
                    or (intent.target_type not in message.lower() and not (intent.target_type == "ris" and "反射面" in message))
                ):
                    raise PlanningError("用户未明确指定目标设备，不能执行硬件任务")
            except PlanningError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})
                return
        if self.path in {"/api/orchestrator/runs", "/api/orchestrator/dispatch"}:
            try:
                pending = self.server.start_run(message.strip(), intent=intent, request_id=request_id)
            except ValueError as exc:
                self._json(HTTPStatus.CONFLICT, {"detail": str(exc)})
                return
            if pending is None:
                self._json(HTTPStatus.TOO_MANY_REQUESTS, {"detail": "编排器正在执行任务，请等待当前任务完成"})
            else:
                self._json(HTTPStatus.ACCEPTED, pending)
            return
        if not self.server.run_lock.acquire(blocking=False):
            self._json(HTTPStatus.TOO_MANY_REQUESTS, {"detail": "编排器正在执行任务，请等待当前任务完成"})
            return
        try:
            result = self.server.execute(message.strip())
        finally:
            self.server.run_lock.release()
        self._json(*result)


def main() -> int:
    parser = build_parser()
    parser.description = "启动供前端调用的编排 HTTP 服务（复用 CLI 的规划器与智能体配置）"
    parser.add_argument("--host", default=os.getenv("ORCHESTRATOR_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ORCHESTRATOR_API_PORT", "18002")))
    args = parser.parse_args()
    if args.request is not None:
        parser.error("HTTP 服务启动时不接受任务文本，请通过 POST 提交任务")
    token = os.getenv("ORCHESTRATOR_API_TOKEN", "").strip()
    if not token:
        parser.error("请在 .env 中设置 ORCHESTRATOR_API_TOKEN")
    if not args.remote_agents and args.fake_outcome is None:
        args.fake_outcome = "succeeded"
    # Validate all CLI configuration before binding the listening socket.
    build_orchestrator(args, parser)
    runtime_info = {
        "planner": args.model_provider if args.planner != "fixed" else "fixed",
        "model_provider": args.model_provider if args.planner != "fixed" else None,
        "model": (args.deepseek_model if args.model_provider == "deepseek" else args.ollama_model) if args.planner != "fixed" else None,
        "intent_service": "local_debug" if args.local_intent else args.intent_service_url,
        "parameter_policy": args.parameter_policy,
        "agents": {
            "knowledge": "http" if args.knowledge_api_url else "rpc" if args.remote_agents else "fake",
            "device_manager": "http_chat" if args.network_chat_url else "rpc" if args.remote_agents else "fake",
            "usrp": "rpc" if args.remote_agents else "fake",
            "ris": "sse" if args.ris_chat_url else "disabled",
        },
    }
    server = OrchestratorHttpServer(
        (args.host, args.port), lambda: build_orchestrator(args, parser)[0],
        token=token, runtime_info=runtime_info,
        model_settings=ModelSettings(args, "orchestration") if args.planner != "fixed" else None,
    )
    host, port = server.server_address
    print(f"编排 HTTP 服务：http://{host}:{port}（PID {os.getpid()}，Ctrl+C 停止）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
