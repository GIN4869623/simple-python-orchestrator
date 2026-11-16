"""Authenticated HTTP entry point for the LAN command-center frontend."""

from __future__ import annotations

import json
import logging
import os
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
    ) -> None:
        if not token:
            raise ValueError("HTTP 接口必须配置 ORCHESTRATOR_API_TOKEN")
        self.factory = factory
        self.token = token
        self.runtime_info = runtime_info
        self.run_lock = threading.Lock()
        self.runs_lock = threading.Lock()
        self.runs: dict[str, tuple[int, dict[str, Any]]] = {}
        super().__init__(address, OrchestratorRequestHandler)

    def execute(self, message: str, request_id: str | None = None) -> tuple[int, dict[str, Any]]:
        try:
            state = self.factory().run(message, request_id=request_id)
            return HTTPStatus.OK, {**serialize_state(state), "runtime": self.runtime_info}
        except PlanningError as exc:
            return HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": f"规划失败：{exc}", "request_id": request_id}
        except Exception:
            logging.exception("Orchestrator HTTP request failed")
            return HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": "编排执行异常，请查看后端日志", "request_id": request_id}

    def start_run(self, message: str) -> dict[str, Any] | None:
        if not self.run_lock.acquire(blocking=False):
            return None
        request_id = f"req-{uuid4().hex}"
        pending = {"request_id": request_id, "status": "running"}
        with self.runs_lock:
            # Only one run can be active. Retain the latest 20 completed/running runs.
            while len(self.runs) >= 20:
                self.runs.pop(next(iter(self.runs)))
            self.runs[request_id] = (HTTPStatus.OK, pending)

        def work() -> None:
            try:
                result = self.execute(message, request_id)
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
        if self.path not in {"/api/orchestrator/run", "/api/orchestrator/runs"}:
            self._json(HTTPStatus.NOT_FOUND, {"detail": "接口不存在"})
            return
        payload = self._read_json()
        if payload is None:
            return
        message = self._read_message(payload)
        if message is None:
            return
        if self.path == "/api/orchestrator/runs":
            pending = self.server.start_run(message.strip())
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
        "planner": args.planner,
        "intent_service": "local_debug" if args.local_intent else args.intent_service_url,
        "parameter_policy": args.parameter_policy,
        "agents": {
            "knowledge": "http" if args.knowledge_api_url else "rpc" if args.remote_agents else "fake",
            "device_manager": "rpc" if args.remote_agents else "fake",
            "usrp": "rpc" if args.remote_agents else "fake",
        },
    }
    server = OrchestratorHttpServer(
        (args.host, args.port), lambda: build_orchestrator(args, parser)[0],
        token=token, runtime_info=runtime_info,
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
