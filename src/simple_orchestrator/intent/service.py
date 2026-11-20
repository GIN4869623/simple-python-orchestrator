"""Independent intent-recognition service, default port 18003."""

from __future__ import annotations

import argparse
import logging
import os
import re
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from typing import Any, Callable
from uuid import uuid4

from ..errors import PlanningError
from ..http_server import JsonRequestHandler
from ..models import ParameterPolicy
from ..model_client import make_model_client, ModelSettings
from ..ports import IntentAgentPort
from .agent import FixedIntentAgent, OllamaIntentAgent
from ..orchestration.api import serialize_state
from ..orchestration.client import OrchestrationHttpClient
from ..orchestration.execution import WorkflowExecutor
from ..orchestration.main import build_executor, build_parser as execution_parser
from ..orchestration.router import RequestRouter
from ..agent_registry import build_default_registry


class IntentHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, address: tuple[str, int], factory: Callable[[], IntentAgentPort],
        *, token: str, runtime_info: dict[str, Any],
        router_factory: Callable[..., RequestRouter] | None = None,
        model_settings: ModelSettings | None = None,
    ) -> None:
        if not token:
            raise ValueError("意图服务必须配置 INTENT_API_TOKEN 或 ORCHESTRATOR_API_TOKEN")
        self.token = token
        self.factory = factory
        self.runtime_info = runtime_info
        self.model_settings = model_settings
        self.analysis_lock = threading.Lock()
        self.router_factory = router_factory
        self.run_lock = threading.Lock()
        self.runs_lock = threading.Lock()
        self.runs: dict[str, dict[str, Any]] = {}
        super().__init__(address, IntentRequestHandler)

    def start_run(self, message: str) -> dict[str, Any] | None:
        if not self.run_lock.acquire(blocking=False):
            return None
        request_id = f"req-{uuid4().hex}"
        pending = {
            **WorkflowExecutor.initial_state(message, request_id),
            "phase": "recognizing", "intent": None, "handling_mode": None,
            "execution_targets": [], "orchestration_request_id": None,
            "clarification_questions": [],
        }
        with self.runs_lock:
            while len(self.runs) >= 20:
                self.runs.pop(next(iter(self.runs)))
            self.runs[request_id] = pending

        def update(**values: Any) -> None:
            with self.runs_lock:
                self.runs[request_id] = {**self.runs[request_id], **values}

        def dispatched(mode, tasks) -> None:
            update(
                handling_mode=mode,
                phase="dispatching" if tasks is None else "executing",
                execution_targets=["orchestrator"] if tasks is None else list(dict.fromkeys(task.agent for task in tasks)),
            )

        def forwarded(target_id: str) -> None:
            update(phase="orchestrating", orchestration_request_id=target_id)

        def work() -> None:
            try:
                with self.analysis_lock:
                    agent = self.factory()
                    intent = agent.analyze_intent(message)
                if intent.provided_parameters.get("parameter_policy") == "use_backend_defaults":
                    raise PlanningError("意图服务只能返回严格分析结果")
                update(intent=asdict(intent), handling_mode=intent.handling_mode, phase="routing")
                router = self.router_factory(agent, forwarded)
                state = router.run_with_intent(message, intent, request_id=request_id, on_dispatch=dispatched)
                update(**serialize_state(state), phase="completed")
            except PlanningError as exc:
                update(status="failed", phase="failed", error=str(exc), final_answer=f"请求处理失败：{exc}")
            except Exception:
                logging.exception("Intent dispatch failed")
                update(status="failed", phase="failed", error="请求处理异常，请查看意图服务日志", final_answer="请求处理异常，请查看意图服务日志")
            finally:
                self.run_lock.release()

        threading.Thread(target=work, name=request_id, daemon=True).start()
        return {"request_id": request_id, "status": "running", "phase": "recognizing"}


class IntentRequestHandler(JsonRequestHandler):
    server: IntentHttpServer
    authentication_error = "意图识别接口认证失败"

    def do_GET(self) -> None:
        if not self._authorized():
            return
        if self.path == "/api/model":
            self._model_get()
            return
        if self.path.startswith("/api/intent/runs/"):
            request_id = self.path.removeprefix("/api/intent/runs/")
            if not re.fullmatch(r"req-[a-f0-9]{32}", request_id):
                self._json(HTTPStatus.BAD_REQUEST, {"detail": "请求编号无效"})
                return
            with self.server.runs_lock:
                state = self.server.runs.get(request_id)
            if state is None:
                self._json(HTTPStatus.NOT_FOUND, {"detail": "请求记录不存在或服务已重启，请检查执行状态，勿自动重试"})
            else:
                self._json(HTTPStatus.OK, state)
            return
        if self.path != "/health":
            self._json(HTTPStatus.NOT_FOUND, {"detail": "接口不存在"})
            return
        self._json(HTTPStatus.OK, {
            "status": "ok", "service": "intent", "pid": os.getpid(),
            "busy": self.server.analysis_lock.locked() or self.server.run_lock.locked(), "runtime": self.server.runtime_info,
        })

    def do_POST(self) -> None:
        if not self._authorized():
            return
        if self.path == "/api/model":
            self._model_post([self.server.run_lock, self.server.analysis_lock])
            return
        if self.path not in {"/api/intent/analyze", "/api/intent/runs"}:
            self._json(HTTPStatus.NOT_FOUND, {"detail": "接口不存在"})
            return
        payload = self._read_json()
        if payload is None:
            return
        message = self._read_message(payload)
        if message is None:
            return
        if self.path == "/api/intent/runs":
            if self.server.router_factory is None:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"detail": "意图服务未配置业务分流"})
                return
            pending = self.server.start_run(message.strip())
            if pending is None:
                self._json(HTTPStatus.TOO_MANY_REQUESTS, {"detail": "请求正在处理中，请等待当前任务完成"})
            else:
                self._json(HTTPStatus.ACCEPTED, pending)
            return
        request_id = payload.get("request_id", f"intent-{uuid4().hex}")
        if not isinstance(request_id, str) or not 1 <= len(request_id.strip()) <= 200:
            self._json(HTTPStatus.BAD_REQUEST, {"detail": "request_id 必须为 1～200 字符的文本"})
            return
        if not self.server.analysis_lock.acquire(blocking=False):
            self._json(HTTPStatus.TOO_MANY_REQUESTS, {"detail": "意图识别服务忙碌，请稍后提交"})
            return
        try:
            # Always preserve missing parameters; the caller owns Fake defaults.
            intent = self.server.factory().analyze_intent(message)
            if intent.provided_parameters.get("parameter_policy") == "use_backend_defaults":
                raise PlanningError("意图服务只能返回严格分析结果")
            result = HTTPStatus.OK, {
                "request_id": request_id, "intent": asdict(intent),
                "parameter_policy": ParameterPolicy.STRICT.value,
            }
        except PlanningError as exc:
            result = HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": f"意图识别失败：{exc}", "request_id": request_id}
        except Exception:
            logging.exception("Intent recognition request failed")
            result = HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": "意图识别异常，请查看服务日志", "request_id": request_id}
        finally:
            self.server.analysis_lock.release()
        self._json(*result)


def build_parser() -> argparse.ArgumentParser:
    parser = execution_parser()
    parser.description = "启动独立意图识别与业务分流服务"
    parser.set_defaults(parameter_policy="strict")
    parser.add_argument("--host", default=os.getenv("INTENT_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("INTENT_API_PORT", "18003")))
    parser.add_argument("--backend", choices=["ollama", "fixed"], default=os.getenv("INTENT_BACKEND", "ollama"))
    parser.set_defaults(
        model_provider=os.getenv("INTENT_MODEL_PROVIDER") or os.getenv("MODEL_PROVIDER", "ollama"),
        deepseek_url=os.getenv("INTENT_DEEPSEEK_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_proxy_url=os.getenv("INTENT_DEEPSEEK_PROXY_URL") or os.getenv("DEEPSEEK_PROXY_URL", ""),
        deepseek_model=os.getenv("INTENT_DEEPSEEK_MODEL") or os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        deepseek_timeout=float(os.getenv("INTENT_DEEPSEEK_TIMEOUT") or os.getenv("DEEPSEEK_TIMEOUT", "120")),
        deepseek_temperature=float(os.getenv("INTENT_DEEPSEEK_TEMPERATURE") or os.getenv("DEEPSEEK_TEMPERATURE", "0.2")),
        deepseek_max_tokens=int(os.getenv("INTENT_DEEPSEEK_MAX_TOKENS") or os.getenv("DEEPSEEK_MAX_TOKENS", "8192")),
        deepseek_reasoning_effort=os.getenv("INTENT_DEEPSEEK_REASONING_EFFORT") or os.getenv("DEEPSEEK_REASONING_EFFORT", "none"),
        ollama_url=os.getenv("INTENT_OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("INTENT_OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL", ""),
        ollama_timeout=float(os.getenv("INTENT_OLLAMA_TIMEOUT") or os.getenv("OLLAMA_TIMEOUT", "120")),
        ollama_temperature=float(os.getenv("INTENT_OLLAMA_TEMPERATURE") or os.getenv("OLLAMA_TEMPERATURE", "0.2")),
        ollama_log_path=os.getenv("INTENT_OLLAMA_LOG_PATH", "logs/intent_responses.json"),
    )
    parser.add_argument("--orchestration-service-url", default=os.getenv("ORCHESTRATION_SERVICE_URL", "http://127.0.0.1:18002"))
    parser.add_argument("--orchestration-timeout", type=float, default=float(os.getenv("ORCHESTRATION_SERVICE_TIMEOUT", "600")))
    return parser


def build_intent_agent(args: argparse.Namespace, parser: argparse.ArgumentParser) -> IntentAgentPort:
    if args.backend == "fixed":
        return FixedIntentAgent(parameter_policy=ParameterPolicy.STRICT)
    try:
        client = make_model_client(args, purpose="intent")
    except PlanningError as exc:
        parser.error(str(exc))
    return OllamaIntentAgent(client, parameter_policy=ParameterPolicy.STRICT)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.request is not None:
        parser.error("请通过 HTTP 提交业务请求")
    if not args.remote_agents and args.fake_outcome is None:
        args.fake_outcome = "succeeded"
    token = (os.getenv("INTENT_API_TOKEN") or os.getenv("ORCHESTRATOR_API_TOKEN", "")).strip()
    if not token:
        parser.error("请配置 INTENT_API_TOKEN 或 ORCHESTRATOR_API_TOKEN")
    build_intent_agent(args, parser)
    def make_executor():
        executor, _ = build_executor(args, parser, knowledge_sender="意图识别智能体")
        return executor

    # Validate business adapters and orchestration transport before binding.
    make_executor()
    orchestration_token = (os.getenv("ORCHESTRATOR_API_TOKEN") or os.getenv("INTENT_API_TOKEN", "")).strip()
    OrchestrationHttpClient(args.orchestration_service_url, orchestration_token, args.orchestration_timeout)

    def make_router(agent, on_submitted):
        return RequestRouter(
            intent_agent=agent, executor=make_executor(),
            orchestrator=OrchestrationHttpClient(
                args.orchestration_service_url, orchestration_token,
                args.orchestration_timeout, on_submitted=on_submitted,
            ),
            registry=build_default_registry(), enable_agent_registry=args.enable_agent_registry,
        )
    server = IntentHttpServer(
        (args.host, args.port), lambda: build_intent_agent(args, parser), token=token,
        runtime_info={"backend": args.model_provider if args.backend != "fixed" else "fixed",
                      "model_provider": args.model_provider if args.backend != "fixed" else None,
                      "model": (args.deepseek_model if args.model_provider == "deepseek" else args.ollama_model) if args.backend != "fixed" else None,
                      "orchestration_service": args.orchestration_service_url,
                      "agents": {"device_manager": "http_chat" if args.network_chat_url else "rpc" if args.remote_agents else "fake"}},
        router_factory=make_router,
        model_settings=ModelSettings(args, "intent") if args.backend != "fixed" else None,
    )
    host, port = server.server_address
    print(f"意图识别服务：http://{host}:{port}（PID {os.getpid()}，Ctrl+C 停止）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
