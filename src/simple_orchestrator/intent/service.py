"""Independent intent-recognition service, default port 18003."""

from __future__ import annotations

import argparse
import logging
import os
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from typing import Any, Callable
from uuid import uuid4

from ..config import load_dotenv
from ..errors import PlanningError
from ..http_server import JsonRequestHandler
from ..models import ParameterPolicy
from ..ollama import OllamaClient
from ..ports import IntentAgentPort
from .agent import FixedIntentAgent, OllamaIntentAgent


class IntentHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, address: tuple[str, int], factory: Callable[[], IntentAgentPort],
        *, token: str, runtime_info: dict[str, Any],
    ) -> None:
        if not token:
            raise ValueError("意图服务必须配置 INTENT_API_TOKEN 或 ORCHESTRATOR_API_TOKEN")
        self.token = token
        self.factory = factory
        self.runtime_info = runtime_info
        self.analysis_lock = threading.Lock()
        super().__init__(address, IntentRequestHandler)


class IntentRequestHandler(JsonRequestHandler):
    server: IntentHttpServer
    authentication_error = "意图识别接口认证失败"

    def do_GET(self) -> None:
        if not self._authorized():
            return
        if self.path != "/health":
            self._json(HTTPStatus.NOT_FOUND, {"detail": "接口不存在"})
            return
        self._json(HTTPStatus.OK, {
            "status": "ok", "service": "intent", "pid": os.getpid(),
            "busy": self.server.analysis_lock.locked(), "runtime": self.server.runtime_info,
        })

    def do_POST(self) -> None:
        if not self._authorized():
            return
        if self.path != "/api/intent/analyze":
            self._json(HTTPStatus.NOT_FOUND, {"detail": "接口不存在"})
            return
        payload = self._read_json()
        if payload is None:
            return
        message = self._read_message(payload)
        if message is None:
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
    load_dotenv()
    parser = argparse.ArgumentParser(description="启动独立意图识别服务")
    parser.add_argument("--host", default=os.getenv("INTENT_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("INTENT_API_PORT", "18003")))
    parser.add_argument("--backend", choices=["ollama", "fixed"], default=os.getenv("INTENT_BACKEND", "ollama"))
    parser.add_argument("--ollama-url", default=os.getenv("INTENT_OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--ollama-model", default=os.getenv("INTENT_OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL", ""))
    parser.add_argument("--ollama-timeout", type=float, default=float(os.getenv("INTENT_OLLAMA_TIMEOUT") or os.getenv("OLLAMA_TIMEOUT", "120")))
    parser.add_argument("--ollama-temperature", type=float, default=float(os.getenv("INTENT_OLLAMA_TEMPERATURE") or os.getenv("OLLAMA_TEMPERATURE", "0.2")))
    parser.add_argument("--ollama-log-path", default=os.getenv("INTENT_OLLAMA_LOG_PATH", "logs/intent_responses.json"))
    return parser


def build_intent_agent(args: argparse.Namespace, parser: argparse.ArgumentParser) -> IntentAgentPort:
    if args.backend == "fixed":
        return FixedIntentAgent(parameter_policy=ParameterPolicy.STRICT)
    if not args.ollama_model.strip():
        parser.error("请配置 INTENT_OLLAMA_MODEL、OLLAMA_MODEL 或 --ollama-model")
    client = OllamaClient(
        model=args.ollama_model, base_url=args.ollama_url,
        timeout=args.ollama_timeout, temperature=args.ollama_temperature,
        log_path=args.ollama_log_path,
    )
    return OllamaIntentAgent(client, parameter_policy=ParameterPolicy.STRICT)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    token = (os.getenv("INTENT_API_TOKEN") or os.getenv("ORCHESTRATOR_API_TOKEN", "")).strip()
    if not token:
        parser.error("请配置 INTENT_API_TOKEN 或 ORCHESTRATOR_API_TOKEN")
    build_intent_agent(args, parser)
    server = IntentHttpServer(
        (args.host, args.port), lambda: build_intent_agent(args, parser), token=token,
        runtime_info={"backend": args.backend, "model": args.ollama_model if args.backend == "ollama" else None},
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
