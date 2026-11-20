"""Small authenticated JSON HTTP primitives shared by the two services."""

from __future__ import annotations

import hmac
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any

MAX_BODY_BYTES = 64 * 1024
MAX_REQUEST_CHARS = 2000


class JsonRequestHandler(BaseHTTPRequestHandler):
    authentication_error = "接口认证失败"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            logging.warning("HTTP client disconnected before receiving its result")

    def _authorized(self) -> bool:
        expected = f"Bearer {self.server.token}"
        value = self.headers.get("Authorization", "")
        if hmac.compare_digest(value.encode(), expected.encode()):
            return True
        self._json(HTTPStatus.UNAUTHORIZED, {"detail": self.authentication_error})
        return False

    def _model_get(self) -> None:
        settings = getattr(self.server, "model_settings", None)
        if settings is None:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"detail": "当前服务未启用模型切换"})
            return
        self._json(HTTPStatus.OK, settings.snapshot())

    def _model_post(self, locks: list) -> None:
        from .errors import PlanningError
        settings = getattr(self.server, "model_settings", None)
        if settings is None:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"detail": "当前服务未启用模型切换"})
            return
        payload = self._read_json()
        if payload is None:
            return
        if set(payload) != {"provider"} or not isinstance(payload["provider"], str):
            self._json(HTTPStatus.BAD_REQUEST, {"detail": "只接受 provider 模型提供方"})
            return
        acquired = []
        try:
            for lock in locks:
                if not lock.acquire(blocking=False):
                    self._json(HTTPStatus.CONFLICT, {"detail": "服务正在处理任务，请完成后再切换模型"})
                    return
                acquired.append(lock)
            try:
                data = settings.select(payload["provider"])
            except PlanningError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})
                return
            self.server.runtime_info.update(model_provider=data["provider"], model=data["model"])
            self.server.runtime_info["backend" if settings.purpose == "intent" else "planner"] = data["provider"]
            self._json(HTTPStatus.OK, data)
        finally:
            for lock in reversed(acquired):
                lock.release()

    def _read_json(self) -> dict[str, Any] | None:
        if self.headers.get_content_type() != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"detail": "请使用 application/json"})
            return None
        length_header = self.headers.get("Content-Length", "")
        if self.headers.get("Transfer-Encoding") or not length_header.isascii() or not length_header.isdecimal():
            self._json(HTTPStatus.BAD_REQUEST, {"detail": "需要有效的 Content-Length"})
            return None
        length = int(length_header)
        if length > MAX_BODY_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"detail": "请求体过大"})
            return None
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("incomplete body")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("expected object")
        except (ValueError, UnicodeDecodeError, OSError):
            self._json(HTTPStatus.BAD_REQUEST, {"detail": "请求必须是完整的 JSON 对象"})
            return None
        return payload

    def _read_message(self, payload: dict[str, Any]) -> str | None:
        message = payload.get("request")
        if not isinstance(message, str) or not 1 <= len(message.strip()) <= MAX_REQUEST_CHARS:
            self._json(HTTPStatus.BAD_REQUEST, {"detail": "request 必须是 1～2000 字符的任务文本"})
            return None
        return message.strip()
