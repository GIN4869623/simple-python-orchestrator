"""Stateless model connection shared by independently configured agents."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import PlanningError


@dataclass
class OllamaClient:
    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    temperature: float = 0.2
    log_path: str | Path | None = None
    max_log_entries: int = 20

    def _chat(
        self,
        *,
        instructions: str,
        user_content: str,
        schema: dict[str, Any],
        user_request: str,
        phase: str,
    ) -> Any:
        if not self.model.strip():
            raise PlanningError("未配置 Ollama 模型名")
        if not 0 <= self.temperature <= 2:
            raise PlanningError("Ollama temperature 必须在 0 到 2 之间")

        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_content},
            ],
            "format": schema,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        request = Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise PlanningError(
                f"Ollama 请求失败（HTTP {exc.code}）：{detail}"
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise PlanningError(
                    f"Ollama {phase} 阶段超过 {self.timeout:g} 秒"
                ) from exc
            raise PlanningError(
                f"无法连接 Ollama：{exc.reason}；请确认 Ollama 服务已启动"
            ) from exc
        except TimeoutError as exc:
            raise PlanningError(
                f"Ollama {phase} 阶段超过 {self.timeout:g} 秒"
            ) from exc
        except json.JSONDecodeError as exc:
            raise PlanningError("Ollama 返回的响应不是有效 JSON") from exc

        self._log_response(user_request, response_body, phase=phase)
        return response_body

    @staticmethod
    def _response_content_as_json(response_body: Any, content_name: str) -> Any:
        try:
            content = response_body["message"]["content"]
            return json.loads(content)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise PlanningError(f"Ollama 响应中没有有效的{content_name}") from exc

    def _log_response(
        self, user_request: str, response_body: Any, *, phase: str
    ) -> None:
        if self.log_path is None:
            return
        if self.max_log_entries < 1:
            raise PlanningError("LLM 日志保留数量必须大于 0")

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "user_request": user_request,
            "phase": phase,
            "response": response_body,
        }
        log_path = Path(self.log_path)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            previous_entries: deque[Any] = deque(
                self._read_log_entries(log_path), maxlen=self.max_log_entries
            )
            previous_entries.append(log_entry)
            log_document = {
                "max_entries": self.max_log_entries,
                "entries": list(previous_entries),
            }
            with log_path.open("w", encoding="utf-8") as log_file:
                json.dump(log_document, log_file, ensure_ascii=False, indent=2)
                log_file.write("\n")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise PlanningError(f"无法写入 LLM 日志 {log_path}：{exc}") from exc

    @staticmethod
    def _read_log_entries(log_path: Path) -> list[Any]:
        source_path = log_path
        legacy_path = log_path.with_suffix(".jsonl")
        if not source_path.exists() and legacy_path.exists():
            source_path = legacy_path
        if not source_path.exists():
            return []

        text = source_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            return [json.loads(line) for line in text.splitlines() if line.strip()]

        if isinstance(document, dict) and isinstance(document.get("entries"), list):
            return document["entries"]
        if isinstance(document, dict) and {
            "timestamp",
            "model",
            "user_request",
            "response",
        }.issubset(document):
            return [document]
        if isinstance(document, list):
            return document
        raise TypeError("日志文件必须是包含 entries 的 JSON 对象")
