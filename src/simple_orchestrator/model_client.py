"""Model provider selection for intent recognition and dynamic orchestration."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen, build_opener, ProxyHandler

from .errors import PlanningError
from .ollama import OllamaClient


@dataclass
class DeepSeekClient(OllamaClient):
    base_url: str = "https://api.deepseek.com"
    api_key: str = field(default="", repr=False)
    max_tokens: int = 8192
    reasoning_effort: str = "none"
    proxy_url: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if (parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise PlanningError("DeepSeek API 地址无效")
        if self.proxy_url:
            proxy = urlsplit(self.proxy_url)
            if proxy.scheme not in {"http", "https"} or not proxy.hostname or proxy.username or proxy.password or proxy.query or proxy.fragment:
                raise PlanningError("DeepSeek 代理地址无效")
        if not self.api_key.strip() or any(c.isspace() for c in self.api_key):
            raise PlanningError("请在后端配置 DEEPSEEK_API_KEY")
        if not self.model.strip() or self.timeout <= 0 or not 0 <= self.temperature <= 2:
            raise PlanningError("DeepSeek 模型名、超时或温度配置无效")
        if not 1 <= self.max_tokens <= 393216 or self.reasoning_effort not in {"none", "low", "high", "max"}:
            raise PlanningError("DeepSeek 输出长度或推理强度配置无效")

    def _chat(self, *, instructions: str, user_content: str, schema: dict[str, Any],
              user_request: str, phase: str) -> Any:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions + "\n只返回一个符合所给 JSON Schema 的 JSON 对象。"},
                {"role": "user", "content": user_content + "\nJSON Schema:\n" + json.dumps(schema, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"}, "stream": False,
            "max_tokens": self.max_tokens, "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "thinking": {"type": "disabled" if self.reasoning_effort == "none" else "enabled"},
        }
        request = Request(self.base_url.rstrip('/') + '/chat/completions',
            data=json.dumps(body, ensure_ascii=False).encode('utf-8'), method='POST',
            headers={'Content-Type': 'application/json', 'Accept': 'application/json',
                     'Authorization': 'Bearer ' + self.api_key})
        try:
            open_request = build_opener(ProxyHandler({'https': self.proxy_url, 'http': self.proxy_url})).open if self.proxy_url else urlopen
            with open_request(request, timeout=self.timeout) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024:
                    raise PlanningError("DeepSeek 响应超过接收限制")
                envelope = json.loads(raw.decode('utf-8'))
        except HTTPError as exc:
            # Upstream bodies may reflect credentials; expose only the status.
            reasons = {401: '密钥认证失败', 402: '账户余额不足', 403: '访问被拒绝', 429: '请求过于频繁'}
            raise PlanningError(f"DeepSeek 请求失败（HTTP {exc.code}）：{reasons.get(exc.code, '请检查服务端配置或稍后重试')}") from None
        except (URLError, TimeoutError, OSError):
            raise PlanningError(f"DeepSeek {phase} 连接失败或超过 {self.timeout:g} 秒") from None
        except (ValueError, UnicodeError):
            raise PlanningError("DeepSeek 返回的响应不是有效 JSON") from None
        try:
            choice = envelope['choices'][0]
            if choice.get('finish_reason') != 'stop':
                raise PlanningError("DeepSeek 未返回完整结果，请检查输出长度或服务状态")
            content = choice['message']['content']
            if not isinstance(content, str) or not content.strip() or not isinstance(json.loads(content), dict):
                raise ValueError('invalid content')
        except (KeyError, IndexError, TypeError, ValueError):
            raise PlanningError("DeepSeek 未返回有效的 JSON 对象") from None
        # Normalize only the final answer; reasoning and credentials never enter task results.
        normalized = {"message": {"content": content}, "provider": "deepseek",
                      "model": envelope.get('model', self.model), "usage": envelope.get('usage', {})}
        self._log_response(user_request, normalized, phase=phase)
        return normalized


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--model-provider', choices=['ollama', 'deepseek'],
                        default=os.getenv('ORCHESTRATOR_MODEL_PROVIDER') or os.getenv('MODEL_PROVIDER', 'ollama'))
    parser.add_argument('--deepseek-url', default=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'))
    parser.add_argument('--deepseek-proxy-url', default=os.getenv('DEEPSEEK_PROXY_URL', ''))
    parser.add_argument('--deepseek-model', default=os.getenv('DEEPSEEK_MODEL', 'deepseek-flash'))
    parser.add_argument('--deepseek-timeout', type=float, default=float(os.getenv('DEEPSEEK_TIMEOUT', '120')))
    parser.add_argument('--deepseek-temperature', type=float, default=float(os.getenv('DEEPSEEK_TEMPERATURE', '0.2')))
    parser.add_argument('--deepseek-max-tokens', type=int, default=int(os.getenv('DEEPSEEK_MAX_TOKENS', '8192')))
    parser.add_argument('--deepseek-reasoning-effort', choices=['none', 'low', 'high', 'max'],
                        default=os.getenv('DEEPSEEK_REASONING_EFFORT', 'none'))


def make_model_client(args: argparse.Namespace, *, purpose: str, provider: str | None = None) -> OllamaClient:
    provider = provider or args.model_provider
    if provider == 'deepseek':
        prefix = 'INTENT_' if purpose == 'intent' else 'ORCHESTRATOR_'
        key = os.getenv(prefix + 'DEEPSEEK_API_KEY') or os.getenv('DEEPSEEK_API_KEY', '')
        return DeepSeekClient(model=args.deepseek_model, base_url=args.deepseek_url, api_key=key,
            timeout=args.deepseek_timeout, temperature=args.deepseek_temperature,
            log_path=args.ollama_log_path, max_tokens=args.deepseek_max_tokens,
            reasoning_effort=args.deepseek_reasoning_effort, proxy_url=args.deepseek_proxy_url)
    if provider != 'ollama':
        raise PlanningError('未知模型提供方')
    if not args.ollama_model.strip():
        raise PlanningError('请配置本地 OLLAMA_MODEL 或选择已配置的云端模型')
    return OllamaClient(model=args.ollama_model, base_url=args.ollama_url,
                        timeout=args.ollama_timeout, temperature=args.ollama_temperature,
                        log_path=args.ollama_log_path)


@dataclass
class ModelSettings:
    args: argparse.Namespace
    purpose: str

    def snapshot(self) -> dict[str, Any]:
        providers = []
        for provider, label, model in [('ollama', '本地 Ollama', self.args.ollama_model),
                                        ('deepseek', '云端 DeepSeek', self.args.deepseek_model)]:
            error = None
            try:
                make_model_client(self.args, purpose=self.purpose, provider=provider)
            except PlanningError as exc:
                error = str(exc)
            providers.append({'id': provider, 'label': label, 'model': model,
                              'configured': error is None, 'detail': error})
        selected = next(p for p in providers if p['id'] == self.args.model_provider)
        return {'provider': self.args.model_provider, 'model': selected['model'], 'providers': providers}

    def select(self, provider: str) -> dict[str, Any]:
        if provider not in {'ollama', 'deepseek'}:
            raise PlanningError('只能选择本地 Ollama 或云端 DeepSeek')
        make_model_client(self.args, purpose=self.purpose, provider=provider)
        self.args.model_provider = provider
        return self.snapshot()
