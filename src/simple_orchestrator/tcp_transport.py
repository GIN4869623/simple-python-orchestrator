"""Length-prefixed TCP transport for the existing tasks.send JSON-RPC contract.

Each connection carries one UTF-8 JSON request and one UTF-8 JSON response.
The first four bytes of each message are its unsigned big-endian byte length.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .remote import JsonRpcAgentAdapter, RemoteCallError


MAX_MESSAGE_BYTES = 1024 * 1024


def send_json(connection: socket.socket, message: Mapping[str, Any]) -> None:
    data = json.dumps(message, ensure_ascii=False).encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("TCP JSON 消息超过 1 MiB 限制")
    connection.sendall(struct.pack("!I", len(data)) + data)


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise RemoteCallError("TCP 连接在消息接收完成前关闭")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_json(
    connection: socket.socket, *, on_body: Callable[[bytes], None] | None = None,
) -> dict[str, Any]:
    size = struct.unpack("!I", _recv_exact(connection, 4))[0]
    if size == 0 or size > MAX_MESSAGE_BYTES:
        raise RemoteCallError("TCP JSON 消息长度无效或超过 1 MiB 限制")
    body = _recv_exact(connection, size)
    if on_body is not None:
        on_body(body)
    try:
        message = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteCallError("TCP 消息不是有效的 UTF-8 JSON") from exc
    if not isinstance(message, dict):
        raise RemoteCallError("TCP JSON 消息必须是对象")
    return message


def tcp_request(
    host: str, port: int, message: Mapping[str, Any], timeout: float = 30.0, *,
    on_response: Callable[[bytes], None] | None = None,
) -> dict[str, Any]:
    if timeout <= 0:
        raise ValueError("TCP 超时必须大于 0")
    try:
        with socket.create_connection((host, port), timeout=timeout) as connection:
            connection.settimeout(timeout)
            send_json(connection, message)
            return recv_json(connection, on_body=on_response)
    except (OSError, TimeoutError) as exc:
        raise RemoteCallError(f"TCP 调用结果未知：{exc}") from exc


@dataclass(frozen=True)
class TcpJsonRpcAgentAdapter(JsonRpcAgentAdapter):
    """Use a fixed tcp://host:port endpoint with the same JSON-RPC payload."""

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("TCP 地址需要有效端口号") from exc
        if (parsed.scheme != "tcp" or not parsed.hostname or port is None
                or parsed.path or parsed.query or parsed.fragment or parsed.username
                or parsed.password):
            raise ValueError("TCP 地址必须是 tcp://主机:端口")
        if self.timeout <= 0:
            raise ValueError("远程智能体超时必须大于 0")
        if self.auth_token:
            raise ValueError("TCP 适配器不支持 HTTP Bearer Token")

    def _exchange(self, payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> Any:
        parsed = urlsplit(self.endpoint)
        if self.message_log is None:
            return tcp_request(parsed.hostname or "", parsed.port or 0, payload, self.timeout)
        return tcp_request(
            parsed.hostname or "", parsed.port or 0, payload, self.timeout,
            on_response=lambda body: self._log_message("response", payload, metadata, body=body),
        )


def serve_tcp(
    host: str, port: int, handler: Callable[[dict[str, Any]], Mapping[str, Any]],
    *, timeout: float = 30.0, max_requests: int | None = None,
) -> None:
    """Serve one request per connection; handler supplies the JSON-RPC response."""
    with socket.create_server((host, port), reuse_port=False) as server:
        served = 0
        while max_requests is None or served < max_requests:
            with server.accept()[0] as connection:
                connection.settimeout(timeout)
                try:
                    request = recv_json(connection)
                    response = handler(request)
                    send_json(connection, response)
                except (RemoteCallError, OSError, ValueError) as exc:
                    print(f"接收失败：{exc}")
            served += 1


def demo_handler(request: dict[str, Any]) -> dict[str, Any]:
    """Display the incoming task and acknowledge it without executing it."""
    print(json.dumps(request, ensure_ascii=False, indent=2), flush=True)
    request_id = request.get("id")
    params = request.get("params")
    if request.get("jsonrpc") != "2.0" or request.get("method") != "tasks.send" or not isinstance(params, dict):
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32600, "message": "无效的 tasks.send 请求"}}
    id_parts = str(request_id).rsplit(":", 2)
    task_id = params.get("task_id") or (id_parts[-2] if len(id_parts) == 3 else None)
    return {
        "jsonrpc": "2.0", "id": request_id,
        "result": {
            "task_id": task_id, "status": "rejected",
            "summary": "示例接收端已收到任务，未执行业务操作", "data": {}, "error": None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TCP JSON-RPC 任务收发示例")
    commands = parser.add_subparsers(dest="command", required=True)
    sender = commands.add_parser("send", help="发送 JSON 文件并打印响应")
    sender.add_argument("host")
    sender.add_argument("port", type=int)
    sender.add_argument("json_file", type=Path)
    sender.add_argument("--timeout", type=float, default=30.0)
    receiver = commands.add_parser("serve", help="启动只确认接收的示例服务端")
    receiver.add_argument("--host", default="127.0.0.1")
    receiver.add_argument("--port", type=int, required=True)
    receiver.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.command == "send":
        try:
            request = json.loads(args.json_file.read_text(encoding="utf-8"))
            if not isinstance(request, dict):
                parser.error("请求 JSON 必须是对象")
            response = tcp_request(args.host, args.port, request, args.timeout)
        except (OSError, ValueError, json.JSONDecodeError, RemoteCallError) as exc:
            parser.exit(1, f"发送失败：{exc}\n")
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        print(f"正在监听 {args.host}:{args.port}", flush=True)
        serve_tcp(args.host, args.port, demo_handler, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
