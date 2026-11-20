"""HTTP access to the standalone Agent registry for the command center."""

import argparse
import json
import logging
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from registry import (
    AGENT_ID, DEFAULT_PATH, AgentNotFound, RegistryConflict, load_registry,
    save_agent, summarize_agents, unique_object,
)

MAX_BODY_BYTES = 64 * 1024


class RegistryHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def authorized(self):
        if self.client_address[0] != self.server.allowed_ip:
            self.send_json(403, {"detail": "此地址不能访问 Agent 仓库"})
            return False
        return True

    def read_payload(self):
        if self.headers.get_content_type() != "application/json":
            self.send_json(415, {"detail": "请使用 application/json"})
            return None
        length = self.headers.get("Content-Length", "")
        if self.headers.get("Transfer-Encoding") or not length.isascii() or not length.isdecimal() or len(length) > 9:
            self.send_json(400, {"detail": "请求长度无效"})
            return None
        size = int(length)
        if size > MAX_BODY_BYTES:
            self.send_json(413, {"detail": "单条 Agent 记录不能超过 64 KB"})
            return None
        try:
            raw = self.rfile.read(size)
            payload = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
            if len(raw) != size or not isinstance(payload, dict):
                raise ValueError("请求必须是完整的 JSON 对象")
        except (OSError, ValueError) as error:
            self.send_json(400, {"detail": f"请求格式无效：{error}"})
            return None
        return payload

    def handle_registry(self, method):
        if not self.authorized():
            return
        path = self.path.split("?", 1)[0]
        prefix = "/agents/"
        agent_id = path[len(prefix):] if path.startswith(prefix) else None
        collection = path == "/agents"
        if not collection and (agent_id is None or not AGENT_ID.fullmatch(agent_id)):
            self.send_json(404, {"detail": "Agent 仓库路径无效"})
            return
        if method == "GET":
            self.read_agents(agent_id)
        elif method == "POST" and collection:
            self.write_agent(None)
        elif method == "PUT" and agent_id is not None:
            self.write_agent(agent_id)
        else:
            self.send_json(405, {"detail": "此路径不支持该操作"})

    def read_agents(self, agent_id):
        try:
            agents = load_registry(self.server.registry_path)["agents"]
        except (OSError, ValueError):
            logging.exception("Unable to read Agent registry")
            self.send_json(503, {"detail": "Agent 仓库暂时无法读取"})
            return
        if agent_id is not None:
            record = agents.get(agent_id)
            if record is None:
                self.send_json(404, {"detail": f"找不到 Agent：{agent_id}"})
            else:
                self.send_json(200, record)
            return
        self.send_json(200, summarize_agents(agents))

    def write_agent(self, agent_id):
        payload = self.read_payload()
        if payload is None:
            return
        adding = agent_id is None
        fields = {"profile"} if adding else {"profile", "expected_revision"}
        if set(payload) != fields or not isinstance(payload["profile"], dict):
            self.send_json(400, {"detail": "请提交完整的 Agent 描述和该操作所需字段"})
            return
        profile = payload["profile"]
        if adding:
            agent_id = profile.get("agent_id")
        if not isinstance(agent_id, str) or not AGENT_ID.fullmatch(agent_id):
            self.send_json(400, {"detail": "Agent ID 必须以小写字母开头，只能包含小写字母、数字和下划线"})
            return
        revision = payload.get("expected_revision")
        if not adding and (type(revision) is not int or revision < 1):
            self.send_json(400, {"detail": "更新需要有效的记录版本"})
            return
        try:
            result = save_agent(self.server.registry_path, agent_id, profile, revision)
        except RegistryConflict as error:
            self.send_json(409, {"detail": str(error)})
        except AgentNotFound as error:
            self.send_json(404, {"detail": str(error)})
        except ValueError as error:
            self.send_json(400, {"detail": str(error)})
        except OSError:
            logging.exception("Unable to write Agent registry")
            self.send_json(503, {"detail": "Agent 仓库暂时无法写入"})
        else:
            self.send_json(201 if adding else 200, result)

    def do_GET(self):
        self.handle_registry("GET")

    def do_POST(self):
        self.handle_registry("POST")

    def do_PUT(self):
        self.handle_registry("PUT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--allow-ip", required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--client-certificate", type=Path, required=True)
    parser.add_argument("--file", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    path = args.file.resolve()
    load_registry(path)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(args.certificate, args.private_key)
    tls.load_verify_locations(cafile=args.client_certificate)
    tls.verify_mode = ssl.CERT_REQUIRED
    with RegistryServer((args.bind, args.port), RegistryHandler) as server:
        server.tls_context = tls
        server.allowed_ip = args.allow_ip
        server.registry_path = path
        server.serve_forever()


class RegistryServer(ThreadingHTTPServer):
    def process_request_thread(self, request, client_address):
        # A TLS handshake happens in the worker, so a slow client cannot block acceptance.
        try:
            request.settimeout(5)
            secure = self.tls_context.wrap_socket(request, server_side=True)
        except OSError:
            self.shutdown_request(request)
            return
        super().process_request_thread(secure, client_address)


if __name__ == "__main__":
    main()
