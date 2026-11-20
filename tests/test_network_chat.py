import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from simple_orchestrator.models import HandlingMode, ResultStatus, WorkflowStatus
from simple_orchestrator.network_chat import NetworkChatAdapter
from simple_orchestrator.orchestration.main import build_orchestrator, build_parser


QUERY = "查询实验室1光照强度"
ANSWER = "实验室1当前光照强度为 321 lx。"


class NetworkChatTests(unittest.TestCase):
    def setUp(self):
        self.messages = []
        self.reply = (
            'data: {"type":"delta","text":"实验室1当前光照强度为 "}\n\n'
            'data: {"type":"delta","text":"321 lx。"}\n\n'
            'data: {"type":"done"}\n\n'
        ).encode("utf-8")
        case = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                case.messages.append((self.path, payload))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(case.reply)))
                self.end_headers()
                self.wfile.write(case.reply)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}/api/chat"
        self.messages_endpoint = f"http://127.0.0.1:{self.server.server_port}/api/agent/messages"
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def test_direct_and_planned_sensor_queries_use_the_real_http_adapter(self):
        self.reply = (
            'data: {"runId":"test-run"}\n\n'
            'data: {"step":{"title":"查询读数"}}\n\n'
            'data: {"delta":"实验室1当前光照强度为 "}\n\n'
            'data: {"delta":"321 lx。"}\n\n'
            'data: {"done":true,"agentId":"query"}\n\n'
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "messages.jsonl"
            parser = build_parser()
            args = parser.parse_args([
                "--planner", "fixed", "--local-intent", "--parameter-policy", "strict",
                "--fake-outcome", "succeeded", "--network-chat-url", self.messages_endpoint,
                "--agent-log-path", str(log),
            ])
            router, _ = build_orchestrator(args, parser)
            self.assertIsInstance(router.executor.device_manager, NetworkChatAdapter)
            for request, mode in (
                (QUERY, HandlingMode.DIRECT),
                (f"请动态编排：{QUERY}", HandlingMode.ORCHESTRATED),
            ):
                with self.subTest(mode=mode):
                    state = router.run(request)
                    self.assertEqual(mode, state["handling_mode"])
                    self.assertEqual(WorkflowStatus.SUCCEEDED, state["status"])
                    self.assertIn(ANSWER, state["final_answer"])
                    result = next(iter(state["results"].values()))[-1].result
                    self.assertEqual({"response_received": True, "run_id": "test-run", "answer": ANSWER}, result.data)
            self.assertEqual(2, len(self.messages))
            for path, payload in self.messages:
                self.assertEqual("/api/agent/messages", path)
                self.assertEqual({"message", "agentId", "sessionId"}, set(payload))
                self.assertEqual("query", payload["agentId"])
                self.assertIn(QUERY, payload["message"])
            self.assertNotEqual(self.messages[0][1]["sessionId"], self.messages[1][1]["sessionId"])
            entries = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(["request", "response", "request", "response"], [e["event"] for e in entries])
            self.assertTrue(all(e["transport"] == "http_chat" and e["agent_id"] == "device_manager" for e in entries))

    def test_legacy_chat_endpoint_keeps_its_stream_request_and_reply_contract(self):
        result = NetworkChatAdapter(self.endpoint).invoke({"message": QUERY})
        self.assertEqual([("/api/chat", {"message": QUERY, "stream": True})], self.messages)
        self.assertEqual(ResultStatus.SUCCEEDED, result.status)
        self.assertEqual(ANSWER, result.data["answer"])

    def test_completed_stream_without_verified_reading_reports_failure(self):
        upstream_reply = "本次未取得有效的查询结果。"
        terminal = {"done": True, "reply": upstream_reply, "verification": {"status": "partial"}}
        self.reply = ("data: " + json.dumps(terminal, ensure_ascii=False) + "\n\n").encode("utf-8")
        result = NetworkChatAdapter(self.messages_endpoint).invoke({"message": QUERY})
        self.assertEqual(ResultStatus.FAILED, result.status)
        self.assertEqual("network_query_unverified", result.error)
        self.assertTrue(result.data["response_received"])
        self.assertNotIn("answer", result.data)
        self.assertEqual(upstream_reply, result.summary)

    def test_cancelled_stream_does_not_return_partial_text(self):
        self.reply = b'data: {"delta":"321"}\n\ndata: {"done":true,"cancelled":true}\n\n'
        result = NetworkChatAdapter(self.messages_endpoint).invoke({"message": QUERY})
        self.assertEqual(ResultStatus.REJECTED, result.status)
        self.assertNotIn("answer", result.data)

    def test_unterminated_stream_does_not_return_a_partial_reading_as_success(self):
        self.reply = 'data: {"type":"delta","text":"321"}\n\n'.encode("utf-8")
        result = NetworkChatAdapter(self.endpoint, timeout=1).invoke({"message": QUERY})
        self.assertEqual(ResultStatus.FAILED, result.status)
        self.assertNotIn("answer", result.data)
        self.assertIn("未收到最终回复", result.summary)

    def test_connection_refusal_is_reported_without_a_simulated_reading(self):
        adapter = NetworkChatAdapter(self.endpoint, timeout=1)
        with patch("simple_orchestrator.network_chat.build_opener") as build_opener:
            build_opener.return_value.open.side_effect = URLError(ConnectionRefusedError())
            result = adapter.invoke({"message": QUERY})
        self.assertEqual(ResultStatus.FAILED, result.status)
        self.assertIn("连接被拒绝", result.summary)
        self.assertEqual({}, result.data)
