import base64
import json
import struct
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from simple_orchestrator.agent_logging import AgentMessageLog, LoggedFakeAgent
from simple_orchestrator.fakes import FakeKnowledgePort
from simple_orchestrator.models import ResultStatus
from simple_orchestrator.remote import JsonRpcAgentAdapter, RemoteCallError
from simple_orchestrator.tcp_transport import TcpJsonRpcAgentAdapter
from simple_orchestrator.task_messages import invocation_metadata, make_agent_request


class AgentLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "nested" / "agent_messages.jsonl"
        self.log = AgentMessageLog(self.path)
        self.context = {
            "request_id": "req-1", "user_request": "查询设备",
            "current_task": {
                "task_id": "T1", "attempt": 1, "description": "查询设备",
                "capability": "device.list", "inputs": {},
            },
        }
        self.response = {
            "jsonrpc": "2.0", "id": "req-1:T1:1",
            "result": {
                "task_id": "T1", "status": "succeeded", "summary": "完成",
                "data": {"network_devices": [{"device_id": "设备甲", "status": "online"}]},
            },
        }

    def entries(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def http_response(self, urlopen, body):
        response = urlopen.return_value.__enter__.return_value
        response.status = 200
        response.read.return_value = body

    @patch("simple_orchestrator.remote.urlopen")
    def test_http_logs_full_bodies_for_both_agents_and_appends(self, urlopen):
        body = json.dumps(self.response, ensure_ascii=False, indent=2).encode()
        self.http_response(urlopen, body)
        for agent_id in ("knowledge", "device_manager"):
            adapter = JsonRpcAgentAdapter(
                "http://agent/rpc", agent_id, auth_token="secret", message_log=self.log,
            )
            request = make_agent_request(self.context)
            self.assertEqual(ResultStatus.SUCCEEDED, adapter.invoke(
                request, metadata=invocation_metadata(self.context),
            ).status)
            sent = json.loads(urlopen.call_args.args[0].data)
            self.assertEqual(request, sent["params"])
            self.assertEqual({"message"}, set(sent["params"]))
            outgoing, incoming = self.entries()[-2:]
            self.assertEqual(sent, outgoing["payload"])
            self.assertEqual(agent_id, outgoing["agent_id"])
            self.assertEqual("req-1:T1:1", outgoing["attempt_id"])
            self.assertEqual(self.response, incoming["payload"])
            self.assertEqual(body.decode(), incoming["raw_body"])
            self.assertEqual(200, incoming["http_status"])
        self.assertEqual(4, len(self.entries()))
        self.assertNotIn("secret", self.path.read_text())

    @patch("simple_orchestrator.remote.urlopen")
    def test_http_records_invalid_response_before_validation(self, urlopen):
        self.http_response(urlopen, b' {"jsonrpc":"2.0", "id":"wrong"} ')
        adapter = JsonRpcAgentAdapter("http://agent/rpc", "knowledge", message_log=self.log)
        with self.assertRaises(RemoteCallError):
            adapter.invoke(self.context)
        entries = self.entries()
        self.assertEqual(["request", "response", "error"], [e["event"] for e in entries])
        self.assertEqual("wrong", entries[1]["payload"]["id"])
        self.assertIn("响应标识不匹配", entries[2]["error"])

    @patch("simple_orchestrator.remote.urlopen")
    def test_http_records_unparseable_and_non_utf8_bodies(self, urlopen):
        adapter = JsonRpcAgentAdapter("http://agent/rpc", "knowledge", message_log=self.log)
        for body in (b'not JSON', b'\xff\xfe\xff'):
            self.http_response(urlopen, body)
            with self.assertRaises(RemoteCallError):
                adapter.invoke(self.context)
            incoming = self.entries()[-2]
            if body == b'not JSON':
                self.assertEqual("not JSON", incoming["raw_body"])
            else:
                self.assertEqual(body, base64.b64decode(incoming["raw_body_base64"]))

    @patch("simple_orchestrator.remote.urlopen")
    def test_http_error_body_and_connection_failure_are_recorded(self, urlopen):
        adapter = JsonRpcAgentAdapter("http://agent/rpc", "device_manager", message_log=self.log)
        urlopen.side_effect = HTTPError("http://agent/rpc", 503, "unavailable", {}, BytesIO(b'busy'))
        with self.assertRaises(RemoteCallError):
            adapter.invoke(self.context)
        entries = self.entries()
        self.assertEqual(503, entries[1]["http_status"])
        self.assertEqual("busy", entries[1]["raw_body"])
        urlopen.side_effect = URLError("connection refused")
        with self.assertRaises(RemoteCallError):
            adapter.invoke(self.context)
        self.assertEqual(["request", "error"], [e["event"] for e in self.entries()[-2:]])

    @patch("simple_orchestrator.tcp_transport.socket.create_connection")
    def test_tcp_logs_actual_frame_body_and_invalid_json(self, connect):
        connection = connect.return_value.__enter__.return_value
        adapter = TcpJsonRpcAgentAdapter("tcp://agent:9000", "device_manager", message_log=self.log)
        for body in (json.dumps(self.response, ensure_ascii=False).encode(), b'invalid JSON'):
            buffer = bytearray(struct.pack("!I", len(body)) + body)

            def recv(size):
                chunk = bytes(buffer[:min(size, 3)])
                del buffer[:len(chunk)]
                return chunk

            connection.recv.side_effect = recv
            if body == b'invalid JSON':
                with self.assertRaises(RemoteCallError):
                    adapter.invoke(self.context)
                incoming = self.entries()[-2]
            else:
                self.assertEqual(ResultStatus.SUCCEEDED, adapter.invoke(self.context).status)
                incoming = self.entries()[-1]
                frame = connection.sendall.call_args.args[0]
                self.assertEqual(json.loads(frame[4:]), self.entries()[-2]["payload"])
            self.assertEqual(body.decode(), incoming["raw_body"])
            self.assertEqual("tcp", incoming["transport"])

    @patch("simple_orchestrator.remote.urlopen")
    def test_usrp_is_excluded_even_if_logger_is_attached(self, urlopen):
        self.http_response(urlopen, json.dumps(self.response).encode())
        JsonRpcAgentAdapter("http://agent/rpc", "usrp", message_log=self.log).invoke(self.context)
        self.assertFalse(self.path.exists())

    def test_fake_logs_actual_input_and_result_including_rejection(self):
        agent = LoggedFakeAgent(FakeKnowledgePort(outcome=ResultStatus.REJECTED), "knowledge", self.log)
        request = make_agent_request(self.context)
        result = agent.invoke(request, metadata=invocation_metadata(self.context))
        outgoing, incoming = self.entries()
        self.assertEqual("fake", outgoing["transport"])
        self.assertEqual(request, outgoing["payload"])
        self.assertEqual({"message"}, set(outgoing["payload"]))
        self.assertEqual("T1", outgoing["task_id"])
        self.assertEqual(result.data, incoming["payload"]["data"])
        self.assertEqual("rejected", incoming["payload"]["status"])
        self.assertEqual(result.error, incoming["payload"]["error"])
        # Preserve the knowledge skill-reading interface used by the USRP Fake.
        self.assertTrue(callable(agent.read_skill))

    def test_parallel_writers_keep_every_entry(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: self.log.record(
                "knowledge" if i % 2 else "device_manager", "request",
                {"attempt_id": str(i)}, payload={"message": "中文" * 2048},
            ), range(40)))
        entries = self.entries()
        self.assertEqual(40, len(entries))
        self.assertEqual({str(i) for i in range(40)}, {e["attempt_id"] for e in entries})

    def test_unwritable_log_warns_without_failing_agent_call(self):
        self.path.parent.mkdir()
        agent = LoggedFakeAgent(FakeKnowledgePort(), "knowledge", AgentMessageLog(self.path.parent))
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(ResultStatus.SUCCEEDED, agent.invoke(
                make_agent_request(self.context), metadata=invocation_metadata(self.context),
            ).status)


if __name__ == "__main__":
    unittest.main()
