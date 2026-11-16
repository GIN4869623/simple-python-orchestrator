import struct
import unittest
from unittest.mock import patch

from simple_orchestrator.models import ResultStatus
from simple_orchestrator.remote import RemoteCallError
from simple_orchestrator.task_messages import invocation_metadata, make_agent_request
from simple_orchestrator.tcp_transport import (
    TcpJsonRpcAgentAdapter,
    demo_handler,
    recv_json,
    send_json,
)


class MemoryConnection:
    def __init__(self, data: bytes = b"") -> None:
        self.data = bytearray(data)

    def sendall(self, data: bytes) -> None:
        self.data.extend(data)

    def recv(self, size: int) -> bytes:
        # Simulate a TCP stream that returns fewer bytes than requested.
        chunk = bytes(self.data[:min(size, 3)])
        del self.data[:len(chunk)]
        return chunk


class TcpTransportTests(unittest.TestCase):
    def test_framing_preserves_unicode_and_rejects_truncated_message(self) -> None:
        connection = MemoryConnection()
        send_json(connection, {"task": "查询设备"})
        self.assertEqual({"task": "查询设备"}, recv_json(connection))
        with self.assertRaises(RemoteCallError):
            recv_json(MemoryConnection(struct.pack("!I", 10) + b"{}"))

    @patch("simple_orchestrator.tcp_transport.tcp_request")
    def test_adapter_sends_contract_and_correlates_response(self, request) -> None:
        sent = []

        def reply(host, port, payload, timeout):
            sent.append((host, port, payload, timeout))
            return demo_handler(payload)

        request.side_effect = reply
        adapter = TcpJsonRpcAgentAdapter("tcp://127.0.0.1:9000", "device_manager")
        context = {
            "request_id": "req-1", "user_request": "查询设备",
            "current_task": {
                "task_id": "T1", "attempt": 1, "description": "查询设备",
                "capability": "device.list", "inputs": {},
            },
        }
        business_request = make_agent_request(context)
        result = adapter.invoke(business_request, metadata=invocation_metadata(context))
        self.assertEqual(ResultStatus.REJECTED, result.status)
        self.assertEqual(("127.0.0.1", 9000), sent[0][:2])
        self.assertEqual("req-1:T1:1", sent[0][2]["id"])
        self.assertEqual("tasks.send", sent[0][2]["method"])
        self.assertEqual({"message"}, set(sent[0][2]["params"]))
        self.assertEqual(business_request, sent[0][2]["params"])
        self.assertIn("请完成以下任务：查询设备", sent[0][2]["params"]["message"])
        self.assertNotIn("task", sent[0][2]["params"])
        self.assertNotIn("context", sent[0][2]["params"])
        request.side_effect = None
        request.return_value = {"jsonrpc": "2.0", "id": "wrong", "result": {"task_id": "T1", "status": "succeeded"}}
        with self.assertRaises(RemoteCallError):
            adapter.invoke({"request_id": "req-1", "current_task": {"task_id": "T1", "attempt": 1}})

    @patch("simple_orchestrator.tcp_transport.tcp_request")
    def test_knowledge_message_survives_tcp_framing(self, request) -> None:
        def reply(host, port, payload, timeout):
            connection = MemoryConnection()
            send_json(connection, payload)
            received = recv_json(connection)
            self.assertEqual("解释 QPSK", received["params"]["message"])
            self.assertNotIn("task", received["params"])
            self.assertNotIn("context", received["params"])
            return demo_handler(received)

        request.side_effect = reply
        adapter = TcpJsonRpcAgentAdapter("tcp://127.0.0.1:9000", "knowledge")
        result = adapter.invoke({
            "request_id": "req-2", "user_request": "解释 QPSK",
            "current_task": {
                "task_id": "K1", "attempt": 1, "description": "解释 QPSK",
                "inputs": {"request": "解释 QPSK"},
            },
        })
        self.assertEqual(ResultStatus.REJECTED, result.status)

    def test_bad_address_and_oversized_frame(self) -> None:
        for endpoint in ("tcp://host", "tcp://host:99999", "http://host:9000", "tcp://host:9000/path"):
            with self.assertRaises(ValueError):
                TcpJsonRpcAgentAdapter(endpoint, "knowledge")
        with self.assertRaises(RemoteCallError):
            recv_json(MemoryConnection(struct.pack("!I", 1024 * 1024 + 1)))


if __name__ == "__main__":
    unittest.main()
