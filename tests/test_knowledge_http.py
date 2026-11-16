import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from simple_orchestrator.agent_logging import AgentMessageLog
from simple_orchestrator.knowledge_http import KnowledgeHttpAdapter
from simple_orchestrator.models import ResultStatus
from simple_orchestrator.remote import RemoteCallError


class KnowledgeHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "messages.jsonl"
        self.adapter = KnowledgeHttpAdapter(message_log=AgentMessageLog(self.path))
        self.metadata = {"request_id": "workflow-1", "task_id": "K1", "attempt": 1, "attempt_id": "workflow-1:K1:1"}
        self.answer = "  发送前先扫描频段。\n再选择合适的频点。\n"
        self.response = {
            "request_id": "ignored-id", "sender": "知识库", "status": "success",
            "summary": "先扫描再选频", "data": {"knowledge": self.answer},
            "error": None, "unused": {"anything": True},
        }

    def mock_response(self, urlopen: MagicMock, body: bytes | None = None) -> None:
        response = urlopen.return_value.__enter__.return_value
        response.status = 200
        response.read.return_value = body if body is not None else json.dumps(self.response, ensure_ascii=False).encode()

    @patch("simple_orchestrator.knowledge_http.urlopen")
    def test_plain_json_contract_preserves_full_answer_and_logs(self, urlopen: MagicMock) -> None:
        self.mock_response(urlopen)
        result = self.adapter.invoke({"message": "  发送前需要做什么？  "}, metadata=self.metadata)
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual("POST", request.get_method())
        self.assertEqual(self.adapter.endpoint, request.full_url)
        self.assertEqual({"request_id", "sender", "question", "knowledge_domain_ids", "top_k", "model_profile"}, set(payload))
        self.assertEqual("发送前需要做什么？", payload["question"])
        self.assertEqual("编排智能体", payload["sender"])
        self.assertEqual([], payload["knowledge_domain_ids"])
        self.assertEqual(3, payload["top_k"])
        self.assertIs(type(payload["top_k"]), int)
        self.assertEqual("local", payload["model_profile"])
        self.assertTrue(1 <= len(payload["request_id"]) <= 128)
        self.assertEqual(ResultStatus.SUCCEEDED, result.status)
        self.assertEqual("先扫描再选频", result.summary)
        self.assertEqual({"knowledge": self.answer}, result.data)
        self.assertIsNone(result.error)
        entries = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(["request", "response"], [e["event"] for e in entries])
        self.assertEqual(self.metadata["attempt_id"], entries[0]["attempt_id"])
        self.assertEqual(payload["request_id"], entries[0]["knowledge_request_id"])
        self.assertEqual(self.response, entries[1]["payload"])
        self.assertEqual(self.answer, json.loads(entries[1]["raw_body"])["data"]["knowledge"])

    @patch("simple_orchestrator.knowledge_http.urlopen")
    def test_deployed_success_status_ignores_retrieval_fields(self, urlopen: MagicMock) -> None:
        envelope = {
            **self.response, "status": "success", "answer": "不读取此字段",
            "hits": [{"text": "不读取检索切片"}], "sources": [], "elapsed_ms": 16000,
        }
        self.mock_response(urlopen, json.dumps(envelope, ensure_ascii=False).encode())
        result = self.adapter.invoke({"message": "问题"})
        self.assertEqual(ResultStatus.SUCCEEDED, result.status)
        self.assertEqual(self.response["summary"], result.summary)
        self.assertEqual({"knowledge": self.answer}, result.data)
        self.assertIsNone(result.error)

    @patch("simple_orchestrator.knowledge_http.urlopen")
    def test_repeated_calls_have_new_request_ids(self, urlopen: MagicMock) -> None:
        self.mock_response(urlopen)
        for _ in range(2):
            self.adapter.invoke({"message": "问题"}, metadata=self.metadata)
        ids = [json.loads(call.args[0].data)["request_id"] for call in urlopen.call_args_list]
        self.assertEqual(2, len(set(ids)))

    @patch("simple_orchestrator.knowledge_http.urlopen")
    def test_api_key_header_is_not_logged_or_in_repr(self, urlopen: MagicMock) -> None:
        self.mock_response(urlopen)
        adapter = KnowledgeHttpAdapter(api_key="test-secret", message_log=AgentMessageLog(self.path))
        adapter.invoke({"message": "问题"})
        self.assertEqual("test-secret", urlopen.call_args.args[0].get_header("X-knowledge-api-key"))
        self.assertIsNone(urlopen.call_args.args[0].get_header("X-api-key"))
        self.assertIsNone(urlopen.call_args.args[0].get_header("Authorization"))
        self.assertNotIn("test-secret", self.path.read_text())
        self.assertNotIn("test-secret", repr(adapter))

    @patch("simple_orchestrator.knowledge_http.urlopen")
    def test_question_limits_do_not_truncate_or_send_invalid_requests(self, urlopen: MagicMock) -> None:
        self.mock_response(urlopen)
        self.adapter.invoke({"message": "问" * 2000})
        self.assertEqual("问" * 2000, json.loads(urlopen.call_args.args[0].data)["question"])
        urlopen.reset_mock()
        for request in ({"message": " \n "}, {"message": "问" * 2001}, {"message": 1}, {"message": "问题", "extra": 1}):
            with self.subTest(request_length=len(str(request))):
                with self.assertRaises(ValueError):
                    self.adapter.invoke(request)
        urlopen.assert_not_called()

    def test_config_constraints(self) -> None:
        for top_k in (True, "3", 0, 21):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                KnowledgeHttpAdapter(top_k=top_k)
        with self.assertRaises(ValueError):
            KnowledgeHttpAdapter(model_profile="other")

    @patch("simple_orchestrator.knowledge_http.urlopen")
    def test_business_failure_does_not_require_answer(self, urlopen: MagicMock) -> None:
        for status in ("failed", "rejected", "unknown"):
            with self.subTest(status=status):
                self.mock_response(urlopen, json.dumps({
                    "status": status, "summary": "服务无法回答", "data": {}, "error": "not_available",
                }).encode())
                result = self.adapter.invoke({"message": "问题"})
                self.assertEqual(ResultStatus(status), result.status)
                self.assertEqual("not_available", result.error)

    @patch("simple_orchestrator.knowledge_http.urlopen")
    def test_invalid_responses_fail_and_are_logged(self, urlopen: MagicMock) -> None:
        bodies = [
            b"not json", b"\xff", b"[]",
            json.dumps({**self.response, "status": "surprise"}).encode(),
            json.dumps({**self.response, "status": "answered"}).encode(),
            json.dumps({**self.response, "status": "succeeded"}).encode(),
            json.dumps({**self.response, "summary": 1}).encode(),
            json.dumps({**self.response, "data": {"knowledge": ["invalid"]}}).encode(),
            json.dumps({**self.response, "data": []}).encode(),
        ]
        for body in bodies:
            with self.subTest(body=body):
                self.mock_response(urlopen, body)
                with self.assertRaises(RemoteCallError):
                    self.adapter.invoke({"message": "问题"})
        entries = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(len(bodies), sum(e["event"] == "error" for e in entries))

    @patch("simple_orchestrator.knowledge_http.urlopen")
    def test_transport_errors_preserve_http_error_body(self, urlopen: MagicMock) -> None:
        for error in (URLError("unreachable"), TimeoutError("timeout"), HTTPError(
            self.adapter.endpoint, 422, "invalid", {}, BytesIO(b'{"detail":"invalid question"}'),
        )):
            with self.subTest(error=error):
                urlopen.side_effect = error
                with self.assertRaises(RemoteCallError):
                    self.adapter.invoke({"message": "问题"})
        entries = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(3, sum(e["event"] == "error" for e in entries))
        response = next(e for e in entries if e["event"] == "response")
        self.assertEqual(422, response["http_status"])
        self.assertEqual({"detail": "invalid question"}, response["payload"])


if __name__ == "__main__":
    unittest.main()
