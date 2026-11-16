import json
import threading
import time
import unittest
from http.client import HTTPConnection
from unittest.mock import Mock

from simple_orchestrator.orchestration.api import MAX_BODY_BYTES, OrchestratorHttpServer
from simple_orchestrator.fakes import FakeDeviceManagerPort, FakeKnowledgePort, FakeUsrpAgent
from simple_orchestrator.orchestration.graph import GraphOrchestrator
from simple_orchestrator.orchestration.execution import WorkflowExecutor
from simple_orchestrator.intent.agent import FixedIntentAgent
from simple_orchestrator.orchestration.router import RequestRouter
from simple_orchestrator.models import ParameterPolicy, ResultStatus
from simple_orchestrator.orchestration.planner import FixedPlanner, PlanningError


class ApiTests(unittest.TestCase):
    def setUp(self):
        def make_runtime():
            knowledge, device, usrp = FakeKnowledgePort(), FakeDeviceManagerPort(), FakeUsrpAgent()
            return RequestRouter(
                FixedIntentAgent(parameter_policy=ParameterPolicy.STRICT),
                WorkflowExecutor(knowledge, device, usrp),
                GraphOrchestrator(FixedPlanner(), knowledge, device, usrp),
            )
        self.factory = Mock(side_effect=make_runtime)
        self.server = OrchestratorHttpServer(
            ("127.0.0.1", 0), self.factory, token="test-token",
            runtime_info={"planner": "fixed", "agents": {"knowledge": "fake"}},
        )
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True,
        )
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, payload=None, *, path="/api/orchestrator/run", token="test-token", raw=None, headers=None):
        connection = HTTPConnection(*self.server.server_address, timeout=3)
        body = raw if raw is not None else json.dumps(payload).encode()
        outgoing = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        outgoing.update(headers or {})
        connection.request("GET" if payload is None and raw is None else "POST", path, body=None if payload is None and raw is None else body, headers=outgoing)
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    def test_authentication_blocks_health_and_execution(self):
        self.assertEqual(401, self.request(path="/health", token="wrong")[0])
        self.assertEqual(401, self.request({"request": "解释 QPSK"}, token="wrong")[0])
        self.factory.assert_not_called()

    def test_health_reports_runtime_and_busy_state(self):
        self.server.run_lock.acquire()
        self.addCleanup(self.server.run_lock.release)
        status, result = self.request(path="/health")
        self.assertEqual(200, status)
        self.assertTrue(result["busy"])
        self.assertEqual("fake", result["runtime"]["agents"]["knowledge"])
        self.factory.assert_not_called()

    def test_knowledge_response_keeps_plan_attempts_and_complete_answer(self):
        status, result = self.request({"request": "解释 QPSK 原理"})
        self.assertEqual(200, status)
        self.assertEqual("succeeded", result["status"])
        self.assertEqual("direct", result["handling_mode"])
        task = result["plan_history"][-1]["tasks"][0]
        self.assertEqual("knowledge", task["agent"])
        attempt = result["results"][task["task_id"]][-1]
        self.assertEqual("QPSK 是一种四相位数字调制方式。", attempt["result"]["data"]["knowledge"])
        self.assertEqual("succeeded", result["task_statuses"][task["task_id"]])
        self.assertTrue(result["reviews"])
        self.assertNotIn("precomputed_plan", result)

    def test_strict_mode_returns_clarification_without_terminal_input(self):
        status, result = self.request({"request": "使用 USRP 发送信号"})
        self.assertEqual(200, status)
        self.assertEqual("needs_clarification", result["status"])
        self.assertIn("补充", result["final_answer"])
        self.assertFalse(result["results"])
        self.assertIn("请提供中心频率", result["clarification_questions"])

    def test_invalid_input_never_starts_a_workflow(self):
        for payload in ([], {}, {"request": 42}, {"request": "  "}, {"request": "a" * 2001}):
            self.assertEqual(400, self.request(payload)[0])
        self.assertEqual(400, self.request(raw=b"{invalid")[0])
        self.assertEqual(413, self.request(raw=b"x" * (MAX_BODY_BYTES + 1))[0])
        self.assertEqual(415, self.request({"request": "test"}, headers={"Content-Type": "text/plain"})[0])
        self.factory.assert_not_called()

    def test_busy_requests_are_rejected_without_executing_twice(self):
        self.server.run_lock.acquire()
        try:
            self.assertEqual(429, self.request({"request": "查询可用设备"})[0])
            self.factory.assert_not_called()
        finally:
            self.server.run_lock.release()
        self.assertEqual(200, self.request({"request": "查询可用设备"})[0])

    def test_planning_failure_releases_execution_slot(self):
        self.factory.side_effect = PlanningError("invalid plan")
        status, result = self.request({"request": "解释 QPSK"})
        self.assertEqual(422, status)
        self.assertIn("规划失败", result["detail"])
        self.assertFalse(self.server.run_lock.locked())

    def test_workflow_failure_is_returned_as_a_result(self):
        self.factory.side_effect = lambda: GraphOrchestrator(
            FixedPlanner(), FakeKnowledgePort(outcome=ResultStatus.REJECTED),
            FakeDeviceManagerPort(), FakeUsrpAgent(),
        )
        status, result = self.request({"request": "解释 QPSK"})
        self.assertEqual(200, status)
        self.assertEqual("rejected", result["status"])
        self.assertIn("停止", result["final_answer"])

    def test_background_run_can_be_polled_after_the_submission_connection_closes(self):
        started = threading.Event()
        release = threading.Event()
        graph = self.factory.side_effect()
        original_run = graph.run

        def slow_run(*args, **kwargs):
            started.set()
            release.wait(timeout=3)
            return original_run(*args, **kwargs)

        graph.run = slow_run
        self.factory.side_effect = lambda: graph
        self.addCleanup(release.set)
        status, pending = self.request({"request": "解释 QPSK"}, path="/api/orchestrator/runs")
        self.assertEqual(202, status)
        self.assertTrue(started.wait(timeout=1))
        path = "/api/orchestrator/runs/" + pending["request_id"]
        self.assertEqual("running", self.request(path=path)[1]["status"])
        self.assertEqual(429, self.request({"request": "查询设备"}, path="/api/orchestrator/runs")[0])
        release.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status, result = self.request(path=path)
            if result["status"] != "running":
                break
            time.sleep(0.01)
        self.assertEqual(200, status)
        self.assertEqual("succeeded", result["status"])
        self.assertEqual(pending["request_id"], result["request_id"])
        self.assertEqual(404, self.request(path="/api/orchestrator/runs/missing")[0])

    def test_background_planning_failure_is_reported_when_polled(self):
        self.factory.side_effect = PlanningError("invalid plan")
        status, pending = self.request({"request": "解释 QPSK"}, path="/api/orchestrator/runs")
        self.assertEqual(202, status)
        path = "/api/orchestrator/runs/" + pending["request_id"]
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status, result = self.request(path=path)
            if status != 200:
                break
            time.sleep(0.01)
        self.assertEqual(422, status)
        self.assertIn("规划失败", result["detail"])


if __name__ == "__main__":
    unittest.main()
