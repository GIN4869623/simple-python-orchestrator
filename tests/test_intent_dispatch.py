"""Prove actual cross-service handoff, not just classification labels."""

import json
import threading
import time
import unittest
from dataclasses import asdict, replace
from http.client import HTTPConnection
from unittest.mock import Mock

from simple_orchestrator.fakes import FakeDeviceManagerPort, FakeKnowledgePort, FakeUsrpAgent
from simple_orchestrator.errors import PlanningError
from simple_orchestrator.intent.agent import FixedIntentAgent
from simple_orchestrator.intent.service import IntentHttpServer
from simple_orchestrator.models import HandlingMode, ParameterPolicy, PlanStatus, Task, TaskPlan
from simple_orchestrator.orchestration.api import OrchestratorHttpServer
from simple_orchestrator.orchestration.client import OrchestrationHttpClient
from simple_orchestrator.orchestration.execution import WorkflowExecutor
from simple_orchestrator.orchestration.graph import GraphOrchestrator
from simple_orchestrator.orchestration.planner import FixedPlanner
from simple_orchestrator.orchestration.router import RequestRouter

TOKEN = "dispatch-integration-test-token"
PHYSICAL = "使用 USRP 以中心频率 2.4GHz、采样率 1MHz、增益 10dB，发射 QPSK 信号 5 秒"


def request(server, path, payload=None, token=TOKEN):
    conn = HTTPConnection(*server.server_address, timeout=3)
    body = json.dumps(payload).encode() if payload is not None else None
    conn.request("POST" if body is not None else "GET", path, body=body,
                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    response = conn.getresponse()
    value = response.status, json.loads(response.read())
    conn.close()
    return value


class IntentDispatchTests(unittest.TestCase):
    def setUp(self):
        self.knowledge, self.device, self.usrp = FakeKnowledgePort(), FakeDeviceManagerPort(), FakeUsrpAgent()
        self.dynamic_knowledge = FakeKnowledgePort()
        self.dynamic_device, self.dynamic_usrp = FakeDeviceManagerPort(), FakeUsrpAgent()
        self.planner = Mock(wraps=FixedPlanner(parameter_policy=ParameterPolicy.STRICT))
        self.forbidden_recognizer = Mock()
        self.forbidden_recognizer.analyze_intent.side_effect = AssertionError("must not recognize twice")
        self.orchestration_factory = Mock(side_effect=lambda: RequestRouter(
            self.forbidden_recognizer,
            WorkflowExecutor(self.dynamic_knowledge, self.dynamic_device, self.dynamic_usrp),
            GraphOrchestrator(self.planner, self.dynamic_knowledge, self.dynamic_device, self.dynamic_usrp),
        ))
        self.orchestration = OrchestratorHttpServer(
            ("127.0.0.1", 0), self.orchestration_factory, token=TOKEN, runtime_info={},
        )
        self.intent_agent = Mock(wraps=FixedIntentAgent(parameter_policy=ParameterPolicy.STRICT))
        self.fail_handoff = False

        def router_factory(agent, on_submitted):
            port = 1 if self.fail_handoff else self.orchestration.server_port
            client = OrchestrationHttpClient(f"http://127.0.0.1:{port}", TOKEN,
                                             timeout=2, poll_interval=.01, on_submitted=on_submitted)
            return RequestRouter(agent, WorkflowExecutor(self.knowledge, self.device, self.usrp), client)

        self.intent = IntentHttpServer(("127.0.0.1", 0), lambda: self.intent_agent,
                                       token=TOKEN, runtime_info={}, router_factory=router_factory)
        self.threads = []
        for server in (self.orchestration, self.intent):
            thread = threading.Thread(target=lambda s=server: s.serve_forever(poll_interval=.01), daemon=True)
            thread.start()
            self.threads.append(thread)

    def tearDown(self):
        for server in (self.intent, self.orchestration):
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=2)

    def wait(self, server, prefix, pending):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status, state = request(server, prefix + pending["request_id"])
            self.assertEqual(200, status, state)
            if state["status"] != "running":
                return state
            time.sleep(.01)
        self.fail("Request never completed")

    def run_request(self, message, **extra):
        status, pending = request(self.intent, "/api/intent/runs", {"request": message, **extra})
        self.assertEqual(202, status, pending)
        return self.wait(self.intent, "/api/intent/runs/", pending)

    def test_direct_device_request_invokes_device_without_contacting_orchestrator(self):
        self.fail_handoff = True
        state = self.run_request("查询在线设备")
        self.assertEqual("succeeded", state["status"])
        self.assertEqual(["device_manager"], state["execution_targets"])
        self.assertIsNone(state["orchestration_request_id"])
        self.assertEqual(1, self.device.call_count)
        self.assertEqual(0, self.knowledge.call_count)
        self.orchestration_factory.assert_not_called()
        self.intent_agent.analyze_intent.assert_called_once()

    def test_direct_knowledge_and_usrp_information_reach_correct_business_ports(self):
        state = self.run_request("解释 QPSK 原理")
        self.assertEqual(["knowledge"], state["execution_targets"])
        self.assertEqual(1, self.knowledge.call_count)
        state = self.run_request("如何使用 USRP")
        self.assertEqual(["usrp"], state["execution_targets"])
        self.assertEqual(1, self.usrp.call_count)
        self.assertEqual(0, self.usrp.physical_execution_count)
        self.orchestration_factory.assert_not_called()

    def test_fixed_workflow_executes_preflight_but_never_calls_dynamic_service(self):
        state = self.run_request(PHYSICAL)
        self.assertEqual("succeeded", state["status"])
        self.assertEqual("fixed_workflow", state["handling_mode"])
        self.assertEqual(1, self.usrp.physical_execution_count)
        self.assertEqual(1, self.device.call_count)
        self.assertEqual(1, self.knowledge.call_count)
        self.orchestration_factory.assert_not_called()

    def test_complex_request_really_crosses_http_and_does_not_repeat_recognition(self):
        state = self.run_request("分别解释 QPSK 和 BPSK")
        self.assertEqual("succeeded", state["status"], state)
        self.assertEqual("orchestrated", state["handling_mode"])
        self.assertEqual(["orchestrator"], state["execution_targets"])
        self.assertEqual(state["request_id"], state["orchestration_request_id"])
        self.assertEqual(0, self.knowledge.call_count)
        self.assertGreater(self.dynamic_knowledge.call_count, 0)
        self.intent_agent.analyze_intent.assert_called_once()
        self.forbidden_recognizer.analyze_intent.assert_not_called()
        self.orchestration_factory.assert_called_once()
        self.planner.create_plan_from_intent.assert_called_once()
        status, remote = request(self.orchestration, "/api/orchestrator/runs/" + state["request_id"])
        self.assertEqual(200, status)
        self.assertEqual("succeeded", remote["status"])

    def test_dispatched_plan_exposes_task_progress_before_final_result(self):
        entered = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]
        for latch in release:
            self.addCleanup(latch.set)
        invoke = self.dynamic_knowledge.invoke

        def slow_knowledge(payload, *, metadata=None):
            slot = 0 if self.dynamic_knowledge.call_count == 0 else 1
            entered[slot].set()
            if not release[slot].wait(2):
                raise AssertionError("test did not release the knowledge task")
            return invoke(payload, metadata=metadata)

        self.dynamic_knowledge.invoke = slow_knowledge

        def plan(message, intent):
            return TaskPlan(PlanStatus.READY, intent, [
                Task("first", "knowledge", "解释 QPSK", capability="knowledge.explain"),
                Task("second", "knowledge", "比较 BPSK", dependencies=["first"], capability="knowledge.explain"),
            ])

        self.planner.create_plan_from_intent.side_effect = plan
        status, pending = request(self.intent, "/api/intent/runs", {"request": "分别解释 QPSK 和 BPSK"})
        self.assertEqual(202, status, pending)
        path = "/api/orchestrator/runs/" + pending["request_id"]
        self.assertTrue(entered[0].wait(2))
        status, state = request(self.orchestration, path)
        self.assertEqual(200, status)
        self.assertEqual("running", state["status"])
        self.assertEqual(["first", "second"], [task["task_id"] for task in state["plan_history"][0]["tasks"]])
        self.assertEqual({"first": "running", "second": "pending"}, state["task_statuses"])
        release[0].set()
        self.assertTrue(entered[1].wait(2))
        status, state = request(self.orchestration, path)
        self.assertEqual(200, status)
        self.assertEqual("running", state["status"])
        self.assertEqual({"first": "succeeded", "second": "running"}, state["task_statuses"])
        self.assertEqual(1, state["results"]["first"][0]["attempt"])
        release[1].set()
        final = self.wait(self.orchestration, "/api/orchestrator/runs/", pending)
        self.assertEqual({"first": "succeeded", "second": "succeeded"}, final["task_statuses"])

    def test_missing_parameters_and_unsupported_requests_stop_before_all_business_calls(self):
        state = self.run_request("使用 USRP 发射信号")
        self.assertEqual("needs_clarification", state["status"])
        self.assertTrue(state["clarification_questions"])
        state = self.run_request("启动 RIS")
        self.assertEqual("unsupported", state["status"])
        self.assertFalse(state["execution_targets"])
        self.assertEqual(0, self.device.call_count + self.knowledge.call_count + self.usrp.call_count)
        self.orchestration_factory.assert_not_called()

    def test_unavailable_orchestrator_is_a_failure_not_a_fake_success(self):
        self.fail_handoff = True
        state = self.run_request("分别解释 QPSK 和 BPSK")
        self.assertEqual("failed", state["status"])
        self.assertIn("编排服务", state["error"])
        self.assertIsNone(state["orchestration_request_id"])
        self.assertEqual(0, self.knowledge.call_count + self.device.call_count)

    def test_remote_planning_failure_preserves_reason_and_accepted_request_id(self):
        self.planner.create_plan_from_intent.side_effect = PlanningError("业务能力不匹配")
        state = self.run_request("分别解释 QPSK 和 BPSK")
        self.assertEqual("failed", state["status"])
        self.assertIn("业务能力不匹配", state["error"])
        self.assertEqual(state["request_id"], state["orchestration_request_id"])
        self.assertEqual(0, self.dynamic_knowledge.call_count)

    def test_browser_supplied_intent_cannot_override_recognized_device_query(self):
        state = self.run_request("查询在线设备", intent=asdict(FixedIntentAgent().analyze_intent(PHYSICAL)))
        self.assertEqual("device_query", state["intent"]["intent_type"])
        self.assertEqual(1, self.device.call_count)
        self.assertEqual(0, self.usrp.physical_execution_count)

    def test_busy_analysis_and_failure_release_the_background_slot(self):
        entered, release = threading.Event(), threading.Event()
        def slow(_):
            entered.set(); release.wait(timeout=2)
            raise ValueError("test failure")
        self.intent_agent.analyze_intent.side_effect = slow
        status, pending = request(self.intent, "/api/intent/runs", {"request": "查询设备"})
        self.assertEqual(202, status)
        self.assertTrue(entered.wait(timeout=1))
        self.assertEqual(429, request(self.intent, "/api/intent/runs", {"request": "另一个请求"})[0])
        self.assertTrue(request(self.intent, "/health")[1]["busy"])
        release.set()
        state = self.wait(self.intent, "/api/intent/runs/", pending)
        self.assertEqual("failed", state["status"])
        self.assertFalse(self.intent.run_lock.locked())

    def test_dispatch_endpoint_validates_intent_and_deduplicates_request_id(self):
        message = "分别解释 QPSK 和 BPSK"
        intent = asdict(FixedIntentAgent().analyze_intent(message))
        payload = {"request": message, "request_id": "req-" + "c" * 32, "intent": intent}
        self.assertEqual(401, request(self.orchestration, "/api/orchestrator/dispatch", payload, token="bad")[0])
        missing_mode = {key: value for key, value in intent.items() if key != "handling_mode"}
        for bad in [None, {}, missing_mode, {**intent, "handling_mode": "direct"}, {**intent, "needs_clarification": True}]:
            self.assertEqual(400, request(self.orchestration, "/api/orchestrator/dispatch", {**payload, "intent": bad})[0])
        status, pending = request(self.orchestration, "/api/orchestrator/dispatch", payload)
        self.assertEqual(202, status)
        self.assertEqual("succeeded", self.wait(self.orchestration, "/api/orchestrator/runs/", pending)["status"])
        self.assertEqual(409, request(self.orchestration, "/api/orchestrator/dispatch", payload)[0])
        self.orchestration_factory.assert_called_once()
        self.forbidden_recognizer.analyze_intent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
