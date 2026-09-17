import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from simple_orchestrator.fakes import (
    FakeDeviceManagerPort,
    FakeKnowledgePort,
    FakeUsrpAgent,
)
from simple_orchestrator.graph_orchestrator import GraphOrchestrator, merge_results
from simple_orchestrator.models import (
    AgentResult,
    BatchReviewDecision,
    NextAction,
    ResultStatus,
    Task,
    TaskPlan,
    TaskStatus,
    WorkflowStatus,
    PlanStatus,
)
from simple_orchestrator.planner import FixedPlanner, OllamaPlanner, PlanningError
from simple_orchestrator.remote import JsonRpcAgentAdapter, RemoteCallError


USRP_REQUEST = (
    "使用 USRP 以中心频率 2.4GHz、采样率 1MHz、增益 10dB，"
    "发射 QPSK 信号 5 秒"
)


class GraphOrchestratorTests(unittest.TestCase):
    def test_usrp_knowledge_query_is_read_only(self) -> None:
        class Planner(FixedPlanner):
            def create_plan_from_intent(self, user_request, intent):
                return TaskPlan(
                    PlanStatus.READY,
                    intent,
                    [Task(
                        "usrp_knowledge", "usrp", "回答 USRP 专属知识",
                        capability="knowledge.explain",
                        inputs={"request": user_request},
                        expected_output=["knowledge"],
                        completion_criteria=["给出只读知识回答"],
                    )],
                )

        knowledge, device, usrp = FakeKnowledgePort(), FakeDeviceManagerPort(), FakeUsrpAgent()
        state = GraphOrchestrator(Planner(), knowledge, device, usrp).run("如何使用 USRP")

        self.assertIs(WorkflowStatus.SUCCEEDED, state["status"])
        self.assertEqual(0, knowledge.call_count)
        self.assertEqual(0, device.call_count)
        self.assertEqual(1, usrp.call_count)
        self.assertEqual(0, usrp.physical_execution_count)
        self.assertEqual(1, len(state["plan_history"][0].tasks))

    def test_device_query_rejects_usrp_task(self) -> None:
        class Planner(FixedPlanner):
            def create_plan_from_intent(self, user_request, intent):
                return TaskPlan(
                    PlanStatus.READY, intent,
                    [Task("wrong", "usrp", "查询设备", capability="usrp.get_status")],
                )

        with self.assertRaisesRegex(PlanningError, "设备查询只能由 DeviceManager 独立执行"):
            GraphOrchestrator(
                Planner(), FakeKnowledgePort(), FakeDeviceManagerPort(), FakeUsrpAgent(),
            ).run("查询可用的 USRP 设备")

    def test_usrp_preflight_runs_in_parallel_then_enters_hardware(self) -> None:
        barrier = threading.Barrier(2)

        class Knowledge(FakeKnowledgePort):
            def invoke(self, request):
                barrier.wait(timeout=2)
                return super().invoke(request)

        class Device(FakeDeviceManagerPort):
            def invoke(self, request):
                barrier.wait(timeout=2)
                return super().invoke(request)

        knowledge, device, usrp = Knowledge(), Device(), FakeUsrpAgent()
        state = GraphOrchestrator(FixedPlanner(), knowledge, device, usrp).run(
            USRP_REQUEST, request_id="req-parallel"
        )

        self.assertIs(WorkflowStatus.SUCCEEDED, state["status"])
        self.assertEqual(1, knowledge.call_count)
        self.assertEqual(1, device.call_count)
        self.assertEqual(1, usrp.physical_execution_count)
        self.assertEqual(2, len(state["reviews"]))
        tasks = state["plan_history"][0].tasks
        self.assertEqual(["knowledge", "device_manager", "usrp"], [task.agent for task in tasks])
        self.assertEqual((tasks[0].task_id, tasks[1].task_id), tasks[2].dependencies)

    def test_no_device_stops_before_usrp(self) -> None:
        usrp = FakeUsrpAgent()
        state = GraphOrchestrator(
            FixedPlanner(), FakeKnowledgePort(),
            FakeDeviceManagerPort(selection="unavailable"), usrp,
        ).run(USRP_REQUEST)

        self.assertIs(WorkflowStatus.FAILED, state["status"])
        self.assertEqual(0, usrp.call_count)
        self.assertIn("没有可用设备", state["final_answer"])
        usrp_task = next(t for t in state["plan_history"][0].tasks if t.agent == "usrp")
        self.assertIs(TaskStatus.SKIPPED, state["task_statuses"][usrp_task.task_id])

    def test_invalid_skill_never_executes_physical_operation(self) -> None:
        knowledge = FakeKnowledgePort(skill_read_failure=True)
        usrp = FakeUsrpAgent()
        state = GraphOrchestrator(
            FixedPlanner(), knowledge, FakeDeviceManagerPort(), usrp,
        ).run(USRP_REQUEST)

        self.assertIs(WorkflowStatus.FAILED, state["status"])
        self.assertEqual(0, usrp.physical_execution_count)
        usrp_task = next(t for t in state["plan_history"][0].tasks if t.agent == "usrp")
        self.assertEqual(1, len(state["results"][usrp_task.task_id]))
        self.assertIn("Skill", state["error"])

    def test_only_failed_task_is_retried_and_attempts_are_preserved(self) -> None:
        class FlakyKnowledge(FakeKnowledgePort):
            def invoke(self, request):
                if self.call_count == 0:
                    self.call_count += 1
                    return AgentResult(ResultStatus.FAILED, "第一次 Skill 查询失败", error="transient")
                return super().invoke(request)

        knowledge, device = FlakyKnowledge(), FakeDeviceManagerPort()
        state = GraphOrchestrator(
            FixedPlanner(), knowledge, device, FakeUsrpAgent(),
        ).run(USRP_REQUEST)

        self.assertIs(WorkflowStatus.SUCCEEDED, state["status"])
        self.assertEqual(2, knowledge.call_count)
        self.assertEqual(1, device.call_count)
        skill_task = next(t for t in state["plan_history"][0].tasks if t.agent == "knowledge")
        attempts = state["results"][skill_task.task_id]
        self.assertEqual([1, 2], [item.attempt for item in attempts])
        self.assertEqual([ResultStatus.FAILED, ResultStatus.SUCCEEDED], [item.result.status for item in attempts])

    def test_one_replan_keeps_history_and_resets_retry_for_new_task(self) -> None:
        class Planner(FixedPlanner):
            def review_batch(self, context):
                if context["failed_task_ids"]:
                    if context["retryable_task_ids"]:
                        return BatchReviewDecision(NextAction.RETRY, "再试一次", context["retryable_task_ids"])
                    if context["can_replan"]:
                        return BatchReviewDecision(
                            NextAction.REPLAN, "换一个知识任务",
                            revised_tasks=[Task("knowledge_replanned", "knowledge", "替代知识任务")],
                        )
                    return BatchReviewDecision(NextAction.STOP, "重规划已用完")
                return BatchReviewDecision(NextAction.COMPLETE, "任务成功")

        class Knowledge(FakeKnowledgePort):
            def invoke(self, request):
                if request["current_task"]["task_id"] == "answer_knowledge":
                    self.call_count += 1
                    return AgentResult(ResultStatus.FAILED, "原知识任务失败", error="source_unavailable")
                return super().invoke(request)

        state = GraphOrchestrator(
            Planner(), Knowledge(), FakeDeviceManagerPort(), FakeUsrpAgent(),
        ).run("解释 QPSK 原理")

        self.assertIs(WorkflowStatus.SUCCEEDED, state["status"])
        self.assertEqual(2, len(state["plan_history"]))
        self.assertEqual(2, len(state["results"]["answer_knowledge"]))
        self.assertEqual(1, len(state["results"]["knowledge_replanned"]))
        self.assertEqual(2, state["results"]["knowledge_replanned"][0].plan_version)

    def test_unknown_usrp_is_not_blindly_retried(self) -> None:
        usrp = FakeUsrpAgent(outcome=ResultStatus.UNKNOWN)
        state = GraphOrchestrator(
            FixedPlanner(), FakeKnowledgePort(), FakeDeviceManagerPort(), usrp,
        ).run(USRP_REQUEST)
        self.assertIs(WorkflowStatus.UNKNOWN, state["status"])
        self.assertEqual(1, usrp.call_count)

    def test_fixed_planner_replans_once_after_retry_exhaustion(self) -> None:
        knowledge = FakeKnowledgePort(outcome=ResultStatus.FAILED)
        state = GraphOrchestrator(
            FixedPlanner(), knowledge, FakeDeviceManagerPort(), FakeUsrpAgent(),
        ).run("解释 QPSK 原理")

        self.assertIs(WorkflowStatus.FAILED, state["status"])
        self.assertEqual(4, knowledge.call_count)
        self.assertEqual(2, len(state["plan_history"]))
        self.assertEqual(1, state["replan_count"])
        self.assertEqual(2, len(state["results"]["answer_knowledge"]))
        self.assertEqual(2, len(state["results"]["answer_knowledge_replan"]))

    def test_replan_reuses_successful_usrp_preflight(self) -> None:
        knowledge, device, usrp = (
            FakeKnowledgePort(), FakeDeviceManagerPort(),
            FakeUsrpAgent(outcome=ResultStatus.FAILED),
        )
        state = GraphOrchestrator(
            FixedPlanner(), knowledge, device, usrp,
        ).run(USRP_REQUEST)

        self.assertIs(WorkflowStatus.FAILED, state["status"])
        self.assertEqual(2, len(state["plan_history"]))
        self.assertEqual(1, knowledge.call_count)
        self.assertEqual(1, device.call_count)
        self.assertEqual(4, usrp.call_count)
        latest = state["plan_history"][-1].tasks
        self.assertEqual(["knowledge", "device_manager", "usrp"], [t.agent for t in latest])

    def test_result_reducer_rejects_duplicate_attempt(self) -> None:
        state = GraphOrchestrator(
            FixedPlanner(), FakeKnowledgePort(), FakeDeviceManagerPort(), FakeUsrpAgent(),
        ).run("解释 QPSK 原理")
        attempts = state["results"]["answer_knowledge"]
        with self.assertRaisesRegex(ValueError, "重复"):
            merge_results({"answer_knowledge": attempts}, {"answer_knowledge": attempts})

    def test_multiple_tasks_can_target_same_node_in_one_batch(self) -> None:
        class TwoKnowledgeTasks(FixedPlanner):
            def create_plan_from_intent(self, user_request, intent):
                return TaskPlan(
                    PlanStatus.READY, intent,
                    [
                        Task("K1", "knowledge", "第一项知识查询"),
                        Task("K2", "knowledge", "第二项知识查询"),
                    ],
                )

        knowledge = FakeKnowledgePort()
        state = GraphOrchestrator(
            TwoKnowledgeTasks(), knowledge, FakeDeviceManagerPort(), FakeUsrpAgent(),
        ).run("解释 QPSK 原理")
        self.assertIs(WorkflowStatus.SUCCEEDED, state["status"])
        self.assertEqual(2, knowledge.call_count)
        self.assertEqual({"K1", "K2"}, set(state["results"]))
        self.assertEqual(1, len(state["reviews"]))

    def test_knowledge_intent_cannot_be_routed_to_usrp(self) -> None:
        class UnsafePlanner(FixedPlanner):
            def create_plan_from_intent(self, user_request, intent):
                return TaskPlan(
                    PlanStatus.READY, intent,
                    [Task("unsafe", "usrp", "错误地调用硬件")],
                )

        with self.assertRaisesRegex(PlanningError, "非 USRP 硬件意图"):
            GraphOrchestrator(
                UnsafePlanner(), FakeKnowledgePort(),
                FakeDeviceManagerPort(), FakeUsrpAgent(),
            ).run("解释 QPSK 原理")

    def test_remote_usrp_physical_execution_is_disabled(self) -> None:
        remote_usrp = JsonRpcAgentAdapter("http://usrp-host/rpc", "usrp")
        state = GraphOrchestrator(
            FixedPlanner(), FakeKnowledgePort(), FakeDeviceManagerPort(), remote_usrp,
        ).run(USRP_REQUEST)
        self.assertIs(WorkflowStatus.REJECTED, state["status"])
        self.assertIn("尚未启用", state["final_answer"])


class JsonRpcAdapterTests(unittest.TestCase):
    @patch("simple_orchestrator.remote.urlopen")
    def test_uses_static_endpoint_and_correlates_attempt(self, urlopen: MagicMock) -> None:
        envelope = {
            "jsonrpc": "2.0", "id": "req-1:T1:1",
            "result": {
                "task_id": "T1", "status": "succeeded", "summary": "已完成",
                "data": {"device_id": "USRP-01"},
            },
        }
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(envelope).encode()
        urlopen.return_value = response
        adapter = JsonRpcAgentAdapter("http://device-host/rpc", "device_manager")
        result = adapter.invoke({
            "request_id": "req-1", "user_request": "查询设备",
            "current_task": {"task_id": "T1", "attempt": 1},
        })
        self.assertIs(ResultStatus.SUCCEEDED, result.status)
        request = urlopen.call_args.args[0]
        self.assertEqual("http://device-host/rpc", request.full_url)
        payload = json.loads(request.data)
        self.assertEqual("req-1:T1:1", payload["id"])
        self.assertEqual("tasks.send", payload["method"])

    @patch("simple_orchestrator.remote.urlopen")
    def test_mismatched_response_id_is_rejected(self, urlopen: MagicMock) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "jsonrpc": "2.0", "id": "wrong", "result": {"task_id": "T1", "status": "succeeded"},
        }).encode()
        urlopen.return_value = response
        adapter = JsonRpcAgentAdapter("http://device-host/rpc", "device_manager")
        with self.assertRaises(RemoteCallError):
            adapter.invoke({"request_id": "req-1", "current_task": {"task_id": "T1", "attempt": 1}})


class OllamaGraphTests(unittest.TestCase):
    @patch("simple_orchestrator.planner.urlopen")
    def test_usrp_device_inventory_uses_only_device_manager(self, urlopen: MagicMock) -> None:
        payloads = [
            {
                "intent_type": "device_query", "domain": "device_management",
                "target_type": "usrp", "action": "查询可用设备",
                "execution_mode": "read_only", "required_capabilities": ["usrp.get_status"],
                "provided_parameters": {}, "missing_parameters": [],
                "needs_clarification": False, "evidence": "查询可用的 USRP 设备",
                "reason": "查询设备状态",
            },
            {"tasks": [{
                "task_id": "llm_inventory_task",
                "agent": "device_manager",
                "capability": "device.list",
                "description": "查询实验室中可用的 USRP 设备",
                "inputs": {"target_type": "usrp", "site": "lab"},
                "expected_output": ["network_devices"],
                "completion_criteria": ["返回设备列表和在线状态"],
                "dependencies": [],
            }]},
            {"action": "complete", "reason": "设备查询完成", "retry_task_ids": [], "revised_tasks": []},
        ]
        responses = []
        for payload in payloads:
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps({
                "message": {"content": json.dumps(payload)}
            }).encode()
            responses.append(response)
        urlopen.side_effect = responses
        knowledge, device, usrp = FakeKnowledgePort(), FakeDeviceManagerPort(), FakeUsrpAgent()

        state = GraphOrchestrator(
            OllamaPlanner(model="local-model"), knowledge, device, usrp,
        ).run("查询可用的USRP设备")

        self.assertIs(WorkflowStatus.SUCCEEDED, state["status"])
        planned_task = state["plan_history"][0].tasks[0]
        self.assertEqual("device_manager", planned_task.agent)
        self.assertEqual("llm_inventory_task", planned_task.task_id)
        self.assertEqual("查询实验室中可用的 USRP 设备", planned_task.description)
        self.assertEqual({"target_type": "usrp", "site": "lab"}, planned_task.inputs)
        planning_request = json.loads(urlopen.call_args_list[1].args[0].data.decode())
        self.assertIn('"available_agents": ["device_manager"]', planning_request["messages"][1]["content"])
        self.assertEqual(0, knowledge.call_count)
        self.assertEqual(1, device.call_count)
        self.assertEqual(0, usrp.call_count)
        self.assertEqual(3, urlopen.call_count)

    @patch("simple_orchestrator.planner.urlopen")
    def test_ollama_device_query_cannot_route_to_usrp(self, urlopen: MagicMock) -> None:
        payloads = [
            {
                "intent_type": "device_query", "domain": "device_management",
                "target_type": "usrp", "action": "查询可用设备",
                "execution_mode": "read_only", "required_capabilities": ["usrp.get_status"],
                "provided_parameters": {}, "missing_parameters": [],
                "needs_clarification": False, "evidence": "查询可用的 USRP 设备",
                "reason": "查询设备状态",
            },
            {"tasks": [{
                "task_id": "wrong", "agent": "usrp", "capability": "usrp.get_status",
                "description": "查询设备", "inputs": {},
                "expected_output": ["devices"], "completion_criteria": ["返回设备"],
                "dependencies": [],
            }]},
        ]
        responses = []
        for payload in payloads:
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps({
                "message": {"content": json.dumps(payload)}
            }).encode()
            responses.append(response)
        urlopen.side_effect = responses
        device, usrp = FakeDeviceManagerPort(), FakeUsrpAgent()

        with self.assertRaisesRegex(PlanningError, "设备查询只能由 DeviceManager 独立执行"):
            GraphOrchestrator(
                OllamaPlanner(model="local-model"), FakeKnowledgePort(), device, usrp,
            ).run("查询可用的USRP设备")
        self.assertEqual(0, device.call_count)
        self.assertEqual(0, usrp.call_count)

    @patch("simple_orchestrator.planner.urlopen")
    def test_analyze_plan_and_batch_review_are_separate_calls(self, urlopen: MagicMock) -> None:
        payloads = [
            {
                "intent_type": "knowledge_query", "domain": "wireless_communication",
                "target_type": None, "action": "explain", "execution_mode": "informational",
                "required_capabilities": ["knowledge.explain"],
                "provided_parameters": {}, "missing_parameters": [],
                "needs_clarification": False, "evidence": "解释 QPSK", "reason": "知识请求",
            },
            {
                "tasks": [{
                    "task_id": "K1", "agent": "knowledge", "capability": "knowledge.explain",
                    "description": "解释 QPSK", "inputs": {"request": "解释 QPSK"},
                    "expected_output": ["knowledge"], "completion_criteria": ["解释完成"],
                    "dependencies": [],
                }]
            },
            {"action": "complete", "reason": "知识已获取", "retry_task_ids": [], "revised_tasks": []},
        ]
        responses = []
        for payload in payloads:
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps({
                "message": {"content": json.dumps(payload)}
            }).encode()
            responses.append(response)
        urlopen.side_effect = responses

        state = GraphOrchestrator(
            OllamaPlanner(model="local-model"),
            FakeKnowledgePort(), FakeDeviceManagerPort(), FakeUsrpAgent(),
        ).run("解释 QPSK")

        self.assertIs(WorkflowStatus.SUCCEEDED, state["status"])
        self.assertEqual(3, urlopen.call_count)


if __name__ == "__main__":
    unittest.main()
