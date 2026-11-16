import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from simple_orchestrator.orchestration.api import serialize_state
from simple_orchestrator.orchestration.execution import WorkflowExecutor
from simple_orchestrator.fakes import FakeDeviceManagerPort, FakeKnowledgePort, FakeUsrpAgent
from simple_orchestrator.orchestration.graph import GraphOrchestrator
from simple_orchestrator.intent.agent import FixedIntentAgent, OllamaIntentAgent
from simple_orchestrator.orchestration.main import build_orchestrator, build_parser
from simple_orchestrator.models import (
    BatchReviewDecision, ExecutionMode, HandlingMode, IntentType, NextAction,
    ParameterPolicy, PlanStatus, ResultStatus, Task, TaskPlan, WorkflowStatus,
)
from simple_orchestrator.ollama import OllamaClient
from simple_orchestrator.orchestration.planner import PlanningError
from simple_orchestrator.ports import GraphPlannerPort, IntentAgentPort
from simple_orchestrator.remote import JsonRpcAgentAdapter
from simple_orchestrator.orchestration.router import RequestRouter


USRP_REQUEST = "使用 USRP 以中心频率 2.4GHz、采样率 1MHz、增益 10dB，发射 QPSK 信号 5 秒"


def response(intent):
    return {"message": {"content": json.dumps(asdict(intent), ensure_ascii=False)}}


class RequestRouterTests(unittest.TestCase):
    def setUp(self):
        self.knowledge = FakeKnowledgePort()
        self.device = FakeDeviceManagerPort()
        self.usrp = FakeUsrpAgent()
        # The dynamic planner has no analyze_intent method in its interface.
        self.planner = Mock(spec=GraphPlannerPort)
        self.orchestrator = GraphOrchestrator(
            self.planner, self.knowledge, self.device, self.usrp,
        )
        self.executor = WorkflowExecutor(self.knowledge, self.device, self.usrp)
        self.intent_agent = Mock(spec=IntentAgentPort)
        self.router = RequestRouter(self.intent_agent, self.executor, self.orchestrator)

    def run_request(self, message, *, mode=None):
        intent = FixedIntentAgent().analyze_intent(message)
        if mode is not None:
            intent = replace(intent, handling_mode=mode)
        self.intent_agent.analyze_intent.return_value = intent
        return self.router.run(message, request_id="req-routing")

    def assert_no_planning(self):
        self.planner.create_plan_from_intent.assert_not_called()
        self.planner.review_batch.assert_not_called()

    def test_simple_knowledge_goes_direct_without_planning_or_model_review(self):
        message = "请用两句话解释 QPSK 原理。"
        state = self.run_request(message)
        self.assertIs(WorkflowStatus.SUCCEEDED, state["status"])
        self.assertIs(HandlingMode.DIRECT, state["handling_mode"])
        self.assertEqual(1, self.knowledge.call_count)
        self.assertEqual(0, self.device.call_count)
        self.assertEqual(0, self.usrp.call_count)
        self.intent_agent.analyze_intent.assert_called_once_with(message)
        task = state["plan_history"][0].tasks[0]
        self.assertEqual(message, task.description)
        self.assertEqual("req-routing", state["results"][task.task_id][0].request_id)
        self.assert_no_planning()

    def test_device_query_goes_direct_without_physical_execution(self):
        state = self.run_request("查询可用的 USRP 设备")
        self.assertIs(HandlingMode.DIRECT, state["handling_mode"])
        self.assertIs(WorkflowStatus.SUCCEEDED, state["status"])
        self.assertEqual(1, self.device.call_count)
        self.assertEqual(0, self.knowledge.call_count)
        self.assertEqual(0, self.usrp.call_count)
        self.assert_no_planning()

    def test_usrp_knowledge_goes_to_read_only_business_agent(self):
        state = self.run_request("如何使用 USRP")
        self.assertIs(WorkflowStatus.SUCCEEDED, state["status"])
        self.assertEqual(1, self.usrp.call_count)
        self.assertEqual(0, self.usrp.physical_execution_count)
        self.assertEqual(0, self.device.call_count)
        self.assert_no_planning()

    def test_fixed_workflow_runs_preflight_without_dynamic_planning(self):
        state = self.run_request(USRP_REQUEST)
        self.assertIs(WorkflowStatus.SUCCEEDED, state["status"])
        self.assertIs(HandlingMode.FIXED_WORKFLOW, state["handling_mode"])
        self.assertEqual(1, self.knowledge.call_count)
        self.assertEqual(1, self.device.call_count)
        self.assertEqual(1, self.usrp.physical_execution_count)
        tasks = state["plan_history"][0].tasks
        hardware = next(task for task in tasks if task.agent == "usrp")
        self.assertEqual(2, len(hardware.dependencies))
        self.assertEqual(0, state["replan_count"])
        self.assert_no_planning()

    def test_physical_direct_suggestion_cannot_skip_fixed_preflight(self):
        state = self.run_request(USRP_REQUEST, mode=HandlingMode.DIRECT)
        self.assertIs(HandlingMode.FIXED_WORKFLOW, state["handling_mode"])
        self.assertEqual(1, self.device.call_count)
        self.assertEqual(1, self.knowledge.call_count)
        self.assertEqual(1, self.usrp.physical_execution_count)
        self.assert_no_planning()

    def test_missing_parameters_stop_before_any_execution_or_planning(self):
        state = self.run_request("使用 USRP 发射信号")
        self.assertIs(WorkflowStatus.NEEDS_CLARIFICATION, state["status"])
        self.assertIn("请提供中心频率", state["clarification_questions"])
        self.assertFalse(state["plan_history"])
        self.assertEqual(0, self.knowledge.call_count + self.device.call_count + self.usrp.call_count)
        self.assert_no_planning()

    def test_unsupported_request_stops_before_planning(self):
        state = self.run_request("启动 RIS")
        self.assertIs(WorkflowStatus.UNSUPPORTED, state["status"])
        self.assert_no_planning()

    def test_forged_hardware_intent_still_requires_explicit_usrp_request(self):
        self.intent_agent.analyze_intent.return_value = FixedIntentAgent().analyze_intent(USRP_REQUEST)
        state = self.router.run("启动 RIS")
        self.assertIs(WorkflowStatus.UNSUPPORTED, state["status"])
        self.assertEqual(0, self.usrp.call_count)
        self.assert_no_planning()

    def test_no_devices_prevents_fixed_hardware_execution(self):
        self.device.outcome = ResultStatus.FAILED
        state = self.run_request(USRP_REQUEST)
        self.assertIs(WorkflowStatus.FAILED, state["status"])
        self.assertEqual(0, self.usrp.call_count)
        self.assert_no_planning()

    def test_unknown_physical_result_is_never_retried(self):
        self.usrp.outcome = ResultStatus.UNKNOWN
        state = self.run_request(USRP_REQUEST)
        self.assertIs(WorkflowStatus.UNKNOWN, state["status"])
        self.assertEqual(1, self.usrp.physical_execution_count)
        self.assertEqual(0, state["replan_count"])
        self.assert_no_planning()

    def test_remote_physical_execution_remains_disabled_in_fixed_workflow(self):
        remote = JsonRpcAgentAdapter("http://usrp.invalid/rpc", "usrp")
        self.router.executor = WorkflowExecutor(self.knowledge, self.device, remote)
        with patch("simple_orchestrator.remote.urlopen") as urlopen:
            state = self.run_request(USRP_REQUEST)
        self.assertIs(WorkflowStatus.REJECTED, state["status"])
        urlopen.assert_not_called()
        self.assert_no_planning()

    def test_direct_failures_use_bounded_retry_without_replanning(self):
        self.knowledge.outcome = ResultStatus.FAILED
        state = self.run_request("解释 QPSK")
        self.assertIs(WorkflowStatus.FAILED, state["status"])
        self.assertEqual(2, self.knowledge.call_count)
        self.assertEqual([NextAction.RETRY, NextAction.STOP], [review.action for review in state["reviews"]])
        self.assert_no_planning()

    def test_direct_rejection_is_not_retried(self):
        self.knowledge.outcome = ResultStatus.REJECTED
        state = self.run_request("解释 QPSK")
        self.assertIs(WorkflowStatus.REJECTED, state["status"])
        self.assertEqual(1, self.knowledge.call_count)
        self.assert_no_planning()

    def configure_dynamic_plan(self):
        tasks = [
            Task("qpsk", "knowledge", "解释 QPSK", capability="knowledge.explain"),
            Task("bpsk", "knowledge", "解释 BPSK", capability="knowledge.explain"),
        ]
        self.planner.create_plan_from_intent.side_effect = lambda request, intent: TaskPlan(PlanStatus.READY, intent, tasks)
        self.planner.review_batch.return_value = BatchReviewDecision(NextAction.COMPLETE, "已完成")

    def test_dynamic_request_uses_upstream_intent_without_analyzing_again(self):
        self.configure_dynamic_plan()
        state = self.run_request("分别解释 QPSK 和 BPSK", mode=HandlingMode.ORCHESTRATED)
        self.assertIs(WorkflowStatus.SUCCEEDED, state["status"])
        self.assertIs(HandlingMode.ORCHESTRATED, state["handling_mode"])
        self.intent_agent.analyze_intent.assert_called_once()
        self.planner.create_plan_from_intent.assert_called_once_with(
            "分别解释 QPSK 和 BPSK", self.intent_agent.analyze_intent.return_value,
        )
        self.planner.review_batch.assert_called_once()
        self.assertEqual(2, self.knowledge.call_count)

    def test_direct_suggestion_with_uncovered_capabilities_enters_orchestration(self):
        self.configure_dynamic_plan()
        intent = replace(
            FixedIntentAgent().analyze_intent("解释 QPSK"),
            required_capabilities=["knowledge.explain", "knowledge.recommend_parameters"],
        )
        self.intent_agent.analyze_intent.return_value = intent
        state = self.router.run("解释 QPSK")
        self.assertIs(HandlingMode.ORCHESTRATED, state["handling_mode"])
        self.planner.create_plan_from_intent.assert_called_once()

    def test_unknown_fixed_workflow_enters_orchestration(self):
        self.configure_dynamic_plan()
        state = self.run_request("解释 QPSK", mode=HandlingMode.FIXED_WORKFLOW)
        self.assertIs(HandlingMode.ORCHESTRATED, state["handling_mode"])
        self.planner.create_plan_from_intent.assert_called_once()

    def test_registry_is_optional_and_only_used_when_enabled(self):
        with patch.object(self.router.registry, "match", wraps=self.router.registry.match) as match:
            self.run_request("解释 QPSK")
            match.assert_not_called()
            self.router.enable_agent_registry = True
            state = self.run_request("解释 QPSK")
            match.assert_called_once()
        self.assertIs(WorkflowStatus.SUCCEEDED, state["status"])
        self.assert_no_planning()

    def test_fixed_workflow_can_use_existing_registry_validation(self):
        self.router.enable_agent_registry = True
        state = self.run_request(USRP_REQUEST)
        self.assertIs(WorkflowStatus.SUCCEEDED, state["status"])
        self.assert_no_planning()

    def test_api_serialization_retains_results_and_exposes_effective_route(self):
        payload = serialize_state(self.run_request("解释 QPSK"))
        self.assertEqual("direct", payload["handling_mode"])
        self.assertEqual("direct", payload["intent"]["handling_mode"])
        self.assertEqual("req-routing", payload["request_id"])
        self.assertIn("knowledge", payload["results"]["answer_knowledge"][0]["result"]["data"])

    @patch("simple_orchestrator.ollama.urlopen")
    def test_cli_runtime_direct_and_fixed_only_call_intent_model_once(self, urlopen):
        for message, mode in (
            ("解释 QPSK", HandlingMode.DIRECT),
            ("查询在线设备", HandlingMode.DIRECT),
            (USRP_REQUEST, HandlingMode.FIXED_WORKFLOW),
        ):
            with self.subTest(message=message):
                urlopen.reset_mock()
                model_response = response(FixedIntentAgent().analyze_intent(message))
                urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(model_response).encode()
                with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"KNOWLEDGE_API_URL": ""}):
                    parser = build_parser()
                    args = parser.parse_args([
                        message, "--local-intent", "--planner", "ollama", "--ollama-model", "test-model",
                        "--fake-outcome", "succeeded", "--ollama-log-path", str(Path(directory) / "model.json"),
                        "--agent-log-path", str(Path(directory) / "agents.jsonl"),
                    ])
                    runtime, _ = build_orchestrator(args, parser)
                    state = runtime.run(args.request)
                    logs = json.loads((Path(directory) / "model.json").read_text())
                self.assertIs(WorkflowStatus.SUCCEEDED, state["status"])
                self.assertIs(mode, state["handling_mode"])
                urlopen.assert_called_once()
                self.assertEqual(["intent_analysis"], [entry["phase"] for entry in logs["entries"]])
                self.assertIs(runtime.intent_agent.client, runtime.orchestrator.planner.client)

    @patch("simple_orchestrator.ollama.urlopen")
    def test_cli_dynamic_path_uses_distinct_intent_plan_and_review_calls(self, urlopen):
        message = "分别解释 QPSK 和 BPSK"
        intent = replace(FixedIntentAgent().analyze_intent(message), handling_mode=HandlingMode.ORCHESTRATED)
        task = {
            "task_id": "answer", "agent": "knowledge", "description": message,
            "capability": "knowledge.explain", "inputs": {}, "dependencies": [],
            "expected_output": ["knowledge"], "completion_criteria": ["回答两个问题"],
        }
        responses = [
            response(intent),
            {"message": {"content": json.dumps({"tasks": [task]})}},
            {"message": {"content": json.dumps({
                "action": "complete", "reason": "目标已满足",
                "retry_task_ids": [], "revised_tasks": [],
            })}},
        ]
        urlopen.return_value.__enter__.return_value.read.side_effect = [json.dumps(item).encode() for item in responses]
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"KNOWLEDGE_API_URL": ""}):
            parser = build_parser()
            args = parser.parse_args([
                message, "--local-intent", "--planner", "ollama", "--ollama-model", "test-model",
                "--fake-outcome", "succeeded", "--ollama-log-path", str(Path(directory) / "model.json"),
                "--agent-log-path", str(Path(directory) / "agents.jsonl"),
            ])
            runtime, _ = build_orchestrator(args, parser)
            state = runtime.run(message)
            entries = json.loads((Path(directory) / "model.json").read_text())["entries"]
        self.assertIs(WorkflowStatus.SUCCEEDED, state["status"])
        self.assertIs(HandlingMode.ORCHESTRATED, state["handling_mode"])
        self.assertEqual(["intent_analysis", "planning", "batch_review"], [entry["phase"] for entry in entries])
        self.assertEqual(3, urlopen.call_count)
        requests = [json.loads(call.args[0].data) for call in urlopen.call_args_list]
        self.assertEqual([2, 2, 2], [len(request["messages"]) for request in requests])
        self.assertNotEqual(requests[0]["messages"][0], requests[1]["messages"][0])


class IntentAgentTests(unittest.TestCase):
    def test_standalone_model_agent_preserves_mode_and_does_not_plan(self):
        client = Mock(spec=OllamaClient)
        # Use the real structured-response parser on a mocked transport.
        client._response_content_as_json.side_effect = OllamaClient._response_content_as_json
        client._chat.return_value = response(FixedIntentAgent().analyze_intent("查询在线设备"))
        intent = OllamaIntentAgent(client).analyze_intent("查询在线设备")
        self.assertIs(HandlingMode.DIRECT, intent.handling_mode)
        self.assertIs(ExecutionMode.READ_ONLY, intent.execution_mode)
        self.assertEqual("intent_analysis", client._chat.call_args.kwargs["phase"])
        self.assertIn("handling_mode", client._chat.call_args.kwargs["schema"]["required"])
        client._chat.assert_called_once()

    def test_unknown_handling_mode_is_rejected(self):
        client = OllamaClient("test-model")
        data = asdict(FixedIntentAgent().analyze_intent("解释 QPSK"))
        data["handling_mode"] = "skip_validation"
        with patch.object(client, "_chat", return_value={"message": {"content": json.dumps(data)}}):
            with self.assertRaises(PlanningError):
                OllamaIntentAgent(client).analyze_intent("解释 QPSK")

    def test_demo_defaults_are_applied_before_routing(self):
        intent = FixedIntentAgent(ParameterPolicy.DEMO_DEFAULTS).analyze_intent("使用 USRP 发射信号")
        self.assertIs(HandlingMode.FIXED_WORKFLOW, intent.handling_mode)
        self.assertFalse(intent.needs_clarification)
        self.assertEqual("use_backend_defaults", intent.provided_parameters["parameter_policy"])
        self.assertIn("center_frequency_hz", intent.provided_parameters["unresolved_parameters"])


if __name__ == "__main__":
    unittest.main()
