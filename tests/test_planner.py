import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from simple_orchestrator.models import (
    AgentResult,
    ExecutionMode,
    IntentAnalysis,
    IntentType,
    NextAction,
    ParameterPolicy,
    PlanStatus,
    ResultStatus,
    Task,
    TaskStatus,
    WorkflowState,
)
from simple_orchestrator.planner import OllamaPlanner, PlanningError


def hardware_intent(*, missing: list[str] | None = None) -> dict:
    missing = missing or []
    return {
        "intent_type": "hardware_execution",
        "domain": "wireless_communication",
        "target_type": "usrp",
        "action": "transmit",
        "execution_mode": "physical",
        "required_capabilities": ["usrp.transmit"],
        "provided_parameters": {"waveform": "qpsk"},
        "missing_parameters": missing,
        "needs_clarification": bool(missing),
        "evidence": "使用 USRP 发射 QPSK 信号",
        "reason": "用户明确要求真实 USRP 操作",
    }


def device_intent() -> IntentAnalysis:
    return IntentAnalysis(
        IntentType.DEVICE_QUERY,
        "device_management",
        "usrp",
        "list",
        ExecutionMode.READ_ONLY,
        required_capabilities=["device.list"],
        evidence="查询 USRP 设备",
    )


def task_data(
    task_id: str,
    agent: str,
    capability: str,
    dependencies: list[str],
) -> dict:
    inputs = {}
    if agent == "usrp":
        inputs = {"operation": capability.removeprefix("usrp.")}
        if dependencies:
            inputs["device_id_from"] = f"{dependencies[-1]}.device_id"
    return {
        "task_id": task_id,
        "agent": agent,
        "capability": capability,
        "description": f"执行 {capability}",
        "inputs": inputs,
        "expected_output": ["result"],
        "completion_criteria": ["返回成功结果"],
        "dependencies": dependencies,
    }


class OllamaPlannerTests(unittest.TestCase):
    @staticmethod
    def _ollama_response(payload: dict) -> MagicMock:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"message": {"role": "assistant", "content": json.dumps(payload)}}
        ).encode("utf-8")
        return response

    @patch("simple_orchestrator.planner.urlopen")
    def test_batch_review_non_replan_requests_empty_revised_tasks(
        self, urlopen: MagicMock
    ) -> None:
        urlopen.return_value = self._ollama_response({
            "action": "continue",
            "reason": "前置任务成功，继续执行现有计划",
            "retry_task_ids": [],
            "revised_tasks": [],
        })

        decision = OllamaPlanner(model="local-model").review_batch({
            "user_request": "使用 USRP 发射信号",
            "retryable_task_ids": [],
        })

        self.assertIs(NextAction.CONTINUE, decision.action)
        self.assertEqual([], decision.revised_tasks)
        request_body = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        instructions = request_body["messages"][0]["content"]
        self.assertIn("revised_tasks 必须是空数组 []", instructions)
        self.assertIn("不得复制当前计划或待执行任务", instructions)

    @patch("simple_orchestrator.planner.urlopen")
    def test_batch_review_still_rejects_non_replan_with_revised_tasks(
        self, urlopen: MagicMock
    ) -> None:
        urlopen.return_value = self._ollama_response({
            "action": "continue",
            "reason": "继续执行现有计划",
            "retry_task_ids": [],
            "revised_tasks": [task_data("T1", "usrp", "usrp.transmit", [])],
        })

        with self.assertRaisesRegex(PlanningError, "非 replan 决策不应包含 revised_tasks"):
            OllamaPlanner(model="local-model").review_batch({
                "user_request": "使用 USRP 发射信号",
                "retryable_task_ids": [],
            })

    @patch("simple_orchestrator.planner.urlopen")
    def test_registry_accepts_usrp_read_only_knowledge_task(
        self, urlopen: MagicMock
    ) -> None:
        intent = {
            "intent_type": "knowledge_query",
            "domain": "wireless_communication",
            "target_type": "usrp",
            "action": "explain",
            "execution_mode": "informational",
            "required_capabilities": ["knowledge.explain"],
            "provided_parameters": {},
            "missing_parameters": [],
            "needs_clarification": False,
            "evidence": "如何使用 USRP",
            "reason": "只询问操作知识",
        }
        urlopen.side_effect = [
            self._ollama_response(intent),
            self._ollama_response({"tasks": [task_data(
                "usrp_knowledge", "usrp", "knowledge.explain", [],
            )]}),
        ]

        plan = OllamaPlanner(
            model="local-model", enable_agent_registry=True,
        ).create_plan("如何使用 USRP")

        self.assertIs(PlanStatus.READY, plan.status)
        self.assertEqual("usrp", plan.tasks[0].agent)
        self.assertEqual("knowledge.explain", plan.tasks[0].capability)

    @patch("simple_orchestrator.planner.urlopen")
    def test_analyzes_intent_then_plans_with_registry_candidates(
        self, urlopen: MagicMock
    ) -> None:
        plan_payload = {
            "tasks": [
                task_data("select_usrp", "device_manager", "device.select", []),
                task_data(
                    "transmit",
                    "usrp",
                    "usrp.transmit",
                    ["select_usrp"],
                ),
            ]
        }
        plan_payload["tasks"][1]["inputs"] = {
            "device_id": "select_usrp.device_id",
            "operation": "transmit",
        }
        urlopen.side_effect = [
            self._ollama_response(hardware_intent()),
            self._ollama_response(plan_payload),
        ]

        plan = OllamaPlanner(
            model="local-model",
            temperature=0.6,
            enable_agent_registry=True,
        ).create_plan("使用 USRP 发射 QPSK 信号")

        self.assertIs(PlanStatus.READY, plan.status)
        self.assertEqual(
            ["device_manager", "usrp"], [task.agent for task in plan.tasks]
        )
        self.assertEqual(2, urlopen.call_count)
        planning_request = json.loads(
            urlopen.call_args_list[1].args[0].data.decode("utf-8")
        )
        planning_content = planning_request["messages"][1]["content"]
        self.assertIn('"agent_id": "device_manager"', planning_content)
        self.assertIn('"agent_id": "usrp"', planning_content)
        self.assertNotIn('"agent_id": "knowledge"', planning_content)
        self.assertEqual(0.6, planning_request["options"]["temperature"])

    @patch("simple_orchestrator.planner.urlopen")
    def test_enabled_registry_injects_device_selection(
        self, urlopen: MagicMock
    ) -> None:
        intent_payload = hardware_intent()
        intent_payload["required_capabilities"] = ["usrp.transmit"]
        plan_payload = {
            "tasks": [
                task_data("select_usrp", "device_manager", "device.select", []),
                task_data(
                    "transmit", "usrp", "usrp.transmit", ["select_usrp"]
                ),
            ]
        }
        urlopen.side_effect = [
            self._ollama_response(intent_payload),
            self._ollama_response(plan_payload),
        ]

        plan = OllamaPlanner(
            model="local-model",
            enable_agent_registry=True,
        ).create_plan("使用 USRP 发射 QPSK 信号")

        self.assertEqual(
            ["device.select", "usrp.transmit"],
            plan.intent.required_capabilities,
        )
        planning_request = json.loads(
            urlopen.call_args_list[1].args[0].data.decode("utf-8")
        )
        planning_content = planning_request["messages"][1]["content"]
        self.assertIn('"agent_id": "device_manager"', planning_content)

    @patch("simple_orchestrator.planner.urlopen")
    def test_incomplete_usrp_intent_stops_before_planning(
        self, urlopen: MagicMock
    ) -> None:
        urlopen.return_value = self._ollama_response(
            hardware_intent(missing=["center_frequency_hz", "gain_db"])
        )

        plan = OllamaPlanner(model="local-model").create_plan("使用 USRP 发射信号")

        self.assertIs(PlanStatus.NEEDS_CLARIFICATION, plan.status)
        self.assertEqual([], plan.tasks)
        self.assertIn("请提供中心频率", plan.clarification_questions)
        self.assertEqual(1, urlopen.call_count)

    @patch("simple_orchestrator.planner.urlopen")
    def test_demo_policy_passes_missing_parameters_to_planner(
        self, urlopen: MagicMock
    ) -> None:
        plan_payload = {
            "tasks": [
                task_data("transmit", "usrp", "usrp.transmit", []),
            ]
        }
        urlopen.side_effect = [
            self._ollama_response(
                hardware_intent(missing=["center_frequency_hz", "gain_db"])
            ),
            self._ollama_response(plan_payload),
        ]

        plan = OllamaPlanner(
            model="local-model",
            parameter_policy=ParameterPolicy.DEMO_DEFAULTS,
        ).create_plan("使用 USRP 发射信号")

        self.assertIs(PlanStatus.READY, plan.status)
        self.assertEqual(2, urlopen.call_count)
        self.assertEqual(
            "use_backend_defaults",
            plan.tasks[0].inputs["parameter_policy"],
        )
        self.assertEqual(
            ["center_frequency_hz", "gain_db"],
            plan.tasks[0].inputs["unresolved_parameters"],
        )
        planning_request = json.loads(
            urlopen.call_args_list[1].args[0].data.decode("utf-8")
        )
        self.assertIn(
            "use_backend_defaults",
            planning_request["messages"][1]["content"],
        )

    @patch("simple_orchestrator.planner.urlopen")
    def test_non_usrp_hardware_is_unsupported_without_planning(
        self, urlopen: MagicMock
    ) -> None:
        intent = hardware_intent()
        intent.update(
            {
                "intent_type": "unsupported",
                "target_type": "ris",
                "required_capabilities": [],
                "reason": "当前系统不支持 RIS 执行",
            }
        )
        urlopen.return_value = self._ollama_response(intent)

        plan = OllamaPlanner(model="local-model").create_plan("启动 RIS")

        self.assertIs(PlanStatus.UNSUPPORTED, plan.status)
        self.assertEqual([], plan.tasks)
        self.assertEqual(1, urlopen.call_count)

    @patch("simple_orchestrator.planner.urlopen")
    def test_hardware_gate_requires_usrp_in_original_request(
        self, urlopen: MagicMock
    ) -> None:
        # Even an incorrect LLM classification cannot turn a RIS request into
        # an eligible USRP hardware operation.
        urlopen.return_value = self._ollama_response(hardware_intent())

        plan = OllamaPlanner(model="local-model").create_plan("启动 RIS")

        self.assertIs(PlanStatus.UNSUPPORTED, plan.status)
        self.assertIn("明确指定 USRP", plan.reason)
        self.assertEqual(1, urlopen.call_count)

    @patch("simple_orchestrator.planner.urlopen")
    def test_registry_disabled_accepts_usrp_without_device_selection(
        self, urlopen: MagicMock
    ) -> None:
        registry = MagicMock()
        urlopen.side_effect = [
            self._ollama_response(hardware_intent()),
            self._ollama_response(
                {
                    "tasks": [
                        task_data("transmit", "usrp", "usrp.transmit", [])
                    ]
                }
            ),
        ]

        plan = OllamaPlanner(
            model="local-model",
            registry=registry,
        ).create_plan("使用 USRP 发射信号")

        self.assertIs(PlanStatus.READY, plan.status)
        self.assertEqual(["usrp.transmit"], plan.intent.required_capabilities)
        self.assertEqual(["usrp"], [task.agent for task in plan.tasks])
        planning_request = json.loads(
            urlopen.call_args_list[1].args[0].data.decode("utf-8")
        )
        planning_content = planning_request["messages"][1]["content"]
        self.assertIn('"agent_registry_enabled": false', planning_content)
        self.assertNotIn('"agent_id": "device_manager"', planning_content)
        registry.match.assert_not_called()

    @patch("simple_orchestrator.planner.urlopen")
    def test_enabled_registry_rejects_usrp_without_device_selection(
        self, urlopen: MagicMock
    ) -> None:
        urlopen.side_effect = [
            self._ollama_response(hardware_intent()),
            self._ollama_response(
                {
                    "tasks": [
                        task_data("transmit", "usrp", "usrp.transmit", [])
                    ]
                }
            ),
        ]

        with self.assertRaisesRegex(PlanningError, "必须依赖 device.select"):
            OllamaPlanner(
                model="local-model",
                enable_agent_registry=True,
            ).create_plan("使用 USRP 发射信号")

    @patch("simple_orchestrator.planner.urlopen")
    def test_reviews_result_with_structured_decision(self, urlopen: MagicMock) -> None:
        urlopen.return_value = self._ollama_response(
            {
                "action": "retry",
                "reason": "设备信息暂时不可用，可以重新查询",
                "revised_tasks": [],
            }
        )
        task = Task(
            "query_devices",
            "device_manager",
            "查询网络设备",
            status=TaskStatus.FAILED,
            capability="device.list",
        )
        state = WorkflowState(
            user_request="查询可用 USRP 设备",
            intent=device_intent(),
            tasks=[task],
            results={
                task.task_id: AgentResult(
                    ResultStatus.FAILED,
                    "设备信息查询失败",
                    error="temporary unavailable",
                )
            },
        )

        decision = OllamaPlanner(model="local-model").review(
            state, task, state.results[task.task_id]
        )

        self.assertIs(NextAction.RETRY, decision.action)
        request_body = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertIn("设备信息查询失败", request_body["messages"][1]["content"])
        self.assertIn('"remaining": 1', request_body["messages"][1]["content"])

    def test_rejects_temperature_outside_supported_range(self) -> None:
        with self.assertRaisesRegex(PlanningError, "必须在 0 到 2 之间"):
            OllamaPlanner(model="local-model", temperature=2.1).create_plan("测试")

    @patch("simple_orchestrator.planner.urlopen")
    def test_planning_timeout_reports_phase_and_limit(
        self, urlopen: MagicMock
    ) -> None:
        urlopen.side_effect = [
            self._ollama_response(hardware_intent()),
            TimeoutError(),
        ]

        with self.assertRaisesRegex(
            PlanningError, "Ollama planning 阶段超过 7 秒"
        ):
            OllamaPlanner(model="local-model", timeout=7).create_plan(
                "使用 USRP 发射 QPSK 信号"
            )

    @patch("simple_orchestrator.planner.urlopen")
    def test_log_keeps_only_the_latest_twenty_responses(
        self, urlopen: MagicMock
    ) -> None:
        urlopen.return_value = self._ollama_response(hardware_intent())

        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "ollama_responses.json"
            planner = OllamaPlanner(model="local-model", log_path=log_path)
            for index in range(22):
                planner.analyze_intent(f"request-{index}")
            log_document = json.loads(log_path.read_text(encoding="utf-8"))
            entries = log_document["entries"]

        self.assertEqual(20, log_document["max_entries"])
        self.assertEqual(20, len(entries))
        self.assertEqual("request-2", entries[0]["user_request"])
        self.assertEqual("request-21", entries[-1]["user_request"])
        self.assertEqual("intent_analysis", entries[-1]["phase"])

    def test_read_log_entries_supports_legacy_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            legacy_path = Path(temporary_directory) / "ollama_responses.jsonl"
            legacy_path.write_text(
                '{"user_request": "first"}\n{"user_request": "second"}\n',
                encoding="utf-8",
            )
            entries = OllamaPlanner._read_log_entries(
                Path(temporary_directory) / "ollama_responses.json"
            )
        self.assertEqual(["first", "second"], [entry["user_request"] for entry in entries])


if __name__ == "__main__":
    unittest.main()
