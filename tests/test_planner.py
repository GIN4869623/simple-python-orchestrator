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
from simple_orchestrator.orchestration.planner import OllamaPlanner, PlanningError


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
            inputs["device_candidates_from"] = f"{dependencies[-1]}.network_devices"
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
    def _replan_context() -> dict:
        return {
            "user_request": "解释QPSK原理",
            "known_task_ids": ["task_1", "failed_task", "pending_task"],
            "successful_task_ids": ["task_1"],
            "retryable_task_ids": [],
            "can_replan": True,
            "intent_analysis": {
                "intent_type": "knowledge_query",
                "domain": "wireless_communication",
                "target_type": None,
                "action": "explain",
                "execution_mode": "read_only",
                "required_capabilities": ["knowledge.explain"],
                "provided_parameters": {},
                "missing_parameters": [],
                "needs_clarification": False,
                "evidence": "解释QPSK原理",
                "reason": "用户请求知识解释",
            },
            "current_plan": [],
        }

    @staticmethod
    def _ollama_response(payload: dict) -> MagicMock:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"message": {"role": "assistant", "content": json.dumps(payload)}}
        ).encode("utf-8")
        return response

    @patch("simple_orchestrator.ollama.urlopen")
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
        self.assertIn("不得使用 known_task_ids 中的任何 ID", instructions)

    @patch("simple_orchestrator.ollama.urlopen")
    def test_replan_corrects_reserved_ids_once_and_logs_correction(self, urlopen):
        for reserved_id in ("task_1", "failed_task", "pending_task"):
            with self.subTest(reserved_id=reserved_id), tempfile.TemporaryDirectory() as directory:
                invalid = task_data(reserved_id, "knowledge", "knowledge.explain", [])
                corrected = task_data("task_2", "knowledge", "knowledge.explain", ["task_1"])
                review = {
                    "action": "replan", "reason": "知识结果不足",
                    "retry_task_ids": [], "revised_tasks": [invalid],
                }
                urlopen.reset_mock()
                urlopen.side_effect = [
                    self._ollama_response(review),
                    self._ollama_response({"tasks": [corrected]}),
                ]
                log_path = Path(directory) / "responses.json"
                decision = OllamaPlanner(model="local-model", log_path=log_path).review_batch(
                    self._replan_context(),
                )
                self.assertIs(NextAction.REPLAN, decision.action)
                self.assertEqual("知识结果不足", decision.reason)
                self.assertEqual("task_2", decision.revised_tasks[0].task_id)
                self.assertEqual(["task_1"], decision.revised_tasks[0].dependencies)
                self.assertEqual(reserved_id, invalid["task_id"])
                self.assertEqual(2, urlopen.call_count)
                correction = json.loads(urlopen.call_args.args[0].data)
                prompt = json.loads(correction["messages"][1]["content"])
                self.assertIn(reserved_id, prompt["validation_error"])
                self.assertEqual([invalid], prompt["invalid_plan"]["tasks"])
                entries = json.loads(log_path.read_text())["entries"]
                self.assertEqual(["batch_review", "replan_id_correction"], [e["phase"] for e in entries])

    @patch("simple_orchestrator.ollama.urlopen")
    def test_replan_corrects_duplicate_ids_within_new_plan(self, urlopen):
        urlopen.side_effect = [
            self._ollama_response({
                "action": "replan", "reason": "拆分问题", "retry_task_ids": [],
                "revised_tasks": [task_data("new", "knowledge", "knowledge.explain", [])] * 2,
            }),
            self._ollama_response({"tasks": [
                task_data("new_1", "knowledge", "knowledge.explain", []),
                task_data("new_2", "knowledge", "knowledge.explain", ["new_1"]),
            ]}),
        ]
        decision = OllamaPlanner(model="local-model").review_batch(self._replan_context())
        self.assertEqual(["new_1", "new_2"], [t.task_id for t in decision.revised_tasks])
        self.assertEqual(["new_1"], decision.revised_tasks[1].dependencies)

    @patch("simple_orchestrator.ollama.urlopen")
    def test_invalid_id_correction_fails_without_a_third_model_call(self, urlopen):
        for corrected, error in (
            ([task_data("task_1", "knowledge", "knowledge.explain", [])], "任务 ID 重复"),
            ([task_data("new", "knowledge", "knowledge.explain", ["failed_task"])], "依赖尚未出现"),
            ([task_data("new", "knowledge", "knowledge.explain", ["pending_task"])], "依赖尚未出现"),
        ):
            with self.subTest(error=error, corrected=corrected):
                urlopen.reset_mock()
                urlopen.side_effect = [
                    self._ollama_response({
                        "action": "replan", "reason": "重规划", "retry_task_ids": [],
                        "revised_tasks": [task_data("task_1", "knowledge", "knowledge.explain", [])],
                    }),
                    self._ollama_response({"tasks": corrected}),
                ]
                with self.assertRaisesRegex(PlanningError, error):
                    OllamaPlanner(model="local-model").review_batch(self._replan_context())
                self.assertEqual(2, urlopen.call_count)

    @patch("simple_orchestrator.ollama.urlopen")
    def test_valid_replan_needs_no_correction(self, urlopen):
        urlopen.return_value = self._ollama_response({
            "action": "replan", "reason": "重规划", "retry_task_ids": [],
            "revised_tasks": [task_data("new", "knowledge", "knowledge.explain", [])],
        })
        decision = OllamaPlanner(model="local-model").review_batch(self._replan_context())
        self.assertEqual("new", decision.revised_tasks[0].task_id)
        self.assertEqual(1, urlopen.call_count)

    @patch("simple_orchestrator.ollama.urlopen")
    def test_other_replan_errors_do_not_trigger_id_correction(self, urlopen):
        urlopen.return_value = self._ollama_response({
            "action": "replan", "reason": "重规划", "retry_task_ids": [],
            "revised_tasks": [task_data("new", "knowledge", "knowledge.explain", ["failed_task"])],
        })
        with self.assertRaisesRegex(PlanningError, "依赖尚未出现"):
            OllamaPlanner(model="local-model").review_batch(self._replan_context())
        self.assertEqual(1, urlopen.call_count)

    @patch("simple_orchestrator.ollama.urlopen")
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

    @patch("simple_orchestrator.ollama.urlopen")
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

    @patch("simple_orchestrator.ollama.urlopen")
    def test_analyzes_intent_then_plans_with_registry_candidates(
        self, urlopen: MagicMock
    ) -> None:
        plan_payload = {
            "tasks": [
                task_data("query_usrp", "device_manager", "device.list", []),
                task_data(
                    "transmit",
                    "usrp",
                    "usrp.transmit",
                    ["query_usrp"],
                ),
            ]
        }
        plan_payload["tasks"][1]["inputs"] = {
            "device_candidates_from": "query_usrp.network_devices",
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

    @patch("simple_orchestrator.ollama.urlopen")
    def test_enabled_registry_injects_device_query(
        self, urlopen: MagicMock
    ) -> None:
        intent_payload = hardware_intent()
        intent_payload["required_capabilities"] = ["usrp.transmit"]
        plan_payload = {
            "tasks": [
                task_data("query_usrp", "device_manager", "device.list", []),
                task_data(
                    "transmit", "usrp", "usrp.transmit", ["query_usrp"]
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
            ["device.list", "usrp.transmit"],
            plan.intent.required_capabilities,
        )
        planning_request = json.loads(
            urlopen.call_args_list[1].args[0].data.decode("utf-8")
        )
        planning_content = planning_request["messages"][1]["content"]
        self.assertIn('"agent_id": "device_manager"', planning_content)

    @patch("simple_orchestrator.ollama.urlopen")
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

    @patch("simple_orchestrator.ollama.urlopen")
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

    @patch("simple_orchestrator.ollama.urlopen")
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

    @patch("simple_orchestrator.ollama.urlopen")
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

    @patch("simple_orchestrator.ollama.urlopen")
    def test_registry_disabled_accepts_usrp_without_device_query(
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

    @patch("simple_orchestrator.ollama.urlopen")
    def test_enabled_registry_rejects_usrp_without_device_query(
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

        with self.assertRaisesRegex(PlanningError, "必须依赖设备查询任务"):
            OllamaPlanner(
                model="local-model",
                enable_agent_registry=True,
            ).create_plan("使用 USRP 发射信号")

    @patch("simple_orchestrator.ollama.urlopen")
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

    @patch("simple_orchestrator.ollama.urlopen")
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

    @patch("simple_orchestrator.ollama.urlopen")
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
