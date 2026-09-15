import unittest

from simple_orchestrator.fakes import (
    FakeDeviceManagerPort,
    FakeKnowledgePort,
    FakeUsrpAgent,
)
from simple_orchestrator.models import (
    AgentResult,
    ExecutionMode,
    IntentAnalysis,
    IntentType,
    NextAction,
    ParameterPolicy,
    PlanStatus,
    ResultStatus,
    ReviewDecision,
    Task,
    TaskPlan,
    TaskStatus,
    WorkflowStatus,
)
from simple_orchestrator.orchestrator import Orchestrator
from simple_orchestrator.planner import FixedPlanner


def knowledge_intent() -> IntentAnalysis:
    return IntentAnalysis(
        IntentType.KNOWLEDGE_QUERY,
        "wireless_communication",
        None,
        "explain",
        ExecutionMode.INFORMATIONAL,
        required_capabilities=["knowledge.explain"],
        evidence="查询知识",
    )


def ready_plan(tasks: list[Task]) -> TaskPlan:
    return TaskPlan(PlanStatus.READY, knowledge_intent(), tasks)


COMPLETE_USRP_REQUEST = (
    "使用 USRP 以中心频率 2.4GHz、采样率 1MHz、增益 10dB，"
    "发射 QPSK 信号 5 秒"
)


class OrchestratorTests(unittest.TestCase):
    def test_planner_decision_can_retry_a_failed_task(self) -> None:
        class RetryPlanner:
            def create_plan(self, user_request: str) -> TaskPlan:
                return ready_plan([Task("knowledge_once", "knowledge", "查询任务知识")])

            def review(self, state, current_task, result) -> ReviewDecision:
                if result.status is ResultStatus.FAILED:
                    return ReviewDecision(
                        current_task.task_id,
                        NextAction.RETRY,
                        "首次查询失败，可以重新执行",
                    )
                return ReviewDecision(
                    current_task.task_id,
                    NextAction.COMPLETE,
                    "重新执行后已经获得结果",
                )

        class FlakyKnowledge:
            def __init__(self) -> None:
                self.call_count = 0

            def invoke(self, request) -> AgentResult:
                self.call_count += 1
                if self.call_count == 1:
                    return AgentResult(ResultStatus.FAILED, "第一次查询失败")
                return AgentResult(ResultStatus.SUCCEEDED, "第二次查询成功")

        knowledge = FlakyKnowledge()
        state = Orchestrator(
            RetryPlanner(),
            knowledge,
            FakeDeviceManagerPort(),
            FakeUsrpAgent(),
            max_retries=1,
        ).run("查询知识")

        self.assertEqual(WorkflowStatus.SUCCEEDED, state.status)
        self.assertEqual(2, knowledge.call_count)
        self.assertEqual(2, state.tasks[0].attempts)

    def test_planner_decision_can_replace_remaining_plan(self) -> None:
        class ReplanningPlanner:
            def create_plan(self, user_request: str) -> TaskPlan:
                return ready_plan(
                    [Task("failed_knowledge", "knowledge", "查询不可用知识源")]
                )

            def review(self, state, current_task, result) -> ReviewDecision:
                if current_task.task_id == "failed_knowledge":
                    return ReviewDecision(
                        current_task.task_id,
                        NextAction.REPLAN,
                        "知识源失败，改为查询设备信息",
                        [
                            Task(
                                "device_fallback",
                                "device_manager",
                                "查询网络中的设备信息",
                            )
                        ],
                    )
                return ReviewDecision(
                    current_task.task_id,
                    NextAction.COMPLETE,
                    "替代计划已经获得所需信息",
                )

        device_manager = FakeDeviceManagerPort()
        state = Orchestrator(
            ReplanningPlanner(),
            FakeKnowledgePort(should_fail=True),
            device_manager,
            FakeUsrpAgent(),
            max_replans=1,
        ).run("获取可用设备信息")

        self.assertEqual(WorkflowStatus.SUCCEEDED, state.status)
        self.assertEqual(2, state.plan_version)
        self.assertEqual(1, device_manager.call_count)

    def test_knowledge_request_does_not_call_hardware_agents(self) -> None:
        knowledge = FakeKnowledgePort()
        device_manager = FakeDeviceManagerPort()
        usrp = FakeUsrpAgent()

        state = Orchestrator(
            FixedPlanner(), knowledge, device_manager, usrp
        ).run("解释 QPSK 原理")

        self.assertEqual(WorkflowStatus.SUCCEEDED, state.status)
        self.assertEqual(1, knowledge.call_count)
        self.assertEqual(0, device_manager.call_count)
        self.assertEqual(0, usrp.call_count)

    def test_how_to_use_usrp_is_knowledge_not_hardware_execution(self) -> None:
        knowledge = FakeKnowledgePort()
        usrp = FakeUsrpAgent()

        state = Orchestrator(
            FixedPlanner(), knowledge, FakeDeviceManagerPort(), usrp
        ).run("如何使用 USRP 发射 QPSK 信号")

        self.assertEqual(WorkflowStatus.SUCCEEDED, state.status)
        self.assertIs(IntentType.KNOWLEDGE_QUERY, state.intent.intent_type)
        self.assertEqual(1, knowledge.call_count)
        self.assertEqual(0, usrp.call_count)

    def test_usrp_status_query_only_calls_device_manager(self) -> None:
        device_manager = FakeDeviceManagerPort()
        usrp = FakeUsrpAgent()

        state = Orchestrator(
            FixedPlanner(), FakeKnowledgePort(), device_manager, usrp
        ).run("查询 USRP 状态")

        self.assertEqual(WorkflowStatus.SUCCEEDED, state.status)
        self.assertIs(IntentType.DEVICE_QUERY, state.intent.intent_type)
        self.assertEqual(1, device_manager.call_count)
        self.assertEqual(0, usrp.call_count)

    def test_complete_usrp_request_runs_usrp_directly_by_default(self) -> None:
        knowledge = FakeKnowledgePort()
        device_manager = FakeDeviceManagerPort(selection="selected")
        usrp = FakeUsrpAgent()

        state = Orchestrator(
            FixedPlanner(), knowledge, device_manager, usrp
        ).run(COMPLETE_USRP_REQUEST)

        self.assertEqual(WorkflowStatus.SUCCEEDED, state.status)
        self.assertEqual(0, knowledge.call_count)
        self.assertEqual(0, device_manager.call_count)
        self.assertEqual(1, usrp.call_count)
        self.assertEqual(["usrp"], [task.agent for task in state.tasks])
        self.assertTrue(all(task.status is TaskStatus.SUCCEEDED for task in state.tasks))

    def test_enabled_registry_runs_device_selection_then_usrp(self) -> None:
        device_manager = FakeDeviceManagerPort(selection="selected")
        usrp = FakeUsrpAgent()

        state = Orchestrator(
            FixedPlanner(enable_agent_registry=True),
            FakeKnowledgePort(),
            device_manager,
            usrp,
        ).run(COMPLETE_USRP_REQUEST)

        self.assertEqual(WorkflowStatus.SUCCEEDED, state.status)
        self.assertEqual(1, device_manager.call_count)
        self.assertEqual(1, usrp.call_count)
        self.assertEqual("USRP-01", state.results["select_usrp"].data["device_id"])
        self.assertEqual(
            ["device_manager", "usrp"], [task.agent for task in state.tasks]
        )

    def test_unavailable_device_never_calls_usrp(self) -> None:
        device_manager = FakeDeviceManagerPort(selection="unavailable")
        usrp = FakeUsrpAgent()

        state = Orchestrator(
            FixedPlanner(enable_agent_registry=True),
            FakeKnowledgePort(),
            device_manager,
            usrp,
        ).run(COMPLETE_USRP_REQUEST)

        self.assertEqual(WorkflowStatus.FAILED, state.status)
        self.assertEqual(0, usrp.call_count)
        self.assertEqual(TaskStatus.FAILED, state.tasks[0].status)
        self.assertEqual(TaskStatus.SKIPPED, state.tasks[1].status)

    def test_unknown_usrp_result_is_not_reported_as_success(self) -> None:
        usrp = FakeUsrpAgent(outcome=ResultStatus.UNKNOWN)
        state = Orchestrator(
            FixedPlanner(),
            FakeKnowledgePort(),
            FakeDeviceManagerPort(),
            usrp,
        ).run(COMPLETE_USRP_REQUEST)

        self.assertEqual(WorkflowStatus.UNKNOWN, state.status)
        self.assertNotIn("执行成功", state.final_answer)

    def test_incomplete_usrp_request_asks_for_parameters_without_execution(self) -> None:
        device_manager = FakeDeviceManagerPort()
        usrp = FakeUsrpAgent()
        state = Orchestrator(
            FixedPlanner(), FakeKnowledgePort(), device_manager, usrp
        ).run("使用 USRP 发送信号")

        self.assertEqual(WorkflowStatus.NEEDS_CLARIFICATION, state.status)
        self.assertIn("中心频率", state.final_answer)
        self.assertEqual([], state.tasks)
        self.assertEqual(0, device_manager.call_count)
        self.assertEqual(0, usrp.call_count)

    def test_demo_parameter_policy_executes_approximate_usrp_request(self) -> None:
        usrp = FakeUsrpAgent()
        state = Orchestrator(
            FixedPlanner(parameter_policy=ParameterPolicy.DEMO_DEFAULTS),
            FakeKnowledgePort(),
            FakeDeviceManagerPort(),
            usrp,
        ).run("使用 USRP 发送信号")

        self.assertEqual(WorkflowStatus.SUCCEEDED, state.status)
        self.assertEqual(1, usrp.call_count)
        configuration = state.results["execute_usrp_operation"].data[
            "actual_configuration"
        ]
        self.assertEqual("fake_defaults", configuration["source"])
        self.assertIn("center_frequency_hz", configuration["defaulted_parameters"])

    def test_ris_execution_is_unsupported_and_never_calls_usrp(self) -> None:
        usrp = FakeUsrpAgent()
        state = Orchestrator(
            FixedPlanner(), FakeKnowledgePort(), FakeDeviceManagerPort(), usrp
        ).run("启动 RIS 对目标进行定位")

        self.assertEqual(WorkflowStatus.UNSUPPORTED, state.status)
        self.assertEqual([], state.tasks)
        self.assertEqual(0, usrp.call_count)

    def test_all_fake_agents_support_rejected_responses(self) -> None:
        outcome = ResultStatus.REJECTED
        results = (
            FakeKnowledgePort(outcome=outcome).invoke({}),
            FakeDeviceManagerPort(outcome=outcome).invoke({}),
            FakeUsrpAgent(outcome=outcome).execute({}),
        )
        self.assertTrue(all(result.status is outcome for result in results))

    def test_retry_and_replan_limits_cannot_exceed_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "只能设置为 0 或 1"):
            Orchestrator(
                FixedPlanner(),
                FakeKnowledgePort(),
                FakeDeviceManagerPort(),
                FakeUsrpAgent(),
                max_retries=2,
            )
        with self.assertRaisesRegex(ValueError, "只能设置为 0 或 1"):
            Orchestrator(
                FixedPlanner(),
                FakeKnowledgePort(),
                FakeDeviceManagerPort(),
                FakeUsrpAgent(),
                max_replans=2,
            )


if __name__ == "__main__":
    unittest.main()
