import unittest
from io import StringIO
from unittest.mock import patch

from simple_orchestrator.main import (
    build_parser,
    choose_fake_outcome,
    describe_intent,
    describe_plan,
    main,
)
from simple_orchestrator.models import (
    ExecutionMode,
    IntentAnalysis,
    IntentType,
    ResultStatus,
    Task,
)


class MainOutputTests(unittest.TestCase):
    @patch("builtins.input", side_effect=["解释 QPSK", "1"])
    @patch("sys.argv", ["simple-orchestrator", "--planner", "fixed"])
    @patch("sys.stdout", new_callable=StringIO)
    def test_interactive_mode_reads_one_request_and_exits(
        self, stdout: StringIO, input_mock: unittest.mock.Mock
    ) -> None:
        exit_code = main()

        self.assertEqual(0, exit_code)
        self.assertEqual(2, input_mock.call_count)
        input_mock.assert_any_call("请输入任务请求：")
        input_mock.assert_any_call("请输入序号（直接回车默认 1）：")
        self.assertIn("knowledge_query", stdout.getvalue())
        self.assertIn("知识智能体负责", stdout.getvalue())

    @patch("builtins.input", return_value="2")
    def test_fake_outcome_can_be_selected_with_one_number(
        self, input_mock: unittest.mock.Mock
    ) -> None:
        with patch("sys.stdout", new_callable=StringIO):
            outcome = choose_fake_outcome(None)

        self.assertIs(ResultStatus.REJECTED, outcome)
        input_mock.assert_called_once()

    def test_configured_fake_outcome_skips_the_menu(self) -> None:
        outcome = choose_fake_outcome("unknown")

        self.assertIs(ResultStatus.UNKNOWN, outcome)

    def test_command_line_can_override_ollama_temperature(self) -> None:
        args = build_parser().parse_args(
            ["测试任务", "--ollama-temperature", "0.8"]
        )

        self.assertEqual(0.8, args.ollama_temperature)

    def test_agent_registry_is_disabled_by_default_and_can_be_enabled(self) -> None:
        disabled = build_parser().parse_args(["测试任务"])
        enabled = build_parser().parse_args(
            ["测试任务", "--enable-agent-registry"]
        )

        self.assertFalse(disabled.enable_agent_registry)
        self.assertTrue(enabled.enable_agent_registry)

    def test_fake_cli_uses_demo_defaults_for_approximate_request(self) -> None:
        argv = [
            "simple-orchestrator",
            "使用 USRP 发送信号",
            "--planner",
            "fixed",
            "--fake-outcome",
            "succeeded",
        ]
        with patch("sys.argv", argv), patch(
            "sys.stdout", new_callable=StringIO
        ) as stdout:
            exit_code = main()

        self.assertEqual(0, exit_code)
        self.assertIn("USRP 硬件智能体：执行成功", stdout.getvalue())
        self.assertNotIn("执行前需要补充信息", stdout.getvalue())

    def test_strict_cli_accepts_follow_up_parameters(self) -> None:
        argv = [
            "simple-orchestrator",
            "使用 USRP 发送信号",
            "--planner",
            "fixed",
            "--fake-outcome",
            "succeeded",
            "--parameter-policy",
            "strict",
        ]
        supplement = (
            "中心频率 2.4GHz、采样率 1MHz、增益 10dB、"
            "QPSK 波形、时长 5 秒"
        )
        with patch("sys.argv", argv), patch(
            "builtins.input", return_value=supplement
        ) as input_mock, patch("sys.stdout", new_callable=StringIO) as stdout:
            exit_code = main()

        self.assertEqual(0, exit_code)
        input_mock.assert_called_once()
        self.assertIn("执行前需要补充信息", stdout.getvalue())
        self.assertIn("USRP 硬件智能体：执行成功", stdout.getvalue())

    def test_describe_plan_uses_natural_language_and_dependencies(self) -> None:
        tasks = [
            Task("get_knowledge", "knowledge", "查询 QPSK 的基本原理"),
            Task(
                "query_network_devices",
                "device_manager",
                "查询网络设备信息并选择可用设备",
                dependencies=["get_knowledge"],
            ),
        ]

        output = describe_plan(tasks, "LLM ")

        self.assertIn("LLM 规划结果（已解析）", output)
        self.assertIn("知识智能体负责：查询 QPSK 的基本原理", output)
        self.assertIn("该任务没有前置依赖", output)
        self.assertIn("设备管理智能体负责", output)
        self.assertIn("第1项任务（get_knowledge）", output)

    def test_describe_intent_shows_structured_routing_result(self) -> None:
        intent = IntentAnalysis(
            IntentType.HARDWARE_EXECUTION,
            "wireless_communication",
            "usrp",
            "transmit",
            ExecutionMode.PHYSICAL,
            required_capabilities=["device.select", "usrp.transmit"],
        )

        output = describe_intent(intent)

        self.assertIn("hardware_execution", output)
        self.assertIn("usrp.transmit", output)


if __name__ == "__main__":
    unittest.main()
