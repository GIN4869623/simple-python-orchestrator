import json
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from simple_orchestrator.orchestration.main import (
    build_executor,
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
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.agent_log_path = Path(temporary.name) / "messages.jsonl"
        environment = patch.dict(os.environ, {
            "ORCHESTRATOR_AGENT_LOG_PATH": str(self.agent_log_path),
            "KNOWLEDGE_API_URL": "",
            "KNOWLEDGE_API_KEY": "",
            "ORCHESTRATOR_LOCAL_INTENT": "1",
        })
        environment.start()
        self.addCleanup(environment.stop)

    def test_executor_keeps_the_actual_knowledge_sender(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--knowledge-api-url", "http://knowledge/answer", "--fake-outcome", "succeeded",
        ])
        orchestrator, _ = build_executor(args, parser)
        intent, _ = build_executor(args, parser, knowledge_sender="意图识别智能体")
        self.assertEqual("编排智能体", orchestrator.knowledge.sender)
        self.assertEqual("意图识别智能体", intent.knowledge.sender)

    @patch("simple_orchestrator.knowledge_http.urlopen")
    def test_real_knowledge_can_run_without_other_remote_agents(self, urlopen) -> None:
        response = urlopen.return_value.__enter__.return_value
        response.status = 200
        response.read.return_value = json.dumps({
            "status": "success", "summary": "知识服务已回答",
            "data": {"knowledge": "第一行\n第二行"}, "error": None,
        }).encode()
        argv = [
            "simple-orchestrator", "请解释：认知选频技能在发送前需要做什么？请用两句话回答。", "--planner", "fixed",
            "--knowledge-api-url", "--fake-outcome", "succeeded",
        ]
        with patch.dict(os.environ, {"KNOWLEDGE_API_KEY": "cli-test-key"}), patch(
            "sys.argv", argv
        ), patch("sys.stdout", new_callable=StringIO) as stdout:
            self.assertEqual(0, main())
        urlopen.assert_called_once()
        self.assertEqual("cli-test-key", urlopen.call_args.args[0].get_header("X-knowledge-api-key"))
        self.assertIn("真实 HTTP 接口", stdout.getvalue())
        entries = [json.loads(line) for line in self.agent_log_path.read_text().splitlines()]
        self.assertEqual({"knowledge"}, {e["agent_id"] for e in entries})
        self.assertEqual("第一行\n第二行", entries[1]["payload"]["data"]["knowledge"])

    def test_knowledge_api_url_can_be_configured_and_overridden(self) -> None:
        with patch.dict(os.environ, {"KNOWLEDGE_API_URL": "http://custom/answer"}):
            self.assertEqual("http://custom/answer", build_parser().parse_args([]).knowledge_api_url)
            self.assertEqual("http://override/answer", build_parser().parse_args(
                ["--knowledge-api-url", "http://override/answer"]
            ).knowledge_api_url)

    @patch("simple_orchestrator.knowledge_http.urlopen")
    def test_configured_knowledge_service_is_used_without_extra_flags_or_menu(self, urlopen) -> None:
        response = urlopen.return_value.__enter__.return_value
        response.status = 200
        response.read.return_value = json.dumps({
            "status": "success", "summary": "真实接口的回答",
            "data": {"knowledge": "完整回答"}, "error": None,
        }).encode()
        argv = ["simple-orchestrator", "解释 QPSK 原理", "--planner", "fixed"]
        with patch.dict(os.environ, {"KNOWLEDGE_API_URL": "http://knowledge/answer"}), patch(
            "sys.argv", argv
        ), patch("builtins.input") as input_mock, patch("sys.stdout", new_callable=StringIO) as stdout:
            self.assertEqual(0, main())
        input_mock.assert_not_called()
        urlopen.assert_called_once()
        self.assertEqual("http://knowledge/answer", urlopen.call_args.args[0].full_url)
        self.assertEqual("解释 QPSK 原理", json.loads(urlopen.call_args.args[0].data)["question"])
        self.assertIn("真实 HTTP 接口", stdout.getvalue())
        self.assertIn("真实接口的回答", stdout.getvalue())

    @patch("simple_orchestrator.knowledge_http.urlopen")
    def test_remote_mode_accepts_knowledge_http_in_place_of_rpc(self, urlopen) -> None:
        response = urlopen.return_value.__enter__.return_value
        response.status = 200
        response.read.return_value = json.dumps({
            "status": "success", "summary": "已回答",
            "data": {"knowledge": "完整知识回答"}, "error": None,
        }).encode()
        argv = [
            "simple-orchestrator", "解释 QPSK", "--planner", "fixed",
            "--remote-agents", "--parameter-policy", "strict", "--knowledge-api-url",
            "--device-rpc-url", "tcp://device:9000", "--usrp-rpc-url", "http://usrp/rpc",
        ]
        with patch("sys.argv", argv), patch("sys.stdout", new_callable=StringIO):
            self.assertEqual(0, main())
        urlopen.assert_called_once()

    def test_cli_logs_fake_messages_and_can_override_path(self) -> None:
        path = self.agent_log_path.with_name("custom.jsonl")
        for request in ("解释 QPSK", "查询有哪些可用设备", "使用 USRP 发送信号"):
            argv = [
                "simple-orchestrator", request, "--planner", "fixed",
                "--fake-outcome", "succeeded", "--agent-log-path", str(path),
            ]
            with patch("sys.argv", argv), patch("sys.stdout", new_callable=StringIO):
                self.assertEqual(0, main())
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual({"knowledge", "device_manager"}, {e["agent_id"] for e in entries})
        self.assertEqual(8, len(entries))
        self.assertEqual(3, len({e["request_id"] for e in entries}))
        for entry in entries:
            if entry["event"] == "request":
                self.assertEqual({"message"}, set(entry["payload"]))
                self.assertNotIn("\n", entry["payload"]["message"])
        self.assertFalse(self.agent_log_path.exists())

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

    def test_rejected_fake_stops_after_one_batch(self) -> None:
        argv = ["simple-orchestrator", "解释 QPSK 原理", "--planner", "fixed"]
        with patch("sys.argv", argv), patch("builtins.input", return_value="2") as input_mock, patch(
            "sys.stdout", new_callable=StringIO
        ) as stdout:
            exit_code = main()

        self.assertEqual(1, exit_code)
        input_mock.assert_called_once()
        self.assertIn("第 1 批：停止工作流——任务被明确拒绝", stdout.getvalue())

    def test_strict_clarification_stops_after_five_rounds(self) -> None:
        argv = [
            "simple-orchestrator", "使用 USRP 发送信号", "--planner", "fixed",
            "--parameter-policy", "strict",
        ]
        with patch("sys.argv", argv), patch(
            "builtins.input", side_effect=["2", *["频率 2.4GHz"] * 5]
        ) as input_mock, patch("sys.stdout", new_callable=StringIO) as stdout:
            exit_code = main()

        self.assertEqual(1, exit_code)
        self.assertEqual(6, input_mock.call_count)
        self.assertIn("已达到最多 5 轮参数补充，本次请求结束", stdout.getvalue())

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
                "查询网络设备信息和在线状态",
                dependencies=["get_knowledge"],
            ),
        ]

        output = describe_plan(tasks, "LLM ")

        self.assertIn("LLM 规划结果（已解析）", output)
        self.assertIn("知识智能体负责：查询 QPSK 的基本原理", output)
        self.assertIn("该任务没有前置依赖", output)
        self.assertIn("网络监管智能体负责", output)
        self.assertIn("第1项任务（get_knowledge）", output)

    def test_describe_intent_shows_structured_routing_result(self) -> None:
        intent = IntentAnalysis(
            IntentType.HARDWARE_EXECUTION,
            "wireless_communication",
            "usrp",
            "transmit",
            ExecutionMode.PHYSICAL,
            required_capabilities=["device.list", "usrp.transmit"],
        )

        output = describe_intent(intent)

        self.assertIn("hardware_execution", output)
        self.assertIn("usrp.transmit", output)


if __name__ == "__main__":
    unittest.main()
