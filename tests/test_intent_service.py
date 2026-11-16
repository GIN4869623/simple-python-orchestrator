import json
import os
import selectors
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import asdict
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from simple_orchestrator.errors import PlanningError
from simple_orchestrator.intent.agent import FixedIntentAgent
from simple_orchestrator.intent.client import IntentHttpClient
from simple_orchestrator.intent.service import IntentHttpServer, build_parser, build_intent_agent
from simple_orchestrator.models import HandlingMode, ParameterPolicy
from simple_orchestrator.orchestration.main import build_orchestrator, build_parser as orchestration_parser


TOKEN = "test-split-services-token"
USRP_REQUEST = "使用 USRP 以中心频率 2.4GHz、采样率 1MHz、增益 10dB，发射 QPSK 信号 5 秒"
ROOT = Path(__file__).resolve().parents[1]


def request(port, path, payload=None, *, token=TOKEN, raw=None, headers=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    outgoing = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    outgoing.update(headers or {})
    body = raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
    connection.request("GET" if body is None else "POST", path, body=body, headers=outgoing)
    response = connection.getresponse()
    result = response.status, json.loads(response.read())
    connection.close()
    return result


class IntentServiceTests(unittest.TestCase):
    def setUp(self):
        self.factory = Mock(side_effect=lambda: FixedIntentAgent())
        self.server = IntentHttpServer(("127.0.0.1", 0), self.factory, token=TOKEN, runtime_info={"backend": "fixed"})
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_analysis_round_trip_returns_typed_intent_and_no_plan(self):
        client = IntentHttpClient(f"http://127.0.0.1:{self.port}", auth_token=TOKEN)
        intent = client.analyze_intent("查询在线设备")
        self.assertIs(HandlingMode.DIRECT, intent.handling_mode)
        self.assertEqual(["device.list"], intent.required_capabilities)
        status, payload = request(self.port, "/api/intent/analyze", {"request_id": "trace-1", "request": "解释 QPSK"})
        self.assertEqual(200, status)
        self.assertEqual("trace-1", payload["request_id"])
        self.assertNotIn("tasks", payload)
        self.assertEqual("strict", payload["parameter_policy"])

    def test_service_preserves_missing_parameters_and_caller_owns_demo_policy(self):
        strict = IntentHttpClient(f"http://127.0.0.1:{self.port}", auth_token=TOKEN)
        demo = IntentHttpClient(f"http://127.0.0.1:{self.port}", auth_token=TOKEN, parameter_policy=ParameterPolicy.DEMO_DEFAULTS)
        strict_intent = strict.analyze_intent("使用 USRP 发射信号")
        demo_intent = demo.analyze_intent("使用 USRP 发射信号")
        self.assertTrue(strict_intent.needs_clarification)
        self.assertFalse(demo_intent.needs_clarification)
        self.assertIn("center_frequency_hz", strict_intent.missing_parameters)
        self.assertEqual("use_backend_defaults", demo_intent.provided_parameters["parameter_policy"])

    def test_authentication_and_invalid_input_do_not_invoke_agent(self):
        self.assertEqual(401, request(self.port, "/health", token="wrong")[0])
        self.assertEqual(401, request(self.port, "/api/intent/analyze", {"request": "解释 QPSK"}, token="wrong")[0])
        for payload in ({}, [], {"request": " "}, {"request": 42}, {"request": "a" * 2001}, {"request": "解释", "request_id": []}):
            self.assertEqual(400, request(self.port, "/api/intent/analyze", payload)[0])
        self.assertEqual(413, request(self.port, "/api/intent/analyze", raw=b"x" * (65536 + 1))[0])
        self.assertEqual(415, request(self.port, "/api/intent/analyze", {"request": "解释"}, headers={"Content-Type": "text/plain"})[0])
        self.factory.assert_not_called()

    def test_busy_and_failed_analysis_release_the_slot(self):
        self.server.analysis_lock.acquire()
        try:
            self.assertEqual(429, request(self.port, "/api/intent/analyze", {"request": "解释"})[0])
        finally:
            self.server.analysis_lock.release()
        self.factory.side_effect = PlanningError("bad intent")
        self.assertEqual(422, request(self.port, "/api/intent/analyze", {"request": "解释"})[0])
        self.assertFalse(self.server.analysis_lock.locked())

    def test_independent_intent_model_configuration_and_strict_policy(self):
        with patch.dict(os.environ, {
            "OLLAMA_MODEL": "planner-model", "INTENT_OLLAMA_MODEL": "intent-model",
            "INTENT_OLLAMA_BASE_URL": "http://intent-model:11434",
        }):
            parser = build_parser()
            args = parser.parse_args([])
            agent = build_intent_agent(args, parser)
        self.assertEqual("intent-model", agent.client.model)
        self.assertEqual("http://intent-model:11434", agent.client.base_url)
        self.assertIs(ParameterPolicy.STRICT, agent.parameter_policy)

    def test_orchestration_defaults_to_remote_intent_client(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "ORCHESTRATOR_LOCAL_INTENT": "0", "INTENT_API_TOKEN": TOKEN,
            "KNOWLEDGE_API_URL": "", "INTENT_SERVICE_URL": f"http://127.0.0.1:{self.port}",
        }):
            parser = orchestration_parser()
            args = parser.parse_args([
                "查询在线设备", "--planner", "fixed", "--fake-outcome", "succeeded",
                "--agent-log-path", str(Path(directory) / "messages.jsonl"),
            ])
            runtime, _ = build_orchestrator(args, parser)
            state = runtime.run(args.request)
        self.assertFalse(args.local_intent)
        self.assertIsInstance(runtime.intent_agent, IntentHttpClient)
        self.assertEqual("succeeded", state["status"])
        self.assertIs(HandlingMode.DIRECT, state["handling_mode"])


class IntentHttpClientValidationTests(unittest.TestCase):
    @patch("simple_orchestrator.intent.client.urlopen")
    def test_invalid_responses_are_rejected_without_local_fallback(self, urlopen):
        valid = {"intent": asdict(FixedIntentAgent().analyze_intent("解释 QPSK")), "parameter_policy": "strict"}
        cases = (
            {**valid, "request_id": "wrong"},
            {**valid, "parameter_policy": "demo_defaults"},
            {**valid, "intent": {}},
            {**valid, "intent": {**valid["intent"], "handling_mode": "unknown"}},
        )
        for payload in cases:
            with self.subTest(payload=payload):
                def read_request(req, **kwargs):
                    data = json.loads(req.data)
                    response = {"request_id": data["request_id"], **payload}
                    context = Mock()
                    context.__enter__ = Mock(return_value=Mock(read=Mock(return_value=json.dumps(response).encode())))
                    context.__exit__ = Mock(return_value=False)
                    return context
                urlopen.side_effect = read_request
                with self.assertRaises(PlanningError):
                    IntentHttpClient(auth_token=TOKEN).analyze_intent("解释 QPSK")

    @patch("simple_orchestrator.intent.client.urlopen")
    def test_transport_failures_are_reported(self, urlopen):
        for error in (
            URLError("connection refused"), TimeoutError("timed out"),
            HTTPError("http://localhost", 401, "Unauthorized", {}, None),
        ):
            with self.subTest(error=type(error).__name__):
                urlopen.side_effect = error
                with self.assertRaises(PlanningError):
                    IntentHttpClient(auth_token=TOKEN).analyze_intent("解释 QPSK")

    def test_configuration_rejects_invalid_endpoint_and_timeout(self):
        for url in ("ftp://localhost", "http://user:secret@localhost", "http://localhost:bad", "http://localhost/path"):
            with self.subTest(url=url), self.assertRaises(PlanningError):
                IntentHttpClient(url, auth_token=TOKEN)
        with self.assertRaises(PlanningError):
            IntentHttpClient(auth_token=TOKEN, timeout=0)


class SeparateProcessTests(unittest.TestCase):
    def test_two_services_run_in_distinct_processes_and_preserve_all_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            # Prevent test services from loading the user's real .env or agents.
            (task_dir / ".env").write_text("")
            env = {
                "PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT / "src"),
                "ORCHESTRATOR_API_TOKEN": TOKEN, "INTENT_API_TOKEN": TOKEN,
                "KNOWLEDGE_API_URL": "", "ORCHESTRATOR_LOCAL_INTENT": "0",
                "PYTHONUNBUFFERED": "1",
            }
            processes = []
            try:
                intent_process, intent_port = self.launch("serve-intent", ["--backend", "fixed"], task_dir, env)
                processes.append(intent_process)
                process, port = self.launch("serve-orchestrator", [
                    "--planner", "fixed", "--fake-outcome", "succeeded", "--parameter-policy", "strict",
                    "--intent-service-url", f"http://127.0.0.1:{intent_port}",
                    "--agent-log-path", str(task_dir / "messages.jsonl"),
                ], task_dir, env)
                processes.append(process)
                intent_health = request(intent_port, "/health")[1]
                health = request(port, "/health")[1]
                self.assertEqual("intent", intent_health["service"])
                self.assertEqual("orchestration", health["service"])
                self.assertEqual(intent_process.pid, intent_health["pid"])
                self.assertEqual(process.pid, health["pid"])
                self.assertNotEqual(intent_health["pid"], health["pid"])
                self.assertNotEqual(intent_port, port)
                for message, mode in (
                    ("查询在线设备", "direct"), (USRP_REQUEST, "fixed_workflow"),
                    ("分别解释 QPSK 和 BPSK", "orchestrated"),
                ):
                    status, state = request(port, "/api/orchestrator/run", {"request": message})
                    self.assertEqual(200, status)
                    self.assertEqual("succeeded", state["status"], state)
                    self.assertEqual(mode, state["handling_mode"])
                status, state = request(port, "/api/orchestrator/run", {"request": "使用 USRP 发射信号"})
                self.assertEqual("needs_clarification", state["status"])
                self.assertFalse(state["results"])
                status, pending = request(port, "/api/orchestrator/runs", {"request": "解释 QPSK"})
                self.assertEqual(202, status)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    status, state = request(port, "/api/orchestrator/runs/" + pending["request_id"])
                    if state["status"] != "running":
                        break
                    time.sleep(0.01)
                self.assertEqual("succeeded", state["status"])
                self.stop(intent_process)
                status, state = request(port, "/api/orchestrator/run", {"request": "解释 QPSK"})
                self.assertEqual(422, status)
                self.assertIn("意图识别服务", state["detail"])
            finally:
                for process in reversed(processes):
                    self.stop(process)

    def launch(self, script, args, directory, env):
        # Each launcher cds to the project root; use an explicit empty test env file
        # by running the Python module directly with the same executable as launchers.
        module = "simple_orchestrator.intent.service" if script == "serve-intent" else "simple_orchestrator.orchestration.api"
        process = subprocess.Popen(
            [sys.executable, "-m", module, "--host", "127.0.0.1", "--port", "0", *args],
            cwd=directory, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        output = []
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if selector.select(timeout=0.2):
                    line = process.stdout.readline()
                    output.append(line)
                    marker = "http://127.0.0.1:"
                    if marker in line:
                        port = int(line.split(marker, 1)[1].split("（", 1)[0])
                        return process, port
                if process.poll() is not None:
                    break
        self.stop(process)
        self.fail("Service did not start: " + "".join(output))

    @staticmethod
    def stop(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if process.stdout is not None:
            process.stdout.close()


if __name__ == "__main__":
    unittest.main()
