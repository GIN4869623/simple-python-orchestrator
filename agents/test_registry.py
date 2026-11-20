import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import registry


def profile(agent_id, description):
    return {
        "agent_id": agent_id,
        "name": agent_id,
        "description": description,
        "capabilities": [{"id": "task.check", "description": "检查状态", "required_inputs": []}],
    }


class RegistryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.catalog = self.directory / "registry.json"
        self.catalog.write_text('{"schema_version": 1, "agents": {}}', encoding="utf-8")
        self.draft("alpha", profile("alpha", "第一条记录"))
        self.draft("beta", profile("beta", "第二条记录"))
        self.cli("add", "alpha", self.directory / "alpha.json")
        self.cli("add", "beta", self.directory / "beta.json")

    def draft(self, name, value):
        path = self.directory / f"{name}.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def run_command(self, *args):
        return subprocess.run(
            [sys.executable, str(Path(registry.__file__).resolve()), "--file", str(self.catalog), *args],
            capture_output=True, text=True, timeout=10,
        )

    def cli(self, *args, status=0):
        result = self.run_command(*args)
        self.assertEqual(result.returncode, status, result.stderr)
        return result.stderr if status else json.loads(result.stdout)

    def parallel(self, *commands):
        start = Barrier(len(commands), timeout=10)

        def launch(args):
            start.wait()
            return self.run_command(*args)

        with ThreadPoolExecutor(max_workers=len(commands)) as executor:
            return list(executor.map(launch, commands))

    def test_invalid_writes_leave_other_agents_intact(self):
        before = self.catalog.read_bytes()
        alpha = self.directory / "alpha.json"
        beta = self.directory / "beta.json"
        missing = self.draft("missing", profile("missing", "仓库中没有此 Agent"))
        duplicate = profile("alpha", "重复能力草稿")
        duplicate["capabilities"].append(duplicate["capabilities"][0].copy())
        invalid = self.draft("duplicate_capabilities", duplicate)
        cases = [
            (("add", "alpha", alpha), "Agent 已存在"),
            (("update", "alpha", beta, "--expect-revision", "1"), "记录 ID 与目标 ID 不一致"),
            (("update", "missing", missing, "--expect-revision", "1"), "找不到 Agent"),
            (("update", "alpha", invalid, "--expect-revision", "1"), "重复能力 ID"),
        ]
        for command, message in cases:
            with self.subTest(command=command):
                self.assertIn(message, self.cli(*command, status=1))
                self.assertEqual(self.catalog.read_bytes(), before)

    def test_concurrent_updates_keep_both_changes(self):
        alpha = self.draft("new_alpha", profile("alpha", "第一条已更新"))
        beta = self.draft("new_beta", profile("beta", "第二条已更新"))
        results = self.parallel(
            ("update", "alpha", alpha, "--expect-revision", "1"),
            ("update", "beta", beta, "--expect-revision", "1"),
        )
        self.assertEqual([result.returncode for result in results], [0, 0], results)
        summaries = self.cli("summaries")
        self.assertEqual(summaries, [
            {"agent_id": "alpha", "description": "第一条已更新"},
            {"agent_id": "beta", "description": "第二条已更新"},
        ])
        agent_ids = [record["agent_id"] for record in summaries]
        full = self.cli("details", *agent_ids)
        self.assertEqual(set(full), set(agent_ids))
        for entry in summaries:
            detail = full[entry["agent_id"]]
            self.assertEqual(detail["revision"], 2)
            self.assertEqual(detail["profile"], profile(entry["agent_id"], entry["description"]))
        self.assertIn("找不到候选 Agent", self.cli("details", "missing", status=1))

    def test_concurrent_updates_to_same_agent_reject_one_draft(self):
        previous_beta = self.cli("details", "beta")
        changes = [self.draft(f"change_{index}", profile("alpha", str(index))) for index in range(2)]
        results = self.parallel(*[
            ("update", "alpha", change, "--expect-revision", "1") for change in changes
        ])
        self.assertEqual(sorted(result.returncode for result in results), [0, 1], results)
        winner = next(index for index, result in enumerate(results) if result.returncode == 0)
        self.assertIn("版本冲突", results[1 - winner].stderr)
        updated = self.cli("details", "alpha")["alpha"]
        self.assertEqual(updated, {"revision": 2, "profile": profile("alpha", str(winner))})
        self.assertEqual(self.cli("details", "beta"), previous_beta)

    def test_duplicate_json_keys_are_rejected_before_indexing(self):
        original = json.loads(self.catalog.read_text(encoding="utf-8"))
        alpha = json.dumps(original["agents"]["alpha"])
        self.catalog.write_text(
            f'{{"schema_version": 1, "agents": {{"alpha": {alpha}, "alpha": {alpha}}}}}',
            encoding="utf-8",
        )
        self.assertIn("重复键：alpha", self.cli("validate", status=1))

    def test_failed_replace_keeps_previous_snapshot(self):
        before = self.catalog.read_bytes()
        with patch.object(registry.os, "replace", side_effect=OSError("test write failure")):
            with self.assertRaises(OSError):
                registry.save_agent(self.catalog, "alpha", profile("alpha", "更新"), 1)
        self.assertEqual(self.catalog.read_bytes(), before)
        self.assertEqual(list(self.directory.glob(".registry.json.*.tmp")), [])

    def test_interfaces_keep_deployment_address_out_of_summary(self):
        description = "带接口的记录"
        value = profile("alpha", description)
        value["interfaces"] = [
            {"protocol": "http", "method": "POST", "url": "https://agent.example:8443/task",
             "description": "提交任务", "verification": "configured"},
            {"protocol": "sse", "method": "POST", "url": "http://agent.example:8000/stream", "description": "流式任务"},
            {"protocol": "tcp", "url": "tcp://[::1]:9000", "description": "任务协议"},
        ]
        other = self.cli("details", "beta")
        self.cli("update", "alpha", self.draft("interfaces", value), "--expect-revision", "1")
        self.assertEqual(self.cli("details", "alpha")["alpha"]["profile"], value)
        self.assertEqual(self.cli("details", "beta"), other)
        self.assertIn({"agent_id": "alpha", "description": description}, self.cli("summaries"))

    def test_invalid_interface_preserves_previous_record(self):
        before = self.catalog.read_bytes()
        value = profile("alpha", "无效的接口草稿")
        invalid = [
            {"protocol": "tcp", "url": "tcp://host:70000"},
            {"protocol": "tcp", "url": "tcp://host:9000", "method": "GET"},
            {"protocol": "http", "url": "https://host/task"},
            {"protocol": "sse", "url": "https://user:password@host/stream", "method": "POST"},
        ]
        for index, interface in enumerate(invalid):
            with self.subTest(interface=interface):
                value["interfaces"] = [{**interface, "description": "无效示例"}]
                draft = self.draft(f"invalid_interface_{index}", value)
                self.assertIn("interfaces[0]", self.cli("update", "alpha", draft, "--expect-revision", "1", status=1))
                self.assertEqual(self.catalog.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
