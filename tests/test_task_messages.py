import unittest

from simple_orchestrator.task_messages import make_knowledge_request


class KnowledgeRequestTests(unittest.TestCase):
    def test_single_question_keeps_user_requirements_without_planner_expansion(self) -> None:
        question = "解释 QPSK 原理。\n请用两句话回答。"
        context = {
            "user_request": question,
            "current_task": {
                "description": "解释调制、星座图、相位映射、应用并比较 BPSK",
                "capability": "knowledge.explain",
                "inputs": {"request": question},
                "expected_output": ["knowledge", "assumptions"],
                "completion_criteria": ["至少解释三个概念"],
            },
        }
        self.assertEqual({"message": question}, make_knowledge_request(context, use_user_question=True))

    def test_subtask_keeps_its_question_and_dependency_facts(self) -> None:
        context = {
            "user_request": "比较两种调制方式，然后查询 USRP 在线状态",
            "current_task": {
                "description": "对比两种调制方式并提供分析",
                "inputs": {"question": "比较 BPSK 和 QPSK 的抗噪性能，请用表格回答。", "snr_db": 5},
                "expected_output": ["不应传给知识智能体的内部字段"],
                "completion_criteria": ["不应拼入问题的内部评审标准"],
            },
            "previous_results": {"measurement": {"knowledge": "链路存在明显噪声"}},
        }
        message = make_knowledge_request(context)["message"]
        self.assertTrue(message.startswith("比较 BPSK 和 QPSK 的抗噪性能，请用表格回答。"))
        self.assertIn("snr_db为5", message)
        self.assertIn("链路存在明显噪声", message)
        self.assertNotIn("USRP 在线状态", message)
        self.assertNotIn("内部", message)

    def test_skill_lookup_asks_for_guidance_and_preserves_parameters(self) -> None:
        context = {
            "user_request": "使用 USRP 发送信号",
            "current_task": {
                "description": "查询 USRP QPSK 操作方法",
                "capability": "knowledge.lookup_skill",
                "inputs": {"frequency_hz": 2_400_000_000},
            },
        }
        message = make_knowledge_request(context)["message"]
        self.assertTrue(message.startswith("请检索可信操作指导：查询 USRP QPSK 操作方法"))
        self.assertIn("频率（Hz）为2400000000", message)
        self.assertNotIn("使用 USRP 发送信号", message)


if __name__ == "__main__":
    unittest.main()
