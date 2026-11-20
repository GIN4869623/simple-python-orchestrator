import io
import json
import unittest
from dataclasses import replace
from email.message import Message
from unittest.mock import Mock, patch

from simple_orchestrator.remote import call_ris_chat
from simple_orchestrator.models import AgentResult, ResultStatus, HandlingMode, WorkflowStatus
from simple_orchestrator.intent.agent import FixedIntentAgent
from simple_orchestrator.orchestration.execution import WorkflowExecutor
from simple_orchestrator.orchestration.graph import GraphOrchestrator
from simple_orchestrator.orchestration.planner import FixedPlanner
from simple_orchestrator.orchestration.router import RequestRouter
from simple_orchestrator.fakes import FakeKnowledgePort, FakeDeviceManagerPort, FakeUsrpAgent

CONTEXT = {'request_id': 'req-ris-test', 'user_request': '请让 RIS 只回复测试成功', 'single_task': True,
           'current_task': {'task_id': 'ris_reply', 'attempt': 1, 'description': 'RIS 文字回复', 'capability': 'knowledge.explain'}}

class Response(io.BytesIO):
    def __init__(self, events, content_type='text/event-stream'):
        super().__init__(events.encode('utf-8'))
        self.headers = Message()
        self.headers['Content-Type'] = content_type

def sse(*events):
    return ''.join('data: ' + json.dumps(e, ensure_ascii=False) + '\r\n\r\n' for e in events)

class RisReplyTests(unittest.TestCase):
    def call(self, body, content_type='text/event-stream'):
        with patch('simple_orchestrator.remote.urlopen', return_value=Response(body, content_type)) as send:
            result = call_ris_chat(CONTEXT, endpoint='http://ris.test/api/chat/stream')
        return result, send.call_args.args[0]

    def test_final_reply_is_authoritative_and_correlated_to_original_task(self):
        result, request = self.call(': heartbeat\r\n\r\n' + sse(
            {'type': 'meta', 'agent_id': 'ris-agent'}, {'type': 'thinking'},
            {'type': 'delta', 'text': '预览'},
            {'type': 'done', 'reply': '完整回复\n第二行', 'tools_total': 0, 'timing': {'total_ms': 10}}))
        self.assertIs(result.status, ResultStatus.SUCCEEDED)
        self.assertEqual(result.data['answer'], '完整回复\n第二行')
        payload = json.loads(request.data)
        self.assertEqual(payload['instruction'], CONTEXT['user_request'])
        self.assertEqual(payload['runId'], 'req-ris-test:ris_reply:1')
        self.assertIn('req-ris-test', payload['session_id'])

    def test_truncated_stream_never_promotes_partial_text_to_success(self):
        result, _ = self.call(sse({'type': 'delta', 'text': '部分回复'}))
        self.assertIs(result.status, ResultStatus.UNKNOWN)
        self.assertEqual(result.error, 'ris_response_unknown')

    def test_invalid_or_empty_final_response_is_not_success(self):
        for body, ct in [('not SSE', 'application/json'), ('data: {oops}\n\n', 'text/event-stream'),
                         (sse({'type': 'done', 'reply': ''}), 'text/event-stream')]:
            with self.subTest(body=body):
                result, _ = self.call(body, ct)
                self.assertIs(result.status, ResultStatus.UNKNOWN)

    def test_ris_explicit_error_is_preserved(self):
        result, _ = self.call(sse({'type': 'error', 'message': 'RIS 服务忙'}))
        self.assertIs(result.status, ResultStatus.FAILED)
        self.assertEqual(result.summary, 'RIS 服务忙')

    def runtimes(self, status=ResultStatus.SUCCEEDED):
        ris = Mock(return_value=AgentResult(status, 'RIS 原文回答', {'answer': 'RIS 原文回答'}))
        ports = (FakeKnowledgePort(), FakeDeviceManagerPort(), FakeUsrpAgent())
        graph = GraphOrchestrator(FixedPlanner(), *ports, ris=ris)
        executor = WorkflowExecutor(*ports, ris=ris)
        return RequestRouter(FixedIntentAgent(), executor, graph), ris

    def test_direct_result_reaches_existing_state_without_calling_other_agents(self):
        router, ris = self.runtimes()
        result = router.run('请 RIS 解释反射面的作用')
        self.assertIs(result['status'], WorkflowStatus.SUCCEEDED)
        self.assertIs(result['handling_mode'], HandlingMode.DIRECT)
        self.assertEqual(result['results']['ris_reply'][0].result.data['answer'], 'RIS 原文回答')
        self.assertIn('RIS 原文回答', result['final_answer'])
        ris.assert_called_once()
        self.assertEqual(router.executor.knowledge.call_count, 0)

    def test_dynamic_result_is_kept_in_orchestrator_state(self):
        router, ris = self.runtimes()
        intent = replace(FixedIntentAgent().analyze_intent('请 RIS 解释反射面'), handling_mode=HandlingMode.ORCHESTRATED)
        result = router.run_with_intent('请 RIS 解释反射面', intent)
        self.assertIs(result['status'], WorkflowStatus.SUCCEEDED)
        self.assertIs(result['handling_mode'], HandlingMode.ORCHESTRATED)
        self.assertEqual(result['results']['ris_reply'][0].agent_id, 'ris')
        ris.assert_called_once()

    def test_explicit_ris_business_request_reaches_ris_not_usrp(self):
        router, ris = self.runtimes()
        result = router.run("启动 RIS 对目标进行定位")
        self.assertIs(result["status"], WorkflowStatus.SUCCEEDED)
        self.assertEqual(result["results"]["ris_reply"][0].agent_id, "ris")
        self.assertEqual(router.executor.usrp.call_count, 0)
        ris.assert_called_once()

    def test_missing_final_reply_stops_without_retrying_ris(self):
        router, ris = self.runtimes(ResultStatus.UNKNOWN)
        result = router.run('请 RIS 解释反射面')
        self.assertIs(result['status'], WorkflowStatus.UNKNOWN)
        ris.assert_called_once()

if __name__ == '__main__':
    unittest.main()
