import io
import json
import os
import tempfile
import threading
import unittest
from dataclasses import asdict
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from simple_orchestrator.errors import PlanningError
from simple_orchestrator.model_client import DeepSeekClient, ModelSettings, make_model_client
from simple_orchestrator.orchestration.main import build_parser, build_orchestrator
from simple_orchestrator.orchestration.api import OrchestratorHttpServer
from simple_orchestrator.intent.service import build_parser as intent_parser, build_intent_agent, IntentHttpServer
from simple_orchestrator.intent.agent import FixedIntentAgent

KEY='unit-test-secret-do-not-expose'

def response(content, finish='stop'):
    from unittest.mock import MagicMock
    result=MagicMock()
    result.__enter__.return_value.read.return_value=json.dumps({
        'model':'test-deepseek', 'choices':[{'finish_reason':finish,'message':{'content':content,'reasoning_content':'internal'}}],
        'usage':{'total_tokens':8},
    }).encode()
    return result

class DeepSeekTests(unittest.TestCase):
    def call(self, client):
        return client._chat(instructions='分类',user_content='用户问题',schema={'type':'object'},user_request='问题',phase='intent_analysis')

    @patch('simple_orchestrator.model_client.urlopen')
    def test_authentication_json_mode_and_normalized_final_response(self, send):
        send.return_value=response('{"answer":"测试"}')
        client=DeepSeekClient(model='test-deepseek',api_key=KEY)
        result=self.call(client)
        request=send.call_args.args[0]
        self.assertEqual(request.full_url,'https://api.deepseek.com/chat/completions')
        self.assertEqual(request.get_header('Authorization'),'Bearer '+KEY)
        body=json.loads(request.data)
        self.assertEqual(body['response_format'],{'type':'json_object'})
        self.assertEqual(body['thinking'],{'type':'disabled'})
        self.assertEqual(body['model'],'test-deepseek')
        self.assertNotIn(KEY,json.dumps(body))
        self.assertEqual(result['message']['content'],'{"answer":"测试"}')
        self.assertNotIn('internal',json.dumps(result))
        self.assertNotIn(KEY,repr(client))

    @patch('simple_orchestrator.model_client.urlopen')
    def test_truncated_empty_non_object_and_invalid_json_are_rejected(self, send):
        for content,finish in [('{}','length'),(' ','stop'),('[]','stop'),('bad','stop')]:
            with self.subTest(content=content,finish=finish):
                send.return_value=response(content,finish)
                with self.assertRaises(PlanningError):self.call(DeepSeekClient(model='test',api_key=KEY))

    @patch('simple_orchestrator.model_client.urlopen')
    def test_errors_hide_upstream_body_and_do_not_fall_back_to_local(self, send):
        send.side_effect=HTTPError('https://api.deepseek.com/chat/completions',401,'unauthorized',{},io.BytesIO(KEY.encode()))
        with self.assertRaises(PlanningError) as error:self.call(DeepSeekClient(model='test',api_key=KEY))
        self.assertIn('401',str(error.exception));self.assertNotIn(KEY,str(error.exception));self.assertEqual(send.call_count,1)

    @patch('simple_orchestrator.model_client.urlopen')
    def test_cloud_log_contains_final_answer_but_no_secret_or_reasoning(self, send):
        send.return_value=response('{"answer":"测试"}')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.json'
            self.call(DeepSeekClient(model='test',api_key=KEY,log_path=path))
            logged=path.read_text()
        self.assertIn('deepseek',logged);self.assertNotIn(KEY,logged);self.assertNotIn('internal',logged)

    def test_invalid_configuration_fails_before_network(self):
        for options in [{'api_key':''},{'api_key':KEY,'base_url':'file:///tmp/test'},
                        {'api_key':KEY,'timeout':0},{'api_key':KEY,'reasoning_effort':'wrong'}]:
            with self.subTest(options=options),self.assertRaises(PlanningError):DeepSeekClient(model='test',**options)

    @patch.dict(os.environ,{'DEEPSEEK_API_KEY':KEY,'OLLAMA_MODEL':'local-test','MODEL_PROVIDER':'ollama',
                          'ORCHESTRATOR_MODEL_PROVIDER':'ollama','INTENT_MODEL_PROVIDER':'deepseek',
                          'INTENT_DEEPSEEK_MODEL':'intent-cloud-test'})
    def test_intent_and_orchestration_can_select_different_providers(self):
        intent_args=intent_parser().parse_args([])
        planner_args=build_parser().parse_args([])
        self.assertIsInstance(build_intent_agent(intent_args,intent_parser()).client,DeepSeekClient)
        self.assertEqual(intent_args.deepseek_model,'intent-cloud-test')
        self.assertNotIsInstance(make_model_client(planner_args,purpose='orchestration'),DeepSeekClient)

    @patch.dict(os.environ,{'DEEPSEEK_API_KEY':KEY,'OLLAMA_MODEL':'local-test','ORCHESTRATOR_LOCAL_INTENT':'1','KNOWLEDGE_API_URL':''})
    def test_cloud_client_is_used_by_intent_and_dynamic_planner(self):
        parser=build_parser();args=parser.parse_args(['--model-provider','deepseek','--fake-outcome','succeeded'])
        with tempfile.TemporaryDirectory() as d:
            args.agent_log_path=str(Path(d)/'messages.jsonl')
            runtime,_=build_orchestrator(args,parser)
        self.assertIsInstance(runtime.intent_agent.client,DeepSeekClient)
        self.assertIs(runtime.orchestrator.planner.client,runtime.intent_agent.client)

class ModelSwitchApiTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{'DEEPSEEK_API_KEY':KEY,'OLLAMA_MODEL':'local-test','ORCHESTRATOR_MODEL_PROVIDER':'ollama'})
        self.env.start();self.addCleanup(self.env.stop)
        self.servers=[]
        for kind in ['intent','orchestration']:
            args=build_parser().parse_args([])
            settings=ModelSettings(args,kind)
            server=(IntentHttpServer(('127.0.0.1',0),lambda:FixedIntentAgent(),token='service-token',runtime_info={},model_settings=settings)
                    if kind=='intent' else OrchestratorHttpServer(('127.0.0.1',0),lambda:None,token='service-token',runtime_info={},model_settings=settings))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            self.servers.append((server,thread))
        self.addCleanup(self.close)

    def close(self):
        for server,thread in self.servers:server.shutdown();server.server_close();thread.join(2)

    def request(self,server,payload=None,token='service-token'):
        conn=HTTPConnection('127.0.0.1',server.server_port,timeout=3)
        conn.request('GET' if payload is None else 'POST','/api/model',body=None if payload is None else json.dumps(payload),
                     headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        response=conn.getresponse();data=json.loads(response.read());status=response.status;conn.close();return status,data

    def test_switching_is_independent_and_metadata_never_contains_credentials(self):
        intent=self.servers[0][0];planner=self.servers[1][0]
        self.assertEqual(self.request(intent,{'provider':'deepseek'})[0],200)
        self.assertEqual(self.request(intent)[1]['provider'],'deepseek')
        self.assertEqual(self.request(planner)[1]['provider'],'ollama')
        self.assertNotIn(KEY,json.dumps(self.request(intent)[1]))
        self.assertEqual(self.request(intent,{'provider':'ollama'})[0],200)

    def test_unauthorized_busy_and_untrusted_configuration_are_rejected(self):
        for server,_ in self.servers:
            self.assertEqual(self.request(server,token='wrong')[0],401)
            for payload in [{'provider':'unknown'},{'provider':'deepseek','api_key':'injected'},{'provider':[]}]:
                self.assertEqual(self.request(server,payload)[0],400)
            server.run_lock.acquire()
            try:self.assertEqual(self.request(server,{'provider':'deepseek'})[0],409)
            finally:server.run_lock.release()
            self.assertEqual(self.request(server)[1]['provider'],'ollama')

    def test_missing_key_does_not_change_active_provider(self):
        server=self.servers[0][0]
        with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'','INTENT_DEEPSEEK_API_KEY':''}):
            self.assertFalse(self.request(server)[1]['providers'][1]['configured'])
            self.assertEqual(self.request(server,{'provider':'deepseek'})[0],400)
        self.assertEqual(self.request(server)[1]['provider'],'ollama')

if __name__=='__main__':unittest.main()
