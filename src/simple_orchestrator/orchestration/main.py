"""Command-line entry point for the learning example."""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial

from ..agent_logging import AgentMessageLog, LoggedFakeAgent
from ..config import load_dotenv
from ..fakes import FakeDeviceManagerPort, FakeKnowledgePort, FakeUsrpAgent
from .graph import GraphOrchestrator
from .execution import WorkflowExecutor
from ..intent.agent import FixedIntentAgent, OllamaIntentAgent
from ..intent.client import IntentHttpClient
from ..model_client import add_model_arguments, make_model_client
from .router import RequestRouter
from ..knowledge_http import DEFAULT_KNOWLEDGE_API_URL, KnowledgeHttpAdapter
from ..network_chat import NetworkChatAdapter
from ..models import (
    HandlingMode,
    IntentAnalysis,
    NextAction,
    ParameterPolicy,
    ResultStatus,
    Task,
    TaskStatus,
    WorkflowStatus,
)
from .planner import FixedPlanner, OllamaPlanner, PlanningError
from ..remote import JsonRpcAgentAdapter, call_ris_chat
from ..tcp_transport import TcpJsonRpcAgentAdapter


AGENT_NAMES = {
    "knowledge": "知识智能体",
    "device_manager": "网络监管智能体",
    "usrp": "USRP 硬件智能体",
    "ris": "RIS 智能体",
}

TASK_STATUS_NAMES = {
    TaskStatus.PENDING: "等待执行",
    TaskStatus.RUNNING: "正在执行",
    TaskStatus.SUCCEEDED: "执行成功",
    TaskStatus.FAILED: "执行失败",
    TaskStatus.REJECTED: "拒绝执行",
    TaskStatus.UNKNOWN: "状态未知",
    TaskStatus.SKIPPED: "已跳过",
}

ACTION_NAMES = {
    NextAction.CONTINUE: "继续执行",
    NextAction.RETRY: "重新执行当前任务",
    NextAction.REPLAN: "重新规划剩余任务",
    NextAction.COMPLETE: "宣布任务成功",
    NextAction.STOP: "停止工作流",
}

FAKE_OUTCOME_NAMES = {
    ResultStatus.SUCCEEDED: "成功",
    ResultStatus.REJECTED: "拒绝",
    ResultStatus.UNKNOWN: "未知",
    ResultStatus.FAILED: "失败",
}

FAKE_OUTCOME_CHOICES = {
    "1": ResultStatus.SUCCEEDED,
    "2": ResultStatus.REJECTED,
    "3": ResultStatus.UNKNOWN,
    "4": ResultStatus.FAILED,
}

MAX_CLARIFICATION_ROUNDS = 5


def describe_plan(tasks: list[Task], planner_name: str, *, heading: str | None = None) -> str:
    """Convert parsed Task objects into a human-readable plan."""

    task_numbers = {task.task_id: index for index, task in enumerate(tasks, start=1)}
    lines = [heading or f"{planner_name}规划结果（已解析）："]
    for index, task in enumerate(tasks, start=1):
        agent_name = AGENT_NAMES.get(task.agent, task.agent)
        lines.append(f"{index}. {agent_name}负责：{task.description}")
        if task.capability:
            lines.append(f"   调用能力：{task.capability}。")
        if task.completion_criteria:
            criteria = "；".join(task.completion_criteria)
            lines.append(f"   完成标准：{criteria}。")
        if task.dependencies:
            dependencies = "、".join(
                f"第{task_numbers.get(task_id, '?')}项任务（{task_id}）"
                for task_id in task.dependencies
            )
            lines.append(f"   开始前需要完成：{dependencies}。")
        else:
            lines.append("   该任务没有前置依赖。")
    return "\n".join(lines)


def describe_intent(intent: IntentAnalysis) -> str:
    target = intent.target_type or "无特定设备"
    capabilities = "、".join(intent.required_capabilities) or "无"
    return (
        "意图分析：\n"
        f"- 类型：{intent.intent_type.value}\n"
        f"- 目标：{target}\n"
        f"- 动作：{intent.action}\n"
        f"- 执行方式：{intent.execution_mode.value}\n"
        f"- 所需能力：{capabilities}"
    )


def build_parser() -> argparse.ArgumentParser:
    load_dotenv()
    parser = argparse.ArgumentParser(description="运行最小编排智能体")
    add_model_arguments(parser)
    parser.add_argument(
        "request",
        nargs="?",
        help="用户任务；省略时程序会在启动后提示输入",
    )
    parser.add_argument(
        "--planner",
        choices=["ollama", "fixed"],
        default="ollama",
        help="动态规划的实现；意图识别默认调用独立服务",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama 服务地址，也可通过 OLLAMA_BASE_URL 配置",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL", ""),
        help="Ollama 模型名，也可通过 OLLAMA_MODEL 配置",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=float(os.getenv("OLLAMA_TIMEOUT", "120")),
        help="等待 Ollama 规划的超时秒数，也可通过 OLLAMA_TIMEOUT 配置",
    )
    parser.add_argument(
        "--ollama-temperature",
        type=float,
        default=float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
        help="Ollama 规划温度（0 到 2），也可通过 OLLAMA_TEMPERATURE 配置",
    )
    parser.add_argument(
        "--ollama-log-path",
        default=os.getenv("OLLAMA_LOG_PATH", "logs/ollama_responses.json"),
        help="最近 20 次 LLM 返回的日志路径，也可通过 OLLAMA_LOG_PATH 配置",
    )
    parser.add_argument(
        "--agent-log-path",
        default=os.getenv("ORCHESTRATOR_AGENT_LOG_PATH", "logs/agent_messages.jsonl"),
        help="知识与网络监管智能体完整收发日志的路径",
    )
    parser.add_argument(
        "--fake-outcome",
        choices=[status.value for status in FAKE_OUTCOME_NAMES],
        help="跳过交互菜单，直接指定所有 Fake 的回复方案",
    )
    parser.add_argument(
        "--remote-agents",
        action="store_true",
        help="通过静态配置的 HTTP(S) 或 TCP 地址调用远程智能体",
    )
    parser.add_argument("--ris-chat-url", default=os.getenv("RIS_AGENT_CHAT_URL", "http://192.168.1.132:8000/api/chat/stream"))
    parser.add_argument("--ris-timeout", type=float, default=float(os.getenv("RIS_AGENT_TIMEOUT", "180")))
    parser.add_argument("--network-chat-url", default=os.getenv("NETWORK_SUPERVISION_CHAT_URL", ""))
    parser.add_argument("--network-chat-timeout", type=float, default=float(os.getenv("NETWORK_SUPERVISION_CHAT_TIMEOUT", "120")))
    parser.add_argument("--knowledge-rpc-url", default=os.getenv("KNOWLEDGE_RPC_URL", ""))
    parser.add_argument(
        "--knowledge-api-url", nargs="?", const=DEFAULT_KNOWLEDGE_API_URL,
        default=os.getenv("KNOWLEDGE_API_URL", ""),
        help="启用真实知识 HTTP JSON 接口；省略地址时使用已记录的局域网地址",
    )
    parser.add_argument("--device-rpc-url", default=os.getenv("DEVICE_MANAGER_RPC_URL", ""))
    parser.add_argument("--usrp-rpc-url", default=os.getenv("USRP_RPC_URL", ""))
    parser.add_argument(
        "--rpc-timeout", type=float,
        default=float(os.getenv("ORCHESTRATOR_RPC_TIMEOUT", "30")),
    )
    parser.add_argument(
        "--device-id",
        default=os.getenv("FAKE_DEVICE_ID", "USRP-01"),
        help="Fake 选中的设备 ID",
    )
    parser.add_argument(
        "--parameter-policy",
        choices=[policy.value for policy in ParameterPolicy],
        default=os.getenv(
            "ORCHESTRATOR_PARAMETER_POLICY",
            ParameterPolicy.DEMO_DEFAULTS.value,
        ),
        help=(
            "缺失硬件参数的处理方式；Fake 默认使用 demo_defaults，"
            "接入真实硬件时应使用 strict"
        ),
    )
    parser.add_argument(
        "--enable-agent-registry",
        action="store_true",
        default=os.getenv("ORCHESTRATOR_ENABLE_AGENT_REGISTRY", "0")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"},
        help=(
            "启用能力仓库匹配、设备查询前置任务和严格计划校验；"
            "默认关闭"
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        choices=[0, 1],
        default=int(os.getenv("ORCHESTRATOR_MAX_RETRIES", "1")),
        help="每个任务在每版计划中最多重试次数，只能为 0 或 1",
    )
    parser.add_argument(
        "--max-replans",
        type=int,
        choices=[0, 1],
        default=int(os.getenv("ORCHESTRATOR_MAX_REPLANS", "1")),
        help="整个工作流最多重新规划次数，只能为 0 或 1",
    )
    parser.add_argument(
        "--intent-service-url", default=os.getenv("INTENT_SERVICE_URL", "http://127.0.0.1:18003"),
        help="独立意图识别服务地址，默认 http://127.0.0.1:18003",
    )
    parser.add_argument(
        "--intent-timeout", type=float, default=float(os.getenv(
            "INTENT_SERVICE_TIMEOUT",
            str(max(180.0, float(os.getenv("INTENT_OLLAMA_TIMEOUT") or os.getenv("OLLAMA_TIMEOUT", "120")) + 30)),
        )),
        help="等待意图识别服务的超时秒数",
    )
    parser.add_argument(
        "--local-intent", action="store_true",
        default=os.getenv("ORCHESTRATOR_LOCAL_INTENT", "0").strip().lower() in {"1", "true", "yes", "on"},
        help="仅供本地调试：在当前进程识别意图，不连接独立意图服务",
    )
    return parser


def choose_fake_outcome(configured_outcome: str | None) -> ResultStatus:
    """Select one response scenario for every Fake used in this workflow."""

    if configured_outcome is not None:
        return ResultStatus(configured_outcome)

    print("\n请选择本次 Fake 回复方案：")
    print("1. 成功")
    print("2. 拒绝")
    print("3. 未知")
    print("4. 失败")
    while True:
        try:
            choice = input("请输入序号（直接回车默认 1）：").strip() or "1"
        except EOFError:
            return ResultStatus.SUCCEEDED
        outcome = FAKE_OUTCOME_CHOICES.get(choice)
        if outcome is not None:
            return outcome
        print("无效序号，请输入 1、2、3 或 4。")


def build_executor(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *,
    knowledge_sender: str = "编排智能体",
) -> tuple[WorkflowExecutor, ResultStatus | None]:
    """Build configured business adapters without constructing a planner."""
    message_log = AgentMessageLog(args.agent_log_path)
    network = None
    if args.network_chat_url:
        try:
            network = NetworkChatAdapter(args.network_chat_url, timeout=args.network_chat_timeout,
                                         sender=knowledge_sender, message_log=message_log)
        except ValueError as exc:
            parser.error(str(exc))
    if args.remote_agents:
        if args.parameter_policy != ParameterPolicy.STRICT.value:
            parser.error("远程硬件模式必须使用 --parameter-policy strict")
        if args.fake_outcome is not None:
            parser.error("远程智能体模式不能同时指定 --fake-outcome")
        if not all((args.knowledge_api_url or args.knowledge_rpc_url, network or args.device_rpc_url, args.usrp_rpc_url)):
            parser.error("远程智能体模式必须配置 Knowledge、DeviceManager 和 USRP 地址")
        fake_outcome = None
        token = os.getenv("ORCHESTRATOR_RPC_TOKEN")
        def make_adapter(endpoint: str, agent_id: str) -> JsonRpcAgentAdapter:
            adapter_type = TcpJsonRpcAgentAdapter if endpoint.startswith("tcp://") else JsonRpcAgentAdapter
            return adapter_type(endpoint, agent_id, timeout=args.rpc_timeout,
                                auth_token=token,
                                message_log=message_log if agent_id != "usrp" else None)

        knowledge = (
            KnowledgeHttpAdapter(
                args.knowledge_api_url, timeout=args.rpc_timeout, sender=knowledge_sender,
                api_key=os.getenv("KNOWLEDGE_API_KEY"), message_log=message_log,
            )
            if args.knowledge_api_url else make_adapter(args.knowledge_rpc_url, "knowledge")
        )
        device_manager = network or make_adapter(args.device_rpc_url, "device_manager")
        usrp = make_adapter(args.usrp_rpc_url, "usrp")
    else:
        configured_fake_outcome = args.fake_outcome
        if configured_fake_outcome is None and (args.knowledge_api_url or network):
            configured_fake_outcome = ResultStatus.SUCCEEDED.value
        fake_outcome = choose_fake_outcome(configured_fake_outcome)
        knowledge = (
            KnowledgeHttpAdapter(
                args.knowledge_api_url, timeout=args.rpc_timeout, sender=knowledge_sender,
                api_key=os.getenv("KNOWLEDGE_API_KEY"), message_log=message_log,
            )
            if args.knowledge_api_url else
            LoggedFakeAgent(FakeKnowledgePort(outcome=fake_outcome), "knowledge", message_log)
        )
        device_manager = network or LoggedFakeAgent(
            FakeDeviceManagerPort(device_id=args.device_id, outcome=fake_outcome),
            "device_manager", message_log,
        )
        usrp = FakeUsrpAgent(outcome=fake_outcome)
    return WorkflowExecutor(
        knowledge=knowledge, device_manager=device_manager, usrp=usrp,
        ris=partial(call_ris_chat, endpoint=args.ris_chat_url, timeout=args.ris_timeout, message_log=message_log) if args.ris_chat_url else None,
        max_retries=args.max_retries,
    ), fake_outcome


def build_orchestrator(
    args: argparse.Namespace, parser: argparse.ArgumentParser,
) -> tuple[RequestRouter, ResultStatus | None]:
    """Build the same runtime for the CLI and HTTP entry points."""

    executor, fake_outcome = build_executor(args, parser)
    knowledge, device_manager, usrp = executor.knowledge, executor.device_manager, executor.usrp
    parameter_policy = ParameterPolicy(args.parameter_policy)

    if args.planner == "ollama":
        try:
            client = make_model_client(args, purpose="orchestration")
        except PlanningError as exc:
            parser.error(str(exc))
        planner = OllamaPlanner(
            model=client.model, base_url=client.base_url, timeout=client.timeout,
            temperature=client.temperature, log_path=client.log_path, client=client,
            parameter_policy=parameter_policy,
            enable_agent_registry=args.enable_agent_registry,
        )
    else:
        planner = FixedPlanner(
            parameter_policy=parameter_policy,
            enable_agent_registry=args.enable_agent_registry,
        )

    if args.local_intent:
        intent_agent = (
            OllamaIntentAgent(client, parameter_policy=parameter_policy)
            if args.planner == "ollama" else FixedIntentAgent(parameter_policy=parameter_policy)
        )
    else:
        try:
            intent_agent = IntentHttpClient(
                base_url=args.intent_service_url, timeout=args.intent_timeout,
                auth_token=(os.getenv("INTENT_API_TOKEN") or os.getenv("ORCHESTRATOR_API_TOKEN", "")).strip(),
                parameter_policy=parameter_policy,
            )
        except PlanningError as exc:
            parser.error(str(exc))

    orchestrator = GraphOrchestrator(
        planner=planner,
        knowledge=knowledge,
        device_manager=device_manager,
        usrp=usrp, ris=executor.ris,
        max_retries=args.max_retries,
        max_replans=args.max_replans,
    )
    runtime = RequestRouter(
        intent_agent=intent_agent, executor=executor, orchestrator=orchestrator,
        registry=planner.registry, enable_agent_registry=args.enable_agent_registry,
    )
    return runtime, fake_outcome


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    user_request = args.request
    if user_request is None:
        try:
            user_request = input("请输入任务请求：").strip()
        except EOFError:
            parser.error("没有读取到任务请求")
        if not user_request:
            parser.error("任务请求不能为空")

    orchestrator, fake_outcome = build_orchestrator(args, parser)
    parameter_policy = ParameterPolicy(args.parameter_policy)
    try:
        state = orchestrator.run(user_request)
        clarification_rounds = 0
        while (
            state["status"] is WorkflowStatus.NEEDS_CLARIFICATION
            and parameter_policy is ParameterPolicy.STRICT
        ):
            print(state["final_answer"])
            if clarification_rounds >= MAX_CLARIFICATION_ROUNDS:
                print(f"已达到最多 {MAX_CLARIFICATION_ROUNDS} 轮参数补充，本次请求结束。")
                break
            try:
                supplement = input(
                    "请补充上述参数（直接回车结束本次请求）："
                ).strip()
            except EOFError:
                break
            if not supplement:
                break
            clarification_rounds += 1
            user_request = f"{user_request}\n用户补充：{supplement}"
            state = orchestrator.run(user_request)
    except PlanningError as exc:
        print(f"规划失败：{exc}", file=sys.stderr)
        return 2

    planner_name = "LLM " if args.planner == "ollama" else "FixedPlanner "
    if state.get("intent") is not None:
        print(describe_intent(state["intent"]))
    tasks = list(state["plan_history"][-1].tasks) if state["plan_history"] else []
    mode = state.get("handling_mode", HandlingMode.ORCHESTRATED)
    heading = {
        HandlingMode.DIRECT: "处理方式：直接调用；执行任务：",
        HandlingMode.FIXED_WORKFLOW: "处理方式：固定流程；执行任务：",
    }.get(mode)
    print(describe_plan(tasks, planner_name, heading=heading))
    if fake_outcome is None:
        print("\n执行结果（当前后端为远程智能体）：")
    elif args.network_chat_url:
        print("\n执行结果（网络监管智能体使用真实 HTTP 接口）：")
    elif args.knowledge_api_url:
        print("\n执行结果（知识智能体使用真实 HTTP 接口，其余后端为 Fake）：")
    else:
        print(
            "\n执行结果（当前后端为 Fake，"
            f"本次方案：{FAKE_OUTCOME_NAMES[fake_outcome]}）："
        )
    for index, task in enumerate(tasks, start=1):
        agent_name = AGENT_NAMES.get(task.agent, task.agent)
        status_name = TASK_STATUS_NAMES[state["task_statuses"][task.task_id]]
        print(f"{index}. {agent_name}：{status_name}")
    print("\n执行判断：")
    for index, review in enumerate(state["reviews"], start=1):
        action_name = ACTION_NAMES[review.action]
        print(f"{index}. 第 {index} 批：{action_name}——{review.reason}")
    print(f"\n工作流汇总：{state['final_answer']}")
    return 0 if state["status"] is WorkflowStatus.SUCCEEDED else 1


if __name__ == "__main__":
    raise SystemExit(main())
