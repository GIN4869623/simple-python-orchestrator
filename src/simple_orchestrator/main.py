"""Command-line entry point for the learning example."""

from __future__ import annotations

import argparse
import os
import sys

from .config import load_dotenv
from .fakes import FakeDeviceManagerPort, FakeKnowledgePort, FakeUsrpAgent
from .graph_orchestrator import GraphOrchestrator
from .models import (
    IntentAnalysis,
    NextAction,
    ParameterPolicy,
    ResultStatus,
    Task,
    TaskStatus,
    WorkflowStatus,
)
from .planner import FixedPlanner, OllamaPlanner, PlanningError
from .remote import JsonRpcAgentAdapter


AGENT_NAMES = {
    "knowledge": "知识智能体",
    "device_manager": "设备管理智能体",
    "usrp": "USRP 硬件智能体",
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


def describe_plan(tasks: list[Task], planner_name: str) -> str:
    """Convert parsed Task objects into a human-readable plan."""

    task_numbers = {task.task_id: index for index, task in enumerate(tasks, start=1)}
    lines = [f"{planner_name}规划结果（已解析）："]
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
    parser.add_argument(
        "request",
        nargs="?",
        help="用户任务；省略时程序会在启动后提示输入",
    )
    parser.add_argument(
        "--planner",
        choices=["ollama", "fixed"],
        default="ollama",
        help="计划来源；默认调用本地 Ollama",
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
        "--fake-outcome",
        choices=[status.value for status in FAKE_OUTCOME_NAMES],
        help="跳过交互菜单，直接指定所有 Fake 的回复方案",
    )
    parser.add_argument(
        "--remote-agents",
        action="store_true",
        help="通过静态配置的 HTTP + JSON-RPC 地址调用远程智能体",
    )
    parser.add_argument("--knowledge-rpc-url", default=os.getenv("KNOWLEDGE_RPC_URL", ""))
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
            "启用能力仓库匹配、device.select 注入和严格计划校验；"
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

    if args.remote_agents:
        if args.parameter_policy != ParameterPolicy.STRICT.value:
            parser.error("远程硬件模式必须使用 --parameter-policy strict")
        if args.fake_outcome is not None:
            parser.error("远程智能体模式不能同时指定 --fake-outcome")
        if not all((args.knowledge_rpc_url, args.device_rpc_url, args.usrp_rpc_url)):
            parser.error("远程智能体模式必须配置 Knowledge、DeviceManager 和 USRP 地址")
        fake_outcome = None
        token = os.getenv("ORCHESTRATOR_RPC_TOKEN")
        knowledge = JsonRpcAgentAdapter(
            args.knowledge_rpc_url, "knowledge", timeout=args.rpc_timeout,
            auth_token=token,
        )
        device_manager = JsonRpcAgentAdapter(
            args.device_rpc_url, "device_manager", timeout=args.rpc_timeout,
            auth_token=token,
        )
        usrp = JsonRpcAgentAdapter(
            args.usrp_rpc_url, "usrp", timeout=args.rpc_timeout,
            auth_token=token,
        )
    else:
        fake_outcome = choose_fake_outcome(args.fake_outcome)
        knowledge = FakeKnowledgePort(outcome=fake_outcome)
        device_manager = FakeDeviceManagerPort(
            device_id=args.device_id,
            outcome=fake_outcome,
        )
        usrp = FakeUsrpAgent(outcome=fake_outcome)
    parameter_policy = ParameterPolicy(args.parameter_policy)

    if args.planner == "ollama":
        if not args.ollama_model:
            parser.error(
                "使用 Ollama Planner 时必须设置 OLLAMA_MODEL 或 --ollama-model"
            )
        planner = OllamaPlanner(
            model=args.ollama_model,
            base_url=args.ollama_url,
            timeout=args.ollama_timeout,
            temperature=args.ollama_temperature,
            log_path=args.ollama_log_path,
            parameter_policy=parameter_policy,
            enable_agent_registry=args.enable_agent_registry,
        )
    else:
        planner = FixedPlanner(
            parameter_policy=parameter_policy,
            enable_agent_registry=args.enable_agent_registry,
        )

    orchestrator = GraphOrchestrator(
        planner=planner,
        knowledge=knowledge,
        device_manager=device_manager,
        usrp=usrp,
        max_retries=args.max_retries,
        max_replans=args.max_replans,
    )
    try:
        state = orchestrator.run(user_request)
        while (
            state["status"] is WorkflowStatus.NEEDS_CLARIFICATION
            and parameter_policy is ParameterPolicy.STRICT
        ):
            print(state["final_answer"])
            try:
                supplement = input(
                    "请补充上述参数（直接回车结束本次请求）："
                ).strip()
            except EOFError:
                break
            if not supplement:
                break
            user_request = f"{user_request}\n用户补充：{supplement}"
            state = orchestrator.run(user_request)
    except PlanningError as exc:
        print(f"规划失败：{exc}", file=sys.stderr)
        return 2

    planner_name = "LLM " if args.planner == "ollama" else "FixedPlanner "
    if state.get("intent") is not None:
        print(describe_intent(state["intent"]))
    tasks = list(state["plan_history"][-1].tasks) if state["plan_history"] else []
    print(describe_plan(tasks, planner_name))
    if fake_outcome is None:
        print("\n执行结果（当前后端为远程 JSON-RPC 智能体）：")
    else:
        print(
            "\n执行结果（当前后端为 Fake，"
            f"本次方案：{FAKE_OUTCOME_NAMES[fake_outcome]}）："
        )
    for index, task in enumerate(tasks, start=1):
        agent_name = AGENT_NAMES.get(task.agent, task.agent)
        status_name = TASK_STATUS_NAMES[state["task_statuses"][task.task_id]]
        print(f"{index}. {agent_name}：{status_name}")
    print("\n规划器执行判断：")
    for index, review in enumerate(state["reviews"], start=1):
        action_name = ACTION_NAMES[review.action]
        print(f"{index}. 第 {index} 批：{action_name}——{review.reason}")
    print(f"\n工作流汇总：{state['final_answer']}")
    return 0 if state["status"] is WorkflowStatus.SUCCEEDED else 1


if __name__ == "__main__":
    raise SystemExit(main())
