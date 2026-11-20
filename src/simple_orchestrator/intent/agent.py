"""Intent agents understand requests; they never plan or execute tasks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

from ..errors import PlanningError
from ..models import ExecutionMode, HandlingMode, IntentAnalysis, IntentType, ParameterPolicy
from ..ollama import OllamaClient


INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "handling_mode": {
            "type": "string",
            "enum": [mode.value for mode in HandlingMode],
        },
        "intent_type": {
            "type": "string",
            "enum": [intent.value for intent in IntentType],
        },
        "domain": {"type": "string"},
        "target_type": {"type": ["string", "null"]},
        "action": {"type": "string"},
        "execution_mode": {
            "type": "string",
            "enum": [mode.value for mode in ExecutionMode],
        },
        "required_capabilities": {
            "type": "array",
            "items": {"type": "string"},
        },
        "provided_parameters": {"type": "object"},
        "missing_parameters": {
            "type": "array",
            "items": {"type": "string"},
        },
        "needs_clarification": {"type": "boolean"},
        "evidence": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": [
        "handling_mode",
        "intent_type",
        "domain",
        "target_type",
        "action",
        "execution_mode",
        "required_capabilities",
        "provided_parameters",
        "missing_parameters",
        "needs_clarification",
        "evidence",
        "reason",
    ],
    "additionalProperties": False,
}

INTENT_INSTRUCTIONS = """你是业务请求意图分析器，理解用户想做什么，并判断处理路径；不生成任务计划，也不执行任务。

处理路径 handling_mode：
- direct：一个专业智能体可以独立完成的明确任务，如知识问答、设备清单或状态查询。
- fixed_workflow：步骤明确、适用已有标准流程的任务。目前仅支持单一 USRP 操作，
  由固定的 Skill 查询、设备查询和硬件执行流程完成。
- orchestrated：需要动态拆解子任务、跨能力协调，或根据中间结果决定后续任务。
  多个问题若同一个专业智能体可以独立回答，仍可 direct；多个智能体也可以走固定流程。
  用户明确要求动态拆解或动态编排时，应遵循该要求，使用 orchestrated。
- 信息不足时标记 needs_clarification，不要猜测参数；澄清完成前不会执行任何路径。

分类定义：
- knowledge_query：解释、咨询、设计方案、操作指导或纯软件仿真，不产生真实设备效果。
- device_query：只读查询设备清单、在线状态、能力、可用性或实时传感器读数。
- hardware_execution：用户明确要求对真实硬件执行操作。
- unsupported：请求执行 USRP、RIS 之外的硬件，或当前系统能力明显无法处理。

识别规则：
1. “如何使用 USRP”“解释 USRP”“仿真 QPSK”属于 knowledge_query，不是硬件执行。
   USRP 专属知识可由 USRP 智能体以只读方式回答，但不能因此执行硬件操作。
   知识问答的 required_capabilities 固定为 ["knowledge.explain"]，USRP 专属知识的 target_type 为 usrp。
   RIS 专属问答或明确交给 RIS 智能体的文字请求，target_type 为 ris，由 RIS 回答。
   通用通信知识仍可交给 Knowledge。
   查询可用 USRP 设备、设备清单或在线状态属于 device_query，能力使用
   device.list 或 device.inspect，不要使用 usrp.get_status。
   “查询实验室1光照强度”等实时光照、温度、湿度读数属于网络监管查询：
   intent_type=device_query，domain=network_monitoring，target_type=environment_sensor，
   action=inspect，execution_mode=read_only，required_capabilities=["device.inspect"]。
   保留用户提供的地点和指标，简单查询使用 direct。
2. 只有用户明确要求真实配置、发射、接收、停止等操作时，execution_mode 才能是 physical。
3. USRP 真实操作的 target_type 必须是 usrp。
4. 明确交给 RIS 的任务 target_type 为 ris，能力为 ris.chat；简单任务 direct，需要动态拆解时 orchestrated。RIS 自行处理专业业务，上游只接收文字回复，不编造其操作参数。频谱仪、信号源等未接入的硬件属于 unsupported。
5. USRP 执行能力只能是 usrp.configure、usrp.transmit、usrp.receive、usrp.stop 或 usrp.get_status。
6. USRP 硬件执行只要求相应的 usrp 能力；不要自动加入设备查询任务。
7. 发射操作至少需要 waveform、center_frequency_hz、sample_rate_sps、gain_db、duration_seconds。缺少时写入 missing_parameters，并令 needs_clarification=true。
8. 不得猜测用户没有提供的硬件参数。
9. evidence 只摘述能够证明分类的用户原始表达，reason 简短说明判断依据。
10. 只返回符合 JSON Schema 的结果。
"""


@dataclass
class FixedIntentAgent:
    """Offline intent recognition for the supported demonstration requests."""

    parameter_policy: ParameterPolicy = ParameterPolicy.STRICT

    def analyze_intent(self, user_request: str) -> IntentAnalysis:
        intent = _analyze_intent_with_rules(user_request)
        mode = (
            HandlingMode.FIXED_WORKFLOW
            if intent.intent_type is IntentType.HARDWARE_EXECUTION and intent.target_type != "ris"
            else HandlingMode.DIRECT
        )
        # The offline recognizer only recognizes these explicit multi-step cues.
        if re.search(r"先.+再|分别|并行|动态|拆解|根据.+(?:决定|选择)", user_request):
            mode = HandlingMode.ORCHESTRATED
        return _apply_parameter_policy(
            replace(intent, handling_mode=mode), self.parameter_policy
        )


@dataclass
class OllamaIntentAgent:
    """Own prompt and schema; the client contains no conversational memory."""

    client: OllamaClient
    parameter_policy: ParameterPolicy = ParameterPolicy.STRICT

    def analyze_intent(self, user_request: str) -> IntentAnalysis:
        response_body = self.client._chat(
            instructions=INTENT_INSTRUCTIONS,
            user_content=(
                f"用户请求：{user_request}\n\n"
                "请使用以下 JSON Schema 返回意图分析：\n"
                f"{json.dumps(INTENT_SCHEMA, ensure_ascii=False)}"
            ),
            schema=INTENT_SCHEMA,
            user_request=user_request,
            phase="intent_analysis",
        )
        data = self.client._response_content_as_json(response_body, "结构化意图分析")
        intent = _parse_intent_analysis(data)
        if intent.intent_type is IntentType.KNOWLEDGE_QUERY:
            # Knowledge requests always use the read-only business capability.
            intent = replace(
                intent,
                execution_mode=ExecutionMode.INFORMATIONAL,
                required_capabilities=["knowledge.explain"],
                target_type=("ris" if re.search(r"(?i)(?<![a-z0-9_])ris(?![a-z0-9_])|反射面", user_request) and "usrp" not in user_request.lower()
                             else "usrp" if "usrp" in user_request.lower() else intent.target_type),
            )
        elif intent.intent_type is IntentType.DEVICE_QUERY:
            capability = (
                "device.inspect"
                if intent.target_type == "environment_sensor" or "状态" in user_request
                or "inspect" in intent.action.lower() or "device.inspect" in intent.required_capabilities
                else "device.list"
            )
            intent = replace(
                intent,
                execution_mode=ExecutionMode.READ_ONLY,
                required_capabilities=[capability],
                missing_parameters=[],
                needs_clarification=False,
            )
        return _apply_parameter_policy(intent, self.parameter_policy)


def _parse_intent_analysis(data: Any) -> IntentAnalysis:
    if not isinstance(data, dict):
        raise PlanningError("LLM 意图分析必须是对象")
    try:
        intent_type = IntentType(data.get("intent_type"))
        execution_mode = ExecutionMode(data.get("execution_mode"))
        handling_mode = HandlingMode(data.get("handling_mode", "orchestrated"))
    except (TypeError, ValueError) as exc:
        raise PlanningError("LLM 意图分析包含未知分类") from exc

    string_fields = ("domain", "action", "evidence", "reason")
    if any(not isinstance(data.get(name), str) for name in string_fields):
        raise PlanningError("LLM 意图分析中的文本字段无效")
    target_type = data.get("target_type")
    if target_type is not None and not isinstance(target_type, str):
        raise PlanningError("LLM 意图分析中的 target_type 无效")
    required_capabilities = _string_list(
        data.get("required_capabilities"), "required_capabilities"
    )
    missing_parameters = _string_list(
        data.get("missing_parameters"), "missing_parameters"
    )
    provided_parameters = data.get("provided_parameters")
    if not isinstance(provided_parameters, dict):
        raise PlanningError("LLM 意图分析中的 provided_parameters 必须是对象")
    needs_clarification = data.get("needs_clarification")
    if not isinstance(needs_clarification, bool):
        raise PlanningError("LLM 意图分析中的 needs_clarification 必须是布尔值")

    return IntentAnalysis(
        intent_type=intent_type,
        domain=data["domain"],
        target_type=target_type,
        action=data["action"],
        execution_mode=execution_mode,
        required_capabilities=required_capabilities,
        provided_parameters=provided_parameters,
        missing_parameters=missing_parameters,
        needs_clarification=needs_clarification,
        evidence=data["evidence"],
        reason=data["reason"],
        handling_mode=handling_mode,
    )


def _analyze_intent_with_rules(user_request: str) -> IntentAnalysis:
    request = user_request.strip()
    lowered = request.lower()
    is_usrp = "usrp" in lowered
    is_ris = bool(re.search(r"(?i)(?<![a-z0-9_])ris(?![a-z0-9_])|反射面", request))
    asks_how = any(
        marker in lowered
        for marker in ("如何", "怎么", "解释", "介绍", "原理", "什么是", "仿真", "模拟")
    )
    has_execution_verb = any(
        marker in lowered
        for marker in ("使用", "启动", "执行", "配置", "发送", "发射", "接收", "停止")
    )
    asks_devices = ("设备" in request or is_usrp) and any(
        marker in request for marker in ("查询", "哪些", "列表", "状态", "在线", "可用", "能力")
    )
    asks_sensor_reading = not (asks_how or has_execution_verb) and (
        any(marker in request for marker in ("查询", "查看", "读取", "当前", "实时", "读数", "多少"))
        and any(metric in request for metric in ("光照", "温度", "湿度", "温湿度"))
    )

    if is_ris and not is_usrp:
        physical = has_execution_verb and not asks_how
        return IntentAnalysis(
            IntentType.HARDWARE_EXECUTION if physical else IntentType.KNOWLEDGE_QUERY,
            "wireless_communication", "ris", "delegate" if physical else "explain",
            ExecutionMode.PHYSICAL if physical else ExecutionMode.INFORMATIONAL,
            required_capabilities=["ris.chat" if physical else "knowledge.explain"],
            evidence=request, reason="将明确的 RIS 请求交给 RIS 智能体并接收文字回复",
        )
    if asks_sensor_reading:
        return IntentAnalysis(
            IntentType.DEVICE_QUERY, "network_monitoring", "environment_sensor", "inspect",
            ExecutionMode.READ_ONLY, required_capabilities=["device.inspect"],
            evidence=request, reason="查询实时传感器读数",
        )
    if asks_devices:
        capability = "device.inspect" if "状态" in request else "device.list"
        return IntentAnalysis(
            IntentType.DEVICE_QUERY,
            "device_management",
            "usrp" if is_usrp else "generic_managed_device",
            "inspect" if capability == "device.inspect" else "list",
            ExecutionMode.READ_ONLY,
            required_capabilities=[capability],
            evidence=request,
            reason="用户要求查询实时设备资源信息",
        )
    if is_usrp and has_execution_verb and not asks_how:
        capability, action = _usrp_capability_for_request(request)
        missing = _missing_usrp_parameters(request, action)
        provided = {"original_request": request}
        return IntentAnalysis(
            IntentType.HARDWARE_EXECUTION,
            "wireless_communication",
            "usrp",
            action,
            ExecutionMode.PHYSICAL,
            required_capabilities=[capability],
            provided_parameters=provided,
            missing_parameters=missing,
            needs_clarification=bool(missing),
            evidence=request,
            reason="用户明确要求执行真实 USRP 硬件操作",
        )
    if has_execution_verb and not asks_how:
        return IntentAnalysis(
            IntentType.UNSUPPORTED,
            "general_hardware",
            None,
            "execute",
            ExecutionMode.PHYSICAL,
            evidence=request,
            reason="当前唯一硬件执行智能体只支持明确的 USRP 操作",
        )
    return IntentAnalysis(
        IntentType.KNOWLEDGE_QUERY,
        "wireless_communication",
        "usrp" if is_usrp else None,
        "explain",
        ExecutionMode.INFORMATIONAL,
        required_capabilities=["knowledge.explain"],
        evidence=request,
        reason="请求属于知识、指导或软件仿真，不产生真实硬件效果",
    )


def _usrp_capability_for_request(request: str) -> tuple[str, str]:
    if any(marker in request for marker in ("发送", "发射")):
        return "usrp.transmit", "transmit"
    if "接收" in request:
        return "usrp.receive", "receive"
    if "停止" in request:
        return "usrp.stop", "stop"
    if any(marker in request for marker in ("状态", "查询")):
        return "usrp.get_status", "get_status"
    return "usrp.configure", "configure"


def _missing_usrp_parameters(request: str, action: str) -> list[str]:
    if action == "configure":
        has_configuration = any(
            marker in request for marker in ("频率", "采样率", "增益", "通道", "天线")
        )
        return [] if has_configuration else ["configuration"]
    if action not in {"transmit", "receive"}:
        return []
    checks = {
        "center_frequency_hz": bool(
            re.search(r"(?:中心)?频率\s*[:：]?\s*\d+(?:\.\d+)?\s*[kmg]?hz", request, re.I)
        ),
        "sample_rate_sps": bool(
            re.search(r"采样率\s*[:：]?\s*\d+(?:\.\d+)?\s*(?:[kmg]?hz|[kmg]?sps)", request, re.I)
        ),
        "gain_db": bool(
            re.search(r"增益\s*[:：]?\s*-?\d+(?:\.\d+)?\s*db", request, re.I)
        ),
        "duration_seconds": bool(
            re.search(r"(?:时长\s*[:：]?\s*)?\d+(?:\.\d+)?\s*(?:秒|s\b)", request, re.I)
        ),
    }
    if action == "transmit":
        checks["waveform"] = any(
            marker.lower() in request.lower()
            for marker in ("QPSK", "BPSK", "OFDM", "波形", "信号类型")
        )
    return [name for name, provided in checks.items() if not provided]


def _clarification_questions(intent: IntentAnalysis) -> list[str]:
    labels = {
        "waveform": "请提供要发射的波形或信号类型",
        "center_frequency_hz": "请提供中心频率",
        "sample_rate_sps": "请提供采样率",
        "gain_db": "请提供增益",
        "duration_seconds": "请提供执行时长",
        "configuration": "请提供需要配置的 USRP 参数",
    }
    return [labels.get(name, f"请提供参数：{name}") for name in intent.missing_parameters]


def _apply_parameter_policy(
    intent: IntentAnalysis, policy: ParameterPolicy
) -> IntentAnalysis:
    if policy is ParameterPolicy.STRICT or not intent.needs_clarification:
        return intent

    provided_parameters = dict(intent.provided_parameters)
    provided_parameters.update(
        {
            "parameter_policy": "use_backend_defaults",
            "unresolved_parameters": list(intent.missing_parameters),
        }
    )
    return replace(
        intent,
        provided_parameters=provided_parameters,
        needs_clarification=False,
        reason=(
            f"{intent.reason}；当前为 Fake 演示模式，缺失参数由后端默认值补全"
        ),
    )


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PlanningError(f"{name} 必须是字符串列表")
    return value
