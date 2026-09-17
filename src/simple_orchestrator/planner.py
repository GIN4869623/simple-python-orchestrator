"""Intent analysis plus fixed and Ollama-backed task planners."""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .agent_registry import AgentRegistry, build_default_registry
from .models import (
    AgentResult,
    BatchReviewDecision,
    ExecutionMode,
    IntentAnalysis,
    IntentType,
    NextAction,
    ParameterPolicy,
    PlanStatus,
    ResultStatus,
    ReviewDecision,
    Task,
    TaskPlan,
    TaskStatus,
    WorkflowState,
)
from .validation import PlanValidationError, PlanValidator


INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
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

TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string"},
        "agent": {"type": "string"},
        "capability": {"type": "string"},
        "description": {"type": "string"},
        "inputs": {"type": "object"},
        "expected_output": {
            "type": "array",
            "items": {"type": "string"},
        },
        "completion_criteria": {
            "type": "array",
            "items": {"type": "string"},
        },
        "dependencies": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "task_id",
        "agent",
        "capability",
        "description",
        "inputs",
        "expected_output",
        "completion_criteria",
        "dependencies",
    ],
    "additionalProperties": False,
}

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": TASK_SCHEMA,
        }
    },
    "required": ["tasks"],
    "additionalProperties": False,
}

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [action.value for action in NextAction],
        },
        "reason": {"type": "string"},
        "revised_tasks": {
            "type": "array",
            "maxItems": 6,
            "items": TASK_SCHEMA,
        },
    },
    "required": ["action", "reason", "revised_tasks"],
    "additionalProperties": False,
}

BATCH_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": [action.value for action in NextAction]},
        "reason": {"type": "string"},
        "retry_task_ids": {"type": "array", "items": {"type": "string"}},
        "revised_tasks": {"type": "array", "items": TASK_SCHEMA},
    },
    "required": ["action", "reason", "retry_task_ids", "revised_tasks"],
    "additionalProperties": False,
}

INTENT_INSTRUCTIONS = """你是业务请求意图分析器，只描述用户想做什么，不选择智能体，也不生成任务。

分类定义：
- knowledge_query：解释、咨询、设计方案、操作指导或纯软件仿真，不产生真实设备效果。
- device_query：查询设备清单、在线状态、能力或可用性，只读访问设备资源信息。
- hardware_execution：用户明确要求对真实硬件执行操作。
- unsupported：请求执行非 USRP 硬件，或当前系统能力明显无法处理。

识别规则：
1. “如何使用 USRP”“解释 USRP”“仿真 QPSK”属于 knowledge_query，不是硬件执行。
   USRP 专属知识可由 USRP 智能体以只读方式回答，但不能因此执行硬件操作。
   通用通信知识仍可交给 Knowledge。
   查询可用 USRP 设备、设备清单或在线状态属于 device_query，能力使用
   device.list 或 device.inspect，不要使用 usrp.get_status。
2. 只有用户明确要求真实配置、发射、接收、停止等操作时，execution_mode 才能是 physical。
3. USRP 真实操作的 target_type 必须是 usrp。
4. RIS、频谱仪、信号源和其他非 USRP 硬件的真实操作属于 unsupported。
5. USRP 执行能力只能是 usrp.configure、usrp.transmit、usrp.receive、usrp.stop 或 usrp.get_status。
6. USRP 硬件执行只要求相应的 usrp 能力；不要自动加入 device.select。
7. 发射操作至少需要 waveform、center_frequency_hz、sample_rate_sps、gain_db、duration_seconds。缺少时写入 missing_parameters，并令 needs_clarification=true。
8. 不得猜测用户没有提供的硬件参数。
9. evidence 只摘述能够证明分类的用户原始表达，reason 简短说明判断依据。
10. 只返回符合 JSON Schema 的结果。
"""

PLANNER_INSTRUCTIONS = """你是多智能体任务规划器。上游已经完成意图分析，并给出当前可用的智能体 ID。

规划规则：
1. 只能使用 available_agents 中列出的智能体，不得创造其他智能体。
2. 选择最少且必要的任务，每个任务只能调用一个 capability。
3. description 必须描述本任务的具体对象、动作和边界。
4. inputs 必须写出执行所需输入；来自前置任务的数据使用“任务ID.字段名”表示。
5. expected_output 和 completion_criteria 必须具体、可检查。
6. device_query 只安排一个 device_manager 任务，不安排 knowledge 或 usrp。
   USRP 专属知识请求可以安排 usrp 的 knowledge.explain 只读任务，不得安排物理操作。
   真实 USRP 物理操作交给 usrp；主图会补齐所需前置任务。
7. knowledge 不得查询实时设备；device_manager 不得审批法规或执行射频操作；usrp 不得执行 RIS、软件仿真或通用硬件任务。
8. 计划必须覆盖 intent_analysis.required_capabilities 中的全部能力。
9. 当 provided_parameters.parameter_policy 为 use_backend_defaults 时，将 unresolved_parameters 原样传给硬件任务，由 Fake 或后端补全；不得自行编造参数。
10. task_id 必须唯一，依赖只能引用更早出现的任务。
11. 仅当 agent_registry_enabled=true 时，available_agents 才包含需要严格遵守的能力仓库资料和前置条件；为 false 时不要自行添加仓库约束。
12. 不输出解释，只返回符合 JSON Schema 的计划。
"""

REVIEW_INSTRUCTIONS = """你是工作流执行评估器。根据用户目标、显式意图、当前计划和结构化执行结果选择下一步。

动作：continue、retry、replan、complete、stop。

规则：
1. 成功且仍有必要任务时选择 continue；目标已满足时选择 complete。
2. retry 只能原样重试当前任务。
3. replan 必须给出完整的新后续计划，且只能使用 available_agents 中的智能体。
4. 非 replan 动作的 revised_tasks 必须为空。
5. USRP 物理任务必须满足真实硬件操作意图；USRP 的 knowledge.explain 是只读知识任务。
6. retry 或 replan 的 remaining 为 0 时不得再次选择对应动作。
7. reason 简短且可检查，不输出长篇思考过程。
"""


class PlanningError(RuntimeError):
    """Raised when intent analysis or planning cannot produce a safe plan."""


@dataclass
class FixedPlanner:
    """Deterministic intent routing and planning for offline use and tests."""

    registry: AgentRegistry = field(default_factory=build_default_registry)
    parameter_policy: ParameterPolicy = ParameterPolicy.STRICT
    enable_agent_registry: bool = False

    def analyze_intent(self, user_request: str) -> IntentAnalysis:
        intent = _analyze_intent_with_rules(user_request)
        if intent.intent_type is IntentType.UNSUPPORTED:
            return intent
        intent = _apply_registry_device_policy(intent, self.enable_agent_registry)
        return _apply_parameter_policy(intent, self.parameter_policy)

    def create_plan(self, user_request: str) -> TaskPlan:
        return self.create_plan_from_intent(user_request, self.analyze_intent(user_request))

    def create_plan_from_intent(
        self, user_request: str, intent: IntentAnalysis
    ) -> TaskPlan:
        if intent.intent_type is IntentType.UNSUPPORTED:
            return TaskPlan(PlanStatus.UNSUPPORTED, intent, reason=intent.reason)
        if intent.needs_clarification:
            return TaskPlan(
                PlanStatus.NEEDS_CLARIFICATION,
                intent,
                clarification_questions=_clarification_questions(intent),
                reason=intent.reason,
            )

        tasks = _build_fixed_tasks(
            intent,
            require_device_selection=self.enable_agent_registry,
        )
        if self.enable_agent_registry:
            candidates = self.registry.match(intent)
            try:
                PlanValidator(self.registry).validate(intent, tasks, candidates)
            except PlanValidationError as exc:
                raise PlanningError(str(exc)) from exc
        return TaskPlan(PlanStatus.READY, intent, tasks)

    def review(
        self,
        state: WorkflowState,
        current_task: Task,
        result: AgentResult,
    ) -> ReviewDecision:
        if result.status is ResultStatus.SUCCEEDED:
            has_pending_task = any(
                task.status is TaskStatus.PENDING for task in state.tasks
            )
            action = NextAction.CONTINUE if has_pending_task else NextAction.COMPLETE
            reason = "当前任务成功，继续固定计划" if has_pending_task else "固定计划已完成"
        else:
            action = NextAction.STOP
            reason = f"固定规划器在任务失败后停止：{result.summary}"
        return ReviewDecision(current_task.task_id, action, reason)

    def review_batch(self, context: dict[str, Any]) -> BatchReviewDecision:
        retryable = context["retryable_task_ids"]
        failed = context["failed_task_ids"]
        if failed:
            if retryable:
                return BatchReviewDecision(
                    NextAction.RETRY, "仅重试本批次失败且尚有机会的任务", retryable
                )
            if context["can_replan"] and all(
                context["batch_results"][task_id]["status"] == "failed"
                for task_id in failed
            ):
                successful = set(context["successful_task_ids"])
                replaced = [
                    task for task in context["current_plan"]
                    if task["task_id"] not in successful
                ]
                used = set(context["known_task_ids"])
                renamed: dict[str, str] = {}
                for task in replaced:
                    base = f"{task['task_id']}_replan"
                    candidate = base
                    suffix = 2
                    while candidate in used:
                        candidate = f"{base}_{suffix}"
                        suffix += 1
                    used.add(candidate)
                    renamed[task["task_id"]] = candidate
                revised = [
                    Task(
                        task_id=renamed[task["task_id"]],
                        agent=task["agent"],
                        description=task["description"],
                        dependencies=[renamed.get(dep, dep) for dep in task["dependencies"]],
                        capability=task["capability"],
                        inputs=dict(task["inputs"]),
                        expected_output=list(task["expected_output"]),
                        completion_criteria=list(task["completion_criteria"]),
                    )
                    for task in replaced
                ]
                return BatchReviewDecision(
                    NextAction.REPLAN,
                    "固定规划器重建失败任务与未执行的下游任务",
                    revised_tasks=revised,
                )
            return BatchReviewDecision(NextAction.STOP, "任务失败且无法继续重试")
        action = NextAction.CONTINUE if context["has_pending_tasks"] else NextAction.COMPLETE
        return BatchReviewDecision(action, "本批次任务成功")


@dataclass
class OllamaPlanner:
    """Analyze intent first, then create a structured task plan."""

    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    temperature: float = 0.2
    log_path: str | Path | None = None
    max_log_entries: int = 20
    registry: AgentRegistry = field(default_factory=build_default_registry)
    parameter_policy: ParameterPolicy = ParameterPolicy.STRICT
    enable_agent_registry: bool = False

    def analyze_intent(self, user_request: str) -> IntentAnalysis:
        response_body = self._chat(
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
        data = self._response_content_as_json(response_body, "结构化意图分析")
        intent = _parse_intent_analysis(data)
        if intent.intent_type is IntentType.DEVICE_QUERY:
            capability = (
                "device.inspect"
                if "状态" in user_request or "inspect" in intent.action.lower()
                else "device.list"
            )
            intent = replace(
                intent,
                execution_mode=ExecutionMode.READ_ONLY,
                required_capabilities=[capability],
                missing_parameters=[],
                needs_clarification=False,
            )
        return intent

    def create_plan(self, user_request: str) -> TaskPlan:
        return self.create_plan_from_intent(user_request, self.analyze_intent(user_request))

    def create_plan_from_intent(
        self, user_request: str, intent: IntentAnalysis
    ) -> TaskPlan:
        if intent.intent_type is IntentType.UNSUPPORTED:
            return TaskPlan(PlanStatus.UNSUPPORTED, intent, reason=intent.reason)
        if (
            intent.intent_type is IntentType.HARDWARE_EXECUTION
            and (
                intent.target_type != "usrp"
                or "usrp" not in user_request.lower()
            )
        ):
            return TaskPlan(
                PlanStatus.UNSUPPORTED,
                intent,
                reason="真实硬件执行必须由用户明确指定 USRP",
            )

        intent = _apply_registry_device_policy(intent, self.enable_agent_registry)
        intent = _apply_parameter_policy(intent, self.parameter_policy)
        if intent.needs_clarification:
            return TaskPlan(
                PlanStatus.NEEDS_CLARIFICATION,
                intent,
                clarification_questions=_clarification_questions(intent),
                reason=intent.reason,
            )
        candidates = self.registry.match(intent) if self.enable_agent_registry else []
        if self.enable_agent_registry and not candidates:
            return TaskPlan(
                PlanStatus.UNSUPPORTED,
                intent,
                reason="能力注册表中没有与该意图匹配的智能体",
            )

        available_agents: list[Any]
        if self.enable_agent_registry:
            available_agents = [
                profile.as_prompt_dict() for profile in candidates
            ]
        elif intent.intent_type is IntentType.DEVICE_QUERY:
            available_agents = ["device_manager"]
        else:
            available_agents = ["knowledge", "device_manager", "usrp"]

        planning_context = {
            "original_request": user_request,
            "intent_analysis": _intent_to_dict(intent),
            "agent_registry_enabled": self.enable_agent_registry,
            "available_agents": available_agents,
        }
        response_body = self._chat(
            instructions=PLANNER_INSTRUCTIONS,
            user_content=(
                "请根据以下上下文生成任务计划：\n"
                f"{json.dumps(planning_context, ensure_ascii=False)}\n\n"
                "请使用以下 JSON Schema 返回计划：\n"
                f"{json.dumps(PLAN_SCHEMA, ensure_ascii=False)}"
            ),
            schema=PLAN_SCHEMA,
            user_request=user_request,
            phase="planning",
        )
        plan_data = self._response_content_as_json(response_body, "结构化计划")
        tasks = _parse_tasks(plan_data)
        _apply_code_owned_task_inputs(intent, tasks)
        if self.enable_agent_registry:
            try:
                PlanValidator(self.registry).validate(intent, tasks, candidates)
            except PlanValidationError as exc:
                raise PlanningError(str(exc)) from exc
        return TaskPlan(PlanStatus.READY, intent, tasks)

    def review(
        self,
        state: WorkflowState,
        current_task: Task,
        result: AgentResult,
    ) -> ReviewDecision:
        if state.intent is None:
            raise PlanningError("工作流缺少意图分析结果")
        candidates = (
            self.registry.match(state.intent) if self.enable_agent_registry else []
        )
        available_agents: list[Any]
        if self.enable_agent_registry:
            available_agents = [
                profile.as_prompt_dict() for profile in candidates
            ]
        else:
            available_agents = ["knowledge", "device_manager", "usrp"]
        review_context = {
            "user_request": state.user_request,
            "intent_analysis": _intent_to_dict(state.intent),
            "agent_registry_enabled": self.enable_agent_registry,
            "available_agents": available_agents,
            "plan_version": state.plan_version,
            "decision_limits": {
                "retry": {
                    "used": state.retry_count,
                    "remaining": state.max_retries - state.retry_count,
                },
                "replan": {
                    "used": state.replan_count,
                    "remaining": state.max_replans - state.replan_count,
                },
            },
            "current_plan": [_task_to_dict(task) for task in state.tasks],
            "current_task": _task_to_dict(current_task),
            "current_result": _result_to_dict(result),
            "all_results": {
                task_id: _result_to_dict(agent_result)
                for task_id, agent_result in state.results.items()
            },
        }
        response_body = self._chat(
            instructions=REVIEW_INSTRUCTIONS,
            user_content=(
                "请评估以下工作流状态并决定下一步：\n"
                f"{json.dumps(review_context, ensure_ascii=False)}\n\n"
                "请使用以下 JSON Schema 返回决策：\n"
                f"{json.dumps(REVIEW_SCHEMA, ensure_ascii=False)}"
            ),
            schema=REVIEW_SCHEMA,
            user_request=state.user_request,
            phase="review",
        )
        decision_data = self._response_content_as_json(response_body, "评估决策")
        successful_tasks = [
            task for task in state.tasks if task.status is TaskStatus.SUCCEEDED
        ]
        decision = _parse_review_decision(
            current_task.task_id,
            decision_data,
            known_task_ids={task.task_id for task in state.tasks},
            dependency_tasks=successful_tasks,
        )
        if decision.action is NextAction.REPLAN:
            _apply_code_owned_task_inputs(state.intent, decision.revised_tasks)
            if self.enable_agent_registry:
                try:
                    PlanValidator(self.registry).validate(
                        state.intent,
                        decision.revised_tasks,
                        candidates,
                        prior_tasks=successful_tasks,
                    )
                except PlanValidationError as exc:
                    raise PlanningError(str(exc)) from exc
        return decision

    def review_batch(self, context: dict[str, Any]) -> BatchReviewDecision:
        response = self._chat(
            instructions=(
                "你是工作流批次评审器。一次评估本批全部结果。成功且有后续任务时 "
                "continue，全部完成时 complete。retry_task_ids 只能包含给定的 "
                "retryable_task_ids。只有 action=replan 且 can_replan=true 时，"
                "revised_tasks 才能填写完整的新后续任务；action 为 continue、retry、"
                "complete 或 stop 时，revised_tasks 必须是空数组 []，不得复制当前计划"
                "或待执行任务。不得放行失败的前置条件。只返回 JSON。"
            ),
            user_content=(
                "请评估批次状态：\n"
                f"{json.dumps(context, ensure_ascii=False, default=str)}\n"
                "输出符合以下 JSON Schema：\n"
                f"{json.dumps(BATCH_REVIEW_SCHEMA, ensure_ascii=False)}"
            ),
            schema=BATCH_REVIEW_SCHEMA,
            user_request=context["user_request"],
            phase="batch_review",
        )
        data = self._response_content_as_json(response, "批次评审决策")
        if not isinstance(data, dict):
            raise PlanningError("批次评审必须返回对象")
        try:
            action = NextAction(data.get("action"))
        except (ValueError, TypeError) as exc:
            raise PlanningError("批次评审返回未知 action") from exc
        reason = data.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise PlanningError("批次评审缺少 reason")
        retry_ids = _string_list(data.get("retry_task_ids"), "retry_task_ids")
        if not set(retry_ids).issubset(context["retryable_task_ids"]):
            raise PlanningError("批次评审要求重试不可重试的任务")
        if action is NextAction.RETRY and not retry_ids:
            raise PlanningError("retry 决策没有任务 ID")
        if action is not NextAction.RETRY and retry_ids:
            raise PlanningError("非 retry 决策不应包含 retry_task_ids")
        raw_revised = data.get("revised_tasks")
        if not isinstance(raw_revised, list):
            raise PlanningError("批次评审缺少 revised_tasks")
        revised: list[Task] = []
        if action is NextAction.REPLAN:
            if not context["can_replan"]:
                raise PlanningError("已达到重规划上限")
            revised = _parse_tasks(
                {"tasks": raw_revised},
                reserved_task_ids=set(context["known_task_ids"]),
                available_dependency_ids=set(context["successful_task_ids"]),
            )
            intent = _parse_intent_analysis(context["intent_analysis"])
            _apply_code_owned_task_inputs(intent, revised)
            if self.enable_agent_registry:
                prior_tasks = [
                    Task(
                        task_id=task["task_id"],
                        agent=task["agent"],
                        description=task["description"],
                        dependencies=list(task["dependencies"]),
                        capability=task["capability"],
                        inputs=task["inputs"],
                    )
                    for task in context["current_plan"]
                    if task["task_id"] in context["successful_task_ids"]
                ]
                try:
                    PlanValidator(self.registry).validate(
                        intent, revised, self.registry.match(intent),
                        prior_tasks=prior_tasks,
                    )
                except PlanValidationError as exc:
                    raise PlanningError(str(exc)) from exc
        elif raw_revised:
            raise PlanningError("非 replan 决策不应包含 revised_tasks")
        return BatchReviewDecision(action, reason, retry_ids, revised)

    def _chat(
        self,
        *,
        instructions: str,
        user_content: str,
        schema: dict[str, Any],
        user_request: str,
        phase: str,
    ) -> Any:
        if not self.model.strip():
            raise PlanningError("未配置 Ollama 模型名")
        if not 0 <= self.temperature <= 2:
            raise PlanningError("Ollama temperature 必须在 0 到 2 之间")

        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_content},
            ],
            "format": schema,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        request = Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise PlanningError(
                f"Ollama 请求失败（HTTP {exc.code}）：{detail}"
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise PlanningError(
                    f"Ollama {phase} 阶段超过 {self.timeout:g} 秒"
                ) from exc
            raise PlanningError(
                f"无法连接 Ollama：{exc.reason}；请确认 Ollama 服务已启动"
            ) from exc
        except TimeoutError as exc:
            raise PlanningError(
                f"Ollama {phase} 阶段超过 {self.timeout:g} 秒"
            ) from exc
        except json.JSONDecodeError as exc:
            raise PlanningError("Ollama 返回的响应不是有效 JSON") from exc

        self._log_response(user_request, response_body, phase=phase)
        return response_body

    @staticmethod
    def _response_content_as_json(response_body: Any, content_name: str) -> Any:
        try:
            content = response_body["message"]["content"]
            return json.loads(content)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise PlanningError(f"Ollama 响应中没有有效的{content_name}") from exc

    def _log_response(
        self, user_request: str, response_body: Any, *, phase: str
    ) -> None:
        if self.log_path is None:
            return
        if self.max_log_entries < 1:
            raise PlanningError("LLM 日志保留数量必须大于 0")

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "user_request": user_request,
            "phase": phase,
            "response": response_body,
        }
        log_path = Path(self.log_path)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            previous_entries: deque[Any] = deque(
                self._read_log_entries(log_path), maxlen=self.max_log_entries
            )
            previous_entries.append(log_entry)
            log_document = {
                "max_entries": self.max_log_entries,
                "entries": list(previous_entries),
            }
            with log_path.open("w", encoding="utf-8") as log_file:
                json.dump(log_document, log_file, ensure_ascii=False, indent=2)
                log_file.write("\n")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise PlanningError(f"无法写入 LLM 日志 {log_path}：{exc}") from exc

    @staticmethod
    def _read_log_entries(log_path: Path) -> list[Any]:
        source_path = log_path
        legacy_path = log_path.with_suffix(".jsonl")
        if not source_path.exists() and legacy_path.exists():
            source_path = legacy_path
        if not source_path.exists():
            return []

        text = source_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            return [json.loads(line) for line in text.splitlines() if line.strip()]

        if isinstance(document, dict) and isinstance(document.get("entries"), list):
            return document["entries"]
        if isinstance(document, dict) and {
            "timestamp",
            "model",
            "user_request",
            "response",
        }.issubset(document):
            return [document]
        if isinstance(document, list):
            return document
        raise TypeError("日志文件必须是包含 entries 的 JSON 对象")


def _parse_intent_analysis(data: Any) -> IntentAnalysis:
    if not isinstance(data, dict):
        raise PlanningError("LLM 意图分析必须是对象")
    try:
        intent_type = IntentType(data.get("intent_type"))
        execution_mode = ExecutionMode(data.get("execution_mode"))
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
    )


def _parse_review_decision(
    task_id: str,
    decision_data: Any,
    *,
    known_task_ids: set[str] | None = None,
    dependency_tasks: list[Task] | None = None,
) -> ReviewDecision:
    if not isinstance(decision_data, dict):
        raise PlanningError("LLM 评估决策必须是对象")
    try:
        action = NextAction(decision_data.get("action"))
    except (TypeError, ValueError) as exc:
        raise PlanningError("LLM 评估返回了未知 action") from exc

    reason = decision_data.get("reason")
    revised_tasks_data = decision_data.get("revised_tasks")
    if not isinstance(reason, str) or not reason.strip():
        raise PlanningError("LLM 评估决策必须包含非空 reason")
    if not isinstance(revised_tasks_data, list):
        raise PlanningError("LLM 评估决策必须包含 revised_tasks 列表")

    if action is NextAction.REPLAN:
        dependency_ids = {task.task_id for task in (dependency_tasks or [])}
        revised_tasks = _parse_tasks(
            {"tasks": revised_tasks_data},
            reserved_task_ids=known_task_ids,
            available_dependency_ids=dependency_ids,
        )
    else:
        if revised_tasks_data:
            raise PlanningError("只有 replan 决策可以包含 revised_tasks")
        revised_tasks = []

    return ReviewDecision(task_id, action, reason, revised_tasks)


def _parse_tasks(
    plan_data: Any,
    *,
    reserved_task_ids: set[str] | None = None,
    available_dependency_ids: set[str] | None = None,
) -> list[Task]:
    if not isinstance(plan_data, dict) or not isinstance(plan_data.get("tasks"), list):
        raise PlanningError("LLM 计划必须包含 tasks 列表")
    raw_tasks = plan_data["tasks"]
    if not 1 <= len(raw_tasks) <= 6:
        raise PlanningError("LLM 计划必须包含 1 到 6 个任务")

    reserved = set(reserved_task_ids or ())
    known_dependency_ids = set(available_dependency_ids or ())
    tasks: list[Task] = []
    new_task_ids: set[str] = set()
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            raise PlanningError("计划中的每一项都必须是任务对象")
        task_id = raw_task.get("task_id")
        agent = raw_task.get("agent")
        capability = raw_task.get("capability")
        description = raw_task.get("description")
        inputs = raw_task.get("inputs")
        dependencies = raw_task.get("dependencies")
        expected_output = raw_task.get("expected_output")
        completion_criteria = raw_task.get("completion_criteria")

        if not isinstance(task_id, str) or not task_id.strip():
            raise PlanningError("每个任务都必须有非空 task_id")
        if task_id in reserved or task_id in new_task_ids:
            raise PlanningError(f"任务 ID 重复：{task_id}")
        if not isinstance(agent, str) or not agent.strip():
            raise PlanningError(f"任务 {task_id} 缺少有效 agent")
        if not isinstance(capability, str) or not capability.strip():
            raise PlanningError(f"任务 {task_id} 缺少 capability")
        if not isinstance(description, str) or not description.strip():
            raise PlanningError(f"任务 {task_id} 缺少有效 description")
        if not isinstance(inputs, dict):
            raise PlanningError(f"任务 {task_id} 的 inputs 必须是对象")
        dependencies = _string_list(dependencies, f"任务 {task_id} 的 dependencies")
        expected_output = _string_list(
            expected_output, f"任务 {task_id} 的 expected_output"
        )
        completion_criteria = _string_list(
            completion_criteria, f"任务 {task_id} 的 completion_criteria"
        )
        if not expected_output or not completion_criteria:
            raise PlanningError(f"任务 {task_id} 必须定义输出和完成标准")

        missing_dependencies = set(dependencies) - known_dependency_ids
        if missing_dependencies:
            missing = "、".join(sorted(missing_dependencies))
            raise PlanningError(f"任务 {task_id} 依赖尚未出现的任务：{missing}")

        tasks.append(
            Task(
                task_id=task_id,
                agent=agent,
                description=description,
                dependencies=dependencies,
                capability=capability,
                inputs=inputs,
                expected_output=expected_output,
                completion_criteria=completion_criteria,
            )
        )
        new_task_ids.add(task_id)
        known_dependency_ids.add(task_id)
    return tasks


def _analyze_intent_with_rules(user_request: str) -> IntentAnalysis:
    request = user_request.strip()
    lowered = request.lower()
    is_usrp = "usrp" in lowered
    is_ris = "ris" in lowered
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

    if is_ris and has_execution_verb and not asks_how:
        return IntentAnalysis(
            IntentType.UNSUPPORTED,
            "wireless_communication",
            "ris",
            "execute",
            ExecutionMode.PHYSICAL,
            evidence=request,
            reason="当前系统没有 RIS 硬件执行智能体",
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


def _build_fixed_tasks(
    intent: IntentAnalysis,
    *,
    require_device_selection: bool = False,
) -> list[Task]:
    if intent.intent_type is IntentType.KNOWLEDGE_QUERY:
        return [
            Task(
                "answer_knowledge",
                "knowledge",
                "回答用户的通信知识、指导或仿真问题",
                capability="knowledge.explain",
                inputs={"request": intent.evidence},
                expected_output=["knowledge", "assumptions"],
                completion_criteria=["回答覆盖用户问题且不声称执行真实硬件"],
            )
        ]
    if intent.intent_type is IntentType.DEVICE_QUERY:
        capability = intent.required_capabilities[0]
        return [
            Task(
                "query_devices",
                "device_manager",
                "查询网络中的设备清单、在线状态和能力",
                capability=capability,
                inputs={"target_type": intent.target_type},
                expected_output=["network_devices"],
                completion_criteria=["返回匹配设备及其当前状态"],
            )
        ]

    usrp_capability = next(
        capability
        for capability in intent.required_capabilities
        if capability.startswith("usrp.")
    )
    if not require_device_selection:
        return [
            Task(
                "execute_usrp_operation",
                "usrp",
                f"执行 USRP {intent.action} 操作",
                capability=usrp_capability,
                inputs={
                    "operation": intent.action,
                    **intent.provided_parameters,
                },
                expected_output=[
                    "operation_id",
                    "actual_configuration",
                    "execution_status",
                ],
                completion_criteria=["操作完成并返回可追踪的 operation_id"],
            )
        ]
    return [
        Task(
            "select_usrp",
            "device_manager",
            "查询在线 USRP，并选择满足本次操作能力的设备",
            capability="device.select",
            inputs={"target_type": "usrp", "required_capability": usrp_capability},
            expected_output=["device_id", "selected_device"],
            completion_criteria=["选中在线且能力匹配的 USRP"],
        ),
        Task(
            "execute_usrp_operation",
            "usrp",
            f"在选定 USRP 上执行 {intent.action} 操作",
            dependencies=["select_usrp"],
            capability=usrp_capability,
            inputs={
                "device_id_from": "select_usrp.device_id",
                "operation": intent.action,
                **intent.provided_parameters,
            },
            expected_output=[
                "operation_id",
                "device_id",
                "actual_configuration",
                "execution_status",
            ],
            completion_criteria=["操作成功且返回可追踪的 operation_id"],
        ),
    ]


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


def _ensure_hardware_device_selection(intent: IntentAnalysis) -> IntentAnalysis:
    """Make device selection a code-owned invariant for hardware execution."""

    if intent.intent_type is not IntentType.HARDWARE_EXECUTION:
        return intent
    capabilities = [
        capability
        for capability in intent.required_capabilities
        if capability != "device.select"
    ]
    return replace(
        intent,
        required_capabilities=["device.select", *capabilities],
    )


def _apply_registry_device_policy(
    intent: IntentAnalysis,
    registry_enabled: bool,
) -> IntentAnalysis:
    """Add legacy device selection only when the capability registry is enabled."""

    if registry_enabled:
        return _ensure_hardware_device_selection(intent)
    if intent.intent_type is not IntentType.HARDWARE_EXECUTION:
        return intent
    return replace(
        intent,
        required_capabilities=[
            capability
            for capability in intent.required_capabilities
            if capability != "device.select"
        ],
    )


def _apply_code_owned_task_inputs(
    intent: IntentAnalysis, tasks: list[Task]
) -> None:
    """Attach orchestration metadata that the model must not be trusted to copy."""

    if intent.provided_parameters.get("parameter_policy") != "use_backend_defaults":
        return
    unresolved = list(
        intent.provided_parameters.get("unresolved_parameters", [])
    )
    for task in tasks:
        if task.agent != "usrp":
            continue
        task.inputs["parameter_policy"] = "use_backend_defaults"
        task.inputs["unresolved_parameters"] = unresolved


def _intent_to_dict(intent: IntentAnalysis) -> dict[str, Any]:
    return {
        "intent_type": intent.intent_type.value,
        "domain": intent.domain,
        "target_type": intent.target_type,
        "action": intent.action,
        "execution_mode": intent.execution_mode.value,
        "required_capabilities": intent.required_capabilities,
        "provided_parameters": intent.provided_parameters,
        "missing_parameters": intent.missing_parameters,
        "needs_clarification": intent.needs_clarification,
        "evidence": intent.evidence,
        "reason": intent.reason,
    }


def _task_to_dict(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "agent": task.agent,
        "capability": task.capability,
        "description": task.description,
        "inputs": task.inputs,
        "expected_output": task.expected_output,
        "completion_criteria": task.completion_criteria,
        "dependencies": task.dependencies,
        "status": task.status.value,
        "attempts": task.attempts,
    }


def _result_to_dict(result: AgentResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "summary": result.summary,
        "data": result.data,
        "error": result.error,
    }


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PlanningError(f"{name} 必须是字符串列表")
    return value
