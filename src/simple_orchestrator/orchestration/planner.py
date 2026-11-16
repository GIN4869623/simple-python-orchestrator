"""Task planning and execution review; intent recognition lives separately."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from ..agent_registry import AgentRegistry, build_default_registry
from ..models import (
    AgentResult,
    BatchReviewDecision,
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
from ..validation import PlanValidationError, PlanValidator
from ..errors import DuplicateTaskIdError, PlanningError
from ..intent.agent import (
    FixedIntentAgent, OllamaIntentAgent, _apply_parameter_policy,
    _clarification_questions, _parse_intent_analysis, _string_list,
)
from ..ollama import OllamaClient
from .workflows import _build_fixed_tasks


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

PLANNER_INSTRUCTIONS = """你是多智能体任务规划器。上游已经完成意图分析，并给出当前可用的智能体 ID。

规划规则：
1. 只能使用 available_agents 中列出的智能体，不得创造其他智能体。
2. 选择最少且必要的业务任务。capability 是路由和权限校验用的业务能力标签，
   不是远端工具名；远端智能体可以自行规划并调用多个内部工具完成一个任务。
3. description 必须描述本任务的具体对象、动作和边界。
4. inputs 必须写出执行所需输入；来自前置任务的数据使用“任务ID.字段名”表示。
5. expected_output 和 completion_criteria 必须具体、可检查。
6. device_query 只安排一个 device_manager 任务，不安排 knowledge 或 usrp。
   USRP 专属知识请求可以安排 usrp 的 knowledge.explain 只读任务，不得安排物理操作。
   真实 USRP 物理操作交给 usrp；主图会补齐所需前置任务。
7. knowledge 不得查询实时设备；device_manager 只负责查询设备清单和状态，不得选择、占用或使用设备，也不得审批法规；usrp 不得执行 RIS、软件仿真或通用硬件任务。
8. 计划必须覆盖 intent_analysis.required_capabilities 中的全部能力。
9. 当 provided_parameters.parameter_policy 为 use_backend_defaults 时，将 unresolved_parameters 原样传给硬件任务，由 Fake 或后端补全；不得自行编造参数。
10. task_id 必须唯一，依赖只能引用更早出现的任务。
11. 仅当 agent_registry_enabled=true 时，available_agents 才包含需要严格遵守的能力仓库资料和前置条件；为 false 时不要自行添加仓库约束。
12. 网络监管返回 network_devices 设备清单。启用能力仓库时，USRP 物理任务必须依赖设备查询任务，并在 inputs.device_candidates_from 中引用“查询任务ID.network_devices”；目标设备由编排器确定。
13. 不输出解释，只返回符合 JSON Schema 的计划。
14. knowledge 任务可在 inputs.question 中提供一句简短、可独立理解的子问题，保留用户明确要求的回答长度和格式。不要把预期输出或验收标准重复写入问题。
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


@dataclass
class FixedPlanner:
    """Deterministic intent routing and planning for offline use and tests."""

    registry: AgentRegistry = field(default_factory=build_default_registry)
    parameter_policy: ParameterPolicy = ParameterPolicy.STRICT
    enable_agent_registry: bool = False

    def analyze_intent(self, user_request: str) -> IntentAnalysis:
        """Compatibility helper; runtime injects a separate intent agent."""
        intent = FixedIntentAgent(self.parameter_policy).analyze_intent(user_request)
        if intent.intent_type is IntentType.UNSUPPORTED:
            return intent
        return _apply_registry_device_policy(intent, self.enable_agent_registry)

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
            require_device_query=self.enable_agent_registry,
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
class OllamaPlanner(OllamaClient):
    """Create and review dynamic plans from an upstream intent."""

    registry: AgentRegistry = field(default_factory=build_default_registry)
    parameter_policy: ParameterPolicy = ParameterPolicy.STRICT
    enable_agent_registry: bool = False

    client: OllamaClient | None = field(default=None, repr=False)

    def _chat(self, **kwargs: Any) -> Any:
        if self.client is not None:
            return self.client._chat(**kwargs)
        return super()._chat(**kwargs)

    def analyze_intent(self, user_request: str) -> IntentAnalysis:
        """Compatibility helper; main runtime uses its own intent agent."""
        return OllamaIntentAgent(self).analyze_intent(user_request)

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
                "或待执行任务。replan 的每个 task_id 必须是全新且互不重复的 ID，"
                "不得使用 known_task_ids 中的任何 ID（包括成功、失败及未执行任务）。"
                "只有 retry 才沿用原任务 ID。重命名时同步更新 dependencies 和 inputs "
                "中的任务引用；仅可引用成功的历史任务或更早出现的新任务。"
                "不得放行失败的前置条件。只返回 JSON。"
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
            revised = self._parse_replanned_tasks(raw_revised, context)
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

    def _parse_replanned_tasks(
        self, raw_tasks: list[Any], context: dict[str, Any],
    ) -> list[Task]:
        reserved = set(context["known_task_ids"])
        available = set(context["successful_task_ids"])
        try:
            return _parse_tasks(
                {"tasks": raw_tasks}, reserved_task_ids=reserved,
                available_dependency_ids=available,
            )
        except DuplicateTaskIdError as exc:
            # Correct the plan once, before dispatch. Do not retry agent execution
            # or silently rename IDs whose dependency references may be ambiguous.
            response = self._chat(
                instructions=(
                    "修正重规划任务的 ID 冲突，只返回符合 JSON Schema 的完整 tasks。"
                    "每个 task_id 必须全新且互不重复，不得使用 known_task_ids 中的 ID。"
                    "保留任务的业务目标、智能体、能力、参数、完成标准及依赖关系，"
                    "同步更新 dependencies 和 inputs 中指向新任务的 ID 引用。"
                    "成功历史任务的引用保持不变；不得依赖失败或未执行的历史任务。"
                    "依赖只能引用 successful_task_ids 或更早出现的新任务。"
                ),
                user_content=json.dumps({
                    "context": context,
                    "invalid_plan": {"tasks": raw_tasks},
                    "validation_error": str(exc),
                    "output_schema": PLAN_SCHEMA,
                }, ensure_ascii=False, default=str),
                schema=PLAN_SCHEMA,
                user_request=context["user_request"],
                phase="replan_id_correction",
            )
        # A second invalid response fails normally; never enter a correction loop.
        return _parse_tasks(
            self._response_content_as_json(response, "任务 ID 纠正计划"),
            reserved_task_ids=reserved, available_dependency_ids=available,
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
            raise DuplicateTaskIdError(f"任务 ID 重复：{task_id}")
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


def _ensure_hardware_device_query(intent: IntentAnalysis) -> IntentAnalysis:
    """Require a read-only device query before hardware execution."""

    if intent.intent_type is not IntentType.HARDWARE_EXECUTION:
        return intent
    capabilities = [
        capability
        for capability in intent.required_capabilities
        if capability != "device.list"
    ]
    return replace(
        intent,
        required_capabilities=["device.list", *capabilities],
    )


def _apply_registry_device_policy(
    intent: IntentAnalysis,
    registry_enabled: bool,
) -> IntentAnalysis:
    """Require a device query only when the capability registry is enabled."""

    if registry_enabled:
        return _ensure_hardware_device_query(intent)
    if intent.intent_type is not IntentType.HARDWARE_EXECUTION:
        return intent
    return replace(
        intent,
        required_capabilities=[
            capability
            for capability in intent.required_capabilities
            if capability != "device.list"
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
        "handling_mode": intent.handling_mode.value,
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
