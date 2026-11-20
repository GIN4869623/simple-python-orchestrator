"""Application entry: clarify, dispatch directly, run a workflow, or orchestrate."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

from ..agent_registry import AgentRegistry, build_default_registry
from ..errors import PlanningError
from .execution import MainState, WorkflowExecutor
from ..intent.agent import _clarification_questions
from ..models import ExecutionMode, HandlingMode, IntentAnalysis, IntentType, PlanStatus, Task, WorkflowStatus
from ..ports import IntentAgentPort, OrchestrationPort
from ..validation import PlanValidationError, PlanValidator
from .workflows import build_direct_task, build_predefined_workflow


@dataclass
class RequestRouter:
    """Route only to explicitly supported handlers; no task planning here."""

    intent_agent: IntentAgentPort
    executor: WorkflowExecutor
    orchestrator: OrchestrationPort
    registry: AgentRegistry = field(default_factory=build_default_registry)
    enable_agent_registry: bool = False

    def run(self, user_request: str, *, request_id: str | None = None) -> MainState:
        intent = self.intent_agent.analyze_intent(user_request)
        return self.run_with_intent(user_request, intent, request_id=request_id)

    def run_with_intent(
        self, user_request: str, intent: IntentAnalysis, *, request_id: str | None = None,
        on_dispatch: Callable[[HandlingMode, list[Task] | None], None] | None = None,
    ) -> MainState:
        """Dispatch an already recognized intent without another model call."""
        state = self.executor.initial_state(user_request, request_id)
        state.update(intent=intent, handling_mode=intent.handling_mode)
        if intent.intent_type is IntentType.UNSUPPORTED:
            return self._unsupported(state, intent.reason)
        if intent.intent_type is IntentType.HARDWARE_EXECUTION and (
            intent.target_type not in {"usrp", "ris"}
            or intent.execution_mode is not ExecutionMode.PHYSICAL
            or (intent.target_type not in user_request.lower() and not (intent.target_type == "ris" and "反射面" in user_request))
        ):
            return self._unsupported(state, "真实硬件执行必须由用户明确指定 USRP 或 RIS")
        if intent.target_type == "ris" and self.executor.ris is None:
            return self._unsupported(state, "RIS 业务接口未配置")
        if intent.needs_clarification:
            questions = _clarification_questions(intent)
            state.update(
                status=WorkflowStatus.NEEDS_CLARIFICATION,
                plan_status=PlanStatus.NEEDS_CLARIFICATION,
                clarification_questions=questions,
                error=intent.reason or "missing required parameters",
                final_answer="执行前需要补充信息：" + "；".join(questions),
            )
            return state

        mode = intent.handling_mode
        tasks = None
        if mode is HandlingMode.DIRECT:
            task = build_direct_task(user_request, intent)
            if task is not None:
                tasks = [task]
            elif intent.intent_type is IntentType.HARDWARE_EXECUTION:
                # Physical operations always use the complete preflight workflow.
                mode = HandlingMode.FIXED_WORKFLOW
        if mode is HandlingMode.FIXED_WORKFLOW:
            tasks = build_predefined_workflow(
                intent, require_device_query=self.enable_agent_registry,
            )
        if tasks is None:
            if on_dispatch:
                on_dispatch(HandlingMode.ORCHESTRATED, None)
            state = self.orchestrator.run(
                user_request, request_id=state["request_id"],
                intent=replace(intent, handling_mode=HandlingMode.ORCHESTRATED),
            )
            state["handling_mode"] = HandlingMode.ORCHESTRATED
            return state

        if self.enable_agent_registry:
            # Include the predefined workflow's dependencies when matching;
            # the user's intent itself still describes only the business goal.
            matching_intent = replace(
                intent,
                required_capabilities=list(dict.fromkeys([
                    *intent.required_capabilities,
                    *(task.capability for task in tasks),
                ])),
            )
            candidates = self.registry.match(matching_intent)
            if not candidates:
                return self._unsupported(state, "能力注册表中没有与该意图匹配的智能体")
            try:
                PlanValidator(self.registry).validate(intent, tasks, candidates)
            except PlanValidationError as exc:
                raise PlanningError(str(exc)) from exc
        if on_dispatch:
            on_dispatch(mode, tasks)
        state = self.executor.run(
            user_request, request_id=state["request_id"], intent=intent, tasks=tasks,
        )
        state["handling_mode"] = mode
        return state

    @staticmethod
    def _unsupported(state: MainState, reason: str) -> MainState:
        state.update(
            status=WorkflowStatus.UNSUPPORTED,
            plan_status=PlanStatus.UNSUPPORTED,
            error=reason,
            final_answer=f"当前智能体无法处理该请求：{reason}",
        )
        return state
