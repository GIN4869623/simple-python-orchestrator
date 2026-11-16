"""Application entry: clarify, dispatch directly, run a workflow, or orchestrate."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..agent_registry import AgentRegistry, build_default_registry
from ..errors import PlanningError
from .execution import MainState, WorkflowExecutor
from .graph import GraphOrchestrator
from ..intent.agent import _clarification_questions
from ..models import ExecutionMode, HandlingMode, IntentType, PlanStatus, WorkflowStatus
from ..ports import IntentAgentPort
from ..validation import PlanValidationError, PlanValidator
from .workflows import build_direct_task, build_predefined_workflow


@dataclass
class RequestRouter:
    """Route only to explicitly supported handlers; no task planning here."""

    intent_agent: IntentAgentPort
    executor: WorkflowExecutor
    orchestrator: GraphOrchestrator
    registry: AgentRegistry = field(default_factory=build_default_registry)
    enable_agent_registry: bool = False

    def run(self, user_request: str, *, request_id: str | None = None) -> MainState:
        state = self.executor.initial_state(user_request, request_id)
        intent = self.intent_agent.analyze_intent(user_request)
        state.update(intent=intent, handling_mode=intent.handling_mode)
        if intent.intent_type is IntentType.UNSUPPORTED:
            return self._unsupported(state, intent.reason)
        if intent.intent_type is IntentType.HARDWARE_EXECUTION and (
            intent.target_type != "usrp"
            or intent.execution_mode is not ExecutionMode.PHYSICAL
            or "usrp" not in user_request.lower()
        ):
            return self._unsupported(state, "真实硬件执行必须由用户明确指定 USRP")
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
            state = self.orchestrator.run(
                user_request, request_id=state["request_id"], intent=intent,
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
