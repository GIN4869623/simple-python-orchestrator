"""Dynamic orchestration; intent recognition is supplied by the application."""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import START

# Re-export state records for existing users of this module.
from .execution import (
    AttemptResult, BranchState, MainState, PlanSnapshot, TaskSpec,
    WorkflowExecutor, append_reviews, merge_results, merge_statuses,
)
from ..errors import PlanningError
from ..models import BatchReviewDecision, IntentAnalysis, PlanStatus, Task, TaskStatus, WorkflowStatus
from ..ports import DeviceManagerPort, GraphPlannerPort, KnowledgeAgentPort, PlannerPort, UsrpAgentPort


class GraphOrchestrator(WorkflowExecutor):
    """Plan and review only requests routed to dynamic orchestration."""

    def __init__(
        self, planner: GraphPlannerPort | PlannerPort,
        knowledge: KnowledgeAgentPort, device_manager: DeviceManagerPort,
        usrp: UsrpAgentPort, max_retries: int = 1, max_replans: int = 1,
        allow_remote_physical_execution: bool = False,
        ris=None,
    ) -> None:
        self.planner = planner
        super().__init__(
            knowledge, device_manager, usrp, max_retries, max_replans,
            allow_remote_physical_execution, ris=ris,
        )

    def run(
        self, user_request: str, *, request_id: str | None = None,
        intent: IntentAnalysis | None = None,
        on_progress: Callable[[MainState], None] | None = None,
    ) -> MainState:
        initial = self.initial_state(user_request, request_id)
        if intent is not None:
            initial["intent"] = intent
        config = {"recursion_limit": 100}
        if on_progress is None:
            return self.graph.invoke(initial, config)
        state = initial
        for state in self.graph.stream(initial, config, stream_mode="values"):
            if state["status"] is WorkflowStatus.RUNNING:
                on_progress(state)
        return state

    def _build_graph(self):
        graph = self._execution_graph()
        graph.add_node("analyze_intent", self._analyze_intent)
        graph.add_node("create_plan", self._create_plan)
        graph.add_edge(START, "analyze_intent")
        graph.add_edge("analyze_intent", "create_plan")
        graph.add_conditional_edges("create_plan", self._route_plan)
        return graph.compile()

    def _analyze_intent(self, state: MainState) -> dict[str, Any]:
        if "intent" in state:
            return {}
        # Compatibility for the old direct GraphOrchestrator entry point.
        analyze = getattr(self.planner, "analyze_intent", None)
        if analyze is not None:
            return {"intent": analyze(state["user_request"])}
        plan = self.planner.create_plan(state["user_request"])
        return {"intent": plan.intent, "precomputed_plan": plan}

    def _create_plan(self, state: MainState) -> dict[str, Any]:
        plan_from_intent = getattr(self.planner, "create_plan_from_intent", None)
        if plan_from_intent is not None:
            plan = plan_from_intent(state["user_request"], state["intent"])
        else:
            plan = state["precomputed_plan"]
        updates: dict[str, Any] = {"plan_status": plan.status, "intent": plan.intent}
        if plan.status is PlanStatus.NEEDS_CLARIFICATION:
            updates.update(
                status=WorkflowStatus.NEEDS_CLARIFICATION,
                error=plan.reason or "missing required parameters",
                final_answer="执行前需要补充信息：" + "；".join(plan.clarification_questions),
            )
            return updates
        if plan.status is PlanStatus.UNSUPPORTED:
            updates.update(
                status=WorkflowStatus.UNSUPPORTED,
                error=plan.reason,
                final_answer=f"当前智能体无法处理该请求：{plan.reason}",
            )
            return updates
        updates.update(self._load_tasks(state["user_request"], plan.intent, plan.tasks))
        return updates

    def _decide_batch(self, context: dict[str, Any]) -> BatchReviewDecision:
        review_batch = getattr(self.planner, "review_batch", None)
        if review_batch is not None:
            return review_batch(context)
        return super()._decide_batch(context)

    def _apply_replan(self, state: MainState, revised: list[Task]) -> dict[str, Any]:
        used = set(state["task_statuses"])
        successful = [
            task for task in self._active_tasks(state)
            if state["task_statuses"][task.task_id] is TaskStatus.SUCCEEDED
        ]
        if any(task.task_id in used for task in revised):
            raise PlanningError("新计划使用了已有任务 ID")
        # Re-check physical preflight for any newly planned USRP task.
        new_specs = self._prepare_tasks(
            revised, state["intent"], reserved_ids=used,
            prior_success_tasks=successful,
        )
        if any(self._is_physical_usrp(task) for task in new_specs) and "usrp" not in state["user_request"].lower():
            raise PlanningError("用户未明确指定 USRP，不允许硬件执行")
        self._validate_tasks(new_specs, {task.task_id for task in successful})
        next_version = state["active_plan_version"] + 1
        snapshot = PlanSnapshot(next_version, tuple([*successful, *new_specs]))
        skipped = {
            task.task_id: TaskStatus.SKIPPED for task in self._active_tasks(state)
            if state["task_statuses"][task.task_id] is TaskStatus.PENDING
        }
        return {
            "plan_history": [*state["plan_history"], snapshot],
            "active_plan_version": next_version,
            "replan_count": state["replan_count"] + 1,
            "task_statuses": {
                **skipped,
                **{task.task_id: TaskStatus.PENDING for task in new_specs},
            },
        }
