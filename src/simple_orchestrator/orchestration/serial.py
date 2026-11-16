"""The minimal, synchronous orchestration loop."""

from __future__ import annotations

from typing import Any

from ..task_messages import invocation_metadata, make_agent_request, make_knowledge_request

from ..models import (
    AgentResult,
    NextAction,
    PlanStatus,
    ResultStatus,
    Task,
    TaskStatus,
    WorkflowState,
    WorkflowStatus,
)
from ..ports import (
    DeviceManagerPort,
    KnowledgeAgentPort,
    PlannerPort,
    UsrpAgentPort,
)


class Orchestrator:
    MAX_RETRIES = 1
    MAX_REPLANS = 1

    def __init__(
        self,
        planner: PlannerPort,
        knowledge: KnowledgeAgentPort,
        device_manager: DeviceManagerPort,
        usrp: UsrpAgentPort,
        max_retries: int = 1,
        max_replans: int = 1,
    ) -> None:
        self.planner = planner
        self.knowledge = knowledge
        self.device_manager = device_manager
        self.usrp = usrp
        if max_retries not in (0, 1):
            raise ValueError("工作流重新执行次数只能设置为 0 或 1")
        if max_replans not in (0, 1):
            raise ValueError("工作流重新规划次数只能设置为 0 或 1")
        self.max_retries = min(max_retries, self.MAX_RETRIES)
        self.max_replans = min(max_replans, self.MAX_REPLANS)

    def run(self, user_request: str) -> WorkflowState:
        state = WorkflowState(
            user_request=user_request,
            max_retries=self.max_retries,
            max_replans=self.max_replans,
        )
        plan = self.planner.create_plan(user_request)
        state.intent = plan.intent
        state.tasks = plan.tasks
        if plan.status is PlanStatus.NEEDS_CLARIFICATION:
            state.status = WorkflowStatus.NEEDS_CLARIFICATION
            questions = "；".join(plan.clarification_questions)
            state.error = plan.reason or "missing required parameters"
            state.final_answer = f"执行前需要补充信息：{questions}"
            return state
        if plan.status is PlanStatus.UNSUPPORTED:
            state.status = WorkflowStatus.UNSUPPORTED
            state.error = plan.reason
            state.final_answer = f"当前智能体无法处理该请求：{plan.reason}"
            return state
        task_index = 0

        while task_index < len(state.tasks):
            task = state.tasks[task_index]
            if not self._dependencies_succeeded(task, state):
                task.status = TaskStatus.SKIPPED
                task_index += 1
                continue

            task.status = TaskStatus.RUNNING
            task.attempts += 1
            result = self._execute(task, state)
            state.results[task.task_id] = result

            if result.status is ResultStatus.SUCCEEDED:
                task.status = TaskStatus.SUCCEEDED
            else:
                task.status = self._task_status_for(result)

            decision = self.planner.review(state, task, result)
            state.reviews.append(decision)

            if decision.action is NextAction.COMPLETE:
                self._skip_remaining_tasks(state)
                state.status = WorkflowStatus.SUCCEEDED
                state.final_answer = self._build_completion_answer(state, decision.reason)
                return state

            if decision.action is NextAction.CONTINUE:
                if result.status is not ResultStatus.SUCCEEDED:
                    self._stop_after_review(
                        state,
                        result,
                        "LLM 在任务未成功时返回 continue，工作流停止",
                    )
                    return state
                task_index += 1
                continue

            if decision.action is NextAction.RETRY:
                if (
                    result.status is not ResultStatus.SUCCEEDED
                    and state.retry_count < state.max_retries
                ):
                    state.retry_count += 1
                    task.status = TaskStatus.PENDING
                    continue
                self._stop_after_review(
                    state,
                    result,
                    f"无法继续重试：{decision.reason}",
                )
                return state

            if decision.action is NextAction.REPLAN:
                if state.replan_count >= state.max_replans:
                    self._stop_after_review(
                        state,
                        result,
                        f"已达到重新规划次数上限：{decision.reason}",
                    )
                    return state
                try:
                    self._append_revised_plan(state, decision.revised_tasks)
                except ValueError as exc:
                    self._stop_after_review(state, result, str(exc))
                    return state
                state.replan_count += 1
                state.plan_version += 1
                task_index = len(state.tasks) - len(decision.revised_tasks)
                continue

            self._stop_after_review(state, result, decision.reason)
            return state

        state.status = WorkflowStatus.SUCCEEDED
        state.final_answer = self._build_final_answer(state)
        return state

    def _execute(self, task: Task, state: WorkflowState) -> AgentResult:
        context: dict[str, Any] = {
            "user_request": state.user_request,
            "current_task": {
                "task_id": task.task_id,
                "agent": task.agent,
                "description": task.description,
                "dependencies": task.dependencies,
                "attempt": task.attempts,
                "capability": task.capability,
                "inputs": task.inputs,
                "expected_output": task.expected_output,
                "completion_criteria": task.completion_criteria,
            },
            "previous_results": {
                task_id: result.data for task_id, result in state.results.items()
            },
        }
        if task.agent == "knowledge":
            request = make_knowledge_request(
                context,
                use_user_question=len(state.tasks) == 1 and task.capability != "knowledge.lookup_skill",
            )
            return self.knowledge.invoke(request, metadata=invocation_metadata(context))
        if task.agent == "device_manager":
            return self.device_manager.invoke(make_agent_request(context), metadata=invocation_metadata(context))
        if task.agent == "usrp":
            return self.usrp.execute(context)
        return AgentResult(
            ResultStatus.FAILED,
            f"未知执行目标：{task.agent}",
            error="unknown agent",
        )

    @staticmethod
    def _dependencies_succeeded(task: Task, state: WorkflowState) -> bool:
        return all(
            dependency in state.results
            and state.results[dependency].status is ResultStatus.SUCCEEDED
            for dependency in task.dependencies
        )

    @staticmethod
    def _task_status_for(result: AgentResult) -> TaskStatus:
        if result.status is ResultStatus.REJECTED:
            return TaskStatus.REJECTED
        if result.status is ResultStatus.UNKNOWN:
            return TaskStatus.UNKNOWN
        return TaskStatus.FAILED

    @staticmethod
    def _stop_workflow(state: WorkflowState, result: AgentResult) -> None:
        if result.status is ResultStatus.REJECTED:
            state.status = WorkflowStatus.REJECTED
        elif result.status is ResultStatus.UNKNOWN:
            state.status = WorkflowStatus.UNKNOWN
        else:
            state.status = WorkflowStatus.FAILED
        state.error = result.error or result.summary
        state.final_answer = f"工作流已停止：{result.summary}"

    @staticmethod
    def _skip_remaining_tasks(state: WorkflowState) -> None:
        for task in state.tasks:
            if task.status is TaskStatus.PENDING:
                task.status = TaskStatus.SKIPPED

    @staticmethod
    def _append_revised_plan(state: WorkflowState, revised_tasks: list[Task]) -> None:
        existing_task_ids = {task.task_id for task in state.tasks}
        repeated_task_ids = existing_task_ids.intersection(
            task.task_id for task in revised_tasks
        )
        if repeated_task_ids:
            repeated = ", ".join(sorted(repeated_task_ids))
            raise ValueError(f"重新规划使用了已有任务 ID：{repeated}")
        Orchestrator._skip_remaining_tasks(state)
        state.tasks.extend(revised_tasks)

    @staticmethod
    def _stop_after_review(
        state: WorkflowState,
        result: AgentResult,
        reason: str,
    ) -> None:
        Orchestrator._stop_workflow(state, result)
        Orchestrator._skip_remaining_tasks(state)
        state.error = reason
        state.final_answer = f"规划器决定停止工作流：{reason}"

    @staticmethod
    def _build_completion_answer(state: WorkflowState, reason: str) -> str:
        summaries = " → ".join(result.summary for result in state.results.values())
        normalized_reason = reason.rstrip("。；; ")
        return f"规划器确认任务成功：{normalized_reason}；执行摘要：{summaries}"

    @staticmethod
    def _build_final_answer(state: WorkflowState) -> str:
        summaries = [result.summary for result in state.results.values()]
        return "工作流执行成功：" + " → ".join(summaries)
