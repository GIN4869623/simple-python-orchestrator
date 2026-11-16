"""Deterministic business validation for LLM-generated plans."""

from __future__ import annotations

from collections.abc import Sequence

from .agent_registry import AgentProfile, AgentRegistry
from .models import ExecutionMode, IntentAnalysis, IntentType, Task


class PlanValidationError(ValueError):
    """Raised when a syntactically valid plan violates agent boundaries."""


class PlanValidator:
    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry

    def validate(
        self,
        intent: IntentAnalysis,
        tasks: Sequence[Task],
        candidates: Sequence[AgentProfile],
        *,
        prior_tasks: Sequence[Task] = (),
    ) -> None:
        if intent.needs_clarification:
            raise PlanValidationError("意图仍缺少必要参数，不能生成可执行计划")
        if not tasks:
            raise PlanValidationError("可执行计划至少需要一个任务")

        candidate_ids = {profile.agent_id for profile in candidates}
        all_tasks = [*prior_tasks, *tasks]
        tasks_by_id = {task.task_id: task for task in all_tasks}

        for task in tasks:
            if task.agent not in candidate_ids:
                raise PlanValidationError(
                    f"任务 {task.task_id} 使用了意图匹配结果之外的智能体：{task.agent}"
                )
            profile = self.registry.get(task.agent)
            if task.capability not in profile.capabilities:
                raise PlanValidationError(
                    f"智能体 {task.agent} 不支持能力：{task.capability}"
                )

            if task.agent == "usrp":
                self._validate_usrp_task(intent, task, tasks_by_id)

        planned_capabilities = {task.capability for task in all_tasks}
        missing_capabilities = (
            set(intent.required_capabilities) - planned_capabilities
        )
        if missing_capabilities:
            missing = "、".join(sorted(missing_capabilities))
            raise PlanValidationError(f"计划没有覆盖意图所需能力：{missing}")

    @staticmethod
    def _validate_usrp_task(
        intent: IntentAnalysis,
        task: Task,
        tasks_by_id: dict[str, Task],
    ) -> None:
        if task.capability == "knowledge.explain":
            if (
                intent.intent_type is not IntentType.KNOWLEDGE_QUERY
                or intent.target_type != "usrp"
                or intent.execution_mode is not ExecutionMode.INFORMATIONAL
            ):
                raise PlanValidationError("USRP 知识查询只允许用于只读 USRP 知识意图")
            return
        if intent.intent_type is not IntentType.HARDWARE_EXECUTION:
            raise PlanValidationError("只有真实硬件执行意图可以启用 USRP 智能体")
        if intent.target_type != "usrp":
            raise PlanValidationError("只有目标设备为 USRP 时可以启用 USRP 智能体")
        if intent.execution_mode is not ExecutionMode.PHYSICAL:
            raise PlanValidationError("软件仿真或知识请求不能启用 USRP 智能体")
        if not intent.evidence.strip():
            raise PlanValidationError("启用 USRP 智能体时必须保留用户意图证据")
        if intent.provided_parameters.get("parameter_policy") == "use_backend_defaults":
            expected_unresolved = intent.provided_parameters.get(
                "unresolved_parameters", []
            )
            if task.inputs.get("parameter_policy") != "use_backend_defaults":
                raise PlanValidationError(
                    f"USRP 任务 {task.task_id} 必须保留 Fake 默认参数策略"
                )
            if task.inputs.get("unresolved_parameters") != expected_unresolved:
                raise PlanValidationError(
                    f"USRP 任务 {task.task_id} 必须原样传递未解析参数"
                )

        pending = list(task.dependencies)
        visited: set[str] = set()
        while pending:
            dependency_id = pending.pop()
            if dependency_id in visited:
                continue
            visited.add(dependency_id)
            dependency = tasks_by_id.get(dependency_id)
            if dependency is None:
                continue
            if (
                dependency.agent == "device_manager"
                and dependency.capability in {"device.list", "device.inspect"}
            ):
                expected_reference = f"{dependency.task_id}.network_devices"
                device_reference = task.inputs.get("device_candidates_from")
                if device_reference != expected_reference:
                    raise PlanValidationError(
                        f"USRP 任务 {task.task_id} 必须引用 "
                        f"{expected_reference}"
                    )
                return
            pending.extend(dependency.dependencies)

        raise PlanValidationError(
            f"USRP 任务 {task.task_id} 必须依赖设备查询任务"
        )
