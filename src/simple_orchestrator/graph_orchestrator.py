"""LangGraph main graph: dynamic task fan-out, batch review and safe routing.

The three agent nodes are transport adapters. They call today's local Ports; an
HTTP/JSON-RPC adapter can replace a Port without changing this graph's State.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Annotated, Any, Mapping, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .models import (
    AgentResult,
    BatchReviewDecision,
    IntentAnalysis,
    IntentType,
    NextAction,
    PlanStatus,
    ResultStatus,
    Task,
    TaskStatus,
    WorkflowStatus,
)
from .planner import PlanningError
from .ports import (
    DeviceManagerPort,
    GraphPlannerPort,
    KnowledgeAgentPort,
    PlannerPort,
    UsrpAgentPort,
)
from .remote import JsonRpcAgentAdapter


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    agent: str
    description: str
    dependencies: tuple[str, ...] = ()
    capability: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    expected_output: tuple[str, ...] = ()
    completion_criteria: tuple[str, ...] = ()

    @classmethod
    def from_task(cls, task: Task) -> TaskSpec:
        return cls(
            task_id=task.task_id,
            agent=task.agent,
            description=task.description,
            dependencies=tuple(task.dependencies),
            capability=task.capability,
            inputs=dict(task.inputs),
            expected_output=tuple(task.expected_output),
            completion_criteria=tuple(task.completion_criteria),
        )


@dataclass(frozen=True)
class PlanSnapshot:
    version: int
    tasks: tuple[TaskSpec, ...]


@dataclass(frozen=True)
class AttemptResult:
    request_id: str
    task_id: str
    agent_id: str
    attempt: int
    attempt_id: str
    plan_version: int
    result: AgentResult

    @property
    def error_type(self) -> str | None:
        return self.result.error

    @property
    def error_message(self) -> str | None:
        return self.result.summary if self.result.status is not ResultStatus.SUCCEEDED else None


def merge_results(
    left: dict[str, list[AttemptResult]], right: dict[str, list[AttemptResult]]
) -> dict[str, list[AttemptResult]]:
    merged = {task_id: list(attempts) for task_id, attempts in left.items()}
    for task_id, additions in right.items():
        existing = merged.setdefault(task_id, [])
        seen = {attempt.attempt_id for attempt in existing}
        for attempt in additions:
            if attempt.task_id != task_id or attempt.attempt_id in seen:
                raise ValueError(f"重复或不匹配的任务尝试：{task_id}/{attempt.attempt_id}")
            existing.append(attempt)
            seen.add(attempt.attempt_id)
    return merged


def merge_statuses(
    left: dict[str, TaskStatus], right: dict[str, TaskStatus]
) -> dict[str, TaskStatus]:
    return {**left, **right}


def append_reviews(
    left: list[BatchReviewDecision], right: list[BatchReviewDecision]
) -> list[BatchReviewDecision]:
    return [*left, *right]


class MainState(TypedDict, total=False):
    request_id: str
    user_request: str
    intent: IntentAnalysis
    precomputed_plan: Any
    plan_status: PlanStatus
    plan_history: list[PlanSnapshot]
    active_plan_version: int
    task_statuses: Annotated[dict[str, TaskStatus], merge_statuses]
    results: Annotated[dict[str, list[AttemptResult]], merge_results]
    reviews: Annotated[list[BatchReviewDecision], append_reviews]
    batch_ids: list[str]
    batch_decision: BatchReviewDecision
    replan_count: int
    status: WorkflowStatus
    final_answer: str
    error: str | None


class BranchState(TypedDict):
    request_id: str
    user_request: str
    intent: IntentAnalysis
    task: TaskSpec
    attempt: int
    plan_version: int
    dependency_results: dict[str, AgentResult]


class GraphOrchestrator:
    """Build and run one stateless, synchronous main graph per request."""

    def __init__(
        self,
        planner: GraphPlannerPort | PlannerPort,
        knowledge: KnowledgeAgentPort,
        device_manager: DeviceManagerPort,
        usrp: UsrpAgentPort,
        max_retries: int = 1,
        max_replans: int = 1,
        allow_remote_physical_execution: bool = False,
    ) -> None:
        if max_retries not in (0, 1) or max_replans not in (0, 1):
            raise ValueError("重试和重规划次数只能设置为 0 或 1")
        self.planner = planner
        self.knowledge = knowledge
        self.device_manager = device_manager
        self.usrp = usrp
        self.max_retries = max_retries
        self.max_replans = max_replans
        self.allow_remote_physical_execution = allow_remote_physical_execution
        # The Fake models the USRP service fetching the Skill itself. The main
        # graph never reads Markdown content or passes it to the hardware node.
        if getattr(usrp, "skill_reader", object()) is None and hasattr(knowledge, "read_skill"):
            usrp.skill_reader = knowledge
        self.graph = self._build_graph()

    def run(self, user_request: str, *, request_id: str | None = None) -> MainState:
        initial: MainState = {
            "request_id": request_id or f"req-{uuid4().hex}",
            "user_request": user_request,
            "plan_history": [],
            "active_plan_version": 1,
            "task_statuses": {},
            "results": {},
            "reviews": [],
            "batch_ids": [],
            "replan_count": 0,
            "status": WorkflowStatus.RUNNING,
            "final_answer": "",
            "error": None,
        }
        return self.graph.invoke(initial, {"recursion_limit": 100})

    def _build_graph(self):
        graph = StateGraph(MainState)
        graph.add_node("analyze_intent", self._analyze_intent)
        graph.add_node("create_plan", self._create_plan)
        graph.add_node("dispatch", self._dispatch)
        graph.add_node("knowledge", self._knowledge)
        graph.add_node("device_manager", self._device_manager)
        graph.add_node("usrp", self._usrp)
        graph.add_node("review_batch", self._review_batch)
        graph.add_node("apply_review", self._apply_review)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "analyze_intent")
        graph.add_edge("analyze_intent", "create_plan")
        graph.add_conditional_edges("create_plan", self._route_plan)
        graph.add_conditional_edges("dispatch", self._route_dispatch)
        for agent in ("knowledge", "device_manager", "usrp"):
            graph.add_edge(agent, "review_batch")
        graph.add_edge("review_batch", "apply_review")
        graph.add_conditional_edges("apply_review", self._route_after_review)
        graph.add_edge("finalize", END)
        return graph.compile()

    def _analyze_intent(self, state: MainState) -> dict[str, Any]:
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
        tasks = self._prepare_tasks(plan.tasks, plan.intent)
        if any(self._is_physical_usrp(task) for task in tasks) and "usrp" not in state["user_request"].lower():
            raise PlanningError("用户未明确指定 USRP，不允许硬件执行")
        self._validate_tasks(tasks)
        updates["plan_history"] = [PlanSnapshot(1, tuple(tasks))]
        updates["task_statuses"] = {task.task_id: TaskStatus.PENDING for task in tasks}
        return updates

    @staticmethod
    def _fresh_id(prefix: str, used: set[str]) -> str:
        candidate = prefix
        index = 2
        while candidate in used:
            candidate = f"{prefix}_{index}"
            index += 1
        used.add(candidate)
        return candidate

    def _prepare_tasks(
        self, tasks: list[Task], intent: IntentAnalysis,
        *, reserved_ids: set[str] | None = None,
        prior_success_tasks: list[TaskSpec] | None = None,
    ) -> list[TaskSpec]:
        """Enforce the physical preflight even if an LLM omits it."""
        specs = [TaskSpec.from_task(task) for task in tasks]
        if intent.intent_type is IntentType.DEVICE_QUERY and (
            len(specs) != 1
            or specs[0].agent != "device_manager"
            or specs[0].capability not in {"device.list", "device.inspect"}
        ):
            raise PlanningError("设备查询只能由 DeviceManager 独立执行")
        for task in specs:
            if task.agent != "usrp":
                continue
            if task.capability == "knowledge.explain":
                if intent.intent_type is not IntentType.KNOWLEDGE_QUERY or intent.target_type != "usrp":
                    raise PlanningError("USRP 知识查询只能用于 USRP 知识意图")
            elif intent.intent_type is not IntentType.HARDWARE_EXECUTION or intent.target_type != "usrp":
                raise PlanningError("非 USRP 硬件意图不得调用 USRP 物理能力")
        if intent.intent_type is not IntentType.HARDWARE_EXECUTION or intent.target_type != "usrp":
            return specs
        if not any(self._is_physical_usrp(task) for task in specs):
            raise PlanningError("USRP 硬件请求的计划缺少 USRP 任务")
        used = {task.task_id for task in specs} | set(reserved_ids or ())
        reusable = [*specs, *(prior_success_tasks or [])]
        prepared: list[TaskSpec] = []
        for task in specs:
            if not self._is_physical_usrp(task):
                prepared.append(task)
                continue
            skill_tasks = [
                candidate for candidate in reusable
                if candidate.agent == "knowledge"
                and candidate.capability == "knowledge.lookup_skill"
                and candidate.inputs.get("operation") == intent.action
            ]
            device_tasks = [
                candidate for candidate in reusable
                if candidate.agent == "device_manager"
                and candidate.capability == "device.select"
                and candidate.inputs.get("required_capability") == task.capability
            ]
            skill_id = skill_tasks[0].task_id if skill_tasks else self._fresh_id(f"{task.task_id}_skill", used)
            device_id = device_tasks[0].task_id if device_tasks else self._fresh_id(f"{task.task_id}_device", used)
            if not skill_tasks:
                skill = TaskSpec(
                        skill_id, "knowledge", "检索可信 USRP 操作 Skill",
                        capability="knowledge.lookup_skill",
                        inputs={"target_agent_id": "usrp", "operation": intent.action},
                        expected_output=("skill_id", "version", "sha256"),
                        completion_criteria=("返回可信 Skill 索引",),
                    )
                prepared.append(skill)
                reusable.append(skill)
            if not device_tasks:
                device = TaskSpec(
                        device_id, "device_manager", "选择可用 USRP 设备",
                        capability="device.select",
                        inputs={"target_type": "usrp", "required_capability": task.capability},
                        expected_output=("device_id", "selected_device"),
                        completion_criteria=("选中在线且能力匹配的 USRP",),
                    )
                prepared.append(device)
                reusable.append(device)
            prepared.append(
                TaskSpec(
                    task.task_id, task.agent, task.description,
                    dependencies=tuple(dict.fromkeys((*task.dependencies, skill_id, device_id))),
                    capability=task.capability, inputs=task.inputs,
                    expected_output=task.expected_output,
                    completion_criteria=task.completion_criteria,
                )
            )
        return prepared

    @staticmethod
    def _validate_tasks(tasks: list[TaskSpec], prior_success: set[str] | None = None) -> None:
        if not tasks:
            raise PlanningError("计划不能为空")
        known = set(prior_success or ())
        for task in tasks:
            if task.agent not in {"knowledge", "device_manager", "usrp"}:
                raise PlanningError(f"未知执行目标：{task.agent}")
            if task.task_id in known:
                raise PlanningError(f"任务 ID 重复：{task.task_id}")
            missing = set(task.dependencies) - known
            if missing:
                raise PlanningError(f"任务 {task.task_id} 依赖尚未出现的任务：{sorted(missing)}")
            known.add(task.task_id)

    @staticmethod
    def _route_plan(state: MainState) -> str:
        return "dispatch" if state["plan_status"] is PlanStatus.READY else "finalize"

    @staticmethod
    def _active_tasks(state: MainState) -> tuple[TaskSpec, ...]:
        return state["plan_history"][-1].tasks

    @staticmethod
    def _is_physical_usrp(task: TaskSpec) -> bool:
        return task.agent == "usrp" and task.capability != "knowledge.explain"

    def _dispatch(self, state: MainState) -> dict[str, Any]:
        statuses = state["task_statuses"]
        pending = [task for task in self._active_tasks(state) if statuses[task.task_id] is TaskStatus.PENDING]
        ready = [
            task for task in pending
            if all(statuses.get(dep) is TaskStatus.SUCCEEDED for dep in task.dependencies)
        ]
        # Physical tasks only run after the read-only preflight has completed.
        if any(self._is_physical_usrp(task) for task in ready):
            ready = [task for task in ready if self._is_physical_usrp(task)][:1]
        if len({task.task_id for task in ready}) != len(ready):
            raise PlanningError("同一批次重复分发任务 ID")
        if ready:
            return {
                "batch_ids": [task.task_id for task in ready],
                "task_statuses": {task.task_id: TaskStatus.RUNNING for task in ready},
            }
        if pending:
            blocked = {task.task_id: TaskStatus.SKIPPED for task in pending}
            return {
                "batch_ids": [], "task_statuses": blocked,
                "status": WorkflowStatus.FAILED,
                "error": "任务依赖未成功，无法继续执行",
                "final_answer": "工作流已停止：任务依赖未成功",
            }
        return {"batch_ids": [], "status": WorkflowStatus.SUCCEEDED}

    def _route_dispatch(self, state: MainState) -> list[Send] | str:
        if not state["batch_ids"]:
            return "finalize"
        task_by_id = {task.task_id: task for task in self._active_tasks(state)}
        sends: list[Send] = []
        for task_id in state["batch_ids"]:
            task = task_by_id[task_id]
            dependency_results = {
                dep: state["results"][dep][-1].result for dep in task.dependencies
            }
            branch: BranchState = {
                "request_id": state["request_id"],
                "user_request": state["user_request"],
                "intent": state["intent"],
                "task": task,
                "attempt": len(state["results"].get(task_id, [])) + 1,
                "plan_version": state["active_plan_version"],
                "dependency_results": dependency_results,
            }
            sends.append(Send(task.agent, branch))
        return sends

    @staticmethod
    def _request_context(branch: BranchState) -> dict[str, Any]:
        task = branch["task"]
        return {
            "request_id": branch["request_id"],
            "user_request": branch["user_request"],
            "current_task": {
                "task_id": task.task_id, "agent": task.agent,
                "description": task.description, "dependencies": list(task.dependencies),
                "attempt": branch["attempt"], "capability": task.capability,
                "inputs": task.inputs, "expected_output": list(task.expected_output),
                "completion_criteria": list(task.completion_criteria),
            },
            "previous_results": {
                task_id: result.data for task_id, result in branch["dependency_results"].items()
            },
        }

    def _knowledge(self, branch: BranchState) -> dict[str, Any]:
        return self._invoke_branch(branch, self.knowledge.invoke)

    def _device_manager(self, branch: BranchState) -> dict[str, Any]:
        return self._invoke_branch(branch, self.device_manager.invoke)

    def _usrp(self, branch: BranchState) -> dict[str, Any]:
        if branch["task"].capability == "knowledge.explain":
            return self._invoke_branch(branch, self.usrp.invoke)
        if isinstance(self.usrp, JsonRpcAgentAdapter) and not self.allow_remote_physical_execution:
            return self._branch_update(
                branch,
                AgentResult(
                    ResultStatus.REJECTED,
                    "远程 USRP 物理执行尚未启用；需要先实现审批、幂等和状态查询",
                    error="remote_physical_execution_disabled",
                ),
            )
        context = self._request_context(branch)
        skills = [
            result.data for result in branch["dependency_results"].values()
            if "skill_id" in result.data
        ]
        devices = [
            result.data for result in branch["dependency_results"].values()
            if "device_id" in result.data
        ]
        if not skills or not devices:
            result = AgentResult(ResultStatus.FAILED, "USRP 未执行：缺少 Skill 或设备", error="missing_preflight")
            return self._branch_update(branch, result)
        skill_ref = skills[0]
        selected = devices[0]
        skill_valid = all(
            isinstance(skill_ref.get(key), str) and skill_ref[key]
            for key in ("skill_id", "version", "sha256")
        )
        device_valid = (
            selected.get("selection") == "selected"
            and isinstance(selected.get("device_id"), str)
            and bool(selected["device_id"])
        )
        device_details = selected.get("selected_device")
        device_valid = (
            device_valid
            and isinstance(device_details, Mapping)
            and device_details.get("device_id") == selected.get("device_id")
            and device_details.get("status") == "online"
            and branch["task"].capability in device_details.get("capabilities", [])
        )
        if not skill_valid or not device_valid:
            result = AgentResult(
                ResultStatus.FAILED,
                "USRP 未执行：Skill 索引或设备能力校验失败",
                error="invalid_preflight",
            )
            return self._branch_update(branch, result)
        context["skill_ref"] = skill_ref
        context["selected_device"] = selected
        return self._invoke_branch(branch, self.usrp.execute, context)

    def _invoke_branch(self, branch: BranchState, invoke, context: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            result = invoke(context or self._request_context(branch))
        except Exception as exc:
            # A remote USRP may have acted before its response was lost.
            status = (
                ResultStatus.UNKNOWN
                if self._is_physical_usrp(branch["task"])
                else ResultStatus.FAILED
            )
            result = AgentResult(status, f"智能体调用异常：{exc}", error=type(exc).__name__)
        return self._branch_update(branch, result)

    @staticmethod
    def _branch_update(branch: BranchState, result: AgentResult) -> dict[str, Any]:
        status = TaskStatus(result.status.value)
        task = branch["task"]
        attempt = AttemptResult(
            branch["request_id"], task.task_id, task.agent, branch["attempt"],
            f"{branch['request_id']}:{task.task_id}:{branch['attempt']}",
            branch["plan_version"], result,
        )
        return {
            "results": {task.task_id: [attempt]},
            "task_statuses": {task.task_id: status},
        }

    def _review_batch(self, state: MainState) -> dict[str, Any]:
        batch = state["batch_ids"]
        statuses = state["task_statuses"]
        failed = [task_id for task_id in batch if statuses[task_id] is not TaskStatus.SUCCEEDED]
        task_by_id = {task.task_id: task for task in self._active_tasks(state)}
        # Hard gate: an explicit no-device result cannot be overturned by an LLM.
        no_device = any(
            task_by_id[task_id].agent == "device_manager"
            and (
                state["results"][task_id][-1].result.error
                in {"no device available", "no_device_available"}
                or state["results"][task_id][-1].result.data.get("selection") == "unavailable"
            )
            for task_id in failed
        )
        rejected = any(statuses[task_id] is TaskStatus.REJECTED for task_id in failed)
        invalid_skill = any(
            state["results"][task_id][-1].result.error
            in {"skill_validation_failed", "invalid_preflight", "missing_preflight"}
            for task_id in failed
        )
        physical_unknown = any(
            self._is_physical_usrp(task_by_id[task_id])
            and statuses[task_id] is TaskStatus.UNKNOWN
            for task_id in failed
        )
        if no_device or rejected or invalid_skill or physical_unknown:
            reason = (
                "没有可用设备" if no_device
                else "任务被明确拒绝" if rejected
                else "可信 Skill 或设备前置校验失败" if invalid_skill
                else "USRP 结果未知，禁止盲目重试"
            )
            decision = BatchReviewDecision(NextAction.STOP, reason)
        else:
            retryable = [
                task_id for task_id in failed
                if statuses[task_id] is TaskStatus.FAILED
                and len(state["results"][task_id]) <= self.max_retries
            ]
            pending = any(
                statuses[task.task_id] is TaskStatus.PENDING
                for task in self._active_tasks(state)
            )
            context = {
                "user_request": state["user_request"],
                "intent_analysis": {
                    "intent_type": state["intent"].intent_type.value,
                    "domain": state["intent"].domain,
                    "target_type": state["intent"].target_type,
                    "action": state["intent"].action,
                    "execution_mode": state["intent"].execution_mode.value,
                    "required_capabilities": state["intent"].required_capabilities,
                    "provided_parameters": state["intent"].provided_parameters,
                    "missing_parameters": state["intent"].missing_parameters,
                    "needs_clarification": state["intent"].needs_clarification,
                    "evidence": state["intent"].evidence,
                    "reason": state["intent"].reason,
                },
                "plan_version": state["active_plan_version"],
                "current_plan": [asdict(task) for task in self._active_tasks(state)],
                "batch_task_ids": batch,
                "batch_results": {
                    task_id: {
                        "status": state["results"][task_id][-1].result.status.value,
                        "summary": state["results"][task_id][-1].result.summary,
                        "data": state["results"][task_id][-1].result.data,
                        "error": state["results"][task_id][-1].result.error,
                    } for task_id in batch
                },
                "failed_task_ids": failed,
                "retryable_task_ids": retryable,
                "has_pending_tasks": pending,
                "can_replan": state["replan_count"] < self.max_replans,
                "known_task_ids": list(state["task_statuses"]),
                "successful_task_ids": [
                    task_id for task_id, status in statuses.items()
                    if status is TaskStatus.SUCCEEDED
                ],
                "successful_results": {
                    task_id: state["results"][task_id][-1].result.data
                    for task_id, status in statuses.items()
                    if status is TaskStatus.SUCCEEDED and task_id in state["results"]
                },
            }
            review_batch = getattr(self.planner, "review_batch", None)
            if review_batch is not None:
                decision = review_batch(context)
            elif retryable:
                decision = BatchReviewDecision(NextAction.RETRY, "重试失败任务", retryable)
            elif failed:
                decision = BatchReviewDecision(NextAction.STOP, "任务失败且无重试机会")
            else:
                decision = BatchReviewDecision(
                    NextAction.CONTINUE if pending else NextAction.COMPLETE,
                    "本批任务已完成",
                )
        return {"batch_decision": decision, "reviews": [decision]}

    def _apply_review(self, state: MainState) -> dict[str, Any]:
        decision = state["batch_decision"]
        action = decision.action
        failed = [
            task_id for task_id in state["batch_ids"]
            if state["task_statuses"][task_id] is not TaskStatus.SUCCEEDED
        ]
        if action is NextAction.RETRY:
            if not decision.task_ids or not set(decision.task_ids).issubset(failed):
                return self._stop(state, "无效的重试任务集合")
            allowed = all(
                state["task_statuses"][task_id] is TaskStatus.FAILED
                and len(state["results"][task_id]) <= self.max_retries
                for task_id in decision.task_ids
            )
            if not allowed or set(decision.task_ids) != set(failed):
                return self._stop(state, "重试必须仅包含本批全部失败且可重试的任务")
            return {"task_statuses": {task_id: TaskStatus.PENDING for task_id in decision.task_ids}}
        if action is NextAction.REPLAN:
            if state["replan_count"] >= self.max_replans or not decision.revised_tasks:
                return self._stop(state, "无法继续重规划")
            try:
                return self._apply_replan(state, decision.revised_tasks)
            except (ValueError, PlanningError) as exc:
                return self._stop(state, f"重规划无效：{exc}")
        if action is NextAction.COMPLETE:
            if failed or any(
                state["task_statuses"][task.task_id] is TaskStatus.PENDING
                for task in self._active_tasks(state)
            ):
                return self._stop(state, "计划尚未完成，不能宣布成功")
            return {"status": WorkflowStatus.SUCCEEDED}
        if action is NextAction.CONTINUE:
            if failed:
                return self._stop(state, "任务未成功，不能继续执行下游")
            return {}
        return self._stop(state, decision.reason)

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

    @staticmethod
    def _stop(state: MainState, reason: str) -> dict[str, Any]:
        failed = [
            state["results"][task_id][-1]
            for task_id in state["batch_ids"] if task_id in state["results"]
            and state["results"][task_id][-1].result.status is not ResultStatus.SUCCEEDED
        ]
        attempt = failed[0] if failed else None
        result = attempt.result if attempt else None
        status = (
            WorkflowStatus.REJECTED if result and result.status is ResultStatus.REJECTED
            else WorkflowStatus.UNKNOWN if result and result.status is ResultStatus.UNKNOWN
            else WorkflowStatus.FAILED
        )
        detail = (
            f"{reason}；任务 {attempt.task_id}（{attempt.agent_id}）"
            f"第 {attempt.attempt} 次尝试失败，"
            f"错误类型：{attempt.error_type or result.status.value}；"
            f"{result.summary}；request_id={attempt.request_id}"
            if attempt and result else reason
        )
        skipped = {
            task.task_id: TaskStatus.SKIPPED
            for task in state["plan_history"][-1].tasks
            if state["task_statuses"][task.task_id] is TaskStatus.PENDING
        }
        return {
            "status": status,
            "error": detail,
            "final_answer": f"工作流已停止：{detail}",
            "task_statuses": skipped,
        }

    @staticmethod
    def _route_after_review(state: MainState) -> str:
        return "dispatch" if state["status"] is WorkflowStatus.RUNNING else "finalize"

    @staticmethod
    def _finalize(state: MainState) -> dict[str, Any]:
        if state.get("final_answer"):
            return {}
        if state["status"] is WorkflowStatus.SUCCEEDED:
            active_ids = [task.task_id for task in state["plan_history"][-1].tasks]
            summaries = [
                state["results"][task_id][-1].result.summary
                for task_id in active_ids if task_id in state["results"]
            ]
            return {"final_answer": "工作流执行成功：" + " → ".join(summaries)}
        return {"final_answer": "工作流已停止：" + (state.get("error") or "未知原因")}
