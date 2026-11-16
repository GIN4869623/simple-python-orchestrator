"""Shared data models used by the whole orchestration workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    UNKNOWN = "unknown"
    SKIPPED = "skipped"


class ResultStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class WorkflowStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    UNKNOWN = "unknown"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNSUPPORTED = "unsupported"


class IntentType(str, Enum):
    KNOWLEDGE_QUERY = "knowledge_query"
    DEVICE_QUERY = "device_query"
    HARDWARE_EXECUTION = "hardware_execution"
    UNSUPPORTED = "unsupported"


class HandlingMode(str, Enum):
    DIRECT = "direct"
    FIXED_WORKFLOW = "fixed_workflow"
    ORCHESTRATED = "orchestrated"


class ExecutionMode(str, Enum):
    INFORMATIONAL = "informational"
    READ_ONLY = "read_only"
    PHYSICAL = "physical"


class PlanStatus(str, Enum):
    READY = "ready"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNSUPPORTED = "unsupported"


class ParameterPolicy(str, Enum):
    DEMO_DEFAULTS = "demo_defaults"
    STRICT = "strict"


class NextAction(str, Enum):
    CONTINUE = "continue"
    RETRY = "retry"
    REPLAN = "replan"
    COMPLETE = "complete"
    STOP = "stop"


@dataclass
class Task:
    task_id: str
    agent: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    capability: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    expected_output: list[str] = field(default_factory=list)
    completion_criteria: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IntentAnalysis:
    intent_type: IntentType
    domain: str
    target_type: str | None
    action: str
    execution_mode: ExecutionMode
    required_capabilities: list[str] = field(default_factory=list)
    provided_parameters: dict[str, Any] = field(default_factory=dict)
    missing_parameters: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    evidence: str = ""
    reason: str = ""
    # Legacy callers without a routing decision retain dynamic planning.
    handling_mode: HandlingMode = HandlingMode.ORCHESTRATED


@dataclass(frozen=True)
class TaskPlan:
    status: PlanStatus
    intent: IntentAnalysis
    tasks: list[Task] = field(default_factory=list)
    clarification_questions: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class ReviewDecision:
    task_id: str
    action: NextAction
    reason: str
    revised_tasks: list[Task] = field(default_factory=list)


@dataclass(frozen=True)
class BatchReviewDecision:
    action: NextAction
    reason: str
    task_ids: list[str] = field(default_factory=list)
    revised_tasks: list[Task] = field(default_factory=list)


@dataclass(frozen=True)
class AgentResult:
    status: ResultStatus
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class WorkflowState:
    user_request: str
    intent: IntentAnalysis | None = None
    tasks: list[Task] = field(default_factory=list)
    results: dict[str, AgentResult] = field(default_factory=dict)
    reviews: list[ReviewDecision] = field(default_factory=list)
    plan_version: int = 1
    retry_count: int = 0
    replan_count: int = 0
    max_retries: int = 1
    max_replans: int = 1
    status: WorkflowStatus = WorkflowStatus.RUNNING
    final_answer: str = ""
    error: str | None = None
