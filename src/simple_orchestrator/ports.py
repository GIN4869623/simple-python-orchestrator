"""Interfaces owned by the orchestrator."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .models import (
    AgentResult,
    BatchReviewDecision,
    IntentAnalysis,
    ReviewDecision,
    Task,
    TaskPlan,
    WorkflowState,
)


class PlannerPort(Protocol):
    """Legacy serial orchestrator contract, retained for comparison."""
    def create_plan(self, user_request: str) -> TaskPlan:
        """Turn one user request into an executable task plan."""

    def review(
        self,
        state: WorkflowState,
        current_task: Task,
        result: AgentResult,
    ) -> ReviewDecision:
        """Choose what the workflow should do after one task result."""


class GraphPlannerPort(Protocol):
    """Main graph contract: separate analysis/planning and batch review."""

    def analyze_intent(self, user_request: str) -> IntentAnalysis:
        """Classify one user request before planning."""

    def create_plan_from_intent(
        self, user_request: str, intent: IntentAnalysis
    ) -> TaskPlan:
        """Create business-level tasks from an already analyzed intent."""

    def review_batch(self, context: dict[str, Any]) -> BatchReviewDecision:
        """Make one decision after every task in a parallel batch has returned."""


class KnowledgeAgentPort(Protocol):
    def invoke(self, request: Mapping[str, Any]) -> AgentResult:
        """Call the single external knowledge-agent interface."""


class DeviceManagerPort(Protocol):
    def invoke(self, request: Mapping[str, Any]) -> AgentResult:
        """Report network device information and select an available device."""


class UsrpAgentPort(Protocol):
    def invoke(self, request: Mapping[str, Any]) -> AgentResult:
        """Answer a USRP-specific knowledge request without hardware effects."""

    def execute(self, request: Mapping[str, Any]) -> AgentResult:
        """Execute one physical operation through the USRP agent."""
