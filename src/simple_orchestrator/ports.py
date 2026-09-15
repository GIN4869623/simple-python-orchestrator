"""Interfaces owned by the orchestrator."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .models import AgentResult, ReviewDecision, Task, TaskPlan, WorkflowState


class PlannerPort(Protocol):
    def create_plan(self, user_request: str) -> TaskPlan:
        """Turn one user request into an executable task plan."""

    def review(
        self,
        state: WorkflowState,
        current_task: Task,
        result: AgentResult,
    ) -> ReviewDecision:
        """Choose what the workflow should do after one task result."""


class KnowledgeAgentPort(Protocol):
    def invoke(self, request: Mapping[str, Any]) -> AgentResult:
        """Call the single external knowledge-agent interface."""


class DeviceManagerPort(Protocol):
    def invoke(self, request: Mapping[str, Any]) -> AgentResult:
        """Report network device information and select an available device."""


class UsrpAgentPort(Protocol):
    def execute(self, request: Mapping[str, Any]) -> AgentResult:
        """Execute one physical operation through the USRP agent."""
