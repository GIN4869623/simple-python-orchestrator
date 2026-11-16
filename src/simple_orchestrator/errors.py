"""Errors shared by intent recognition, planning and execution."""


class PlanningError(RuntimeError):
    """An intent, plan or model response cannot be used safely."""


class DuplicateTaskIdError(PlanningError):
    """A plan reuses a reserved or newly defined task ID."""
