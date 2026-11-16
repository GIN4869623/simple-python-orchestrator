"""Intent and orchestration services with shared data and transport modules."""

__all__ = ["GraphOrchestrator", "Orchestrator"]


def __getattr__(name: str):
    # Starting the intent service must not import the orchestration graph.
    if name == "GraphOrchestrator":
        from .orchestration.graph import GraphOrchestrator
        return GraphOrchestrator
    if name == "Orchestrator":
        from .orchestration.serial import Orchestrator
        return Orchestrator
    raise AttributeError(name)
