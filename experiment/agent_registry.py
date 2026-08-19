"""Agent registry: maps design-matrix agent labels to concrete classes + configs.

Update the AGENT_REGISTRY dict as agent implementations are finalized.
The runner uses this to instantiate the correct agent for a given design row.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentSpec:
    """Specification for instantiating an agent."""

    agent_module: str
    agent_class: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self) -> Any:
        """Import and instantiate the agent class with stored kwargs."""
        import importlib

        mod = importlib.import_module(self.agent_module)
        cls = getattr(mod, self.agent_class)
        return cls(**self.kwargs)


AGENT_REGISTRY: dict[str, AgentSpec] = {
    "agent_1": AgentSpec(
        agent_module="agents.naive_rag",
        agent_class="NaiveRAGAgent",
        kwargs={"store_backend": "chroma"},
    ),
    "agent_2": AgentSpec(
        agent_module="agents.naive_rag",
        agent_class="NaiveRAGAgent",
        kwargs={"store_backend": "mempalace"},
    ),
    "agent_3": AgentSpec(
        agent_module="agents.corrective_rag_v1",
        agent_class="CorrectiveRAGAgent",
        kwargs={"store_backend": "chroma"},
    ),
    "agent_4": AgentSpec(
        agent_module="agents.naive_rag",
        agent_class="NaiveRAGAgent",
        kwargs={"store_backend": "unified"},
    ),
    "agent_3_v2": AgentSpec(
        agent_module="agents.corrective_rag",
        agent_class="CorrectiveRAGAgent",
        kwargs={"store_backend": "chroma", "min_score": 0.25},
    ),
}


def available_agents() -> list[str]:
    return sorted(AGENT_REGISTRY.keys())


def get_agent_spec(agent_id: str) -> AgentSpec:
    if agent_id not in AGENT_REGISTRY:
        raise KeyError(
            f"Unknown agent {agent_id!r}. Available: {available_agents()}"
        )
    return AGENT_REGISTRY[agent_id]
