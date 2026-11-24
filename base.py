"""Base class: an agent = skills (typed handlers) + an A2A agent card. Agents are stateless
request handlers; durable state lives in the workflow, memory in the memory store."""
from __future__ import annotations

from typing import Any

from surgeshield.a2a import protocol as P
from surgeshield.config import get_settings


class BaseAgent:
    name: str = "agent"
    description: str = ""
    skills: list[P.AgentSkill] = []
    uses_a2ui: bool = False

    @property
    def default_skill(self) -> str:
        return self.skills[0].id

    def card(self) -> P.AgentCard:
        base = get_settings().api.public_base_url
        ext = [P.AgentExtension(uri=P.A2UI_EXTENSION_URI, description="Emits A2UI v0.8 surfaces")] if self.uses_a2ui else []
        return P.AgentCard(name=self.name, description=self.description, url=f"{base}/agents/{self.name}/",
                           provider=P.AgentProvider(organization="SurgeShield reference project",
                                                    url="https://github.com/your-org/surgeshield"),
                           capabilities=P.AgentCapabilities(extensions=ext), skills=self.skills)

    def handle(self, skill: str, payload: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(self, f"skill_{skill}", None)
        if fn is None:
            raise ValueError(f"unknown skill {skill}")
        return fn(**payload)
