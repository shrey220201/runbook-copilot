"""Data schemas for Runbook Copilot evaluation harness."""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class EvaluationTicket:
    ticket_id: str
    incident_description: str
    category: str
    expected_sources: List[str]
    expected_key_facts: List[str]
    ground_truth_answer: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationTicket":
        return cls(
            ticket_id=data["ticket_id"],
            incident_description=data["incident_description"],
            category=data["category"],
            expected_sources=data.get("expected_sources", []),
            expected_key_facts=data.get("expected_key_facts", []),
            ground_truth_answer=data["ground_truth_answer"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
