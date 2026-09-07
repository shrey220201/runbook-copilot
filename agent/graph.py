"""Agent Orchestrator graph with confidence-gated escalation."""

import re
import json
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from rag.retriever import RetrievedChunk
from rag.prompts import build_rag_prompt
from rag.llm import llm_client
from agent.tools import search_runbooks, check_system_status, escalate_to_human

# Deterministic safety gate keywords
DESTRUCTIVE_KEYWORDS = ["delete", "drop", "rm -rf", "format", "wipe", "destroy"]
CONFIDENCE_THRESHOLD = 0.50


class AgentState(TypedDict, total=False):
    query: str
    top_k: int
    source_filter: Optional[str]
    hostname: Optional[str]
    retrieved_chunks: List[RetrievedChunk]
    is_escalated: bool
    escalation_reason: Optional[str]
    system_status: Optional[Dict[str, Any]]
    response: Optional[str]
    escalation_result: Optional[Dict[str, Any]]


def check_destructive_action(query: str) -> Optional[str]:
    """Check query against predefined destructive action keywords."""
    query_lower = query.lower()
    for kw in DESTRUCTIVE_KEYWORDS:
        if kw == "rm -rf":
            if "rm -rf" in query_lower or "rm  -rf" in query_lower:
                return kw
        else:
            # Word boundary check for standard single words
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, query_lower):
                return kw
    return None


def classify_confidence_and_retrieve(state: AgentState) -> Dict[str, Any]:
    """Deterministic classification node: checks destructive keywords and retrieval confidence."""
    query = state.get("query", "")
    top_k = state.get("top_k", 4)
    source_filter = state.get("source_filter")

    # 1. Safety rule check: Destructive commands immediately escalate
    destructive_match = check_destructive_action(query)
    if destructive_match:
        return {
            "retrieved_chunks": [],
            "is_escalated": True,
            "escalation_reason": f"Destructive action keyword detected: '{destructive_match}'",
        }

    # 2. Retrieve candidate chunks using search_runbooks tool
    chunks = search_runbooks(
        query=query,
        top_k=top_k,
        source_filter=source_filter,
    )

    # 3. Confidence rule check: Low score or empty chunks trigger escalation
    if not chunks:
        return {
            "retrieved_chunks": [],
            "is_escalated": True,
            "escalation_reason": "No relevant runbook documentation found in knowledge base",
        }

    top_score = chunks[0].score
    if top_score < CONFIDENCE_THRESHOLD:
        return {
            "retrieved_chunks": chunks,
            "is_escalated": True,
            "escalation_reason": f"Low retrieval confidence (top similarity score {top_score:.4f} < {CONFIDENCE_THRESHOLD})",
        }

    return {
        "retrieved_chunks": chunks,
        "is_escalated": False,
        "escalation_reason": None,
    }


def route_decision(state: AgentState) -> str:
    """Conditional edge routing decision."""
    if state.get("is_escalated", False):
        return "escalate_node"
    return "generate_node"


def escalate_node(state: AgentState) -> Dict[str, Any]:
    """Escalation node: calls escalate_to_human tool and sets escalation response."""
    incident = state.get("query", "")
    reason = state.get("escalation_reason", "Manual escalation triggered")

    escalation_result = escalate_to_human(
        incident_description=incident,
        reason=reason,
    )

    return {
        "escalation_result": escalation_result,
        "response": escalation_result["message"],
    }


def generate_node(state: AgentState) -> Dict[str, Any]:
    """Generation node: deterministically gathers telemetry and invokes grounded LLM generation."""
    query = state.get("query", "")
    chunks = state.get("retrieved_chunks", [])
    hostname = state.get("hostname") or "cluster-node-01"

    # 1. Deterministic tool call: Fetch live system telemetry
    system_status = check_system_status(hostname=hostname)

    # 2. Append telemetry context to query without modifying rag/prompts.py signature
    telemetry_block = json.dumps(system_status, indent=2)
    augmented_query = (
        f"{query}\n\n"
        f"--- Live System Telemetry ({hostname}) ---\n"
        f"{telemetry_block}"
    )

    # 3. Build prompt and generate response
    messages = build_rag_prompt(query=augmented_query, chunks=chunks)
    response = llm_client.chat(messages=messages, temperature=0.2)

    return {
        "system_status": system_status,
        "response": response,
    }


# Construct StateGraph
workflow = StateGraph(AgentState)

workflow.add_node("classify_confidence_and_retrieve", classify_confidence_and_retrieve)
workflow.add_node("generate_node", generate_node)
workflow.add_node("escalate_node", escalate_node)

workflow.add_edge(START, "classify_confidence_and_retrieve")
workflow.add_conditional_edges(
    "classify_confidence_and_retrieve",
    route_decision,
    {
        "generate_node": "generate_node",
        "escalate_node": "escalate_node",
    },
)
workflow.add_edge("generate_node", END)
workflow.add_edge("escalate_node", END)

app = workflow.compile()


def run_agent_workflow(
    query: str,
    top_k: int = 4,
    source_filter: Optional[str] = None,
    hostname: Optional[str] = None,
) -> Dict[str, Any]:
    """Helper function to execute the compiled LangGraph workflow."""
    initial_state: AgentState = {
        "query": query,
        "top_k": top_k,
        "source_filter": source_filter,
        "hostname": hostname,
    }
    return app.invoke(initial_state)
