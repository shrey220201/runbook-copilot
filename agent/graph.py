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
# Stems, not dictionary forms, for words with a silent trailing "e" - matching
# "delete" as a literal substring misses "deleting" (the "e" drops before
# "-ing": delete -> deleting, not delete+ing). Using the stem ("delet",
# "wip", "purg", "eras") catches every inflected form in one regex instead
# of enumerating each tense separately.
DESTRUCTIVE_KEYWORDS = ["delet", "drop", "rm -rf", "format", "wip", "destroy", "purg", "eras"]

# Multi-word destructive phrases, checked as substrings rather than word-boundary
# regex, since phrases like "zero out" do not tokenize cleanly as a single word.
DESTRUCTIVE_PHRASES = ["zero out", "zero the drives", "factory reset"]

# Prompt-injection patterns: phrases attempting to override system instructions
# or claim unearned authority to bypass safety gating. Pattern-based and not
# exhaustive - a broader, actively-maintained list or an LLM-based intent
# classifier would be the production-grade version (see README Limitations).
INJECTION_PATTERNS = [
    r"ignore (your|the) (system prompt|previous instructions|safety guidelines)",
    r"unrestricted mode",
    r"i'?m the (system )?administrator",
    r"i am the (system )?administrator",
    r"skip (any|the) warnings",
    r"full authorization",
    r"disregard (your|the) (instructions|guidelines|rules)",
]

# Domain vocabulary used for the vague-query heuristic: a short query with no
# recognizable technical term lacks enough context to safely answer, so it
# routes to a human instead of the agent guessing at intent.
DOMAIN_TERMS = [
    "active directory", "domain controller", " ad ", "group policy", "gpo",
    "dns", "dhcp", "hyper-v", "hyperv", "vm", "virtual machine",
    "proxmox", "pve", "lxc", "container", "nakivo", "backup", "replication",
    "repository", "transporter", "qnap", "nas", "raid", "volume", "snapshot",
    "storage", "disk", "ssd", "hdd", "cisco", "router", "switch", "vlan",
    "firewall", "network", "cluster", "quorum", "corosync", "server",
    "database", "service", "certificate", "authentication", "cifs", "smb",
    "nfs", "ntfs", "acl", "permission",
]
VAGUE_WORD_COUNT_THRESHOLD = 10

CONFIDENCE_THRESHOLD = 0.50


class AgentState(TypedDict, total=False):
    query: str
    top_k: int
    source_filter: Optional[str]
    hostname: Optional[str]
    conversation_history: List[Dict[str, str]]
    retrieved_chunks: List[RetrievedChunk]
    is_escalated: bool
    escalation_reason: Optional[str]
    system_status: Optional[Dict[str, Any]]
    response: Optional[str]
    escalation_result: Optional[Dict[str, Any]]


def check_prompt_injection(query: str) -> Optional[str]:
    """Check query against known prompt-injection / authority-override patterns."""
    query_lower = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            return pattern
    return None


def check_destructive_action(query: str) -> Optional[str]:
    """Check query against destructive action keywords and multi-word phrases."""
    query_lower = query.lower()

    for phrase in DESTRUCTIVE_PHRASES:
        if phrase in query_lower:
            return phrase

    for kw in DESTRUCTIVE_KEYWORDS:
        if kw == "rm -rf":
            if "rm -rf" in query_lower or "rm  -rf" in query_lower:
                return kw
        else:
            # Prefix match on word boundary, not exact-word match, so
            # inflected forms (erasing, deleted, wiping, destroyed) are
            # still caught, not just the dictionary form of the verb.
            pattern = rf"\b{re.escape(kw)}\w*"
            if re.search(pattern, query_lower):
                return kw
    return None


def check_vague_query(query: str, history: Optional[List[Dict[str, str]]] = None) -> bool:
    """Flag queries that are both short and lack any recognizable domain term.

    A short follow-up like "what about the second option?" is not actually
    vague if the conversation already established a domain (e.g. a prior
    turn discussed a NAKIVO transporter error) - so recent history is
    checked for domain terms too, not just the current message in isolation.
    This does NOT apply to the injection or destructive-keyword checks,
    which always evaluate the current message alone.
    """
    query_lower = f" {query.lower()} "
    word_count = len(query.split())
    has_domain_term = any(term in query_lower for term in DOMAIN_TERMS)

    if has_domain_term or word_count >= VAGUE_WORD_COUNT_THRESHOLD:
        return False

    if history:
        recent_text = " ".join(turn.get("content", "") for turn in history[-4:]).lower()
        recent_text = f" {recent_text} "
        if any(term in recent_text for term in DOMAIN_TERMS):
            return False

    return True


def classify_confidence_and_retrieve(state: AgentState) -> Dict[str, Any]:
    """Deterministic classification node: checks injection patterns, destructive
    keywords, query vagueness, and retrieval confidence, in that order."""
    query = state.get("query", "")
    top_k = state.get("top_k", 4)
    source_filter = state.get("source_filter")
    history = state.get("conversation_history", []) or []

    # 1. Prompt injection check - highest priority, before any other logic
    # (always checked against the current message alone - history never
    # softens this or the destructive-keyword check below)
    injection_match = check_prompt_injection(query)
    if injection_match:
        return {
            "retrieved_chunks": [],
            "is_escalated": True,
            "escalation_reason": "Possible prompt injection or authority-override attempt detected",
        }

    # 2. Safety rule check: Destructive commands immediately escalate
    destructive_match = check_destructive_action(query)
    if destructive_match:
        return {
            "retrieved_chunks": [],
            "is_escalated": True,
            "escalation_reason": f"Destructive action keyword detected: '{destructive_match}'",
        }

    # 3. Vagueness check: too little context to safely answer, considering
    # recent conversation history for follow-up questions
    if check_vague_query(query, history=history):
        return {
            "retrieved_chunks": [],
            "is_escalated": True,
            "escalation_reason": "Query too vague or ambiguous - insufficient technical context to safely answer",
        }

    # 4. Retrieve candidate chunks using search_runbooks tool
    chunks = search_runbooks(
        query=query,
        top_k=top_k,
        source_filter=source_filter,
    )

    # 5. Confidence rule check: Low score or empty chunks trigger escalation
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
    history = state.get("conversation_history", []) or []
    history = state.get("conversation_history", []) or []

    # 1. Deterministic tool call: Fetch live system telemetry
    system_status = check_system_status(hostname=hostname)

    # 2. Append telemetry context to query without modifying rag/prompts.py signature
    telemetry_block = json.dumps(system_status, indent=2)
    augmented_query = (
        f"{query}\n\n"
        f"--- Live System Telemetry ({hostname}) ---\n"
        f"{telemetry_block}"
    )

    # 3. Build prompt (with prior conversation turns, if any) and generate
    messages = build_rag_prompt(query=augmented_query, chunks=chunks, history=history)
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
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Helper function to execute the compiled LangGraph workflow."""
    initial_state: AgentState = {
        "query": query,
        "top_k": top_k,
        "source_filter": source_filter,
        "hostname": hostname,
        "conversation_history": conversation_history or [],
    }
    return app.invoke(initial_state)