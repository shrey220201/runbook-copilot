"""Tools for the Runbook Copilot Agent Orchestrator."""

import os
import json
import random
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from rag.retriever import retriever, RetrievedChunk


def search_runbooks(
    query: str,
    top_k: int = 4,
    source_filter: Optional[str] = None,
) -> List[RetrievedChunk]:
    """Retrieve relevant runbook documentation chunks from Qdrant vector store.
    
    Calls QdrantRetriever.retrieve() with matching signature parameters.
    """
    return retriever.retrieve(
        query=query,
        top_k=top_k,
        source_filter=source_filter,
    )


def check_system_status(hostname: Optional[str] = None) -> Dict[str, Any]:
    """Deterministic monitoring tool returning structured mock telemetry for a given host."""
    host = hostname or "cluster-node-01"
    
    return {
        "hostname": host,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cpu_percent": round(random.uniform(25.0, 78.0), 1),
        "memory_used_percent": round(random.uniform(40.0, 85.0), 1),
        "disk_free_gb": round(random.uniform(50.0, 220.0), 1),
        "service_status": random.choice(["running", "running", "degraded"]),
        "load_average_15m": round(random.uniform(0.8, 3.5), 2),
        "network_status": "connected",
    }


def escalate_to_human(
    incident_description: str,
    reason: str,
    log_path: str = "eval/escalations.log",
) -> Dict[str, Any]:
    """Escalation tool that records incident details and reasons to an audit log file."""
    timestamp = datetime.now(timezone.utc).isoformat()
    log_entry = {
        "timestamp": timestamp,
        "incident": incident_description,
        "escalation_reason": reason,
        "status": "ESCALATED_PENDING_REVIEW",
    }

    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "status": "escalated",
        "reason": reason,
        "timestamp": timestamp,
        "log_path": log_path,
        "message": f"🚨 [ESCALATION REQUIRED] Incident escalated to on-call human engineer.\nReason: {reason}",
    }
