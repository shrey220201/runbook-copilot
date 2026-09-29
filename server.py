import time
import os
import requests
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    EMBEDDING_MODEL_NAME,
)
from agent.graph import run_agent_workflow
from rag.retriever import retriever
from rag.prompts import build_rag_prompt
from rag.llm import llm_client

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Runbook Copilot Web UI",
    description="AI-powered on-call incident response and runbook copilot",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


class QueryRequest(BaseModel):
    query: str = Field(..., description="Incident troubleshooting query or error message")
    mode: str = Field("agent", description="'agent' (confidence-gated with telemetry) or 'rag' (direct retrieval)")
    top_k: int = Field(4, ge=1, le=10, description="Number of runbook chunks to retrieve")
    source_filter: Optional[str] = Field(None, description="Optional runbook source filter")
    hostname: Optional[str] = Field("cluster-node-01", description="Target host for mock telemetry")


class ChunkResponse(BaseModel):
    id: str
    score: float
    title: str
    source: str
    url: str
    header_path: str
    content: str


class QueryResponse(BaseModel):
    query: str
    mode: str
    response: str
    is_escalated: bool
    escalation_reason: Optional[str] = None
    system_status: Optional[Dict[str, Any]] = None
    retrieved_chunks: List[ChunkResponse]
    latency_ms: float


@app.get("/api/status")
async def get_status():
    """Return live status of Qdrant, Ollama, and vector collection metadata."""
    qdrant_ok = False
    qdrant_count = 0
    sources = []
    ollama_ok = False
    ollama_models = []

    # Check Qdrant
    try:
        col_info = retriever.client.get_collection(QDRANT_COLLECTION_NAME)
        qdrant_ok = True
        qdrant_count = col_info.points_count or 0
        sources = ["All Sources", "Proxmox VE Wiki", "NAKIVO Help Center", "Microsoft Learn", "ServerFault"]
    except Exception as e:
        qdrant_ok = False

    # Check Ollama
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if r.status_code == 200:
            ollama_ok = True
            ollama_models = [m.get("name") for m in r.json().get("models", [])]
    except Exception:
        ollama_ok = False

    return {
        "status": "ready" if (qdrant_ok and ollama_ok) else "degraded",
        "qdrant": {
            "connected": qdrant_ok,
            "host": f"{QDRANT_HOST}:{QDRANT_PORT}",
            "collection": QDRANT_COLLECTION_NAME,
            "indexed_chunks": qdrant_count,
            "embedding_model": EMBEDDING_MODEL_NAME,
        },
        "ollama": {
            "connected": ollama_ok,
            "host": OLLAMA_HOST,
            "active_model": OLLAMA_MODEL,
            "available_models": ollama_models,
        },
        "sources": sources,
    }


@app.post("/api/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    """Handle query via agent workflow or direct RAG."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start_time = time.time()
    source_arg = None if req.source_filter in [None, "", "All Sources", "all"] else req.source_filter

    try:
        if req.mode == "rag":
            # Direct RAG Pipeline
            chunks = retriever.retrieve(
                query=req.query,
                top_k=req.top_k,
                source_filter=source_arg,
            )
            messages = build_rag_prompt(query=req.query, chunks=chunks)
            response_text = llm_client.chat(messages=messages, temperature=0.2)
            is_escalated = False
            escalation_reason = None
            system_status = None
        else:
            # Agent Orchestrator with Confidence Gate and System Telemetry
            result = run_agent_workflow(
                query=req.query,
                top_k=req.top_k,
                source_filter=source_arg,
                hostname=req.hostname or "cluster-node-01",
            )
            chunks = result.get("retrieved_chunks", [])
            is_escalated = result.get("is_escalated", False)
            escalation_reason = result.get("escalation_reason")
            system_status = result.get("system_status")
            response_text = result.get("response", "")
    except Exception as exc:
        # Graceful fallback on LLM/Ollama hiccups
        chunks = []
        is_escalated = True
        escalation_reason = f"LLM Generation Error: {str(exc)}"
        system_status = None
        response_text = (
            f"⚠️ An error occurred while generating the remediation: {str(exc)}.\n\n"
            "Please check that Ollama is active and has sufficient memory resources."
        )

    chunk_dtos = [
        ChunkResponse(
            id=c.id,
            score=round(c.score, 4),
            title=c.title,
            source=c.source,
            url=c.url,
            header_path=c.header_path,
            content=c.content,
        )
        for c in chunks
    ]

    latency_ms = round((time.time() - start_time) * 1000, 1)

    return QueryResponse(
        query=req.query,
        mode=req.mode,
        response=response_text,
        is_escalated=is_escalated,
        escalation_reason=escalation_reason,
        system_status=system_status,
        retrieved_chunks=chunk_dtos,
        latency_ms=latency_ms,
    )


# Serve static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(STATIC_DIR / "index.html")
