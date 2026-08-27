"""Prompt templates for Runbook Copilot RAG queries."""

from typing import List, Dict
from rag.retriever import RetrievedChunk

SYSTEM_PROMPT = """You are Runbook Copilot, an expert Site Reliability Engineer (SRE) and DevOps troubleshooting assistant.
Your job is to assist engineers during on-call incidents, system outages, and infrastructure troubleshooting by synthesizing official runbooks, wiki guides, and knowledge base documentation.

Guidelines:
1. Ground your answers strictly in the provided Runbook Context. Do not invent commands, flags, or configuration options that contradict the reference material.
2. Provide step-by-step diagnostic and remediation instructions formatted clearly with Markdown headings and bullet points.
3. Place all executable shell commands, scripts, and configuration blocks in fenced code blocks with appropriate syntax highlighting (`bash`, `json`, `python`, etc.).
4. Include safety precautions, prerequisites, and rollback instructions where relevant.
5. Explicitly cite your sources by referencing the document title and source name (e.g. `[Source: Proxmox VE Wiki - Cluster Quorum Loss]`).
6. If the provided context is insufficient to answer the question confidently, state what information is missing and what diagnostic commands the engineer should run first.
"""


def format_context_chunks(chunks: List[RetrievedChunk]) -> str:
    """Format a list of retrieved chunks into an organized context string."""
    if not chunks:
        return "No relevant runbook documentation found."

    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        block = f"""---
[Context Reference {i}]
Source: {chunk.source}
Title: {chunk.title}
Section: {chunk.header_path}
URL: {chunk.url}

{chunk.content}
---"""
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def build_rag_prompt(query: str, chunks: List[RetrievedChunk]) -> List[Dict[str, str]]:
    """Build chat messages payload for LLMClient containing system prompt and formatted context."""
    formatted_context = format_context_chunks(chunks)

    user_message = f"""Incident Question / Problem:
{query}

==================== RUNBOOK CONTEXT ====================
{formatted_context}
=========================================================

Based on the runbook context provided above, provide a comprehensive, step-by-step resolution plan with exact diagnostic and remediation commands, caveats, and citations.
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_message.strip()},
    ]
