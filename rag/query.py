"""CLI Query Entrypoint for Runbook Copilot.

Retrieves relevant runbook chunks from Qdrant and generates a grounded
remediation answer using the centralized LLMClient from rag.llm.
"""

import sys
import argparse
from typing import Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    console = Console(legacy_windows=False)
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False

from rag.retriever import retriever, RetrievedChunk
from rag.prompts import build_rag_prompt
from rag.llm import llm_client


def display_retrieved_chunks(chunks: list[RetrievedChunk]):
    """Display retrieved runbook chunks with metadata in terminal."""
    if not chunks:
        if HAS_RICH:
            console.print("[yellow]⚠️ No relevant runbook documents retrieved.[/yellow]")
        else:
            print("No relevant runbook documents retrieved.")
        return

    if HAS_RICH:
        table = Table(title="🔍 Retrieved Runbook Documents", show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Source", style="green", width=18)
        table.add_column("Title / Section", style="bold white")
        table.add_column("Similarity Score", justify="right", style="magenta", width=16)

        for i, chunk in enumerate(chunks, 1):
            table.add_row(
                str(i),
                chunk.source,
                f"{chunk.title}\n[dim]{chunk.header_path}[/dim]",
                f"{chunk.score:.4f}",
            )
        console.print(table)
        console.print()
    else:
        print("\n=== RETRIEVED RUNBOOK DOCUMENTS ===")
        for i, chunk in enumerate(chunks, 1):
            print(f"[{i}] {chunk.source} | {chunk.title} | Score: {chunk.score:.4f}")
            print(f"    Section: {chunk.header_path}")
            print(f"    URL: {chunk.url}\n")


def query_runbook(
    question: str,
    top_k: int = 4,
    source_filter: Optional[str] = None,
    stream: bool = True,
) -> str:
    """Execute end-to-end RAG query pipeline for a question."""
    if HAS_RICH:
        console.print(Panel(f"[bold white]{question}[/bold white]", title="❓ User Incident Query", border_style="blue"))
    else:
        print(f"\nUser Query: {question}\n" + "=" * 50)

    # 1. Retrieve relevant chunks
    chunks = retriever.retrieve(
        query=question,
        top_k=top_k,
        source_filter=source_filter,
    )

    # 2. Display retrieved chunks
    display_retrieved_chunks(chunks)

    # 3. Build prompt
    messages = build_rag_prompt(question, chunks)

    # 4. Generate response using centralized LLMClient
    if HAS_RICH:
        console.print("[bold cyan]🤖 Generating Runbook Copilot Solution...[/bold cyan]\n")
    else:
        print("Generating Solution...\n")

    if stream:
        collected_text = []
        try:
            for token in llm_client.stream_chat(messages=messages, temperature=0.2):
                sys.stdout.write(token)
                sys.stdout.flush()
                collected_text.append(token)
            print("\n")
            return "".join(collected_text)
        except Exception as e:
            if HAS_RICH:
                console.print(f"\n[red]Streaming failed ({e}), falling back to standard completion...[/red]")
            else:
                print(f"\nStreaming failed ({e}), falling back to standard completion...")
            answer = llm_client.chat(messages=messages, temperature=0.2)
            if HAS_RICH:
                console.print(Markdown(answer))
            else:
                print(answer)
            return answer
    else:
        answer = llm_client.chat(messages=messages, temperature=0.2)
        if HAS_RICH:
            console.print(Panel(Markdown(answer), title="📋 Remediation Plan", border_style="green"))
        else:
            print("\n=== REMEDIATION PLAN ===")
            print(answer)
        return answer


def main():
    parser = argparse.ArgumentParser(description="Query the Runbook Copilot RAG pipeline")
    parser.add_argument("query", type=str, help="Incident query or troubleshooting question")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")
    parser.add_argument("--source", type=str, default=None, help="Filter by source (e.g. 'Proxmox VE Wiki')")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming response")

    args = parser.parse_args()

    query_runbook(
        question=args.query,
        top_k=args.top_k,
        source_filter=args.source,
        stream=not args.no_stream,
    )


if __name__ == "__main__":
    main()
