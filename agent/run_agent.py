"""CLI entrypoint for Runbook Copilot Agent Orchestrator.

Runs queries through the LangGraph agent workflow with confidence-gated escalation,
deterministic system health telemetry, and grounded runbook remediation.
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

from agent.graph import run_agent_workflow
from rag.query import display_retrieved_chunks


def main():
    parser = argparse.ArgumentParser(description="Run incident query through Runbook Copilot Agent")
    parser.add_argument("query", type=str, help="Incident query or troubleshooting command")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")
    parser.add_argument("--source", type=str, default=None, help="Filter by runbook source")
    parser.add_argument("--hostname", type=str, default="cluster-node-01", help="Target hostname for telemetry")

    args = parser.parse_args()

    if HAS_RICH:
        console.print(Panel(f"[bold white]{args.query}[/bold white]", title="🤖 Runbook Copilot Agent Query", border_style="cyan"))
    else:
        print(f"\nAgent Query: {args.query}\n" + "=" * 50)

    result = run_agent_workflow(
        query=args.query,
        top_k=args.top_k,
        source_filter=args.source,
        hostname=args.hostname,
    )

    is_escalated = result.get("is_escalated", False)
    retrieved_chunks = result.get("retrieved_chunks", [])
    response = result.get("response", "")
    system_status = result.get("system_status")

    if retrieved_chunks:
        display_retrieved_chunks(retrieved_chunks)

    if system_status:
        if HAS_RICH:
            telemetry_table = Table(title=f"📊 System Health Telemetry ({system_status.get('hostname')})", header_style="bold blue")
            telemetry_table.add_column("Metric", style="cyan")
            telemetry_table.add_column("Value", style="bold green")
            for k, v in system_status.items():
                telemetry_table.add_row(str(k), str(v))
            console.print(telemetry_table)
            console.print()
        else:
            print(f"\n--- System Telemetry ({system_status.get('hostname')}) ---")
            for k, v in system_status.items():
                print(f"  {k}: {v}")
            print()

    if is_escalated:
        reason = result.get("escalation_reason", "Safety / confidence policy triggered")
        if HAS_RICH:
            console.print(Panel(
                f"[bold red]⚠️ INCIDENT ESCALATED TO HUMAN ON-CALL[/bold red]\n\n"
                f"[yellow]Reason:[/yellow] {reason}\n\n"
                f"[dim]Logged to: eval/escalations.log[/dim]",
                title="🚨 Escalation Notice",
                border_style="red",
            ))
        else:
            print("\n" + "!" * 50)
            print(f"INCIDENT ESCALATED: {reason}")
            print("Logged to: eval/escalations.log")
            print("!" * 50 + "\n")
    else:
        if HAS_RICH:
            console.print(Panel(Markdown(response), title="📋 Remediation Plan", border_style="green"))
        else:
            print("\n=== REMEDIATION PLAN ===")
            print(response)


if __name__ == "__main__":
    main()
