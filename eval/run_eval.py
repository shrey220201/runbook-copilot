"""Evaluation harness for Runbook Copilot RAG pipeline.

Evaluates retrieval quality and generation quality against synthetic ground-truth
tickets using RAGAS metrics (faithfulness, answer_relevancy, context_precision)
and source recall/precision metrics with an Ollama-backed LLM judge.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.run_config import RunConfig

from config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    EMBEDDING_MODEL_NAME,
)
from eval.schema import EvaluationTicket
from rag.retriever import retriever, RetrievedChunk
from rag.prompts import build_rag_prompt
from rag.llm import llm_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TICKETS_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_tickets" / "tickets.json"


def load_tickets(tickets_path: Path) -> List[EvaluationTicket]:
    """Load ground-truth evaluation tickets from JSON."""
    if not tickets_path.exists():
        raise FileNotFoundError(f"Evaluation tickets file not found: {tickets_path}")

    with open(tickets_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [EvaluationTicket.from_dict(item) for item in data]


def run_rag_pipeline_for_ticket(
    ticket: EvaluationTicket,
    top_k: int = 4,
) -> Dict[str, Any]:
    """Run a single evaluation ticket through the retriever and LLM client."""
    # 1. Retrieve chunks
    chunks: List[RetrievedChunk] = retriever.retrieve(
        query=ticket.incident_description,
        top_k=top_k,
    )

    # 2. Build prompt and generate response
    messages = build_rag_prompt(ticket.incident_description, chunks)
    generated_answer = llm_client.chat(messages=messages, temperature=0.2)

    # 3. Source analysis
    retrieved_sources = list({c.source for c in chunks})
    expected_sources = ticket.expected_sources
    source_match = any(src in retrieved_sources for src in expected_sources)

    # 4. Contexts extraction
    contexts = [c.content for c in chunks]

    return {
        "ticket": ticket,
        "chunks": chunks,
        "contexts": contexts,
        "retrieved_sources": retrieved_sources,
        "source_match": source_match,
        "generated_answer": generated_answer,
    }


def compute_ragas_evaluation(
    eval_results: List[Dict[str, Any]],
    ollama_host: Optional[str] = None,
    ollama_model: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
) -> Any:
    """Run RAGAS evaluation on pipeline outputs using Ollama and local HuggingFace embeddings."""
    host = ollama_host or OLLAMA_HOST
    model = ollama_model or OLLAMA_MODEL
    emb_model = embedding_model_name or EMBEDDING_MODEL_NAME

    logger.info(f"Initializing RAGAS evaluator LLM (ChatOllama: {model} @ {host})")
    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            base_url=host,
            model=model,
            temperature=0.0,
        )
    )

    logger.info(f"Initializing RAGAS evaluator Embeddings (HuggingFace: {emb_model})")
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=emb_model)
    )

    # Build RAGAS dataset samples
    samples = []
    for item in eval_results:
        ticket: EvaluationTicket = item["ticket"]
        samples.append(
            SingleTurnSample(
                user_input=ticket.incident_description,
                retrieved_contexts=item["contexts"] if item["contexts"] else ["No context retrieved."],
                response=item["generated_answer"],
                reference=ticket.ground_truth_answer,
            )
        )

    dataset = EvaluationDataset(samples=samples)

    logger.info("Executing RAGAS metric calculations (faithfulness, answer_relevancy, context_precision)...")
    metrics = [faithfulness, answer_relevancy, context_precision]
    run_config = RunConfig(
        timeout=600,
        max_workers=1,
    )
    evaluation_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )

    return evaluation_result


def print_evaluation_summary(
    eval_results: List[Dict[str, Any]],
    ragas_df: Any,
):
    """Print a Rich summary table per ticket and average baseline scores."""
    if HAS_RICH:
        table = Table(
            title="📊 Runbook Copilot RAG Evaluation Baseline",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Ticket ID", style="bold white", width=10)
        table.add_column("Category", style="dim", width=16)
        table.add_column("Expected Sources", width=20)
        table.add_column("Retrieved Sources", width=20)
        table.add_column("Faithfulness", justify="right", style="green", width=14)
        table.add_column("Answer Relevancy", justify="right", style="magenta", width=16)
        table.add_column("Context Precision", justify="right", style="yellow", width=18)

        for i, item in enumerate(eval_results):
            ticket: EvaluationTicket = item["ticket"]
            row = ragas_df.iloc[i] if ragas_df is not None and i < len(ragas_df) else {}

            f_score = row.get("faithfulness", float("nan"))
            ar_score = row.get("answer_relevancy", float("nan"))
            cp_score = row.get("context_precision", float("nan"))

            table.add_row(
                ticket.ticket_id,
                ticket.category,
                ", ".join(ticket.expected_sources),
                ", ".join(item["retrieved_sources"]) or "None",
                f"{f_score:.4f}" if not (f_score != f_score) else "N/A",
                f"{ar_score:.4f}" if not (ar_score != ar_score) else "N/A",
                f"{cp_score:.4f}" if not (cp_score != cp_score) else "N/A",
            )

        console.print(table)
        console.print()

        # Overall Averages Panel
        avg_f = ragas_df["faithfulness"].mean() if "faithfulness" in ragas_df else 0.0
        avg_ar = ragas_df["answer_relevancy"].mean() if "answer_relevancy" in ragas_df else 0.0
        avg_cp = ragas_df["context_precision"].mean() if "context_precision" in ragas_df else 0.0

        stats_text = (
            f"[bold green]Faithfulness:[/bold green] {avg_f:.4f}\n"
            f"[bold magenta]Answer Relevancy:[/bold magenta] {avg_ar:.4f}\n"
            f"[bold yellow]Context Precision:[/bold yellow] {avg_cp:.4f}"
        )
        console.print(Panel(stats_text, title="🎯 Overall Baseline Averages", border_style="cyan"))
    else:
        print("\n=== RUNBOOK COPILOT RAG EVALUATION BASELINE ===")
        for i, item in enumerate(eval_results):
            ticket: EvaluationTicket = item["ticket"]
            row = ragas_df.iloc[i] if ragas_df is not None and i < len(ragas_df) else {}
            print(f"[{ticket.ticket_id}] Category: {ticket.category}")
            print(f"  Expected: {ticket.expected_sources} | Retrieved: {item['retrieved_sources']}")
            print(f"  Faithfulness: {row.get('faithfulness', 'N/A')}")
            print(f"  Answer Relevancy: {row.get('answer_relevancy', 'N/A')}")
            print(f"  Context Precision: {row.get('context_precision', 'N/A')}\n")


def run_evaluation(
    tickets_path: Path = DEFAULT_TICKETS_PATH,
    top_k: int = 4,
    save_output: Optional[Path] = None,
):
    """Run full evaluation suite across all seed tickets."""
    tickets = load_tickets(tickets_path)
    logger.info(f"Loaded {len(tickets)} evaluation tickets from {tickets_path}")

    eval_results = []
    for ticket in tickets:
        logger.info(f"Running pipeline for ticket: {ticket.ticket_id} ({ticket.category})")
        res = run_rag_pipeline_for_ticket(ticket, top_k=top_k)
        eval_results.append(res)

    ragas_result = compute_ragas_evaluation(eval_results)
    ragas_df = ragas_result.to_pandas()

    print_evaluation_summary(eval_results, ragas_df)

    if save_output:
        save_data = []
        for i, item in enumerate(eval_results):
            ticket: EvaluationTicket = item["ticket"]
            row = ragas_df.iloc[i].to_dict() if ragas_df is not None and i < len(ragas_df) else {}
            save_data.append({
                "ticket_id": ticket.ticket_id,
                "incident_description": ticket.incident_description,
                "category": ticket.category,
                "expected_sources": ticket.expected_sources,
                "retrieved_sources": item["retrieved_sources"],
                "generated_answer": item["generated_answer"],
                "ground_truth_answer": ticket.ground_truth_answer,
                "metrics": {
                    "faithfulness": row.get("faithfulness"),
                    "answer_relevancy": row.get("answer_relevancy"),
                    "context_precision": row.get("context_precision"),
                }
            })

        save_output.parent.mkdir(parents=True, exist_ok=True)
        with open(save_output, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)
        logger.info(f"Evaluation report saved to {save_output}")


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation harness for Runbook Copilot")
    parser.add_argument(
        "--tickets",
        type=str,
        default=str(DEFAULT_TICKETS_PATH),
        help="Path to synthetic tickets JSON file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of retrieved chunks per query (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save evaluation results JSON",
    )
    args = parser.parse_args()

    run_evaluation(
        tickets_path=Path(args.tickets),
        top_k=args.top_k,
        save_output=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
