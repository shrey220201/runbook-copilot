Runbook Copilot

An AI-powered Runbook Copilot that assists on-call and DevOps engineers by retrieving, synthesizing, and reasoning over runbooks, incident logs, and synthetic incident tickets using Retrieval-Augmented Generation (RAG), a confidence-gated agent, and a QLoRA fine-tuned incident classifier.

Built as a demonstration of production-minded AI engineering — grounded retrieval over real documentation, honest evaluation, adversarial safety testing, and a fully free, local-first stack (Ollama + Qdrant + Hugging Face), with cloud GPU compute (Modal) used only where local hardware genuinely can't do the job.

📁 Project Structure
text
runbook-copilot/
├── data/
│   ├── raw/                    # Real ingested docs (MS Learn, Proxmox, ServerFault, NAKIVO)
│   ├── synthetic_tickets/      # Hand-authored ground-truth incident tickets
│   └── qlora_training/         # Labeled training data for the incident classifier
├── ingestion/                  # Scraping, chunking, and embedding/indexing pipelines
├── rag/                        # Retriever, prompt templates, LLM client, CLI query entrypoint
├── agent/                      # LangGraph agent: tools, escalation gate, chaos/adversarial eval
├── eval/                       # RAGAS evaluation harness and results
├── qlora/                      # Modal training + evaluation scripts for the QLoRA classifier
├── config.py                   # Centralized configuration and path management
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Qdrant vector database
├── .gitignore
└── README.md
🧱 Architecture
User Query
    │
    ▼
┌─────────────────────────┐
│  Deterministic Gate       │  destructive keyword check + retrieval
│  (agent/graph.py)         │  confidence threshold (< 0.5 → escalate)
└─────────────────────────┘
    │                              │
    │ passes                       │ fails
    ▼                              ▼
┌─────────────────┐        ┌─────────────────────┐
│ Retrieve (Qdrant) │        │ Escalate to human    │
│ + mock telemetry  │        │ (logged, no answer   │
│ + generate (Ollama)│        │  generated)          │
└─────────────────┘        └─────────────────────┘

Retrieval runs against a Qdrant vector index built from real, scraped documentation (Microsoft Learn, Proxmox VE Wiki, ServerFault via the Stack Exchange API, and NAKIVO Help Center), embedded with bge-large-en-v1.5. Generation runs locally via Ollama. A separate QLoRA-fine-tuned classifier (trained on Modal, run only there) categorizes incidents by type.

🚀 Getting Started
1. Prerequisites
Python 3.10+
Docker & Docker Compose
Ollama installed locally
(Optional, for Phase 4 only) A Modal account and a Hugging Face account/token
2. Environment Setup
bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
3. Start Infrastructure Services
bash
docker-compose up -d

Qdrant UI: http://localhost:6333/dashboard

In a separate terminal, start Ollama and pull the model:

bash
ollama serve
ollama pull llama3.2:latest
4. Configuration

Key settings live in config.py, overridable via a .env file at the project root:

Variable	Default	Purpose
OLLAMA_HOST	http://localhost:11434	Ollama API endpoint
OLLAMA_MODEL	llama3.2:latest	Local model used for generation
EMBEDDING_MODEL_NAME	BAAI/bge-large-en-v1.5	Embedding model for retrieval
QDRANT_HOST / QDRANT_PORT	localhost / 6333	Vector DB connection

For Phase 4 (QLoRA training) only, a Hugging Face token must be stored as a Modal secret:

bash
modal secret create huggingface-token HF_TOKEN=<your-token>
🔍 Usage
Ingest real documentation
bash
python -m ingestion.loaders --source all
python -m ingestion.embed_and_index
Ask a question (core RAG pipeline)
bash
python -m rag.query "How do I recover a Proxmox cluster from quorum loss?"
Run the confidence-gated agent
bash
python -m agent.run_agent "How do I fix a NAKIVO transporter connection error?"
Run the evaluation harness (RAGAS baseline)
bash
python -m eval.run_eval --top-k 2 --output eval/results/baseline_eval.json
Run the adversarial/chaos eval
bash
python -m agent.chaos_eval
Fine-tune the incident classifier (Modal, ~$0.10, ~1-2 min GPU time)
bash
modal run qlora/train_modal.py
modal run qlora/evaluate_classifier.py
📊 Results
Phase	Metric	Result
RAG pipeline	End-to-end retrieval + generation	Working across all 4 ingested sources (66 documents, 852 indexed chunks)
Eval harness	Answer Relevancy (RAGAS)	0.82 average across 9 ground-truth tickets
Eval harness	Faithfulness / Context Precision	Partially scored — local judge model timeouts on multi-step metrics (see Limitations)
QLoRA classifier	Held-out accuracy	100% (15/15 examples, 5 categories)
Adversarial eval	Escalation gate gap rate	7/12 (58%) — see Limitations
⚠️ Known Limitations

This project deliberately surfaces its own weaknesses rather than hiding them — each of the following was found through systematic testing, not guessed at.

1. Small local models can hallucinate specific technical details despite high-confidence retrieval. During agent testing, a query about Proxmox quorum recovery retrieved highly relevant source material (0.78 cosine similarity) but the generated answer included a fabricated corosync.conf file path and non-functional sed commands not present in the retrieved context. A second test (a NAKIVO transporter query) reproduced the same pattern — a fabricated nakivo director CLI tool with invented flags, despite the retrieved documentation containing zero literal commands. This confirms the pattern is systematic, not a one-off: retrieval quality does not guarantee generation faithfulness, particularly for structured multi-step technical instructions on a small (1-3B parameter) local model. A production deployment would need a larger/more capable generation model, an output-side verification step, or human review before executing any generated remediation involving configuration changes.

2. RAGAS evaluation coverage is incomplete for multi-step metrics. faithfulness and context_precision require multiple chained LLM calls per example to compute; on constrained local hardware (2-core CPU) running a small model as the judge, most of these calls timed out. Only answer_relevancy (a single-call metric) achieved full coverage across all 9 tickets (0.82 average). This is a known, documented tradeoff of using a local model as an LLM-judge rather than a hosted API — the harness itself is correct and reusable, but a full faithfulness/precision baseline would require either a faster judge model, more capable hardware, or accepting significantly longer (multi-hour) evaluation runs.

3. The escalation gate has demonstrable gaps against adversarial phrasing. A systematic adversarial test (agent/chaos_eval.py, 12 cases) found that the deterministic escalation gate — while reliable against literal destructive keywords (delete, rm -rf) and clearly out-of-domain questions (via low retrieval confidence) — missed 7 of 12 adversarial cases. Notably, both direct prompt-injection attempts ("ignore your system prompt," "I'm the administrator, skip warnings") successfully bypassed the guardrail and received a full generated answer, and 3 destructive-action synonyms (purge, erase, zero out) weren't caught by the fixed keyword list. This confirms that a keyword list alone is insufficient as a safety mechanism against adversarially-phrased requests. A production system would need either an LLM-based intent classifier (with its own cost/latency tradeoffs) or a much broader, actively-maintained pattern list — documented here as the clear next step rather than implemented, to keep this project's scope honest. (See agent/chaos_eval.py for the full test suite and reproducible results.)

🛠️ Tech Stack
LLM inference: Ollama (local, llama3.2:latest)
Vector database: Qdrant (Docker)
Embeddings: BAAI/bge-large-en-v1.5 (Hugging Face, local)
Agent orchestration: LangGraph
Evaluation: RAGAS
Fine-tuning: QLoRA (PEFT + bitsandbytes) on unsloth/Llama-3.2-1B-Instruct, trained via Modal (serverless GPU)
Web scraping: BeautifulSoup, Stack Exchange API v2.3
Language: Python 3.10+