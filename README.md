# Runbook Copilot

An AI-powered Runbook Copilot that assists on-call and DevOps engineers by retrieving, synthesizing, and reasoning over runbooks, incident logs, and synthetic incident tickets using Retrieval-Augmented Generation (RAG).

---

## 📁 Project Structure

```text
runbook-copilot/
├── data/
│   ├── raw/                  # Raw runbooks, markdown documents, and runbook dumps
│   └── synthetic_tickets/    # Generated synthetic incident tickets and scenarios
├── ingestion/                # Ingestion, parsing, chunking, and indexing pipelines
├── rag/                      # Retrieval-Augmented Generation pipeline & query engine
├── config.py                 # Centralized configuration and path management
├── requirements.txt          # Python dependencies
├── docker-compose.yml        # Infrastructure setup (e.g., Qdrant vector database)
├── .gitignore                # Git ignore configuration
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Docker & Docker Compose

### 2. Environment Setup

Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Start Infrastructure Services

Spin up the Qdrant vector database:
```bash
docker-compose up -d
```
Qdrant UI will be accessible at `http://localhost:6333/dashboard`.

### 4. Configuration

Configure your environment settings in `config.py` or create a `.env` file at the root.
