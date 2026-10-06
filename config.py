import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SYNTHETIC_TICKETS_DIR = DATA_DIR / "synthetic_tickets"

# Vector Store / Qdrant Settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "runbooks")

# Embedding Settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")

# LLM / Ollama Settings
# NOTE: deliberately named RUNBOOK_OLLAMA_HOST, not OLLAMA_HOST - the Ollama
# installer sets a system-level OLLAMA_HOST env var (0.0.0.0) to control which
# interface the Ollama *server* binds to. Reusing that name here collided with
# it silently, since os.getenv only falls back to the default when the var is
# completely unset, not when it holds an unexpected value.
#
# Default is localhost rather than a hardcoded LAN IP - a prior default of
# 192.168.2.47 went stale after a DHCP address change and silently broke any
# code path reaching the LLM, undetected until chaos-eval testing exercised
# it directly. localhost is immune to that class of drift since Ollama binds
# to 0.0.0.0, which is reachable via localhost regardless of the LAN IP.
OLLAMA_HOST = os.getenv("RUNBOOK_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")