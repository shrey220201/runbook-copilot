"""
Modal-hosted Ollama server for Runbook Copilot generation.

Runs Ollama on a Modal GPU container (A10G), exposed as a token-protected
HTTP endpoint shaped identically to Ollama's own API, so rag/llm.py needs
no code changes - only RUNBOOK_OLLAMA_HOST (and a new auth header) point
at this URL instead of localhost.

Scale-to-zero by default (min_containers=0): costs nothing while idle,
pays only for active request time plus a short post-request keep-warm
window, at the cost of a cold-start delay (model load) on the first
request after idle time.

Deploy:
    modal deploy deploy/ollama_modal.py
    (prints the public URL on success - copy it into RUNBOOK_OLLAMA_HOST)
"""

import os
import subprocess
import time

import modal

MODEL_NAME = "llama3.2:latest"
OLLAMA_PORT = 11434

app = modal.App("runbook-copilot-ollama")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("fastapi", "httpx", "uvicorn")
)

# Persistent volume so the pulled model survives container restarts -
# avoids re-downloading ~2GB every cold start.
model_volume = modal.Volume.from_name("ollama-models", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    scaledown_window=120,  # keep container warm 2 min after last request
    min_containers=0,      # scale to zero when idle - no cost while unused
    timeout=600,
    volumes={"/root/.ollama": model_volume},
    secrets=[modal.Secret.from_name("ollama-auth-token")],
)
@modal.asgi_app()
def serve():
    import httpx
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import Response

    # Start the actual Ollama server as a background process in this container
    subprocess.Popen(["ollama", "serve"])

    # Wait for it to come up, then ensure the model is present
    for _ in range(30):
        try:
            httpx.get(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=2)
            break
        except Exception:
            time.sleep(1)
    subprocess.run(["ollama", "pull", MODEL_NAME], check=False)

    web_app = FastAPI()
    AUTH_TOKEN = os.environ["OLLAMA_AUTH_TOKEN"]

    @web_app.middleware("http")
    async def check_token(request: Request, call_next):
        auth_header = request.headers.get("authorization", "")
        if auth_header != f"Bearer {AUTH_TOKEN}":
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")
        return await call_next(request)

    @web_app.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        """Transparent proxy to the local Ollama process inside this container."""
        body = await request.body()
        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.request(
                request.method,
                f"http://127.0.0.1:{OLLAMA_PORT}/{path}",
                content=body,
                headers={"content-type": request.headers.get("content-type", "application/json")},
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    return web_app