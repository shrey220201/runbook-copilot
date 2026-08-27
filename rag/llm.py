"""Abstracted LLM interface for Runbook Copilot."""

from typing import Generator, List, Dict, Any, Optional
import requests
import json

from config import OLLAMA_HOST, OLLAMA_MODEL


class LLMClient:
    """Client for interacting with the LLM backend (Ollama).
    
    Acts as a centralized abstraction layer so switching models or providers
    does not affect retriever or query pipeline logic.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
    ):
        self.host = (host or OLLAMA_HOST).rstrip("/")
        self.model = model or OLLAMA_MODEL
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        """Generate a single text completion for a prompt."""
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                **kwargs,
            },
        }
        if system:
            payload["system"] = system

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request with structured message history."""
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                **kwargs,
            },
        }

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Stream a chat completion response chunk by chunk."""
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                **kwargs,
            },
        }

        with requests.post(url, json=payload, timeout=self.timeout, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content


# Global default instance
llm_client = LLMClient()
