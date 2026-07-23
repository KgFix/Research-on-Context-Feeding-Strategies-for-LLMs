"""Thin wrapper around the Ollama REST API.

Handles all LLM calls: query-answering, intent classification (S2),
summarisation (S4), and relevance judging. Temperature is fixed low
for reproducibility — the strategy, not sampling randomness, should
explain differences between runs.

All calls go through generate(), which is the single point of contact
with Ollama. This makes it easy to log prompts and swap models.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from ctxfeed.config import LLMConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for the Ollama local LLM REST API.

    Args:
        config: LLM configuration (model name, base URL, decoding params).
    """

    def __init__(self, config: LLMConfig) -> None:
        self._model = config.model
        self._base_url = config.base_url.rstrip("/")
        self._temperature = config.temperature
        self._top_p = config.top_p
        self._seed = config.seed

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
    ) -> tuple[str, float]:
        """Send a prompt to Ollama and return the response text + latency.

        Args:
            prompt: The user/query prompt.
            system: Optional system prompt prepended to the conversation.
            model_override: Use a different model for this call (e.g. judge).
            temperature_override: Override the default temperature.

        Returns:
            Tuple of (response_text, latency_seconds).

        Raises:
            ConnectionError: If Ollama is not reachable.
            RuntimeError: If the API returns an error.
        """
        model = model_override or self._model
        temperature = temperature_override if temperature_override is not None else self._temperature

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": self._top_p,
                "seed": self._seed,
            },
        }
        if system:
            payload["system"] = system

        url = f"{self._base_url}/api/generate"

        logger.debug("Ollama request: model=%s, prompt_len=%d", model, len(prompt))

        start = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
        except requests.ConnectionError as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self._base_url}. "
                f"Is 'ollama serve' running? Error: {e}"
            ) from e
        except requests.HTTPError as e:
            raise RuntimeError(
                f"Ollama API error: {resp.status_code} — {resp.text}"
            ) from e

        latency = time.perf_counter() - start
        result = resp.json()
        response_text = result.get("response", "").strip()

        logger.debug(
            "Ollama response: %d chars in %.2fs", len(response_text), latency
        )

        return response_text, latency

    def is_available(self) -> bool:
        """Check whether Ollama is reachable and the model is loaded."""
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check if our model (or a prefix match) is available
            return any(self._model in m for m in models)
        except (requests.ConnectionError, requests.Timeout):
            return False

    def __repr__(self) -> str:
        return (
            f"OllamaClient(model='{self._model}', "
            f"base_url='{self._base_url}', "
            f"temperature={self._temperature})"
        )
