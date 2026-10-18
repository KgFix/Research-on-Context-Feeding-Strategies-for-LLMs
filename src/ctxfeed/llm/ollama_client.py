# wrapper around ollama local rest api for mistral / phi queries
# all model calls go through generate() to measure latency accurately

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from ctxfeed.config import LLMConfig

logger = logging.getLogger(__name__)


class OllamaClient:
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
        # post to /api/generate and measure inference time

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
                f"Ollama API error: {resp.status_code} - {resp.text}"
            ) from e

        latency = time.perf_counter() - start
        result = resp.json()
        response_text = result.get("response", "").strip()

        logger.debug(
            "Ollama response: %d chars in %.2fs", len(response_text), latency
        )

        return response_text, latency

    def is_available(self) -> bool:
        # check whether ollama is running and model is loaded
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
