"""Cliente mínimo de la API de Ollama (``/api/chat``) con salidas estructuradas."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import requests

from biblio_llm.config import Settings

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
_FENCE_RE = re.compile(r"^```\w*\s*|\s*```$")


class OllamaError(Exception):
    """Ollama no responde, el modelo no existe o la respuesta no es utilizable."""


@dataclass
class ChatResult:
    content: str  # respuesta limpia (sin razonamiento ni vallas de código)
    thinking: str  # razonamiento del modelo (vacío si no pensó)
    raw: dict[str, Any]  # respuesta completa de la API
    truncated: bool  # se cortó por num_predict


def split_thinking(content: str, thinking: str = "") -> tuple[str, str]:
    """Separa un bloque ``<think>…</think>`` incrustado en el contenido.

    Ollama moderno devuelve el razonamiento en un campo aparte, pero algunas
    versiones o modelos lo dejan dentro del texto.
    """
    found = _THINK_RE.findall(content)
    if found:
        thinking = "\n".join(filter(None, [thinking, *(f.strip() for f in found)]))
        content = _THINK_RE.sub("", content)
    return _FENCE_RE.sub("", content.strip()).strip(), thinking.strip()


class OllamaClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.s = settings
        self.session = session or requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.s.ollama_url}{path}"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.session.request(
                method, self._url(path), timeout=kwargs.pop("timeout", 30), **kwargs
            )
        except requests.ConnectionError as exc:
            raise OllamaError(
                f"No se pudo conectar con Ollama en {self.s.ollama_url}. "
                "¿Está abierto? (app de Ollama o `ollama serve`)"
            ) from exc
        except requests.Timeout as exc:
            raise OllamaError(
                f"Ollama no respondió a tiempo ({self.s.request_timeout:.0f} s). "
                "Sube REQUEST_TIMEOUT o usa un modelo más pequeño."
            ) from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("error", response.text)
            except ValueError:
                detail = response.text
            raise OllamaError(f"Ollama respondió {response.status_code}: {detail}")
        data: dict[str, Any] = response.json()
        return data

    # ------------------------------------------------------------ información
    def version(self) -> str:
        return str(self._request("GET", "/api/version").get("version", "?"))

    def model_digest(self) -> str | None:
        """Digest del modelo instalado (identifica la versión exacta), o None."""
        for model in self._request("GET", "/api/tags").get("models", []):
            if model.get("name") == self.s.model or model.get("model") == self.s.model:
                return str(model.get("digest"))
        return None

    # ------------------------------------------------------------------ chat
    def chat(
        self, system: str, user: str, schema: dict[str, Any] | None = None
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.s.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": self.s.think,
            "options": {
                "temperature": self.s.temperature,
                "seed": self.s.seed,
                "num_ctx": self.s.num_ctx,
                "num_predict": self.s.num_predict,
            },
        }
        if schema is not None:
            payload["format"] = schema
        data = self._request(
            "POST", "/api/chat", json=payload, timeout=self.s.request_timeout
        )
        message = data.get("message") or {}
        content, thinking = split_thinking(
            message.get("content", ""), message.get("thinking", "")
        )
        truncated = data.get("done_reason") == "length"
        if truncated:
            logger.warning(
                "La respuesta se cortó por NUM_PREDICT=%d (el razonamiento también "
                "cuenta). Súbelo en .env.",
                self.s.num_predict,
            )
        return ChatResult(content, thinking, data, truncated)
