"""Fixtures: configuración aislada y un Ollama simulado."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import responses

from biblio_llm.config import Settings

ROOT = Path(__file__).resolve().parents[1]
URL = "http://ollama.test"

KEYWORDS = {
    g: [{"termino": f"{g}-{i}", "nota": f"nota {i}"} for i in range(5)]
    for g in ("nucleo", "sinonimos", "exclusiones")
}
FICHA = {
    "tema": "t",
    "audiencia": "a",
    "objetivo": "o",
    "alcance": {"incluye": ["x"], "excluye": ["y"]},
    "subpreguntas": [f"p{i}" for i in range(5)],
    "palabras_clave": [f"k{i}" for i in range(5)],
    "riesgos_y_sesgos": ["r1", "r2", "r3"],
}
PLAN = {
    "titulo": "T",
    "resumen_100_palabras": "R",
    "secciones": [
        {"nombre": f"S{i}", "objetivo": "o", "puntos": ["p"]} for i in range(3)
    ],
    "criterios_de_calidad": ["c1", "c2", "c3"],
}


@pytest.fixture
def settings(tmp_path) -> Settings:
    templates = tmp_path / "plantillas"
    shutil.copytree(ROOT / "plantillas", templates)
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        ollama_url=URL,
        templates_dir=templates,
        runs_dir=tmp_path / "runs",
        log_dir=tmp_path / "logs",
    )


@pytest.fixture
def ollama():
    """Ollama falso: responde a cada plantilla con un JSON válido para su esquema."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        mock.get(f"{URL}/api/version", json={"version": "0.40.8"})
        mock.get(
            f"{URL}/api/tags",
            json={"models": [{"name": "deepseek-r1:14b", "digest": "abc123def456789"}]},
        )

        def chat(request):
            body = json.loads(request.body)
            schema = body["format"]
            match = next(
                v
                for k, v in (
                    ("nucleo", KEYWORDS),
                    ("audiencia", FICHA),
                    ("titulo", PLAN),
                )
                if k in schema["properties"]
            )
            answer = {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(match),
                    "thinking": "razonando…",
                },
                "done_reason": "stop",
                "eval_count": 100,
                "prompt_eval_count": 50,
            }
            return 200, {}, json.dumps(answer)

        mock.add_callback(responses.POST, f"{URL}/api/chat", callback=chat)
        yield mock
