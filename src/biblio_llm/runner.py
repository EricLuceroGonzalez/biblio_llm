"""Ejecución de un run: una carpeta nueva por consulta, nada se sobrescribe."""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from biblio_llm import __version__
from biblio_llm.config import Settings
from biblio_llm.ollama_client import OllamaClient, OllamaError
from biblio_llm.prompts import load_template, render, system_prompt
from biblio_llm.tasks import TASKS, Task
from biblio_llm.validation import validate_run

logger = logging.getLogger(__name__)


@dataclass
class RunParams:
    tema: str
    idioma: str
    audiencia: str


@dataclass
class RunSummary:
    run_dir: Path
    errors: dict[str, str] = field(default_factory=dict)
    validation: list[str] = field(default_factory=list)


def slugify(text: str, max_len: int = 50) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "consulta"


def new_run_dir(runs_dir: Path, tema: str) -> Path:
    """``runs/AAAA-MM-DD_HHMMSS_<tema>``; si ya existe, añade un sufijo."""
    base = runs_dir / f"{datetime.now():%Y-%m-%d_%H%M%S}_{slugify(tema)}"
    candidate, n = base, 1
    while candidate.exists():
        n += 1
        candidate = base.with_name(f"{base.name}-{n}")
    candidate.mkdir(parents=True)
    return candidate


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def run_task(
    client: OllamaClient,
    settings: Settings,
    task: Task,
    params: RunParams,
    run_dir: Path,
) -> dict[str, Any]:
    """Ejecuta una plantilla y guarda prompt, razonamiento, respuesta y salida."""
    folder = run_dir / task.folder
    folder.mkdir()
    values = {
        "TEMA": params.tema,
        "IDIOMA": params.idioma,
        "AUDIENCIA": params.audiencia,
    }

    user = render(load_template(settings.templates_dir, task.template), values)
    system = system_prompt(params.idioma)
    _write(folder / "prompt.txt", f"### SYSTEM\n{system}\n\n### USER\n{user}\n")

    started = time.monotonic()
    result = client.chat(system, user, task.schema)
    elapsed = time.monotonic() - started

    _write(
        folder / "respuesta_cruda.json",
        json.dumps(result.raw, ensure_ascii=False, indent=2),
    )
    if result.thinking:
        _write(folder / "razonamiento.txt", result.thinking + "\n")
    info: dict[str, Any] = {
        "seconds": round(elapsed, 1),
        "done_reason": result.raw.get("done_reason"),
        "prompt_tokens": result.raw.get("prompt_eval_count"),
        "output_tokens": result.raw.get("eval_count"),
        "truncated": result.truncated,
    }
    try:
        data = json.loads(result.content)
    except json.JSONDecodeError as exc:
        _write(folder / "respuesta.txt", result.content)
        raise OllamaError(
            f"la respuesta no es JSON ({exc}); "
            + (
                "se cortó por NUM_PREDICT"
                if result.truncated
                else "revisa respuesta.txt"
            )
        ) from exc
    _write(
        folder / "respuesta.json", json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
    _write(folder / task.output, task.render(data))
    return info


def execute(
    settings: Settings,
    params: RunParams,
    task_keys: list[str],
    client: OllamaClient | None = None,
) -> RunSummary:
    client = client or OllamaClient(settings)
    # Comprobaciones antes de crear nada: Ollama y el modelo, y las plantillas.
    digest = client.model_digest()
    if digest is None:
        raise OllamaError(
            f"el modelo {settings.model} no está instalado: ejecuta "
            f"`ollama pull {settings.model}`"
        )
    for key in task_keys:
        load_template(settings.templates_dir, TASKS[key].template)

    run_dir = new_run_dir(settings.runs_dir, params.tema)
    summary = RunSummary(run_dir)
    meta: dict[str, Any] = {
        "tool": f"biblio-llm {__version__}",
        "date": datetime.now().isoformat(timespec="seconds"),
        "params": params.__dict__,
        "model": settings.model,
        "model_digest": digest,
        "ollama_version": client.version(),
        "options": {
            "think": settings.think,
            "temperature": settings.temperature,
            "seed": settings.seed,
            "num_ctx": settings.num_ctx,
            "num_predict": settings.num_predict,
        },
        "tasks": {},
    }
    for key in task_keys:
        task = TASKS[key]
        logger.info("Plantilla %s (%s)…", key.upper(), task.template)
        try:
            meta["tasks"][task.folder] = run_task(
                client, settings, task, params, run_dir
            )
            logger.info("  → %s/%s", task.folder, task.output)
        except OllamaError as exc:
            summary.errors[task.folder] = str(exc)
            meta["tasks"][task.folder] = {"error": str(exc)}
            logger.error("  %s: %s", task.folder, exc)
        _write(
            run_dir / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
        )
    summary.validation = validate_run(run_dir)
    return summary
