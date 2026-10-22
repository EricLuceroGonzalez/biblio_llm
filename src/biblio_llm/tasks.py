"""Las tres tareas (plantillas A, B y C) y cómo se guarda cada salida."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from biblio_llm import schemas

GROUP_LABELS = {
    "nucleo": "Nucleo",
    "sinonimos": "Sinonimos",
    "exclusiones": "Exclusiones",
}


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def keywords_to_markdown(data: dict[str, Any]) -> str:
    """Tabla Markdown con cabeceras fijas (no dependen del idioma del modelo)."""
    lines = ["| Grupo | Termino | Nota |", "|---|---|---|"]
    for key, label in GROUP_LABELS.items():
        for item in data.get(key, []):
            lines.append(
                f"| {label} | {_cell(item['termino'])} | {_cell(item['nota'])} |"
            )
    return "\n".join(lines) + "\n"


def _pretty_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


@dataclass(frozen=True)
class Task:
    key: str  # a | b | c
    folder: str  # subcarpeta dentro del run
    template: str  # archivo en plantillas/
    schema: dict[str, Any]
    output: str  # nombre del archivo final
    render: Callable[[dict[str, Any]], str]


TASKS: dict[str, Task] = {
    "a": Task(
        "a",
        "a_keywords",
        "plantilla_a_keywords.txt",
        schemas.KEYWORDS,
        "tabla_keywords.md",
        keywords_to_markdown,
    ),
    "b": Task(
        "b",
        "b_ficha",
        "plantilla_b_ficha.txt",
        schemas.FICHA,
        "ficha_tema.json",
        _pretty_json,
    ),
    "c": Task(
        "c",
        "c_sota",
        "plantilla_c_sota.txt",
        schemas.PLAN_SOTA,
        "plan_sota.json",
        _pretty_json,
    ),
}
