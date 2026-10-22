"""Validación de las salidas de un run (equivale a validate_outputs.py del curso)."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from biblio_llm.tasks import TASKS

EXPECTED_COLUMNS = ["Grupo", "Termino", "Nota"]


def validate_markdown_table(path: Path) -> str:
    if not path.is_file():
        return f"FAIL: No se encontro {path.name}"
    lines = [
        ln
        for ln in path.read_text(encoding="utf-8").strip().splitlines()
        if "|" in ln and "---" not in ln
    ]
    if not lines:
        return "FAIL: No se encontro tabla Markdown valida"
    header = [c.strip() for c in lines[0].split("|") if c.strip()]
    if not all(col in header for col in EXPECTED_COLUMNS):
        return f"FAIL: Columnas esperadas {EXPECTED_COLUMNS}, encontradas {header}"
    return f"PASS: Tabla Markdown con columnas correctas ({len(lines) - 1} terminos)"


def validate_json(path: Path, schema: dict) -> str:
    if not path.is_file():
        return f"FAIL: No se encontro {path.name}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"FAIL: {path.name} no es un JSON valido - {exc}"
    errors = sorted(
        Draft202012Validator(schema).iter_errors(data), key=lambda e: e.path
    )
    if errors:
        detail = "; ".join(
            f"{'/'.join(map(str, e.path)) or '(raiz)'}: {e.message}" for e in errors[:5]
        )
        return f"FAIL: {path.name} no cumple el esquema - {detail}"
    return f"PASS: {path.name} valido segun su esquema"


def validate_run(run_dir: Path) -> list[str]:
    """Valida las salidas presentes en un run y escribe ``verificacion.md``."""
    results = []
    for task in TASKS.values():
        folder = run_dir / task.folder
        if not folder.exists():
            continue  # tarea no ejecutada en este run (p. ej. --only b)
        path = folder / task.output
        if task.key == "a":
            results.append(validate_markdown_table(path))
            raw_json = folder / "respuesta.json"
            if raw_json.exists():
                results.append(validate_json(raw_json, task.schema))
        else:
            results.append(validate_json(path, task.schema))
    (run_dir / "verificacion.md").write_text(
        "# Resultado de validacion\n\n" + "\n".join(f"- {r}" for r in results) + "\n",
        encoding="utf-8",
    )
    return results
