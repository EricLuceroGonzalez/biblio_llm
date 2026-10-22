"""Carga y relleno de plantillas."""

from __future__ import annotations

import re
from pathlib import Path

_PLACEHOLDER = re.compile(r"\{([A-Z_]+)\}")


class TemplateError(Exception):
    """Plantilla inexistente o con marcadores sin rellenar."""


def render(template: str, values: dict[str, str]) -> str:
    """Sustituye ``{CLAVE}`` por su valor. Falla si queda alguno sin rellenar.

    Se usa una sustitución explícita (no ``str.format``) para que las llaves de
    los ejemplos JSON de una plantilla no se interpreten como marcadores.
    """
    text = template
    for key, value in values.items():
        text = text.replace(f"{{{key}}}", value)
    missing = sorted(set(_PLACEHOLDER.findall(text)))
    if missing:
        raise TemplateError(f"marcadores sin valor: {', '.join(missing)}")
    return text


def load_template(templates_dir: Path, filename: str) -> str:
    path = templates_dir / filename
    if not path.is_file():
        raise TemplateError(f"no existe la plantilla {path}")
    return path.read_text(encoding="utf-8")


def system_prompt(idioma: str) -> str:
    """Mensaje de sistema: refuerza el idioma, que es lo que más se desobedece."""
    return (
        "Eres un asistente de investigación académica riguroso. "
        f"Escribe TODOS los valores de texto en {idioma}, aunque el tema o las "
        "instrucciones estén en otro idioma. Nunca traduzcas las claves del JSON. "
        "Responde únicamente con el JSON que se te pide."
    )
