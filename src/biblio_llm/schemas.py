"""JSON Schema de cada salida.

Se envían a Ollama en el parámetro ``format``: el modelo queda obligado a
producir exactamente esta estructura. Los mismos esquemas se usan después para
validar los archivos guardados.
"""

from __future__ import annotations

from typing import Any


def _strings(n_min: int, n_max: int | None = None) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "minItems": n_min,
        "maxItems": n_max if n_max is not None else n_min,
    }


_KEYWORD = {
    "type": "object",
    "properties": {
        "termino": {"type": "string", "minLength": 1},
        "nota": {"type": "string", "minLength": 1},
    },
    "required": ["termino", "nota"],
    "additionalProperties": False,
}


def _group(n: int) -> dict[str, Any]:
    return {"type": "array", "items": _KEYWORD, "minItems": n, "maxItems": n}


KEYWORDS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "nucleo": _group(5),
        "sinonimos": _group(5),
        "exclusiones": _group(5),
    },
    "required": ["nucleo", "sinonimos", "exclusiones"],
    "additionalProperties": False,
}

FICHA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tema": {"type": "string", "minLength": 1},
        "audiencia": {"type": "string", "minLength": 1},
        "objetivo": {"type": "string", "minLength": 1},
        "alcance": {
            "type": "object",
            "properties": {"incluye": _strings(1, 10), "excluye": _strings(1, 10)},
            "required": ["incluye", "excluye"],
            "additionalProperties": False,
        },
        "subpreguntas": _strings(5),
        "palabras_clave": _strings(5),
        "riesgos_y_sesgos": _strings(3),
    },
    "required": [
        "tema",
        "audiencia",
        "objetivo",
        "alcance",
        "subpreguntas",
        "palabras_clave",
        "riesgos_y_sesgos",
    ],
    "additionalProperties": False,
}

PLAN_SOTA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "titulo": {"type": "string", "minLength": 1},
        "resumen_100_palabras": {"type": "string", "minLength": 1},
        "secciones": {
            "type": "array",
            "minItems": 3,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "nombre": {"type": "string", "minLength": 1},
                    "objetivo": {"type": "string", "minLength": 1},
                    "puntos": _strings(1, 8),
                },
                "required": ["nombre", "objetivo", "puntos"],
                "additionalProperties": False,
            },
        },
        "criterios_de_calidad": _strings(3),
    },
    "required": ["titulo", "resumen_100_palabras", "secciones", "criterios_de_calidad"],
    "additionalProperties": False,
}
