"""Extracción, temas, vacíos y síntesis a partir de abstracts (Módulo 4).

Sigue la Guía Operativa (plantillas 4.1–4.5) y la Guía Práctica (pasos 1–7 y
formato de entrega). Flujo de ``biblio-llm extract``:

1. **Fichas** (plantilla 4.1): una ficha JSON por artículo. ``paper_id``,
   ``title``, ``year`` y ``source`` salen del archivo de entrada, no del modelo,
   así que no se pueden inventar. El modelo solo extrae los campos de contenido
   y, si un dato no aparece en el abstract, escribe ``not_specified``.
2. **Verificación**: cada ficha se valida con JSON Schema (determinista) y, salvo
   ``--sin-verificador``, con el verificador PASS/FAIL (plantilla 4.3), que
   compara la ficha con su abstract.
3. **Normalización** (paso 4): variantes de una misma keyword (mayúsculas,
   guiones, plurales) se unifican en todas las fichas; ``normalizacion.md``
   registra cada fusión.
4. **Tabla comparativa** (paso 3): se genera desde las fichas, sin otra llamada
   al modelo; así tabla y fichas nunca se contradicen.
5. **Temas** (paso 5, plantilla 4.4) y **familias de método**, en una llamada;
   con 3 fichas o más. Se comprueba con código que cada ``paper_id`` exista y que
   cada evidencia apunte a un campo real de una ficha del tema.
6. **Vacíos** (paso 6, plantilla 4.5) y **síntesis** de 5–10 líneas.
7. **Matriz de cobertura** (paso 7): tema × método, generada con código a partir
   de los temas y las familias de método.

Todo queda en una carpeta nueva de ``runs/``; nunca se sobrescribe nada. Con
``--entrega`` se copia además lo que pide la Guía Práctica, con sus nombres.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator

from biblio_llm import __version__
from biblio_llm.classify import (
    ClassifyError,
    _find_column,
    _sha256,
    _text,
    load_articles,
)
from biblio_llm.config import Settings
from biblio_llm.ollama_client import OllamaClient, OllamaError
from biblio_llm.runner import new_run_dir

logger = logging.getLogger(__name__)

NS = "not_specified"
CONTENT_FIELDS = [
    "problem",
    "contribution",
    "method",
    "data_setup",
    "results",
    "limitations",
]
EVIDENCE_FIELDS = [*CONTENT_FIELDS, "keywords"]
GAP_TYPES = ["coverage", "method", "evidence", "generalization", "consensus"]
MIN_FICHAS_GRUPOS = 3  # la Guía Práctica trabaja con 3 artículos
TEMAS = (3, 5)  # temas transversales (Guía Práctica, paso 5)
VACIOS = (2, 3)  # paso 6
METODOS = (2, 6)  # familias de método para la matriz
LINEAS_SINTESIS = (5, 10)  # entregable "5–10 líneas de síntesis"
TABLE_COLUMNS = [
    "paper_id",
    "title",
    "problem",
    "method",
    "data_setup",
    "key_result",
    "limitations",
    "keywords",
]
MAX_KEYWORDS = 8
ANSWER_RESERVE = 4096  # tokens para razonamiento + respuesta en grupos/vacíos

ID_COLUMNS = ["paper_id", "id", "ID", "Id", "PMID"]
YEAR_COLUMNS = ["Year", "year", "Año", "año", "Ano", "Publication Year", "PubYear"]
DOI_COLUMNS = ["DOI", "doi", "Doi"]
URL_COLUMNS = ["URL", "url", "Link", "link"]

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PAPER_RE = re.compile(r"\bP\d{2,}\b")
_EVIDENCE_RE = re.compile(r"^\s*(P\d{2,})\.([a-z_]+)\s*=\s*(.*)$", re.DOTALL)

# --------------------------------------------------------------- esquemas
_STR = {"type": "string", "minLength": 1}

FICHA_LLM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **{f: {"type": "string"} for f in CONTENT_FIELDS},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": [*CONTENT_FIELDS, "keywords"],
}

FICHA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "paper_id": {"type": "string", "pattern": r"^P\d{2,}$"},
        "title": _STR,
        "year": {"type": ["integer", "null"]},
        "source": {
            "type": "object",
            "properties": {"doi": {"type": "string"}, "url": {"type": "string"}},
            "required": ["doi", "url"],
            "additionalProperties": False,
        },
        **{f: _STR for f in CONTENT_FIELDS},
        "keywords": {
            "type": "array",
            "items": _STR,
            "minItems": 3,
            "maxItems": MAX_KEYWORDS,
        },
    },
    "required": ["paper_id", "title", "year", "source", *EVIDENCE_FIELDS],
    "additionalProperties": False,
}

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}

GROUPS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "minItems": TEMAS[0],
            "maxItems": TEMAS[1],
            "items": {
                "type": "object",
                "properties": {
                    "label": _STR,
                    "rule": _STR,
                    "papers": {"type": "array", "items": _STR, "minItems": 1},
                    "evidence": {"type": "array", "items": _STR, "minItems": 1},
                },
                "required": ["label", "rule", "papers", "evidence"],
            },
        },
        "methods": {
            "type": "array",
            "minItems": METODOS[0],
            "maxItems": METODOS[1],
            "items": {
                "type": "object",
                "properties": {
                    "label": _STR,
                    "papers": {"type": "array", "items": _STR, "minItems": 1},
                },
                "required": ["label", "papers"],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["groups", "methods", "notes"],
}

GAPS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "gaps": {
            "type": "array",
            "minItems": VACIOS[0],
            "maxItems": VACIOS[1],
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": GAP_TYPES},
                    "description": _STR,
                    "evidence": _STR,
                    "how_to_cover": {
                        "type": "array",
                        "items": _STR,
                        "minItems": 1,
                        "maxItems": 2,
                    },
                },
                "required": ["type", "description", "evidence", "how_to_cover"],
            },
        }
    },
    "required": ["gaps"],
}

SYNTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": _STR,
            "minItems": LINEAS_SINTESIS[0],
            "maxItems": LINEAS_SINTESIS[1],
        }
    },
    "required": ["lines"],
}


def schema_errors(data: Any, schema: dict[str, Any]) -> list[str]:
    """Errores de JSON Schema legibles (``ruta: mensaje``); lista vacía si es válido."""
    errors = sorted(
        Draft202012Validator(schema).iter_errors(data), key=lambda e: list(e.path)
    )
    return [
        f"{'.'.join(str(p) for p in e.path) or '(raíz)'}: {e.message}" for e in errors
    ]


# ---------------------------------------------------------------- prompts
def ficha_system(idioma: str) -> str:
    return (
        "You are an assistant that extracts information from academic papers. "
        "Do NOT summarise. Do NOT invent. If a piece of information does not "
        f'appear in the text, use "{NS}". Return ONLY valid JSON. '
        f"Write the values in {idioma}."
    )


def ficha_user(paper_id: str, title: str, abstract: str) -> str:
    return (
        "Extract these fields from the abstract:\n"
        "- problem: the question or problem the paper addresses (1–2 sentences).\n"
        "- contribution: what the paper contributes (1 sentence).\n"
        "- method: the main approach (1 sentence).\n"
        "- data_setup: data, scenario or protocol, only if stated.\n"
        "- results: the main reported result (1 sentence), only if stated.\n"
        "- limitations: limitations stated explicitly by the authors, only if stated.\n"
        f"- keywords: 3–{MAX_KEYWORDS} short keywords.\n"
        f'Any field not stated in the abstract must be "{NS}".\n\n'
        f"INPUT:\npaper_id: {paper_id}\ntitle: {title}\nabstract: {abstract}"
    )


VERIFIER_SYSTEM = (
    "Act as a strict verifier. You compare an extraction record with the abstract "
    "it was extracted from and answer only PASS or FAIL with a brief reason."
)


def verifier_user(abstract: str, ficha: dict[str, Any]) -> str:
    return (
        "CRITERIA:\n"
        "- If required keys are missing -> FAIL\n"
        "- If a field contains information that is clearly invented (not present "
        "in the abstract) or contradicts it -> FAIL\n"
        f'- A field marked "{NS}" is never a reason to FAIL.\n'
        "- Otherwise -> PASS\n\n"
        f"ABSTRACT:\n{abstract}\n\n"
        f"RECORD:\n{json.dumps(ficha, ensure_ascii=False, indent=2)}"
    )


def groups_system(idioma: str) -> str:
    return (
        "You group academic papers using only the records extracted from them. "
        f"Do not invent. Return ONLY valid JSON. Write in {idioma}."
    )


def groups_user(records: list[dict[str, Any]], feedback: str = "") -> str:
    text = (
        f"1. groups: propose {TEMAS[0]}–{TEMAS[1]} cross-cutting themes by problem, "
        "method, data or type of contribution (not just by shared words). A paper "
        "may support more than one theme. Each theme must be justified using ONLY "
        "the extracted fields.\n"
        "   - label: a short name.\n"
        "   - rule: the explicit criterion (e.g. same problem or same method).\n"
        "   - papers: the paper_id of every paper that supports the theme.\n"
        "   - evidence: at least one item for every paper in the theme, of the "
        "form P01.method=<text copied from that field>. Valid fields: "
        + ", ".join(EVIDENCE_FIELDS)
        + ".\n"
        f"2. methods: {METODOS[0]}–{METODOS[1]} method families (e.g. exact, "
        "metaheuristic, hybrid, machine learning) based on the method field. "
        "Put every paper in exactly one family.\n"
        "3. notes: ambiguous cases, or papers that fit no theme.\n\n"
        f"INPUT (records):\n{json.dumps(records, ensure_ascii=False, indent=2)}"
    )
    if feedback:
        text += "\n\nYour previous answer had these problems; fix them:\n" + feedback
    return text


def gaps_system(idioma: str) -> str:
    return (
        "You identify research gaps from grouped paper records. Do not invent. "
        f"Return ONLY valid JSON. Write in {idioma}."
    )


def gaps_user(groups: dict[str, Any], records: list[dict[str, Any]]) -> str:
    return (
        f"From the themes, identify {VACIOS[0]}–{VACIOS[1]} gaps. Do not invent. "
        "Use ONLY what appears in the records. Classify each gap as "
        + " / ".join(GAP_TYPES)
        + ".\n"
        "- description: a brief description.\n"
        "- evidence: what is missing and why, based on the records; cite paper_id "
        "(P01) or theme labels.\n"
        "- how_to_cover: 1–2 concrete ideas.\n\n"
        f"GROUPS:\n{json.dumps(groups, ensure_ascii=False, indent=2)}\n\n"
        f"RECORDS:\n{json.dumps(records, ensure_ascii=False, indent=2)}"
    )


def synthesis_system(idioma: str) -> str:
    return (
        "You write a short synthesis of a mini state of the art from extracted "
        f"records, themes and gaps. Do not invent. Return ONLY valid JSON. "
        f"Write in {idioma}."
    )


def synthesis_user(
    groups: dict[str, Any], gaps: dict[str, Any], records: list[dict[str, Any]]
) -> str:
    return (
        f"Write a synthesis of {LINEAS_SINTESIS[0]}–{LINEAS_SINTESIS[1]} lines "
        "(one sentence each) covering: what the papers address, the main themes "
        "and methods, what the evidence shows, and the main gaps. Cite paper_id "
        "(P01) for every claim about specific papers. Use ONLY the information "
        "below.\n\n"
        f"THEMES:\n{json.dumps(groups, ensure_ascii=False, indent=2)}\n\n"
        f"GAPS:\n{json.dumps(gaps, ensure_ascii=False, indent=2)}\n\n"
        f"RECORDS:\n{json.dumps(records, ensure_ascii=False, indent=2)}"
    )


# ---------------------------------------------------------------- entrada
@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    year: int | None
    doi: str
    url: str
    original_id: str = ""


def normalize_doi(value: str) -> str:
    """DOI limpio (``10.xxxx/...``) o ``''`` si la celda no contiene un DOI."""
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I)
    return doi if _DOI_RE.match(doi) else ""


def parse_year(value: Any) -> int | None:
    if isinstance(value, float) and not math.isnan(value):
        return int(value)
    if isinstance(value, int):
        return value
    match = _YEAR_RE.search(_text(value))
    return int(match.group(0)) if match else None


def select_rows(df: pd.DataFrame, yes_columns: list[str]) -> pd.DataFrame:
    """Filas cuya respuesta empieza por YES en todas las columnas indicadas."""
    for column in yes_columns:
        if column not in df.columns:
            raise ClassifyError(
                f"no existe la columna {column!r}. Columnas: {list(df.columns)}"
            )
        mask = df[column].map(lambda v: _text(v).upper().startswith("YES"))
        df = df[mask]
    return df


def load_papers(
    path: Path, yes_columns: list[str] | None = None, limit: int | None = None
) -> tuple[list[Paper], list[str]]:
    """Artículos de un .xlsx/.csv con ``paper_id`` P01, P02… y avisos de datos."""
    df, abstract_col, title_col = load_articles(path)
    df = select_rows(df, yes_columns or [])
    if limit:
        df = df.head(limit)
    if df.empty:
        raise ClassifyError("no queda ningún artículo que procesar")
    cols = {
        "id": _find_column(df, ID_COLUMNS),
        "year": _find_column(df, YEAR_COLUMNS),
        "doi": _find_column(df, DOI_COLUMNS),
        "url": _find_column(df, URL_COLUMNS),
    }
    width = max(2, len(str(len(df))))
    papers, warnings = [], []
    for n, (_, row) in enumerate(df.iterrows(), start=1):
        pid = f"P{n:0{width}d}"
        raw_doi = _text(row.get(cols["doi"])) if cols["doi"] else ""
        doi = normalize_doi(raw_doi)
        if raw_doi and not doi:
            warnings.append(
                f"{pid}: la columna DOI no contiene un DOI ({raw_doi[:60]!r}); "
                "se deja vacío"
            )
        url = f"https://doi.org/{doi}" if doi else ""
        if not url and cols["url"]:
            url = _text(row.get(cols["url"]))
        papers.append(
            Paper(
                paper_id=pid,
                title=_text(row.get(title_col)) if title_col else "",
                abstract=_text(row.get(abstract_col)),
                year=parse_year(row.get(cols["year"])) if cols["year"] else None,
                doi=doi,
                url=url,
                original_id=_text(row.get(cols["id"])) if cols["id"] else "",
            )
        )
    return papers, warnings


# ---------------------------------------------------------------- llamadas
def _clean(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return NS if text.lower() in {"", "null", "none", "n/a", "na", NS} else text


def normalize_keywords(values: Any) -> list[str]:
    """Sin vacíos ni duplicados (sin distinguir mayúsculas), máximo 8."""
    seen, out = set(), []
    for value in values if isinstance(values, list) else []:
        kw = re.sub(r"\s+", " ", str(value)).strip()
        if kw and kw.lower() != NS and kw.lower() not in seen:
            seen.add(kw.lower())
            out.append(kw)
    return out[:MAX_KEYWORDS]


def _keyword_key(keyword: str) -> str:
    """Clave para detectar variantes: minúsculas, sin guiones, plural simple."""
    words = re.sub(r"[-_/]+", " ", keyword.lower()).split()
    return " ".join(
        w[:-1]
        if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "is", "us"))
        else w
        for w in words
    )


def normalize_across(
    fichas: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Unifica variantes de keywords entre fichas (paso 4 de la Guía Práctica).

    Para cada grupo de variantes (``Genetic Algorithm``, ``genetic-algorithms``…)
    se elige la forma más frecuente; a igualdad, la primera que aparece. Las
    siglas (``GA`` frente a ``genetic algorithm``) no se fusionan: es una
    ambigüedad que decide quien investiga.

    Returns:
        Fichas nuevas y ``{forma elegida: [variantes sustituidas]}``.
    """
    counts: dict[str, dict[str, int]] = {}
    for f in fichas:
        for kw in f["keywords"]:
            forms = counts.setdefault(_keyword_key(kw), {})
            forms[kw] = forms.get(kw, 0) + 1
    canonical = {
        key: max(forms, key=lambda k: (forms[k], -list(forms).index(k)))
        for key, forms in counts.items()
    }
    merged = {
        canonical[key]: sorted(k for k in forms if k != canonical[key])
        for key, forms in counts.items()
        if len(forms) > 1
    }
    out = []
    for f in fichas:
        keywords: list[str] = []
        for kw in f["keywords"]:
            chosen = canonical[_keyword_key(kw)]
            if chosen not in keywords:
                keywords.append(chosen)
        out.append({**f, "keywords": keywords})
    return out, merged


def _chat_json(
    client: OllamaClient,
    system: str,
    user: str,
    schema: dict[str, Any],
    retries: int = 1,
) -> dict[str, Any]:
    """JSON de la respuesta; reintenta si Ollama falla o el JSON no es válido."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            data = json.loads(client.chat(system, user, schema).content)
            if not isinstance(data, dict):
                raise ValueError("la respuesta no es un objeto JSON")
            return data
        except (OllamaError, ValueError) as exc:
            last = exc
            if attempt < retries:
                logger.warning("    Reintentando (error: %s)…", exc)
    raise OllamaError(str(last))


def build_ficha(paper: Paper, extracted: dict[str, Any]) -> dict[str, Any]:
    """Ficha completa: campos del archivo + campos extraídos por el modelo."""
    return {
        "paper_id": paper.paper_id,
        "title": paper.title or NS,
        "year": paper.year,
        "source": {"doi": paper.doi, "url": paper.url},
        **{f: _clean(extracted.get(f)) for f in CONTENT_FIELDS},
        "keywords": normalize_keywords(extracted.get("keywords")),
    }


@dataclass
class FichaResult:
    paper_id: str
    status: str  # OK | ERROR | SKIPPED
    ficha: dict[str, Any] | None = None
    schema: list[str] = field(default_factory=list)
    verifier: str = ""  # "PASS" | "FAIL: ..." | "ERROR: ..." | ""
    error: str = ""


def extract_one(client: OllamaClient, paper: Paper, idioma: str) -> FichaResult:
    if not paper.abstract:
        return FichaResult(paper.paper_id, "SKIPPED", error="abstract vacío")
    try:
        extracted = _chat_json(
            client,
            ficha_system(idioma),
            ficha_user(paper.paper_id, paper.title, paper.abstract),
            FICHA_LLM_SCHEMA,
        )
    except OllamaError as exc:
        return FichaResult(paper.paper_id, "ERROR", error=str(exc))
    ficha = build_ficha(paper, extracted)
    return FichaResult(paper.paper_id, "OK", ficha, schema_errors(ficha, FICHA_SCHEMA))


def verify_one(client: OllamaClient, abstract: str, ficha: dict[str, Any]) -> str:
    try:
        data = _chat_json(
            client, VERIFIER_SYSTEM, verifier_user(abstract, ficha), VERDICT_SCHEMA
        )
    except OllamaError as exc:
        return f"ERROR: {exc}"
    verdict = str(data.get("verdict", "")).strip().upper()
    reason = str(data.get("reason", "")).strip()
    if verdict == "PASS":
        return "PASS"
    return f"FAIL: {reason or 'sin motivo'}"


# ----------------------------------------------------------- comprobaciones
def _norm(text: str) -> str:
    return re.sub(r"[\s\W_]+", " ", text).strip().lower()


def validate_groups(
    groups: dict[str, Any], fichas: dict[str, dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """(problemas, avisos) de los temas y familias de método frente a las fichas.

    Problemas (FAIL): esquema, ``paper_id`` inexistente, evidencia mal formada,
    sobre un campo inexistente o sobre un artículo que no está en el tema.
    Avisos: evidencia que no es copia literal del campo, artículos sin tema,
    artículos sin familia de método o en más de una.
    """
    problems = schema_errors(groups, GROUPS_SCHEMA)
    warnings: list[str] = []
    if problems:
        return problems, warnings
    grouped: set[str] = set()
    for g in groups["groups"]:
        label = g["label"]
        members = set(g["papers"])
        for pid in sorted(members - fichas.keys()):
            problems.append(f"tema {label!r}: {pid} no existe")
        grouped |= members & fichas.keys()
        with_evidence: set[str] = set()
        for item in g["evidence"]:
            match = _EVIDENCE_RE.match(item)
            if not match:
                problems.append(
                    f"tema {label!r}: evidencia sin formato P01.campo=…: {item!r}"
                )
                continue
            pid, fname, quoted = match.groups()
            with_evidence.add(pid)
            if pid not in members:
                problems.append(
                    f"tema {label!r}: la evidencia cita {pid}, que no está en el tema"
                )
            elif fname not in EVIDENCE_FIELDS:
                problems.append(
                    f"tema {label!r}: campo inexistente {fname!r} en {item!r}"
                )
            elif pid in fichas:
                value = fichas[pid][fname]
                value = " ".join(value) if isinstance(value, list) else value
                if _norm(quoted.rstrip(".… ")) not in _norm(value):
                    warnings.append(
                        f"tema {label!r}: {pid}.{fname} no es copia literal de la ficha"
                    )
        missing = sorted((members & fichas.keys()) - with_evidence)
        if missing:
            warnings.append(
                f"tema {label!r}: sin evidencia propia para {', '.join(missing)}"
            )
    for pid in sorted(fichas.keys() - grouped):
        warnings.append(f"{pid} no está en ningún tema")
    families: dict[str, list[str]] = {}
    for m in groups["methods"]:
        for pid in m["papers"]:
            if pid not in fichas:
                problems.append(f"método {m['label']!r}: {pid} no existe")
            else:
                families.setdefault(pid, []).append(m["label"])
    for pid in sorted(fichas):
        labels = families.get(pid, [])
        if not labels:
            warnings.append(f"{pid} no está en ninguna familia de método")
        elif len(labels) > 1:
            warnings.append(f"{pid} está en varias familias de método: {labels}")
    return problems, warnings


def validate_gaps(
    gaps: dict[str, Any], fichas: dict[str, Any], labels: list[str]
) -> tuple[list[str], list[str]]:
    problems = schema_errors(gaps, GAPS_SCHEMA)
    warnings: list[str] = []
    if problems:
        return problems, warnings
    lower_labels = [lbl.lower() for lbl in labels]
    for n, gap in enumerate(gaps["gaps"], start=1):
        cited = set(_PAPER_RE.findall(gap["evidence"]))
        for pid in sorted(cited - fichas.keys()):
            problems.append(f"vacío {n}: cita {pid}, que no existe")
        mentions_group = any(lbl in gap["evidence"].lower() for lbl in lower_labels)
        if not cited and not mentions_group:
            warnings.append(f"vacío {n}: la evidencia no cita ningún artículo ni tema")
    return problems, warnings


def validate_synthesis(
    synthesis: dict[str, Any], fichas: dict[str, Any]
) -> tuple[list[str], list[str]]:
    problems = schema_errors(synthesis, SYNTHESIS_SCHEMA)
    warnings: list[str] = []
    if problems:
        return problems, warnings
    cited: set[str] = set()
    for n, line in enumerate(synthesis["lines"], start=1):
        ids = set(_PAPER_RE.findall(line))
        cited |= ids
        for pid in sorted(ids - fichas.keys()):
            problems.append(f"línea {n}: cita {pid}, que no existe")
    if not cited:
        warnings.append("la síntesis no cita ningún artículo")
    return problems, warnings


# ------------------------------------------------------------------ salidas
Check = tuple[list[str], list[str]]  # (problemas, avisos)


def _cell(value: Any) -> str:
    text = ", ".join(value) if isinstance(value, list) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def table_markdown(fichas: list[dict[str, Any]]) -> str:
    """Tabla comparativa con columnas fijas; ``key_result`` es el campo ``results``."""
    source = {c: c for c in TABLE_COLUMNS} | {"key_result": "results"}
    lines = [
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "|" + "---|" * len(TABLE_COLUMNS),
    ]
    for f in fichas:
        lines.append(
            "| " + " | ".join(_cell(f[source[c]]) for c in TABLE_COLUMNS) + " |"
        )
    return "\n".join(lines) + "\n"


def check_table(text: str) -> list[str]:
    """Problemas de la tabla: cabecera distinta de ``TABLE_COLUMNS`` o filas cojas."""
    rows = [
        [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
        for line in text.splitlines()
        if line.strip()
    ]
    if not rows or rows[0] != TABLE_COLUMNS:
        return [f"la cabecera no es exactamente {TABLE_COLUMNS}"]
    return [
        f"la fila {n} tiene {len(r)} celdas"
        for n, r in enumerate(rows[2:], start=1)
        if len(r) != len(TABLE_COLUMNS)
    ]


def themes_markdown(groups: dict[str, Any]) -> list[str]:
    lines = ["## Temas detectados", ""]
    for n, g in enumerate(groups["groups"], start=1):
        lines += [
            f"### Tema {n}: {g['label']}",
            f"- papers: {', '.join(g['papers'])}",
            f"- regla: {g['rule']}",
            f"- evidencia: {'; '.join(g['evidence'])}",
            "",
        ]
    if groups.get("notes", "").strip():
        lines += [f"> Notas: {groups['notes'].strip()}", ""]
    return lines


def gaps_lines(gaps: dict[str, Any]) -> list[str]:
    lines = ["## Vacíos detectados", ""]
    for gap in gaps["gaps"]:
        lines += [
            f"- [{gap['type']}] {gap['description']}",
            f"  - evidencia: {gap['evidence']}",
            "  - cómo se podría cubrir: "
            + "; ".join(i.strip().rstrip(".") for i in gap["how_to_cover"])
            + ".",
        ]
    return [*lines, ""]


def synthesis_markdown(
    groups: dict[str, Any] | None,
    gaps: dict[str, Any] | None,
    synthesis: dict[str, Any] | None,
) -> str:
    """``sintesis.md``: temas + vacíos + síntesis (formato de la Guía Práctica)."""
    lines = ["# Síntesis: temas derivados y vacíos", ""]
    lines += (
        themes_markdown(groups)
        if groups
        else ["## Temas detectados", "", "No se generaron.", ""]
    )
    lines += (
        gaps_lines(gaps)
        if gaps
        else ["## Vacíos detectados", "", "No se generaron.", ""]
    )
    lines += ["## Síntesis", ""]
    lines += synthesis["lines"] if synthesis else ["No se generó."]
    lines += [
        "",
        "> Antes de declarar un vacío, comprueba si es real o si lo causa la "
        "búsqueda (sinónimos, otras bases de datos, otros venues).",
    ]
    return "\n".join(lines) + "\n"


def matrix_markdown(groups: dict[str, Any], paper_ids: list[str]) -> str:
    """Matriz tema × familia de método; cada celda lista los artículos de ambos."""
    known = set(paper_ids)
    methods = [
        (m["label"], [p for p in m["papers"] if p in known]) for m in groups["methods"]
    ]
    header = ["Tema \\ Método", *(_cell(label) for label, _ in methods)]
    lines = [
        "# Matriz de cobertura: tema × método",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for g in groups["groups"]:
        members = set(g["papers"])
        cells = [", ".join(p for p in papers if p in members) for _, papers in methods]
        lines.append("| " + " | ".join([_cell(g["label"]), *cells]) + " |")
    lines += ["", "Familias de método:", ""]
    lines += [f"- **{label}**: {', '.join(papers) or '—'}" for label, papers in methods]
    lines += [
        "",
        "Una celda vacía indica una combinación tema × método que ningún artículo "
        "cubre: es candidata a vacío de tipo *method* o *coverage*.",
    ]
    return "\n".join(lines) + "\n"


def normalization_markdown(merged: dict[str, list[str]]) -> str:
    lines = ["# Normalización de keywords", ""]
    if merged:
        lines += ["| forma usada | variantes sustituidas |", "|---|---|"]
        lines += [f"| {_cell(k)} | {_cell(v)} |" for k, v in sorted(merged.items())]
    else:
        lines.append("No había variantes que unificar.")
    lines += [
        "",
        "Se unifican mayúsculas, guiones y plurales simples. Las siglas (p. ej. "
        "`GA` frente a `genetic algorithm`) no se fusionan automáticamente: si "
        "hay ambigüedad real, anótala aquí.",
    ]
    return "\n".join(lines) + "\n"


def _status_line(problems: list[str], warnings: list[str]) -> str:
    head = "PASS" if not problems else "FAIL"
    return head + (f" ({len(warnings)} avisos)" if warnings else "")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    _write(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _prompt_file(path: Path, system: str, user: str) -> None:
    _write(path, f"### SYSTEM\n{system}\n\n### USER\n{user}\n")


# ------------------------------------------------------------------ proceso
def client_for_prompt(client: OllamaClient, *texts: str) -> OllamaClient:
    """Cliente con ``num_ctx`` suficiente para un prompt largo (temas, vacíos…).

    Si el prompt más la respuesta no caben en el contexto, Ollama recorta el
    principio del prompt sin avisar: el modelo agruparía sin ver todas las fichas.
    Se estima ~3 caracteres por token (por lo alto), más ``ANSWER_RESERVE`` para
    el razonamiento y la respuesta, y se sube a múltiplos de 4096.
    """
    s = client.s
    needed = sum(len(t) for t in texts) // 3 + ANSWER_RESERVE
    if needed <= s.num_ctx:
        return client
    num_ctx = math.ceil(needed / 4096) * 4096
    logger.info("Prompt largo: num_ctx %d → %d para esta llamada", s.num_ctx, num_ctx)
    return OllamaClient(s.model_copy(update={"num_ctx": num_ctx}), client.session)


def _record(ficha: dict[str, Any]) -> dict[str, Any]:
    """Lo que ve el modelo al agrupar: sin título largo, fuente ni año."""
    return {"paper_id": ficha["paper_id"], **{f: ficha[f] for f in EVIDENCE_FIELDS}}


@dataclass
class ExtractSummary:
    run_dir: Path
    results: list[FichaResult]
    data_warnings: list[str] = field(default_factory=list)
    table_problems: list[str] = field(default_factory=list)
    merged: dict[str, list[str]] = field(default_factory=dict)
    groups: dict[str, Any] | None = None
    gaps: dict[str, Any] | None = None
    synthesis: dict[str, Any] | None = None
    groups_check: Check | None = None
    gaps_check: Check | None = None
    synthesis_check: Check | None = None
    interrupted: bool = False
    entrega: Path | None = None

    def count(self, status: str) -> int:
        return sum(r.status == status for r in self.results)

    @property
    def fichas(self) -> list[dict[str, Any]]:
        return [r.ficha for r in self.results if r.ficha is not None]

    @property
    def checks(self) -> list[tuple[str, Check | None]]:
        return [
            ("Temas y métodos", self.groups_check),
            ("Vacíos", self.gaps_check),
            ("Síntesis", self.synthesis_check),
        ]

    @property
    def ok(self) -> bool:
        return (
            not self.interrupted
            and self.count("ERROR") == 0
            and not self.table_problems
            and all(
                not r.schema and not r.verifier.startswith(("FAIL", "ERROR"))
                for r in self.results
            )
            and all(not c[0] for _, c in self.checks if c)
        )


def verification_markdown(s: ExtractSummary) -> str:
    lines = [
        "# Verificación",
        "",
        "## Fichas",
        "",
        "| paper_id | estado | JSON Schema | verificador LLM |",
        "|---|---|---|---|",
    ]
    for r in s.results:
        schema = "—" if r.ficha is None else ("PASS" if not r.schema else "FAIL")
        state = r.status if not r.error else f"{r.status}: {r.error}"
        lines.append(
            f"| {r.paper_id} | {_cell(state)} | {schema} | {_cell(r.verifier or '—')} |"
        )
    detail = [f"- {r.paper_id} · {e}" for r in s.results for e in r.schema]
    if detail:
        lines += ["", "Errores de JSON Schema:", "", *detail]
    lines += ["", "## Tabla comparativa", ""]
    lines.append(f"**{_status_line(s.table_problems, [])}**")
    lines += [f"- ✗ {p}" for p in s.table_problems]
    for title, check in s.checks:
        lines += ["", f"## {title}", ""]
        if check is None:
            lines.append("No se generaron.")
            continue
        problems, warnings = check
        lines.append(f"**{_status_line(problems, warnings)}**")
        lines += [f"- ✗ {p}" for p in problems] + [f"- ⚠ {w}" for w in warnings]
    if s.data_warnings:
        lines += ["", "## Datos de entrada", "", *(f"- ⚠ {w}" for w in s.data_warnings)]
    return "\n".join(lines) + "\n"


def _ask_checked(
    client: OllamaClient,
    path: Path,
    system: str,
    make_user: Any,
    schema: dict[str, Any],
    validate: Any,
    what: str,
) -> tuple[dict[str, Any] | None, Check]:
    """Pide un JSON y lo valida; si falla, reintenta una vez con los problemas."""
    feedback, data, check = "", None, ([], [])
    for attempt in range(2):
        user = make_user(feedback)
        _prompt_file(path, system, user)
        try:
            data = _chat_json(client_for_prompt(client, user), system, user, schema)
        except OllamaError as exc:
            return data, ([f"el modelo no devolvió {what}: {exc}"], [])
        check = validate(data)
        if not check[0]:
            break
        if attempt == 0:
            logger.warning("%s con problemas, se reintenta: %s", what, check[0])
            feedback = "\n".join(f"- {p}" for p in check[0])
    return data, check


def _with_feedback(text: str, feedback: str) -> str:
    if not feedback:
        return text
    return text + "\n\nYour previous answer had these problems; fix them:\n" + feedback


def _themes_gaps_synthesis(
    client: OllamaClient, s: ExtractSummary, idioma: str
) -> None:
    prompts = s.run_dir / "prompts"
    by_id = {f["paper_id"]: f for f in s.fichas}
    records = [_record(f) for f in s.fichas]

    logger.info("Temas y familias de método (%d fichas)…", len(records))
    s.groups, s.groups_check = _ask_checked(
        client,
        prompts / "temas.txt",
        groups_system(idioma),
        lambda fb: groups_user(records, fb),
        GROUPS_SCHEMA,
        lambda d: validate_groups(d, by_id),
        "temas",
    )
    if s.groups is not None:
        _write_json(s.run_dir / "grupos.json", s.groups)
    if s.groups is None or s.groups_check[0]:
        logger.warning("Los temas tienen problemas; no se buscan vacíos.")
        s.groups = None
        return
    _write(s.run_dir / "matriz_cobertura.md", matrix_markdown(s.groups, list(by_id)))

    logger.info("Vacíos…")
    labels = [g["label"] for g in s.groups["groups"]]
    groups = s.groups
    s.gaps, s.gaps_check = _ask_checked(
        client,
        prompts / "vacios.txt",
        gaps_system(idioma),
        lambda fb: _with_feedback(gaps_user(groups, records), fb),
        GAPS_SCHEMA,
        lambda d: validate_gaps(d, by_id, labels),
        "vacíos",
    )
    if s.gaps is not None:
        _write_json(s.run_dir / "vacios.json", s.gaps)
    if s.gaps is None or s.gaps_check[0]:
        s.gaps = None
        return

    logger.info("Síntesis…")
    gaps = s.gaps
    s.synthesis, s.synthesis_check = _ask_checked(
        client,
        prompts / "sintesis.txt",
        synthesis_system(idioma),
        lambda fb: _with_feedback(synthesis_user(groups, gaps, records), fb),
        SYNTHESIS_SCHEMA,
        lambda d: validate_synthesis(d, by_id),
        "síntesis",
    )
    if s.synthesis_check[0]:
        s.synthesis = None


def extract(
    settings: Settings,
    input_path: Path,
    *,
    yes_columns: list[str] | None = None,
    limit: int | None = None,
    idioma: str = "English",
    verify: bool = True,
    group: bool = True,
    entrega: Path | None = None,
    clasificador: Path | None = None,
    tema: str = "",
    command: str = "",
    client: OllamaClient | None = None,
) -> ExtractSummary:
    if entrega is not None and entrega.exists() and any(entrega.iterdir()):
        raise ClassifyError(f"{entrega} ya existe y no está vacía; elige otro nombre")
    if clasificador is None and (input_path.parent / "clasificador.txt").is_file():
        clasificador = input_path.parent / "clasificador.txt"  # run de classify
    if clasificador is not None and not clasificador.is_file():
        raise ClassifyError(f"no existe {clasificador}")
    papers, data_warnings = load_papers(input_path, yes_columns, limit)
    for w in data_warnings:
        logger.warning(w)

    client = client or OllamaClient(settings)
    digest = client.model_digest()
    if digest is None:
        raise OllamaError(
            f"el modelo {settings.model} no está instalado: "
            f"ejecuta `ollama pull {settings.model}`"
        )

    run_dir = new_run_dir(settings.runs_dir, f"extraccion {input_path.stem}")
    _prompt_file(
        run_dir / "prompts" / "ficha.txt",
        ficha_system(idioma),
        ficha_user("{PAPER_ID}", "{TITLE}", "{ABSTRACT}"),
    )
    _prompt_file(
        run_dir / "prompts" / "verificador.txt",
        VERIFIER_SYSTEM,
        verifier_user("{ABSTRACT}", {"...": "ficha"}),
    )
    for p in papers:
        _write(
            run_dir / "inputs" / f"{p.paper_id}.txt",
            f"paper_id: {p.paper_id}\ntitle: {p.title}\nyear: {p.year or ''}\n"
            f"doi: {p.doi}\nurl: {p.url}\n"
            + (f"original_id: {p.original_id}\n" if p.original_id else "")
            + f"\nabstract:\n{p.abstract}\n",
        )
    logger.info(
        "Artículos: %d · idioma: %s · verificador: %s", len(papers), idioma, verify
    )

    s = ExtractSummary(run_dir, [], data_warnings)
    try:
        for n, paper in enumerate(papers, start=1):
            logger.info(
                "[%d/%d] %s %s", n, len(papers), paper.paper_id, paper.title[:60]
            )
            result = extract_one(client, paper, idioma)
            if result.ficha is not None and verify:
                result.verifier = verify_one(client, paper.abstract, result.ficha)
                logger.info("    verificador → %s", result.verifier[:70])
            elif result.ficha is None:
                logger.warning("    %s: %s", result.status, result.error)
            s.results.append(result)
    except KeyboardInterrupt:
        s.interrupted = True
        logger.warning("Interrumpido: se guardan las fichas ya extraídas.")

    # Paso 4: normalización entre fichas, antes de la tabla y de agrupar.
    normalized, s.merged = normalize_across(s.fichas)
    by_id = {f["paper_id"]: f for f in normalized}
    for r in s.results:
        if r.ficha is not None:
            r.ficha = by_id[r.paper_id]
            r.schema = schema_errors(r.ficha, FICHA_SCHEMA)
            _write_json(run_dir / "fichas" / f"{r.paper_id}.json", r.ficha)
    _write(run_dir / "normalizacion.md", normalization_markdown(s.merged))
    table = table_markdown(s.fichas)
    s.table_problems = check_table(table)
    _write(run_dir / "tabla_comparativa.md", table)

    if group and not s.interrupted:
        if len(s.fichas) < MIN_FICHAS_GRUPOS:
            logger.warning(
                "Solo hay %d fichas: hacen falta %d para derivar temas; se omiten.",
                len(s.fichas),
                MIN_FICHAS_GRUPOS,
            )
        else:
            _themes_gaps_synthesis(client, s, idioma)
    if s.groups_check is not None:
        _write(
            run_dir / "sintesis.md", synthesis_markdown(s.groups, s.gaps, s.synthesis)
        )

    _write(run_dir / "verificacion.md", verification_markdown(s))
    meta = {
        "tool": f"biblio-llm {__version__}",
        "date": datetime.now().isoformat(timespec="seconds"),
        "command": command,
        "input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
            "yes_columns": yes_columns or [],
            "limit": limit,
            "clasificador": str(clasificador) if clasificador else None,
        },
        "model": settings.model,
        "model_digest": digest,
        "options": {
            "think": settings.think,
            "temperature": settings.temperature,
            "seed": settings.seed,
            "num_ctx": settings.num_ctx,
            "idioma": idioma,
            "verifier": verify,
        },
        "papers": {
            p.paper_id: {"original_id": p.original_id, "doi": p.doi, "title": p.title}
            for p in papers
        },
        "fichas": {r.paper_id: r.status for r in s.results},
        "keywords_merged": s.merged,
        "themes": len(s.groups["groups"]) if s.groups else 0,
        "gaps": len(s.gaps["gaps"]) if s.gaps else 0,
        "interrupted": s.interrupted,
    }
    _write_json(run_dir / "meta.json", meta)

    if entrega is not None and not s.interrupted:
        s.entrega = write_entrega(s, entrega, meta, clasificador, tema)
    return s


# ------------------------------------------------------------------ entrega
ENTREGA_FILES = ["tabla_comparativa.md", "matriz_cobertura.md", "sintesis.md"]


def _check_item(ok: bool | None, text: str) -> str:
    mark = {True: "x", False: " ", None: " "}[ok]
    return f"- [{mark}] {text}" + (" — *no comprobado*" if ok is None else "")


def entrega_readme(
    s: ExtractSummary, name: str, meta: dict[str, Any], tema: str, has_clasif: bool
) -> str:
    n = len(s.fichas)
    schema_ok = sum(1 for r in s.results if r.ficha is not None and not r.schema)
    verdicts = [r.verifier for r in s.results if r.verifier]
    passed = sum(v == "PASS" for v in verdicts)
    opts = meta["options"]
    filtro = meta["input"]["yes_columns"]
    groups_ok = None if s.groups_check is None else not s.groups_check[0]
    gaps_ok = None if s.gaps_check is None else not s.gaps_check[0]
    lines = [
        f"# Módulo 4 — Entrega {name}",
        "",
        "## Contexto",
        "",
        f"- **Tema (Módulo 1):** {tema or '*(completa aquí el tema)*'}",
        f"- **Artículos:** {n} fichas a partir de `{Path(meta['input']['path']).name}`"
        + (f", filtrados con YES en: {', '.join(filtro)}" if filtro else "")
        + ".",
        f"- **Modelo:** `{meta['model']}` en local con Ollama "
        f"(digest `{meta['model_digest'][:12]}`), temperature {opts['temperature']}, "
        f"seed {opts['seed']}, razonamiento {'sí' if opts['think'] else 'no'}.",
        f"- **Fecha:** {meta['date']}",
        "",
        "## Cómo se ejecutó",
        "",
        "Con la herramienta `biblio-llm` (subcomandos `classify` y `extract`):",
        "",
        "```bash",
        meta["command"] or "biblio-llm extract …",
        "```",
        "",
        "1. **Entradas:** una por artículo, con `paper_id` (P01, P02…), título, "
        "abstract y DOI/URL. `paper_id`, `title`, `year` y `source` se copian del "
        "archivo, no los genera el modelo.",
        "2. **Fichas:** plantilla de extracción a JSON de la Guía Operativa (4.1), con "
        "salida forzada por JSON Schema. Lo que no aparece en el abstract queda como "
        "`not_specified`.",
        "3. **Tabla comparativa:** generada con código desde las fichas.",
        "4. **Normalización:** variantes de keywords unificadas entre fichas "
        f"({len(s.merged)} {'fusión' if len(s.merged) == 1 else 'fusiones'}; "
        "detalle en `normalizacion.md` del run).",
        "5. **Temas:** plantilla de agrupación (4.4); cada tema cita evidencia "
        "`P01.campo=…` comprobada con código contra las fichas.",
        "6. **Vacíos:** plantilla de gap analysis (4.5), con tipo, evidencia y "
        "propuesta.",
        "7. **Matriz:** tema × familia de método, generada con código.",
        "",
        "## Contenido",
        "",
        "```",
        f"{name}/",
        "├── README.md              este archivo",
        "├── clasificador.txt       preguntas YES/NO usadas en el cribado"
        + ("" if has_clasif else " (no incluido)"),
        "├── fichas/                "
        + ", ".join(f"{f['paper_id']}.json" for f in s.fichas[:3])
        + (" …" if n > 3 else ""),
        "├── tabla_comparativa.md",
        "├── matriz_cobertura.md",
        "└── sintesis.md            temas derivados + vacíos + síntesis",
        "```",
        "",
        "## Checklist (comprobado automáticamente)",
        "",
        _check_item(
            schema_ok == n and n > 0,
            f"Los {n} JSON son parseables y contienen todas las claves "
            f"(JSON Schema: {schema_ok}/{n}).",
        ),
        _check_item(
            not s.table_problems, "La tabla tiene exactamente las columnas definidas."
        ),
        _check_item(
            passed == len(verdicts) if verdicts else None,
            "No hay detalles inventados"
            + (
                f" (verificador PASS/FAIL: {passed}/{len(verdicts)} PASS)."
                if verdicts
                else "."
            ),
        ),
        _check_item(
            groups_ok, "Los temas están apoyados por evidencia (campos concretos)."
        ),
        _check_item(gaps_ok, "Los vacíos indican tipo + justificación + propuesta."),
    ]
    notes = [
        *s.data_warnings,
        *(w for _, c in s.checks if c for w in c[1]),
        *(
            f"{r.paper_id}: {r.verifier}"
            for r in s.results
            if r.verifier.startswith("FAIL")
        ),
    ]
    if notes:
        lines += ["", "## Avisos", "", *(f"- {w}" for w in notes)]
    lines += [
        "",
        "## Pensamiento crítico",
        "",
        "La IA generativa puede completar huecos por estilo. Por eso: regla de "
        "`not_specified`, esquema fijo, verificador PASS/FAIL y comprobaciones con "
        "código de las evidencias. Los vacíos se han revisado frente a la búsqueda: "
        "*(añade aquí tu revisión manual)*.",
    ]
    return "\n".join(lines) + "\n"


def write_entrega(
    s: ExtractSummary,
    dest: Path,
    meta: dict[str, Any],
    clasificador: Path | None,
    tema: str,
) -> Path:
    """Copia lo que pide la Guía Práctica (§12) con sus nombres exactos."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(s.run_dir / "fichas", dest / "fichas", dirs_exist_ok=True)
    for name in ENTREGA_FILES:
        if (s.run_dir / name).is_file():
            shutil.copy2(s.run_dir / name, dest / name)
        else:
            logger.warning("Entrega: falta %s", name)
    if clasificador is not None:
        shutil.copy2(clasificador, dest / "clasificador.txt")
    else:
        logger.warning("Entrega: falta clasificador.txt (usa --clasificador)")
    _write(
        dest / "README.md",
        entrega_readme(s, dest.name, meta, tema, clasificador is not None),
    )
    return dest


# ----------------------------------------------------------------------- CLI
def add_parser(sub: Any) -> None:
    """Registra el subcomando ``extract`` en la CLI."""
    e = sub.add_parser(
        "extract", help="fichas, tabla, temas, vacíos, matriz y síntesis"
    )
    e.add_argument("input", type=Path, help="Excel (.xlsx) o CSV con los artículos")
    e.add_argument(
        "--si",
        action="append",
        default=[],
        metavar="COLUMNA",
        help="solo filas cuya columna empieza por YES (repetible; p. ej. una "
        "pregunta de articulos_clasificado_abstract.xlsx)",
    )
    e.add_argument("--limit", type=int, help="procesar solo los N primeros")
    e.add_argument(
        "--idioma", default="English", help="idioma de las fichas (por defecto English)"
    )
    e.add_argument(
        "--sin-verificador",
        dest="verify",
        action="store_false",
        help="no pasar el verificador PASS/FAIL (la mitad de llamadas)",
    )
    e.add_argument(
        "--sin-grupos",
        dest="group",
        action="store_false",
        help="solo fichas y tabla, sin temas, vacíos, matriz ni síntesis",
    )
    e.add_argument(
        "--entrega",
        type=Path,
        metavar="CARPETA",
        help="copiar la entrega del Módulo 4 a CARPETA (p. ej. M4_entrega_LuceroEric)",
    )
    e.add_argument(
        "--clasificador",
        type=Path,
        help="clasificador.txt para la entrega (por defecto, el del run de "
        "classify si la entrada sale de uno)",
    )
    e.add_argument("--tema", default="", help="tema del Módulo 1, para el README")
    think = e.add_mutually_exclusive_group()
    think.add_argument("--think", dest="think", action="store_true", default=None)
    think.add_argument("--no-think", dest="think", action="store_false")


def run_cli(args: Any, settings: Settings) -> int:
    """Ejecuta ``biblio-llm extract``; devuelve el código de salida."""
    if args.think is not None:
        settings = settings.model_copy(update={"think": args.think})
    try:
        s = extract(
            settings,
            args.input,
            yes_columns=args.si,
            limit=args.limit,
            idioma=args.idioma,
            verify=args.verify,
            group=args.group,
            entrega=args.entrega,
            clasificador=args.clasificador,
            tema=args.tema,
            command="biblio-llm " + shlex.join(sys.argv[1:]),
        )
    except (ClassifyError, OllamaError) as exc:
        print(f"✗ {exc}")
        return 2
    verdicts = [r.verifier for r in s.results if r.verifier]
    print(
        f"\nFichas: {len(s.results)} (OK {s.count('OK')} · ERROR {s.count('ERROR')}"
        f" · SKIPPED {s.count('SKIPPED')})"
        + (" (interrumpido)" if s.interrupted else "")
    )
    schema_ok = sum(1 for r in s.results if r.ficha is not None and not r.schema)
    print(f"JSON Schema: {schema_ok}/{s.count('OK')} PASS")
    if verdicts:
        passed = sum(v == "PASS" for v in verdicts)
        print(f"Verificador LLM: {passed}/{len(verdicts)} PASS")
    print(f"Keywords unificadas: {len(s.merged)}")
    for name, check in s.checks:
        if check is not None:
            print(f"{name}: {_status_line(*check)}")
    print(f"\nResultado en {s.run_dir} (detalle en verificacion.md)")
    if s.entrega:
        print(f"Entrega en {s.entrega}")
    return 0 if s.ok else 1
