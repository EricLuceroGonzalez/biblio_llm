"""Extracción, agrupación y vacíos a partir de abstracts (Módulo 4, secciones 3–5).

Flujo de ``biblio-llm extract``:

1. **Fichas** (plantilla 4.1): una ficha JSON por artículo. ``paper_id``,
   ``title``, ``year`` y ``source`` salen del archivo de entrada, no del modelo,
   así que no se pueden inventar. El modelo solo extrae los campos de contenido
   y, si un dato no aparece en el abstract, escribe ``not_specified``.
2. **Verificación**: cada ficha se valida con JSON Schema (determinista) y, salvo
   ``--sin-verificador``, con el verificador PASS/FAIL (plantilla 4.3), que
   compara la ficha con su abstract.
3. **Tabla** (3.1): ``tabla.md`` se genera desde las fichas, sin otra llamada al
   modelo; así tabla y fichas nunca se contradicen.
4. **Grupos** (4.4) y **vacíos** (4.5): con 5 fichas o más. Se comprueba con
   código que cada ``paper_id`` citado exista y que cada evidencia apunte a un
   campo real de una ficha del grupo.

Todo queda en una carpeta nueva de ``runs/``; nunca se sobrescribe nada.
"""

from __future__ import annotations

import json
import logging
import math
import re
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
MIN_FICHAS_GRUPOS = 5  # la guía recomienda agrupar con 5–30 fichas
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
            "minItems": 3,
            "maxItems": 6,
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
        "notes": {"type": "string"},
    },
    "required": ["groups", "notes"],
}

GAPS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "gaps": {
            "type": "array",
            "minItems": 2,
            "maxItems": 4,
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
        "Group the papers into 3–6 groups by problem, method, data or type of "
        "contribution (not just by shared words). Each group must be justified "
        "using ONLY the extracted fields.\n"
        "- rule: the explicit criterion (e.g. same problem or same method).\n"
        "- papers: the paper_id of every paper in the group.\n"
        "- evidence: items of the form P01.method=<text copied from that field>, "
        "only for papers in the group. Valid fields: "
        + ", ".join(EVIDENCE_FIELDS)
        + ".\n"
        "- notes: ambiguous cases, or papers that fit no group.\n\n"
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
        "From the groups, identify 2–4 gaps. Do not invent. Classify each gap as "
        + " / ".join(GAP_TYPES)
        + ".\n"
        "- description: a brief description.\n"
        "- evidence: what is missing and why, based on the records; cite paper_id "
        "(P01) or group labels.\n"
        "- how_to_cover: 1–2 concrete ideas.\n\n"
        f"GROUPS:\n{json.dumps(groups, ensure_ascii=False, indent=2)}\n\n"
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
    """(problemas, avisos) de los grupos frente a las fichas.

    Problemas (FAIL): esquema, ``paper_id`` inexistente, evidencia mal formada,
    sobre un campo inexistente o sobre un artículo que no está en el grupo.
    Avisos: evidencia que no es copia literal del campo, artículos sin grupo.
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
            problems.append(f"grupo {label!r}: {pid} no existe")
        grouped |= members & fichas.keys()
        for item in g["evidence"]:
            match = _EVIDENCE_RE.match(item)
            if not match:
                problems.append(
                    f"grupo {label!r}: evidencia sin formato P01.campo=…: {item!r}"
                )
                continue
            pid, fname, quoted = match.groups()
            if pid not in members:
                problems.append(
                    f"grupo {label!r}: la evidencia cita {pid}, que no está en el grupo"
                )
            elif fname not in EVIDENCE_FIELDS:
                problems.append(
                    f"grupo {label!r}: campo inexistente {fname!r} en {item!r}"
                )
            elif pid in fichas:
                value = fichas[pid][fname]
                value = " ".join(value) if isinstance(value, list) else value
                if _norm(quoted.rstrip(".… ")) not in _norm(value):
                    warnings.append(
                        f"grupo {label!r}: {pid}.{fname} no es copia literal "
                        "de la ficha"
                    )
    for pid in sorted(fichas.keys() - grouped):
        warnings.append(f"{pid} no está en ningún grupo")
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
            warnings.append(f"vacío {n}: la evidencia no cita ningún artículo ni grupo")
    return problems, warnings


# ------------------------------------------------------------------ salidas
def _cell(value: Any) -> str:
    text = ", ".join(value) if isinstance(value, list) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def table_markdown(fichas: list[dict[str, Any]]) -> str:
    """Tabla mínima de la guía (3.1); ``key_result`` es el campo ``results``."""
    columns = [
        ("paper_id", "paper_id"),
        ("title", "title"),
        ("problem", "problem"),
        ("method", "method"),
        ("data_setup", "data_setup"),
        ("key_result", "results"),
        ("limitations", "limitations"),
        ("keywords", "keywords"),
    ]
    lines = [
        "| " + " | ".join(c for c, _ in columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for f in fichas:
        lines.append("| " + " | ".join(_cell(f[key]) for _, key in columns) + " |")
    return "\n".join(lines) + "\n"


def gaps_markdown(gaps: dict[str, Any]) -> str:
    lines = ["## Vacíos detectados", ""]
    for gap in gaps["gaps"]:
        lines += [
            f"- [{gap['type']}] {gap['description']}",
            f"  - evidencia: {gap['evidence']}",
            f"  - cómo se podría cubrir: {'; '.join(gap['how_to_cover'])}",
        ]
    lines += [
        "",
        "> Antes de declarar un vacío, comprueba si es real o si lo causa la "
        "búsqueda (sinónimos, otras bases de datos, otros venues).",
    ]
    return "\n".join(lines) + "\n"


def _status_line(problems: list[str], warnings: list[str]) -> str:
    head = "PASS" if not problems else "FAIL"
    return head + (f" ({len(warnings)} avisos)" if warnings else "")


def verification_markdown(
    results: list[FichaResult],
    groups_check: tuple[list[str], list[str]] | None,
    gaps_check: tuple[list[str], list[str]] | None,
    data_warnings: list[str],
) -> str:
    lines = [
        "# Verificación",
        "",
        "## Fichas",
        "",
        "| paper_id | estado | JSON Schema | verificador LLM |",
        "|---|---|---|---|",
    ]
    for r in results:
        schema = "—" if r.ficha is None else ("PASS" if not r.schema else "FAIL")
        state = r.status if not r.error else f"{r.status}: {r.error}"
        lines.append(
            f"| {r.paper_id} | {_cell(state)} | {schema} | {_cell(r.verifier or '—')} |"
        )
    detail = [f"- {r.paper_id} · {e}" for r in results for e in r.schema]
    if detail:
        lines += ["", "Errores de JSON Schema:", "", *detail]
    for title, check in (("Grupos", groups_check), ("Vacíos", gaps_check)):
        lines += ["", f"## {title}", ""]
        if check is None:
            lines.append("No se generaron.")
            continue
        problems, warnings = check
        lines.append(f"**{_status_line(problems, warnings)}**")
        lines += [f"- ✗ {p}" for p in problems] + [f"- ⚠ {w}" for w in warnings]
    if data_warnings:
        lines += ["", "## Datos de entrada", "", *(f"- ⚠ {w}" for w in data_warnings)]
    return "\n".join(lines) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    _write(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


# ------------------------------------------------------------------ proceso
def client_for_prompt(client: OllamaClient, *texts: str) -> OllamaClient:
    """Cliente con ``num_ctx`` suficiente para un prompt largo (grupos, vacíos).

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
    groups_check: tuple[list[str], list[str]] | None = None
    gaps_check: tuple[list[str], list[str]] | None = None
    n_groups: int = 0
    n_gaps: int = 0
    interrupted: bool = False

    def count(self, status: str) -> int:
        return sum(r.status == status for r in self.results)

    @property
    def ok(self) -> bool:
        checks = [c for c in (self.groups_check, self.gaps_check) if c]
        return (
            not self.interrupted
            and self.count("ERROR") == 0
            and all(
                not r.schema and not r.verifier.startswith(("FAIL", "ERROR"))
                for r in self.results
            )
            and all(not problems for problems, _ in checks)
        )


def extract(
    settings: Settings,
    input_path: Path,
    *,
    yes_columns: list[str] | None = None,
    limit: int | None = None,
    idioma: str = "English",
    verify: bool = True,
    group: bool = True,
    client: OllamaClient | None = None,
) -> ExtractSummary:
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
    _write(
        run_dir / "prompts" / "ficha.txt",
        f"### SYSTEM\n{ficha_system(idioma)}\n\n### USER\n"
        + ficha_user("{PAPER_ID}", "{TITLE}", "{ABSTRACT}")
        + "\n",
    )
    _write(
        run_dir / "prompts" / "verificador.txt",
        f"### SYSTEM\n{VERIFIER_SYSTEM}\n\n### USER\n"
        + verifier_user("{ABSTRACT}", {"...": "ficha"})
        + "\n",
    )
    for p in papers:
        _write(
            run_dir / "inputs" / f"{p.paper_id}.txt",
            f"title: {p.title}\nyear: {p.year or ''}\ndoi: {p.doi}\nurl: {p.url}\n"
            + (f"original_id: {p.original_id}\n" if p.original_id else "")
            + f"\nabstract:\n{p.abstract}\n",
        )
    logger.info(
        "Artículos: %d · idioma: %s · verificador: %s", len(papers), idioma, verify
    )

    summary = ExtractSummary(run_dir, [])
    try:
        for n, paper in enumerate(papers, start=1):
            logger.info(
                "[%d/%d] %s %s", n, len(papers), paper.paper_id, paper.title[:60]
            )
            result = extract_one(client, paper, idioma)
            if result.ficha is not None:
                _write_json(
                    run_dir / "fichas_json" / f"{paper.paper_id}.json", result.ficha
                )
                if verify:
                    result.verifier = verify_one(client, paper.abstract, result.ficha)
                    logger.info("    verificador → %s", result.verifier[:70])
            else:
                logger.warning("    %s: %s", result.status, result.error)
            summary.results.append(result)
    except KeyboardInterrupt:
        summary.interrupted = True
        logger.warning("Interrumpido: se guardan las fichas ya extraídas.")

    fichas = [r.ficha for r in summary.results if r.ficha is not None]
    _write(run_dir / "tabla.md", table_markdown(fichas))

    if group and not summary.interrupted:
        if len(fichas) < MIN_FICHAS_GRUPOS:
            logger.warning(
                "Solo hay %d fichas: hacen falta %d para agrupar; "
                "se omiten grupos y vacíos.",
                len(fichas),
                MIN_FICHAS_GRUPOS,
            )
        else:
            _group_and_gaps(client, summary, fichas, idioma)

    _write(
        run_dir / "verificacion.md",
        verification_markdown(
            summary.results, summary.groups_check, summary.gaps_check, data_warnings
        ),
    )
    _write_json(
        run_dir / "meta.json",
        {
            "tool": f"biblio-llm {__version__}",
            "date": datetime.now().isoformat(timespec="seconds"),
            "input": {
                "path": str(input_path),
                "sha256": _sha256(input_path),
                "yes_columns": yes_columns or [],
                "limit": limit,
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
                p.paper_id: {
                    "original_id": p.original_id,
                    "doi": p.doi,
                    "title": p.title,
                }
                for p in papers
            },
            "fichas": {r.paper_id: r.status for r in summary.results},
            "groups": summary.n_groups,
            "gaps": summary.n_gaps,
            "interrupted": summary.interrupted,
        },
    )
    return summary


def _group_and_gaps(
    client: OllamaClient,
    summary: ExtractSummary,
    fichas: list[dict[str, Any]],
    idioma: str,
) -> None:
    run_dir = summary.run_dir
    by_id = {f["paper_id"]: f for f in fichas}
    records = [_record(f) for f in fichas]

    logger.info("Agrupando %d fichas…", len(fichas))
    feedback, groups = "", None
    for attempt in range(2):  # un segundo intento con los problemas como feedback
        user = groups_user(records, feedback)
        _write(
            run_dir / "prompts" / "grupos.txt",
            f"### SYSTEM\n{groups_system(idioma)}\n\n### USER\n{user}\n",
        )
        try:
            groups = _chat_json(
                client_for_prompt(client, user),
                groups_system(idioma),
                user,
                GROUPS_SCHEMA,
            )
        except OllamaError as exc:
            summary.groups_check = ([f"el modelo no devolvió grupos: {exc}"], [])
            return
        summary.groups_check = validate_groups(groups, by_id)
        if not summary.groups_check[0]:
            break
        if attempt == 0:
            logger.warning(
                "Grupos con problemas, se reintenta: %s", summary.groups_check[0]
            )
            feedback = "\n".join(f"- {p}" for p in summary.groups_check[0])
    assert groups is not None
    _write_json(run_dir / "grupos.json", groups)
    summary.n_groups = len(groups.get("groups", []))
    if summary.groups_check and summary.groups_check[0]:
        logger.warning("Los grupos siguen con problemas; no se buscan vacíos.")
        return

    logger.info("Buscando vacíos…")
    user = gaps_user(groups, records)
    _write(
        run_dir / "prompts" / "vacios.txt",
        f"### SYSTEM\n{gaps_system(idioma)}\n\n### USER\n{user}\n",
    )
    try:
        gaps = _chat_json(
            client_for_prompt(client, user), gaps_system(idioma), user, GAPS_SCHEMA
        )
    except OllamaError as exc:
        summary.gaps_check = ([f"el modelo no devolvió vacíos: {exc}"], [])
        return
    labels = [g["label"] for g in groups["groups"]]
    summary.gaps_check = validate_gaps(gaps, by_id, labels)
    _write_json(run_dir / "vacios.json", gaps)
    if not schema_errors(gaps, GAPS_SCHEMA):
        _write(run_dir / "vacios.md", gaps_markdown(gaps))
        summary.n_gaps = len(gaps["gaps"])


# ----------------------------------------------------------------------- CLI
def add_parser(sub: Any) -> None:
    """Registra el subcomando ``extract`` en la CLI."""
    e = sub.add_parser(
        "extract", help="fichas JSON, tabla, grupos y vacíos a partir de abstracts"
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
        help="solo fichas y tabla, sin grupos ni vacíos",
    )
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
    for name, check, n in (
        ("Grupos", s.groups_check, s.n_groups),
        ("Vacíos", s.gaps_check, s.n_gaps),
    ):
        if check is not None:
            print(f"{name}: {n} · {_status_line(*check)}")
    print(f"\nResultado en {s.run_dir} (detalle en verificacion.md)")
    return 0 if s.ok else 1
