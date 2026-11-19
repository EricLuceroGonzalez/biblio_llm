"""Clasificación de abstracts con preguntas YES/NO (Módulo 4 del curso).

Para cada artículo y cada pregunta de ``clasificador.txt`` se pregunta al modelo
si el abstract responde YES o NO. La respuesta está forzada por un JSON Schema,
así que siempre es exactamente ``YES`` o ``NO`` más una explicación breve.

Entrada: el Excel del Módulo 2 (``.xlsx``) o un CSV de biblio-search
(``results.csv`` / ``merged.csv``). Salida: una carpeta nueva en ``runs/`` con
``articulos_clasificado_abstract.xlsx`` (el nombre que espera el Módulo 5), los
prompts y ``meta.json``; nunca se sobrescribe nada.

Formato del Excel (``--formato``):

* ``curso`` (por defecto): una columna por pregunta, como el script del curso:
  ``YES - explicación``, ``NO - explicación``, ``Error: ...`` o
  ``SKIPPED - abstract vacío``. Se filtra con "empieza por YES".
* ``detallado``: dos columnas por pregunta, la respuesta sola y
  ``<pregunta> (motivo)``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from biblio_llm import __version__
from biblio_llm.config import Settings
from biblio_llm.ollama_client import OllamaClient, OllamaError
from biblio_llm.runner import new_run_dir

logger = logging.getLogger(__name__)

OUTPUT_NAME = "articulos_clasificado_abstract.xlsx"
FORMATS = ("curso", "detallado")

# Nombres de columna aceptados (Excel del curso, biblio-search, variantes comunes)
TITLE_COLUMNS = ["Título", "Titulo", "Title", "title", "título", "titulo"]
ABSTRACT_COLUMNS = ["Abstract", "abstract", "Resumen", "resumen", "Summary"]

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["YES", "NO"]},
        "explanation": {"type": "string"},
    },
    "required": ["answer", "explanation"],
}

SYSTEM_PROMPT = (
    "You are a careful research assistant screening academic abstracts for a "
    "systematic review. Answer only from the information in the abstract; if the "
    "abstract does not say it, answer NO. Keep the explanation to one short sentence."
)


def user_prompt(abstract: str, question: str) -> str:
    """El mismo prompt del curso, adaptado a la salida JSON."""
    return (
        f"Based on the following academic abstract, answer the question: '{question}'\n"
        "Answer with 'YES' or 'NO' and a very short explanation.\n\n"
        f"Abstract:\n{abstract}"
    )


class ClassifyError(Exception):
    """Archivo de entrada o de preguntas inválido."""


# ------------------------------------------------------------------- entrada
def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def _read_text(path: Path) -> str:
    """UTF-8 (con o sin BOM); si falla, Windows-1252 (CSV antiguos de Excel)."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def read_csv(path: Path) -> pd.DataFrame:
    """Lee un CSV detectando el separador (``,`` ``;`` o tabulador).

    Excel en español exporta con ``;``; Scopus y biblio-search con ``,``. Si el
    archivo está mal formado, se explica la causa en lugar de mostrar el error
    interno de pandas. No se descartan filas en silencio: en una revisión
    sistemática perder artículos sin saberlo es peor que detenerse.
    """
    text = _read_text(path)
    sample = "\n".join(text.splitlines()[:20])
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        sep = ","
    try:
        return pd.read_csv(io.StringIO(text), sep=sep)
    except pd.errors.ParserError as exc:
        match = re.search(r"Expected (\d+) fields in line (\d+), saw (\d+)", str(exc))
        where = (
            f"la cabecera tiene {match[1]} columnas pero la línea {match[2]} "
            f"tiene {match[3]}"
            if match
            else str(exc)
        )
        name = {",": "coma", ";": "punto y coma", "\t": "tabulador"}[sep]
        raise ClassifyError(
            f"{path.name} no es un CSV válido: {where} (separador detectado: {name}). "
            "Probablemente un campo (p. ej. el abstract) contiene el separador sin ir "
            "entre comillas. Solución: guarda el archivo desde Excel como .xlsx, o "
            "como 'CSV UTF-8', y vuelve a intentarlo."
        ) from exc


def load_articles(path: Path) -> tuple[pd.DataFrame, str, str | None]:
    """Lee .xlsx o .csv y devuelve (tabla, columna de abstract, columna de título)."""
    if not path.is_file():
        raise ClassifyError(f"no existe {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = read_csv(path)
    else:
        raise ClassifyError(f"formato no soportado: {path.suffix} (usa .xlsx o .csv)")
    abstract_col = _find_column(df, ABSTRACT_COLUMNS)
    if abstract_col is None:
        raise ClassifyError(
            f"{path.name} no tiene columna de abstract "
            f"({' / '.join(ABSTRACT_COLUMNS)}). "
            f"Columnas encontradas: {list(df.columns)}"
        )
    return df, abstract_col, _find_column(df, TITLE_COLUMNS)


def load_questions(path: Path) -> list[str]:
    """Una pregunta por línea; se ignoran las vacías y las que empiezan por ``#``."""
    if not path.is_file():
        raise ClassifyError(f"no existe {path}")
    questions = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not questions:
        raise ClassifyError(f"{path.name} no contiene ninguna pregunta")
    return questions


def _text(value: Any) -> str:
    """Celda de pandas → texto ('' si está vacía o es NaN)."""
    return "" if value is None or pd.isna(value) else str(value).strip()


# ------------------------------------------------------------------ pregunta
@dataclass
class Answer:
    answer: str  # YES | NO | SKIPPED (sin abstract) | ERROR
    explanation: str
    seconds: float = 0.0

    def cell(self) -> str:
        """Texto de una sola celda, como en el script del curso."""
        if self.answer == "ERROR":
            return f"Error: {self.explanation}"
        if not self.explanation:
            return self.answer
        return f"{self.answer} - {self.explanation}"


def ask(client: OllamaClient, abstract: str, question: str, retries: int = 1) -> Answer:
    """Pregunta YES/NO sobre un abstract. Nunca lanza: los fallos quedan como ERROR."""
    if not abstract:
        return Answer("SKIPPED", "abstract vacío")
    last_error = ""
    for attempt in range(retries + 1):
        started = time.monotonic()
        try:
            result = client.chat(
                SYSTEM_PROMPT, user_prompt(abstract, question), ANSWER_SCHEMA
            )
            data = json.loads(result.content)
            answer = str(data.get("answer", "")).strip().upper()
            if answer not in {"YES", "NO"}:
                raise ValueError(f"respuesta inesperada: {data.get('answer')!r}")
            return Answer(
                answer,
                str(data.get("explanation", "")).strip(),
                round(time.monotonic() - started, 1),
            )
        except (OllamaError, ValueError) as exc:  # JSONDecodeError es ValueError
            last_error = str(exc)
            if attempt < retries:
                logger.warning("    Reintentando (error: %s)…", exc)
    return Answer("ERROR", last_error)


# ------------------------------------------------------------------- proceso
@dataclass
class ClassifySummary:
    run_dir: Path
    output: Path
    articles: int
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    interrupted: bool = False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(
    settings: Settings,
    input_path: Path,
    questions_path: Path,
    limit: int | None = None,
    client: OllamaClient | None = None,
    fmt: str = "curso",
) -> ClassifySummary:
    """Clasifica cada abstract con cada pregunta y guarda todo en un run nuevo."""
    if fmt not in FORMATS:
        raise ClassifyError(f"formato desconocido {fmt!r} (usa: {', '.join(FORMATS)})")
    df, abstract_col, title_col = load_articles(input_path)
    questions = load_questions(questions_path)
    if limit:
        df = df.head(limit).copy()

    client = client or OllamaClient(settings)
    digest = client.model_digest()
    if digest is None:
        raise OllamaError(
            f"el modelo {settings.model} no está instalado: "
            f"ejecuta `ollama pull {settings.model}`"
        )

    run_dir = new_run_dir(settings.runs_dir, f"clasificacion {input_path.stem}")
    (run_dir / "clasificador.txt").write_text(
        "\n".join(questions) + "\n", encoding="utf-8"
    )
    (run_dir / "prompt.txt").write_text(
        f"### SYSTEM\n{SYSTEM_PROMPT}\n\n### USER (por artículo y pregunta)\n"
        + user_prompt("<abstract>", "<pregunta>")
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Artículos: %d · preguntas: %d · columna de abstract: %r",
        len(df),
        len(questions),
        abstract_col,
    )

    results: dict[str, list[Answer]] = {q: [] for q in questions}
    interrupted = False
    try:
        for n, (_, row) in enumerate(df.iterrows(), start=1):
            title = _text(row.get(title_col)) if title_col else ""
            logger.info("[%d/%d] %s", n, len(df), (title or f"Item {n}")[:70])
            abstract = _text(row.get(abstract_col))
            for question in questions:
                result = ask(client, abstract, question)
                results[question].append(result)
                logger.info("    %s → %s", question[:60], result.answer)
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Interrumpido: se guardan los artículos ya clasificados.")

    done = len(results[questions[0]])
    out = df.head(done).copy()
    for question in questions:
        if fmt == "curso":
            out[question] = [r.cell() for r in results[question]]
        else:
            out[question] = [r.answer for r in results[question]]
            out[f"{question} (motivo)"] = [r.explanation for r in results[question]]
    output = run_dir / OUTPUT_NAME
    out.to_excel(output, index=False)

    counts = {
        q: {
            k: sum(r.answer == k for r in results[q])
            for k in ("YES", "NO", "SKIPPED", "ERROR")
        }
        for q in questions
    }
    meta = {
        "tool": f"biblio-llm {__version__}",
        "date": datetime.now().isoformat(timespec="seconds"),
        "input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
            "abstract_column": abstract_col,
            "title_column": title_col,
        },
        "output": {"file": OUTPUT_NAME, "format": fmt},
        "model": settings.model,
        "model_digest": digest,
        "options": {
            "think": settings.think,
            "temperature": settings.temperature,
            "seed": settings.seed,
        },
        "articles": done,
        "interrupted": interrupted,
        "counts": counts,
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return ClassifySummary(run_dir, output, done, counts, interrupted)


# ----------------------------------------------------------------------- CLI
def add_parser(sub: Any) -> None:
    """Registra el subcomando ``classify`` en la CLI."""
    c = sub.add_parser("classify", help="clasificar abstracts con preguntas YES/NO")
    c.add_argument("input", type=Path, help="Excel (.xlsx) o CSV con los artículos")
    c.add_argument(
        "--preguntas",
        type=Path,
        default=Path("clasificador.txt"),
        help="archivo con una pregunta por línea (por defecto clasificador.txt)",
    )
    c.add_argument(
        "--limit", type=int, help="clasificar solo los N primeros (para probar)"
    )
    c.add_argument(
        "--formato",
        choices=FORMATS,
        default="curso",
        help="curso: una columna por pregunta ('YES - motivo'); "
        "detallado: respuesta y motivo en columnas separadas",
    )
    think = c.add_mutually_exclusive_group()
    think.add_argument("--think", dest="think", action="store_true", default=None)
    think.add_argument(
        "--no-think",
        dest="think",
        action="store_false",
        help="sin razonamiento: mucho más rápido para YES/NO",
    )


def run_cli(args: Any, settings: Settings) -> int:
    """Ejecuta ``biblio-llm classify``; devuelve el código de salida."""
    if args.think is not None:
        settings = settings.model_copy(update={"think": args.think})
    try:
        summary = classify(
            settings, args.input, args.preguntas, args.limit, fmt=args.formato
        )
    except (ClassifyError, OllamaError) as exc:
        print(f"✗ {exc}")
        return 2
    print(
        f"\nArtículos clasificados: {summary.articles}"
        + (" (interrumpido)" if summary.interrupted else "")
    )
    for question, c in summary.counts.items():
        extra = "".join(f" · {k}: {c[k]}" for k in ("SKIPPED", "ERROR") if c[k])
        print(f"  {question[:60]:<60} YES: {c['YES']} · NO: {c['NO']}{extra}")
    print(f"\nResultado en {summary.output}")
    errors = sum(c["ERROR"] for c in summary.counts.values())
    return 1 if errors or summary.interrupted else 0
