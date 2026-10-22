"""CLI: ``uv run biblio-llm <comando>``.

Comandos:
    check                       Comprueba Ollama, el modelo y las plantillas.
    run --tema "…" [opciones]   Ejecuta las plantillas en una carpeta nueva de runs/.
    validate CARPETA_RUN        Vuelve a validar un run (escribe verificacion.md).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from biblio_llm import __version__
from biblio_llm.config import get_settings
from biblio_llm.logging_setup import setup_logging
from biblio_llm.ollama_client import OllamaClient, OllamaError
from biblio_llm.prompts import TemplateError, load_template
from biblio_llm.runner import RunParams, execute
from biblio_llm.tasks import TASKS
from biblio_llm.validation import validate_run

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="biblio-llm",
        description="Consultas bibliográficas con un LLM local (Ollama).",
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="comprobar Ollama, modelo y plantillas")
    r = sub.add_parser("run", help="ejecutar las plantillas")
    r.add_argument("--tema", required=True, help="tema de investigación")
    r.add_argument(
        "--idioma",
        default="English",
        help="idioma de la respuesta (p. ej. español, english)",
    )
    r.add_argument("--audiencia", help="audiencia de la ficha (plantilla B)")
    r.add_argument("--only", default="a,b,c", help="plantillas a ejecutar (p. ej. b,c)")
    think = r.add_mutually_exclusive_group()
    think.add_argument(
        "--think",
        dest="think",
        action="store_true",
        default=None,
        help="dejar que el modelo razone (por defecto: THINK del .env)",
    )
    think.add_argument("--no-think", dest="think", action="store_false")
    v = sub.add_parser("validate", help="validar un run existente")
    v.add_argument("run_dir", type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = get_settings()
    except ValidationError as exc:
        print(f"Configuración inválida:\n{exc}", file=sys.stderr)
        return 2
    setup_logging(
        log_dir=settings.log_dir,
        log_filename="biblio_llm.log",
        level=settings.log_level,
    )
    for noisy in ("urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.command == "validate":
        if not args.run_dir.is_dir():
            print(f"No existe la carpeta {args.run_dir}", file=sys.stderr)
            return 2
        results = validate_run(args.run_dir)
        for line in results:
            print(f"  {line}")
        return 0 if all(r.startswith("PASS") for r in results) else 1

    client = OllamaClient(settings)

    if args.command == "check":
        ok = True
        try:
            print(f"Ollama {client.version()} en {settings.ollama_url}")
            digest = client.model_digest()
            if digest:
                print(f"Modelo {settings.model} ({digest[:12]})")
            else:
                ok = False
                print(f"Falta el modelo: ollama pull {settings.model}")
        except OllamaError as exc:
            ok = False
            print(f"{exc}")
        for task in TASKS.values():
            try:
                print(f"Plantilla {task.template}")
                load_template(settings.templates_dir, task.template)
            except TemplateError as exc:
                ok = False
                print(f"{exc}")
        return 0 if ok else 1

    keys = [k.strip().lower() for k in args.only.split(",") if k.strip()]
    unknown = [k for k in keys if k not in TASKS]
    if unknown:
        print(f"Plantillas desconocidas: {unknown} (usa a, b, c)", file=sys.stderr)
        return 2
    if args.think is not None:
        settings = settings.model_copy(update={"think": args.think})
    params = RunParams(
        tema=args.tema,
        idioma=args.idioma or settings.default_idioma,
        audiencia=args.audiencia or settings.default_audiencia,
    )
    try:
        summary = execute(settings, params, keys, client)
    except (OllamaError, TemplateError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    print("\n" + "=" * 50)
    print("RESUMEN DE VALIDACION")
    print("=" * 50)
    for line in summary.validation:
        print(f"  {line}")
    for folder, error in summary.errors.items():
        print(f"  ERROR {folder}: {error}")
    print(f"\nResultados en {summary.run_dir}/")
    ok = not summary.errors and all(r.startswith("PASS") for r in summary.validation)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
