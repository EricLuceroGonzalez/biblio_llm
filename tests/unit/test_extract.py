import argparse
import json

import pandas as pd
import pytest
import responses

from biblio_llm import extract as ex
from biblio_llm.config import Settings
from biblio_llm.ollama_client import OllamaClient

URL = "http://ollama.test"

FICHA = {
    "problem": "Nurse rostering in hospitals.",
    "contribution": "A hybrid genetic algorithm.",
    "method": "Genetic algorithm with tabu search.",
    "data_setup": "",
    "results": "Reduces overtime by 12%.",
    "limitations": "null",
    "keywords": ["nurse rostering", "Genetic Algorithms", "genetic algorithm", "tabu"],
}

GAPS = {
    "gaps": [
        {
            "type": "evidence",
            "description": "No real-world validation.",
            "evidence": "P01 and P03 report no data_setup.",
            "how_to_cover": ["Test on hospital data."],
        },
        {
            "type": "method",
            "description": "No exact methods.",
            "evidence": "The themes Genetic algorithms and Overtime use heuristics.",
            "how_to_cover": ["Compare with MIP.", "Benchmark."],
        },
    ]
}

SYNTHESIS = {
    "lines": [
        "The papers address nurse rostering (P01, P02, P03).",
        "Genetic algorithms dominate (P01).",
        "Overtime is the main reported result (P03).",
        "No paper validates on real data.",
        "Exact methods are missing.",
    ]
}


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, ollama_url=URL, runs_dir=tmp_path / "runs")  # type: ignore[call-arg]


def groups_for(ids):
    return {
        "groups": [
            {
                "label": "Genetic algorithms",
                "rule": "same method",
                "papers": ids[:2],
                "evidence": [f"{ids[0]}.method=Genetic algorithm with tabu search"],
            },
            {
                "label": "Overtime",
                "rule": "same result",
                "papers": ids[1:],
                "evidence": [f"{ids[-1]}.results=Reduces overtime by 12%"],
            },
            {
                "label": "Rostering",
                "rule": "same problem",
                "papers": ids,
                "evidence": [f"{ids[0]}.problem=Nurse rostering in hospitals"],
            },
        ],
        "methods": [
            {"label": "Metaheuristic", "papers": ids[:-1]},
            {"label": "Hybrid", "papers": ids[-1:]},
        ],
        "notes": "P02 is ambiguous.",
    }


def fake_ollama(
    mock, ficha=FICHA, verdict="PASS", groups=None, gaps=GAPS, synth=SYNTHESIS, log=None
):
    """Ollama falso: responde según el prompt de sistema de cada llamada."""
    mock.get(
        f"{URL}/api/tags",
        json={"models": [{"name": "deepseek-r1:14b", "digest": "abc123def4567890"}]},
    )

    def chat(request):
        body = json.loads(request.body)
        system = body["messages"][0]["content"]
        user = body["messages"][1]["content"]
        if log is not None:
            log.append(body)
        if system.startswith("You are an assistant that extracts"):
            content = ficha if isinstance(ficha, str) else json.dumps(ficha)
        elif system.startswith("Act as a strict verifier"):
            content = json.dumps({"verdict": verdict, "reason": "invented result"})
        elif system.startswith("You group"):
            records = user.split("INPUT (records):\n")[1].split("\n\nYour previous")[0]
            ids = [r["paper_id"] for r in json.loads(records)]
            content = json.dumps(groups or groups_for(ids))
        elif system.startswith("You identify research gaps"):
            content = json.dumps(gaps)
        else:
            content = json.dumps(synth)
        return (
            200,
            {},
            json.dumps({"message": {"content": content}, "done_reason": "stop"}),
        )

    mock.add_callback(responses.POST, f"{URL}/api/chat", callback=chat)


def articles(tmp_path, n=3, **extra):
    path = tmp_path / "articulos.xlsx"
    data = {
        "Title": [f"Paper {i}" for i in range(n)],
        "Abstract": [f"Abstract {i} about nurse rostering." for i in range(n)],
        "Publication Year": (["2021 Mar"] + [2020] * n)[:n],
        "DOI": (["10.1000/abc", "Nurse Scheduling with Open-Source Tools."] + [""] * n)[
            :n
        ],
        **extra,
    }
    pd.DataFrame(data).to_excel(path, index=False)
    return path


def test_full_pipeline_three_papers(settings, tmp_path) -> None:
    path = articles(tmp_path, n=3)
    log: list = []
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        fake_ollama(mock, log=log)
        s = ex.extract(settings, path)
    run = s.run_dir
    assert s.ok, ex.verification_markdown(s)
    # 3 fichas + 3 verificaciones + temas + vacíos + síntesis
    assert len(log) == 9
    f1 = json.loads((run / "fichas" / "P01.json").read_text())
    assert f1["year"] == 2021 and f1["source"]["doi"] == "10.1000/abc"
    assert f1["data_setup"] == ex.NS and f1["limitations"] == ex.NS
    # normalización: "Genetic Algorithms" y "genetic algorithm" → una sola forma
    assert f1["keywords"] == ["nurse rostering", "Genetic Algorithms", "tabu"]
    assert s.merged == {"Genetic Algorithms": ["genetic algorithm"]}
    assert (
        "| Genetic Algorithms | genetic algorithm |"
        in (run / "normalizacion.md").read_text()
    )
    f2 = json.loads((run / "fichas" / "P02.json").read_text())
    assert f2["source"]["doi"] == ""  # el título en la columna DOI se descarta
    table = (run / "tabla_comparativa.md").read_text()
    assert ex.check_table(table) == [] and len(table.splitlines()) == 5
    matrix = (run / "matriz_cobertura.md").read_text()
    assert "| Tema \\ Método | Metaheuristic | Hybrid |" in matrix
    assert "| Overtime | P02 | P03 |" in matrix
    synth = (run / "sintesis.md").read_text()
    assert "### Tema 1: Genetic algorithms" in synth
    assert "- papers: P01, P02" in synth and "> Notas: P02 is ambiguous." in synth
    assert "- [evidence] No real-world validation." in synth
    assert "Exact methods are missing." in synth
    assert "tema 'Overtime': sin evidencia propia para P02" in s.groups_check[1]
    verif = (run / "verificacion.md").read_text()
    assert "| P01 | OK | PASS | PASS |" in verif
    assert "la columna DOI no contiene un DOI" in verif
    meta = json.loads((run / "meta.json").read_text())
    assert meta["themes"] == 3 and meta["gaps"] == 2
    for name in ("ficha", "verificador", "temas", "vacios", "sintesis"):
        assert (run / "prompts" / f"{name}.txt").is_file()


def test_entrega(settings, tmp_path) -> None:
    path = articles(tmp_path, n=3)
    clasif = tmp_path / "clasificador.txt"
    clasif.write_text("Does it use a metaheuristic?\n")
    dest = tmp_path / "M4_entrega_LuceroEric"
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        fake_ollama(mock)
        s = ex.extract(
            settings,
            path,
            entrega=dest,
            tema="Personnel scheduling with metaheuristics",
            command="biblio-llm extract papers.csv --entrega M4_entrega_LuceroEric",
        )
    assert s.entrega == dest
    names = sorted(
        p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()
    )
    assert names == [
        "README.md",
        "clasificador.txt",  # encontrado junto a la entrada
        "fichas/P01.json",
        "fichas/P02.json",
        "fichas/P03.json",
        "matriz_cobertura.md",
        "sintesis.md",
        "tabla_comparativa.md",
    ]
    readme = (dest / "README.md").read_text()
    assert "# Módulo 4 — Entrega M4_entrega_LuceroEric" in readme
    assert "Personnel scheduling with metaheuristics" in readme
    assert "biblio-llm extract papers.csv --entrega" in readme
    assert "- [x] Los 3 JSON son parseables" in readme
    assert "(verificador PASS/FAIL: 3/3 PASS)" in readme
    assert "- [x] Los vacíos indican tipo" in readme
    assert "abc123def456" in readme
    # una entrega existente no se sobrescribe
    with pytest.raises(ex.ClassifyError, match="ya existe"):
        ex.extract(settings, path, entrega=dest)


def test_yes_filter_and_too_few_fichas(settings, tmp_path) -> None:
    path = articles(tmp_path, n=4, **{"Q1": ["YES – ok", "NO – x", "yes", "Error: t"]})
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        fake_ollama(mock)
        s = ex.extract(settings, path, yes_columns=["Q1"], verify=False)
    assert [r.paper_id for r in s.results] == ["P01", "P02"]
    assert s.groups_check is None and not (s.run_dir / "sintesis.md").exists()
    with pytest.raises(ex.ClassifyError, match="no existe la columna"):
        ex.extract(settings, path, yes_columns=["Q9"])


def test_ficha_error_and_skipped(settings, tmp_path) -> None:
    path = tmp_path / "a.csv"
    pd.DataFrame({"title": ["A", "B"], "abstract": ["Text.", ""]}).to_csv(
        path, index=False
    )
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        fake_ollama(mock, ficha="not json")
        s = ex.extract(settings, path)
    assert [r.status for r in s.results] == ["ERROR", "SKIPPED"]
    assert not s.ok


def test_verifier_fail_makes_exit_code_1(settings, tmp_path) -> None:
    path = articles(tmp_path, n=1)
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        fake_ollama(mock, verdict="FAIL")
        s = ex.extract(settings, path)
    assert s.results[0].verifier == "FAIL: invented result" and not s.ok


def test_invalid_themes_retry_then_stop(settings, tmp_path) -> None:
    path = articles(tmp_path, n=3)
    bad = groups_for(["P01", "P02", "P99"])
    log: list = []
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        fake_ollama(mock, groups=bad, log=log)
        s = ex.extract(settings, path, verify=False)
    calls = [b for b in log if b["messages"][0]["content"].startswith("You group")]
    assert len(calls) == 2  # segundo intento con feedback
    assert "P99 no existe" in calls[1]["messages"][1]["content"]
    assert s.groups_check and s.groups_check[0] and s.gaps_check is None
    assert (s.run_dir / "grupos.json").exists()
    assert not (s.run_dir / "matriz_cobertura.md").exists()
    assert "No se generaron." in (s.run_dir / "sintesis.md").read_text()
    assert not s.ok


def test_synthesis_citing_unknown_paper(settings, tmp_path) -> None:
    path = articles(tmp_path, n=3)
    bad = {"lines": ["P07 is great."] + ["Line (P01)."] * 4}
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        fake_ollama(mock, synth=bad)
        s = ex.extract(settings, path, verify=False)
    assert s.synthesis_check == (["línea 1: cita P07, que no existe"], [])
    assert "## Síntesis\n\nNo se generó." in (s.run_dir / "sintesis.md").read_text()
    assert not s.ok


def test_validate_groups() -> None:
    fichas = {
        "P01": {"method": "Genetic algorithm", "keywords": ["ga"], "problem": "x"},
        "P02": {"method": "Tabu search", "keywords": [], "problem": "y"},
        "P03": {"method": "MIP", "keywords": [], "problem": "z"},
    }
    groups = {
        "groups": [
            {
                "label": "A",
                "rule": "r",
                "papers": ["P01", "P09"],
                "evidence": [
                    "P01.method=Genetic algorithm",
                    "P01.method=Simulated annealing",
                    "P02.method=Tabu search",
                    "P01.venue=x",
                    "method is GA",
                ],
            },
            {
                "label": "B",
                "rule": "r",
                "papers": ["P02"],
                "evidence": ["P02.keywords=..."],
            },
            {
                "label": "C",
                "rule": "r",
                "papers": ["P02"],
                "evidence": ["P02.problem=y"],
            },
        ],
        "methods": [
            {"label": "Exact", "papers": ["P01", "P08"]},
            {"label": "Meta", "papers": ["P01"]},
        ],
        "notes": "",
    }
    problems, warnings = ex.validate_groups(groups, fichas)
    assert any("P09 no existe" in p for p in problems)
    assert any("cita P02, que no está en el tema" in p for p in problems)
    assert any("campo inexistente 'venue'" in p for p in problems)
    assert any("sin formato" in p for p in problems)
    assert "método 'Exact': P08 no existe" in problems
    assert any("P01.method no es copia literal" in w for w in warnings)
    assert "P03 no está en ningún tema" in warnings
    assert "P01 está en varias familias de método: ['Exact', 'Meta']" in warnings
    assert "P02 no está en ninguna familia de método" in warnings
    assert "tema 'A': sin evidencia propia para P01" not in warnings
    bad = ex.validate_groups({"groups": []}, fichas)[0]
    assert bad


def test_validate_gaps() -> None:
    gaps = {
        "gaps": [
            {
                "type": "method",
                "description": "d",
                "evidence": "P07 lacks data",
                "how_to_cover": ["x"],
            },
            {
                "type": "coverage",
                "description": "d",
                "evidence": "nothing cited",
                "how_to_cover": ["x"],
            },
        ]
    }
    problems, warnings = ex.validate_gaps(gaps, {"P01": {}}, ["Genetic"])
    assert problems == ["vacío 1: cita P07, que no existe"]
    assert warnings == ["vacío 2: la evidencia no cita ningún artículo ni tema"]
    too_many = {"gaps": gaps["gaps"] * 2}
    assert ex.validate_gaps(too_many, {"P01": {}}, [])[0]  # máximo 3 vacíos


def test_normalize_across() -> None:
    fichas = [
        {"paper_id": "P01", "keywords": ["Nurse-Rostering", "GA", "analysis"]},
        {"paper_id": "P02", "keywords": ["nurse rostering", "Genetic algorithm"]},
        {"paper_id": "P03", "keywords": ["nurse rosterings", "genetic algorithms"]},
        {"paper_id": "P04", "keywords": ["nurse rostering"]},
    ]
    out, merged = ex.normalize_across(fichas)
    assert out[0]["keywords"] == [
        "nurse rostering",
        "GA",
        "analysis",
    ]  # GA no se fusiona
    assert out[2]["keywords"] == ["nurse rostering", "Genetic algorithm"]
    assert merged == {
        "nurse rostering": ["Nurse-Rostering", "nurse rosterings"],
        "Genetic algorithm": ["genetic algorithms"],
    }
    assert fichas[0]["keywords"][0] == "Nurse-Rostering"  # no muta la entrada


def test_helpers() -> None:
    assert (
        ex.normalize_doi("https://doi.org/10.1016/J.EJOR.2020.01.001")
        == "10.1016/J.EJOR.2020.01.001"
    )
    assert ex.normalize_doi("doi: 10.3390/a15010001") == "10.3390/a15010001"
    assert ex.normalize_doi("Nurse Scheduling with State-of-the Art Tools.") == ""
    assert ex.parse_year("2019 Dec 3") == 2019 and ex.parse_year(2020.0) == 2020
    assert ex.parse_year("") is None
    row = {
        **{k: "a|b" for k in ex.CONTENT_FIELDS},
        "paper_id": "P01",
        "title": "t",
        "keywords": ["x", "y"],
    }
    table = ex.table_markdown([row])
    assert (
        table.splitlines()[2]
        == "| P01 | t | a\\|b | a\\|b | a\\|b | a\\|b | a\\|b | x, y |"
    )
    assert ex.check_table(table) == []
    assert ex.check_table("| paper_id | title |\n|---|---|\n") != []
    assert ex.check_table(table + "| P02 | t |\n") == ["la fila 2 tiene 2 celdas"]


def test_client_for_long_prompt(settings) -> None:
    client = OllamaClient(settings)
    assert ex.client_for_prompt(client, "short") is client
    big = ex.client_for_prompt(client, "x" * 30000)
    assert big.s.num_ctx == 16384


def test_cli(settings, tmp_path, capsys) -> None:
    path = articles(tmp_path, n=3)
    parser = argparse.ArgumentParser()
    ex.add_parser(parser.add_subparsers(dest="command"))
    dest = tmp_path / "entrega"
    args = parser.parse_args(
        [
            "extract",
            str(path),
            "--sin-verificador",
            "--no-think",
            "--entrega",
            str(dest),
            "--tema",
            "T",
        ]
    )
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        fake_ollama(mock)
        code = ex.run_cli(args, settings)
        assert json.loads(mock.calls[1].request.body)["think"] is False
    out = capsys.readouterr().out
    assert code == 0, out
    assert "Fichas: 3 (OK 3 · ERROR 0 · SKIPPED 0)" in out
    assert "Temas y métodos: PASS" in out and "Síntesis: PASS" in out
    assert f"Entrega en {dest}" in out
    assert (dest / "README.md").is_file()
