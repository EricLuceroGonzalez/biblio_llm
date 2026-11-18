import argparse
import json

import pandas as pd
import pytest
import responses

from biblio_llm import classify as cl
from biblio_llm.config import Settings

URL = "http://ollama.test"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, ollama_url=URL, runs_dir=tmp_path / "runs")  # type: ignore[call-arg]


@pytest.fixture
def questions(tmp_path):
    path = tmp_path / "clasificador.txt"
    path.write_text(
        "# comentario\nDoes it use a metaheuristic?\n\nIs it about nurses?\n",
        encoding="utf-8",
    )
    return path


def _ollama(mock, answers):
    """Ollama falso: responde en orden la lista ``answers`` (dicts o texto)."""
    mock.get(
        f"{URL}/api/tags",
        json={"models": [{"name": "deepseek-r1:14b", "digest": "abc123"}]},
    )
    queue = list(answers)

    def chat(request):
        item = queue.pop(0)
        content = item if isinstance(item, str) else json.dumps(item)
        return (
            200,
            {},
            json.dumps({"message": {"content": content}, "done_reason": "stop"}),
        )

    mock.add_callback(responses.POST, f"{URL}/api/chat", callback=chat)


def test_excel_with_empty_abstract(settings, questions, tmp_path) -> None:
    xlsx = tmp_path / "articulos.xlsx"
    pd.DataFrame(
        {
            "Título": ["Nurse rostering with GA", "Sin abstract"],
            "Abstract": ["We apply a genetic algorithm to nurse rostering.", None],
        }
    ).to_excel(xlsx, index=False)
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        _ollama(
            mock,
            [
                {"answer": "YES", "explanation": "genetic algorithm"},
                {"answer": "yes", "explanation": "nurse rostering"},
            ],
        )
        summary = cl.classify(settings, xlsx, questions)
        body = json.loads(mock.calls[1].request.body)
    assert body["format"] == cl.ANSWER_SCHEMA
    assert body["messages"][0]["content"] == cl.SYSTEM_PROMPT
    assert "Does it use a metaheuristic?" in body["messages"][1]["content"]
    out = pd.read_excel(summary.output)
    assert list(out["Does it use a metaheuristic?"]) == ["YES", "SKIPPED"]
    assert list(out["Is it about nurses?"]) == ["YES", "SKIPPED"]
    assert out["Is it about nurses? (motivo)"][1] == "abstract vacío"
    assert summary.counts["Is it about nurses?"] == {
        "YES": 1,
        "NO": 0,
        "SKIPPED": 1,
        "ERROR": 0,
    }
    meta = json.loads((summary.run_dir / "meta.json").read_text())
    assert (
        meta["model_digest"] == "abc123" and meta["input"]["title_column"] == "Título"
    )
    assert (summary.run_dir / "clasificador.txt").read_text().splitlines() == [
        "Does it use a metaheuristic?",
        "Is it about nurses?",
    ]


def test_biblio_search_csv_and_retry(settings, questions, tmp_path) -> None:
    csv_file = tmp_path / "merged.csv"
    pd.DataFrame({"title": ["A"], "abstract": ["Tabu search for timetabling."]}).to_csv(
        csv_file, index=False, encoding="utf-8-sig"
    )
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        _ollama(
            mock,
            [
                "not json",
                {"answer": "NO", "explanation": "x"},
                {"answer": "MAYBE", "explanation": "?"},
                "still bad",
            ],
        )
        summary = cl.classify(settings, csv_file, questions)
    out = pd.read_excel(summary.output)
    assert out["Does it use a metaheuristic?"][0] == "NO"  # falló y reintentó
    assert out["Is it about nurses?"][0] == "ERROR"  # falló dos veces
    assert summary.counts["Is it about nurses?"]["ERROR"] == 1


def test_missing_abstract_column(settings, questions, tmp_path) -> None:
    xlsx = tmp_path / "x.xlsx"
    pd.DataFrame({"Title": ["a"], "Summary_text": ["b"]}).to_excel(xlsx, index=False)
    with pytest.raises(cl.ClassifyError, match="Columnas encontradas"):
        cl.classify(settings, xlsx, questions)


def test_new_run_dir_each_time_and_limit(settings, questions, tmp_path) -> None:
    xlsx = tmp_path / "a.xlsx"
    pd.DataFrame({"Abstract": ["one", "two", "three"]}).to_excel(xlsx, index=False)
    ok = {"answer": "NO", "explanation": "-"}
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        _ollama(mock, [ok] * 8)
        first = cl.classify(settings, xlsx, questions, limit=2)
        second = cl.classify(settings, xlsx, questions, limit=2)
    assert first.run_dir != second.run_dir
    assert first.articles == 2


def test_interrupt_saves_partial(settings, questions, tmp_path, monkeypatch) -> None:
    xlsx = tmp_path / "a.xlsx"
    pd.DataFrame({"Abstract": ["one", "two"]}).to_excel(xlsx, index=False)
    calls = {"n": 0}

    def fake_ask(client, abstract, question, retries=1):
        calls["n"] += 1
        if calls["n"] > 2:
            raise KeyboardInterrupt
        return cl.Answer("YES", "ok")

    monkeypatch.setattr(cl, "ask", fake_ask)
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        _ollama(mock, [])
        summary = cl.classify(settings, xlsx, questions)
    assert summary.interrupted and summary.articles == 1
    assert len(pd.read_excel(summary.output)) == 1


def test_cli(settings, questions, tmp_path, capsys) -> None:
    xlsx = tmp_path / "a.xlsx"
    pd.DataFrame({"Abstract": ["one"]}).to_excel(xlsx, index=False)
    parser = argparse.ArgumentParser()
    cl.add_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        ["classify", str(xlsx), "--preguntas", str(questions), "--no-think"]
    )
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        _ollama(
            mock,
            [
                {"answer": "YES", "explanation": "a"},
                {"answer": "NO", "explanation": "b"},
            ],
        )
        code = cl.run_cli(args, settings)
        assert json.loads(mock.calls[1].request.body)["think"] is False
    assert code == 0
    assert "YES: 1 · NO: 0" in capsys.readouterr().out
