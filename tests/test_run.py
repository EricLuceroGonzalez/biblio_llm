import json

import responses

from biblio_llm.__main__ import main
from biblio_llm.runner import RunParams, execute
from tests.conftest import URL


def _params():
    return RunParams(
        "Personnel scheduling using metaheuristics", "español", "doctorandos"
    )


def test_run_creates_new_folder_each_time(settings, ollama) -> None:
    first = execute(settings, _params(), ["a", "b", "c"])
    second = execute(settings, _params(), ["a", "b", "c"])
    assert first.run_dir != second.run_dir  # nada se sobrescribe
    assert first.run_dir.name.endswith("personnel-scheduling-using-metaheuristics")
    assert not first.errors
    assert all(r.startswith("PASS") for r in first.validation), first.validation
    for folder, output in [
        ("a_keywords", "tabla_keywords.md"),
        ("b_ficha", "ficha_tema.json"),
        ("c_sota", "plan_sota.json"),
    ]:
        d = first.run_dir / folder
        assert (d / output).exists() and (d / "prompt.txt").exists()
        assert (d / "razonamiento.txt").read_text().strip() == "razonando…"
    meta = json.loads((first.run_dir / "meta.json").read_text())
    assert meta["model_digest"].startswith("abc123")
    assert meta["params"]["idioma"] == "español"
    assert meta["options"]["think"] is True
    prompt = (first.run_dir / "b_ficha" / "prompt.txt").read_text()
    assert "IDIOMA DE RESPUESTA: español" in prompt
    assert "{" not in prompt  # ningún marcador sin rellenar
    assert (first.run_dir / "verificacion.md").exists()


def test_only_selected_tasks(settings, ollama) -> None:
    summary = execute(settings, _params(), ["c"])
    assert not (summary.run_dir / "a_keywords").exists()
    assert len(summary.validation) == 1


def test_missing_model_aborts_before_creating_run(settings) -> None:
    with responses.RequestsMock() as mock:
        mock.get(f"{URL}/api/tags", json={"models": []})
        try:
            execute(settings, _params(), ["a"])
        except Exception as exc:  # noqa: BLE001
            assert "ollama pull" in str(exc)
    assert not settings.runs_dir.exists()


def test_bad_answer_is_recorded_and_others_continue(settings) -> None:
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        mock.get(f"{URL}/api/version", json={"version": "x"})
        mock.get(
            f"{URL}/api/tags",
            json={"models": [{"name": "deepseek-r1:14b", "digest": "d"}]},
        )
        mock.post(
            f"{URL}/api/chat",
            json={"message": {"content": "no json"}, "done_reason": "stop"},
        )
        summary = execute(settings, _params(), ["b"])
    assert "no es JSON" in summary.errors["b_ficha"]
    assert (summary.run_dir / "b_ficha" / "respuesta.txt").read_text() == "no json"
    assert summary.validation[0].startswith("FAIL")


def test_cli_validate(settings, ollama, monkeypatch, capsys) -> None:
    summary = execute(settings, _params(), ["a"])
    monkeypatch.setattr("biblio_llm.__main__.get_settings", lambda: settings)
    assert main(["validate", str(summary.run_dir)]) == 0
    assert "PASS: Tabla Markdown" in capsys.readouterr().out
