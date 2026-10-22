import json

import pytest
import responses

from biblio_llm import schemas
from biblio_llm.ollama_client import OllamaClient, OllamaError, split_thinking
from biblio_llm.prompts import TemplateError, render
from biblio_llm.runner import slugify
from biblio_llm.tasks import keywords_to_markdown
from biblio_llm.validation import validate_json, validate_markdown_table
from tests.conftest import FICHA, KEYWORDS, URL


def test_render_detects_missing_placeholders() -> None:
    assert render('T={TEMA} {"k": 1}', {"TEMA": "x"}) == 'T=x {"k": 1}'
    with pytest.raises(TemplateError, match="IDIOMA"):
        render("{TEMA} en {IDIOMA}", {"TEMA": "x"})


def test_split_thinking_inline_and_fences() -> None:
    content, thinking = split_thinking('<think>hmm</think>\n```json\n{"a": 1}\n```')
    assert content == '{"a": 1}' and thinking == "hmm"
    assert split_thinking('{"a": 1}', "aparte") == ('{"a": 1}', "aparte")


def test_slugify() -> None:
    assert slugify("Personnel scheduling / timetabling: ¿metaheurísticas?") == (
        "personnel-scheduling-timetabling-metaheuristicas"
    )
    assert slugify("!!!") == "consulta"


def test_keywords_markdown_has_fixed_headers(tmp_path) -> None:
    md = keywords_to_markdown(KEYWORDS)
    assert md.splitlines()[0] == "| Grupo | Termino | Nota |"
    assert md.count("| Exclusiones |") == 5
    path = tmp_path / "t.md"
    path.write_text(md, encoding="utf-8")
    assert validate_markdown_table(path).startswith("PASS")


def test_validate_json_reports_schema_errors(tmp_path) -> None:
    bad = dict(FICHA, subpreguntas=["solo una"])
    del bad["objetivo"]
    path = tmp_path / "f.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    msg = validate_json(path, schemas.FICHA)
    assert msg.startswith("FAIL") and "objetivo" in msg and "subpreguntas" in msg
    path.write_text("<think>x</think>{", encoding="utf-8")
    assert "no es un JSON valido" in validate_json(path, schemas.FICHA)


@responses.activate
def test_chat_payload_and_truncation(settings) -> None:
    responses.post(
        f"{URL}/api/chat",
        json={"message": {"content": '{"a"', "thinking": "t"}, "done_reason": "length"},
    )
    result = OllamaClient(settings).chat("sys", "user", {"type": "object"})
    body = json.loads(responses.calls[0].request.body)
    assert body["think"] is True and body["format"] == {"type": "object"}
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["options"]["seed"] == 42 and body["options"]["temperature"] == 0
    assert result.truncated


def test_connection_error_is_friendly(settings) -> None:
    settings = settings.model_copy(update={"ollama_url": "http://127.0.0.1:9"})
    with pytest.raises(OllamaError, match="¿Está abierto"):
        OllamaClient(settings).version()
