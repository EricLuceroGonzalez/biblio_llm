# biblio-llm

Consultas bibliográficas con un **LLM local** (Ollama + `deepseek-r1:14b`) a partir de tres plantillas:

| Plantilla | Produce | Archivo |
|---|---|---|
| A | 15 palabras clave en tres grupos: núcleo, sinónimos y exclusiones | `tabla_keywords.md` |
| B | Ficha del tema: objetivo, alcance, subpreguntas, palabras clave, riesgos | `ficha_tema.json` |
| C | Plan de un mini estado del arte: título, resumen, secciones, criterios | `plan_sota.json` |

Cada salida está **forzada por un JSON Schema** (salidas estructuradas de Ollama) y se valida al terminar. Cada ejecución va a su propia carpeta: **nunca se sobrescribe una consulta anterior**.

## Requisitos

- [Ollama](https://ollama.com) instalado y en marcha (con la app abierta, o `ollama serve`).
- El modelo descargado: `ollama pull deepseek-r1:14b` (unos 9 GB).

## Inicio rápido

```bash
make setup
cp -n .env.example .env
uv run biblio-llm check
uv run biblio-llm run \
  --tema "Personnel scheduling or timetabling using metaheuristic algorithms" \
  --idioma english \
  --audiencia "PhD students and research personnel"
```

| Comando | Qué hace |
|---|---|
| `check` | Comprueba que Ollama responde, que el modelo está instalado y que existen las plantillas. |
| `run --tema "…" [--idioma] [--audiencia] [--only a,b,c] [--think/--no-think]` | Ejecuta las plantillas y guarda todo en una carpeta nueva de `runs/`. |
| `validate runs/<carpeta>` | Vuelve a validar un run y reescribe su `verificacion.md`. |

`--idioma` y `--audiencia` toman sus valores por defecto de `DEFAULT_IDIOMA` y `DEFAULT_AUDIENCIA` en el `.env`.

## Qué se guarda en cada run

```
runs/2026-10-22_153012_personnel-scheduling-or-timetabling-using-metaheur/
├── meta.json                  parámetros, modelo + digest, versión de Ollama, opciones, tiempos y tokens
├── verificacion.md            PASS/FAIL de cada salida
├── a_keywords/
│   ├── prompt.txt             mensajes de sistema y de usuario enviados, exactos
│   ├── razonamiento.txt       lo que el modelo "pensó" (si THINK=true)
│   ├── respuesta.json         JSON devuelto por el modelo
│   ├── respuesta_cruda.json   respuesta completa de la API de Ollama
│   └── tabla_keywords.md      salida final
├── b_ficha/   (… ficha_tema.json)
└── c_sota/    (… plan_sota.json)
```

- **`meta.json` guarda el `digest` del modelo.** Si actualizas `deepseek-r1:14b`, el nombre sigue igual pero el modelo cambia; el digest lo deja registrado.
- **Con `TEMPERATURE=0` y `SEED` fijos**, repetir la consulta con el mismo modelo debería dar la misma respuesta.
- **Si una plantilla falla**, las demás siguen. El error queda en `meta.json`, y lo que devolvió el modelo, en `respuesta.txt`.

## Idioma

El idioma se pide tres veces: en el mensaje de sistema, en la plantilla (`IDIOMA DE RESPUESTA`) y en las instrucciones. Las **claves** del JSON y las **cabeceras** de la tabla (`Grupo | Termino | Nota`) no se traducen nunca, para que la validación funcione en cualquier idioma.

`deepseek-r1` razona en inglés aunque responda en español. Es normal que `razonamiento.txt` esté en inglés.

## Ajustes (`.env`)

| Variable | Por defecto | Notas |
|---|---|---|
| `MODEL` | `deepseek-r1:14b` | Cualquier modelo instalado en Ollama |
| `THINK` | `true` | Razonamiento del modelo; más lento, a menudo mejor resultado |
| `NUM_PREDICT` | `8192` | Máximo de tokens **incluido el razonamiento**. Si `meta.json` dice `"truncated": true`, súbelo |
| `NUM_CTX` | `8192` | Ventana de contexto |
| `REQUEST_TIMEOUT` | `900` | Segundos por respuesta |
| `TEMPERATURE`, `SEED` | `0`, `42` | Reproducibilidad |

Si un modelo no respeta bien el esquema mientras razona, prueba con `--no-think`.

## Estructura

```
plantillas/                  editables: se rellenan {TEMA}, {IDIOMA} y {AUDIENCIA}
src/biblio_llm/
├── __main__.py              CLI
├── config.py                ajustes del .env
├── ollama_client.py         llamadas a /api/chat, errores claros, separación del razonamiento
├── prompts.py               carga y relleno de plantillas (falla si queda un {MARCADOR})
├── schemas.py               JSON Schema de cada salida
├── tasks.py                 las tres tareas y el paso de JSON a tabla Markdown
├── runner.py                carpeta del run, ejecución y meta.json
└── validation.py            validación (equivale a validate_outputs.py del curso)
```

## Desarrollo

```bash
make test     # tests con Ollama simulado: no requiere GPU ni modelo
make check    # lint + tests
```
