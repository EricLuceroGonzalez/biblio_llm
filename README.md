# biblio-llm

Herramientas bibliográficas con un **LLM local** (Ollama + `deepseek-r1:14b`). Cubre tres pasos de un estado del arte:

| Comando | Para qué | Produce |
|---|---|---|
| `run` | Preparar la búsqueda a partir de un tema (plantillas A, B, C) | `tabla_keywords.md`, `ficha_tema.json`, `plan_sota.json` |
| `classify` | Cribar abstracts con preguntas YES/NO | `articulos_clasificado_abstract.xlsx` |
| `extract` | Fichas por artículo, temas, vacíos, matriz y síntesis (entrega del Módulo 4) | `fichas/`, `tabla_comparativa.md`, `matriz_cobertura.md`, `sintesis.md` |

Las salidas JSON están **forzadas por un JSON Schema** (salidas estructuradas de Ollama) y se validan con código al terminar. Cada ejecución va a su propia carpeta de `runs/`: **nunca se sobrescribe una ejecución anterior**.

## Requisitos

- [Ollama](https://ollama.com) instalado y en marcha (con la app abierta, o `ollama serve`).
- El modelo descargado: `ollama pull deepseek-r1:14b` (unos 9 GB).

## Inicio rápido

```bash
make setup
cp -n .env.example .env
uv run biblio-llm check
```

| Comando | Qué hace |
|---|---|
| `check` | Comprueba que Ollama responde, que el modelo está instalado y que existen las plantillas. |
| `run --tema "…" [--idioma] [--audiencia] [--only a,b,c] [--think/--no-think]` | Ejecuta las plantillas A, B y C. |
| `validate runs/<carpeta>` | Vuelve a validar un run de `run` y reescribe su `verificacion.md`. |
| `classify ARCHIVO [--preguntas] [--limit N] [--formato] [--think/--no-think]` | Clasifica cada abstract con cada pregunta YES/NO. |
| `extract ARCHIVO [--si COLUMNA] [--limit N] [--idioma] [--sin-verificador] [--sin-grupos] [--entrega CARPETA] [--clasificador] [--tema] [--think/--no-think]` | Fichas, tabla, temas, vacíos, matriz y síntesis. |

`ARCHIVO` es un Excel (`.xlsx`) o un CSV con, al menos, una columna de abstract.

## Flujo de trabajo

```
tema ──run──▶ keywords ──(búsqueda: Scopus, PubMed…)──▶ articulos.xlsx / papers.csv
                                                              │
                                         classify + clasificador.txt
                                                              ▼
                                         articulos_clasificado_abstract.xlsx
                                                              │
                                              extract --si "<pregunta>"
                                                              ▼
                    fichas → tabla → temas → vacíos → síntesis + matriz
```

Ejemplo completo:

```bash
# 1. Plantillas A, B y C a partir del tema
uv run biblio-llm run \
  --tema "Personnel scheduling or timetabling using metaheuristic algorithms" \
  --idioma english \
  --audiencia "PhD students and research personnel"

# 2. Cribado YES/NO (primero una prueba con 3 artículos)
uv run biblio-llm classify plantillas/papers.csv --preguntas plantillas/clasificador.txt --limit 3 --no-think
uv run biblio-llm classify plantillas/papers.csv --preguntas plantillas/clasificador.txt --no-think

# 3. Extracción de todos los artículos y entrega del Módulo 4
uv run biblio-llm extract plantillas/papers.csv \
  --clasificador plantillas/clasificador.txt \
  --tema "Personnel scheduling or timetabling using metaheuristic algorithms" \
  --entrega M4_entrega_LuceroEric

# 3 (alternativa). Solo los artículos que respondieron YES en classify
uv run biblio-llm extract runs/<fecha>_clasificacion-papers/articulos_clasificado_abstract.xlsx \
  --si "Does the paper use a metaheuristic algorithm?"
```

## Plantillas A, B y C (`run`)

| Plantilla | Produce | Archivo |
|---|---|---|
| A | 15 palabras clave en tres grupos: núcleo, sinónimos y exclusiones | `tabla_keywords.md` |
| B | Ficha del tema: objetivo, alcance, subpreguntas, palabras clave, riesgos | `ficha_tema.json` |
| C | Plan de un mini estado del arte: título, resumen, secciones, criterios | `plan_sota.json` |

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

Si una plantilla falla, las demás siguen. El error queda en `meta.json`, y lo que devolvió el modelo, en `respuesta.txt`.

## Cribado YES/NO (`classify`)

Pregunta al modelo, para cada artículo y cada línea de `clasificador.txt`, si el abstract responde YES o NO.

**`clasificador.txt`**: una pregunta por línea, en inglés (el modelo responde de forma más estable). Las líneas vacías y las que empiezan por `#` se ignoran. Por defecto se busca en el directorio desde el que ejecutas el comando; con `--preguntas` indicas otra ruta.

```text
# Preguntas del Módulo 1: personnel scheduling con metaheurísticas
Does the paper use a metaheuristic algorithm?
Is the paper about nurse or staff scheduling?
```

**Entrada:**
- **Excel** del curso (`articulos.xlsx`).
- **CSV** de PubMed, Scopus o biblio-search. El separador (`,` `;` o tabulador) y la codificación (UTF-8 o Windows-1252) se detectan solos. Si el CSV está mal formado, el error dice en qué línea y por qué.

**Salida:** `runs/<fecha>_clasificacion-<archivo>/`

```
articulos_clasificado_abstract.xlsx   todas las columnas de entrada + una por pregunta
clasificador.txt                      copia de las preguntas usadas
prompt.txt                            mensajes enviados al modelo
meta.json                             sha256 de la entrada, modelo + digest, opciones, recuento
```

**Formato de las celdas** (`--formato`):

| `curso` (por defecto) | `detallado` |
|---|---|
| Una columna por pregunta: `YES – uses a genetic algorithm`, `NO – …`, `Error: …`, `SKIPPED – abstract vacío` | Dos columnas por pregunta: `YES`/`NO` y `<pregunta> (motivo)` |

El formato `curso` es el que espera el Módulo 5: se filtra con "empieza por YES".

**Robustez:**
- La respuesta está forzada a `YES` o `NO`; si Ollama falla o responde otra cosa, se reintenta una vez y, si vuelve a fallar, la celda queda como `Error: …`.
- Un abstract vacío no se envía al modelo: queda como `SKIPPED`.
- Con `Ctrl+C` se guardan los artículos ya clasificados.
- `--no-think` es mucho más rápido y suficiente para preguntas YES/NO.

## Extracción, temas y vacíos (`extract`)

Implementa la Guía Operativa (plantillas 4.1–4.5) y la Guía Práctica (pasos 1–7) del Módulo 4:

| Paso | Qué hace | Quién lo hace |
|---|---|---|
| 1. Entradas | `paper_id` (P01, P02…), `title`, `year` y `source` de cada artículo | código, desde el archivo: no se pueden inventar |
| 2. Fichas | `problem`, `contribution`, `method`, `data_setup`, `results`, `limitations`, `keywords`; lo que no está en el abstract queda `not_specified` | modelo (plantilla 4.1, JSON Schema) |
| — Verificación | cada ficha contra su JSON Schema y con el verificador PASS/FAIL (4.3), que también recibe el abstract | código + modelo |
| 4. Normalización | variantes de una keyword (mayúsculas, guiones, plurales) se unifican en todas las fichas; las siglas no | código |
| 3. Tabla comparativa | `paper_id \| title \| problem \| method \| data_setup \| key_result \| limitations \| keywords` | código, desde las fichas |
| 5. Temas | 3–5 temas transversales con regla y evidencia `P01.method=…`, y familias de método | modelo (4.4) + comprobación con código |
| 6. Vacíos | 2–3 vacíos con tipo (coverage, method, evidence, generalization, consensus), evidencia y propuesta | modelo (4.5) + comprobación con código |
| 7. Matriz | tema × familia de método; cada celda lista los artículos | código |
| — Síntesis | 5–10 líneas que citan `paper_id` | modelo + comprobación con código |

**Comprobaciones con código** (FAIL si no se cumplen):
- cada `paper_id` citado en temas, métodos, vacíos y síntesis existe;
- cada evidencia tiene la forma `P01.campo=…`, con un campo real, de un artículo del tema;
- la tabla tiene exactamente las columnas fijas.

**Avisos** (no son FAIL): evidencia que no es copia literal de la ficha, artículos de un tema sin evidencia propia, artículos sin tema o sin familia de método.

Si los temas, los vacíos o la síntesis fallan, se reintenta una vez pasándole al modelo la lista de problemas. Los temas necesitan al menos 3 fichas; `--sin-grupos` deja solo fichas y tabla.

**Entrada:** las mismas que `classify`. Columnas que se reconocen si existen: año (`Year`, `Publication Year`, `Año`…), DOI y URL. Un DOI que no tiene forma de DOI (p. ej. un título desplazado en un export de PubMed) se deja vacío y se avisa en `verificacion.md`. `--si COLUMNA` (repetible) procesa solo las filas cuya columna empieza por YES: así se encadena con la salida de `classify`.

**Salida:** `runs/<fecha>_extraccion-<archivo>/`

```
inputs/P01.txt …         paper_id, title, year, doi/url y abstract
fichas/P01.json …        una ficha por artículo (ya normalizada)
normalizacion.md         keywords unificadas
tabla_comparativa.md     tabla con columnas fijas
grupos.json              temas, familias de método y notas
matriz_cobertura.md      tema × método
vacios.json
sintesis.md              temas detectados + vacíos detectados + síntesis
verificacion.md          resultado de todas las comprobaciones
prompts/                 los prompts usados (ficha, verificador, temas, vacíos, síntesis)
meta.json                comando, sha256 de la entrada, modelo + digest, opciones
```

**Ejemplo de ficha:**

```json
{
  "paper_id": "P01",
  "title": "A matheuristic for customized multi-level multi-criteria university timetabling",
  "year": 2023,
  "source": {"doi": "10.1007/s10479-023-05325-2", "url": "https://doi.org/10.1007/s10479-023-05325-2"},
  "problem": "…",
  "contribution": "…",
  "method": "…",
  "data_setup": "not_specified",
  "results": "…",
  "limitations": "not_specified",
  "keywords": ["university timetabling", "matheuristic", "…"]
}
```

### Entrega del Módulo 4 (`--entrega`)

`--entrega M4_entrega_LuceroEric` copia, al terminar, lo que pide la Guía Práctica (§12) con sus nombres exactos:

```
M4_entrega_LuceroEric/
├── README.md              contexto, comando ejecutado, pasos y checklist comprobado
├── clasificador.txt       --clasificador, o el del run de classify si la entrada sale de uno
├── fichas/P01.json …
├── tabla_comparativa.md
├── matriz_cobertura.md
└── sintesis.md            temas derivados + vacíos
```

El checklist del README (§10 de la guía) se marca según las comprobaciones reales, y los avisos se listan debajo. Antes de entregar, revisa `sintesis.md` y completa la sección "Pensamiento crítico" del README. Si la carpeta ya existe y no está vacía, el comando se detiene antes de llamar al modelo. Comprimir el `.zip` queda en tus manos.

**Contexto del modelo:** los prompts de temas, vacíos y síntesis incluyen todas las fichas. Si no caben en `NUM_CTX`, Ollama recortaría el principio sin avisar; por eso esas llamadas amplían `num_ctx` automáticamente (en múltiplos de 4096) y lo registran en el log.

**Códigos de salida** (`classify` y `extract`): `0` todo correcto · `1` algún `Error`, FAIL o interrupción · `2` entrada inválida u Ollama no disponible.

## Reproducibilidad

- **`meta.json` guarda el `digest` del modelo.** Si actualizas `deepseek-r1:14b`, el nombre sigue igual pero el modelo cambia; el digest lo deja registrado.
- **`meta.json` guarda el `sha256` del archivo de entrada** (`classify` y `extract`): identifica exactamente qué datos se procesaron.
- **Con `TEMPERATURE=0` y `SEED` fijos**, repetir la consulta con el mismo modelo debería dar la misma respuesta.
- **Los prompts exactos** se guardan en cada run.

## Idioma

En `run`, el idioma se pide tres veces: en el mensaje de sistema, en la plantilla (`IDIOMA DE RESPUESTA`) y en las instrucciones. Las **claves** del JSON y las **cabeceras** de las tablas no se traducen nunca, para que la validación funcione en cualquier idioma.

En `classify`, las preguntas van en inglés. En `extract`, las fichas se escriben en inglés por defecto (`--idioma` lo cambia); como los abstracts suelen estar en inglés, así se evita que el modelo traduzca y pierda precisión.

`deepseek-r1` razona en inglés aunque responda en español. Es normal que `razonamiento.txt` esté en inglés.

## Ajustes (`.env`)

Todas las variables llevan el prefijo `BIBLIO_LLM_`, para que variables genéricas de otros proyectos (`MODEL`, `TEMPLATES_DIR`…) no se cuelen en esta configuración.

| Variable | Por defecto | Notas |
|---|---|---|
| `BIBLIO_LLM_MODEL` | `deepseek-r1:14b` | Cualquier modelo instalado en Ollama |
| `BIBLIO_LLM_THINK` | `true` | Razonamiento del modelo; más lento, a menudo mejor resultado. `--think`/`--no-think` lo cambian en cada comando |
| `BIBLIO_LLM_NUM_PREDICT` | `8192` | Máximo de tokens **incluido el razonamiento**. Si `meta.json` dice `"truncated": true`, súbelo |
| `BIBLIO_LLM_NUM_CTX` | `8192` | Ventana de contexto (en `extract` se amplía sola para temas, vacíos y síntesis) |
| `BIBLIO_LLM_REQUEST_TIMEOUT` | `900` | Segundos por respuesta |
| `BIBLIO_LLM_TEMPERATURE`, `BIBLIO_LLM_SEED` | `0`, `42` | Reproducibilidad |
| `BIBLIO_LLM_TEMPLATES_DIR`, `BIBLIO_LLM_RUNS_DIR` | `plantillas`, `runs` | Carpetas de plantillas y de resultados |
| `BIBLIO_LLM_DEFAULT_IDIOMA`, `BIBLIO_LLM_DEFAULT_AUDIENCIA` | | Valores por defecto de `run --idioma` y `--audiencia` |

Si un modelo no respeta bien el esquema mientras razona, prueba con `--no-think`.

## Tiempo orientativo

Con `deepseek-r1:14b` en un Mac con Apple Silicon, cada llamada tarda de unos segundos (`--no-think`) a más de un minuto (con razonamiento).

- `classify`: artículos × preguntas llamadas. 10 artículos × 4 preguntas = 40.
- `extract`: 2 llamadas por artículo (ficha + verificador) + 3 (temas, vacíos, síntesis). 12 artículos ≈ 27.

Prueba siempre primero con `--limit 3`.

## Estructura

```
plantillas/                  editables: se rellenan {TEMA}, {IDIOMA} y {AUDIENCIA}
src/biblio_llm/
├── __main__.py              CLI
├── config.py                ajustes del .env (prefijo BIBLIO_LLM_)
├── ollama_client.py         llamadas a /api/chat, errores claros, separación del razonamiento
├── prompts.py               carga y relleno de plantillas (falla si queda un {MARCADOR})
├── schemas.py               JSON Schema de cada salida de `run`
├── tasks.py                 las tres tareas y el paso de JSON a tabla Markdown
├── runner.py                carpeta del run, ejecución y meta.json
├── validation.py            validación (equivale a validate_outputs.py del curso)
├── classify.py              cribado YES/NO de abstracts; lectura de .xlsx y .csv
└── extract.py               fichas, tabla, temas, vacíos, matriz, síntesis y entrega
```

## Desarrollo

```bash
make test     # tests con Ollama simulado: no requiere GPU ni modelo
make check    # lint + tests
```
