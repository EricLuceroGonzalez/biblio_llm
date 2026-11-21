# Módulo 4 — Entrega M4_entrega_LuceroEric

## Contexto

- **Tema (Módulo 1):** Personnel scheduling or timetabling using metaheuristic algorithms
- **Artículos:** 12 fichas a partir de `papers.csv`.
- **Modelo:** `deepseek-r1:14b` en local con Ollama (digest `c333b7232bdb`), temperature 0.0, seed 42, razonamiento sí.
- **Fecha:** 2026-11-21T18:40:36

## Cómo se ejecutó

Con la herramienta `biblio-llm` (subcomandos `classify` y `extract`):

```bash
biblio-llm extract plantillas/papers.csv --clasificador plantillas/clasificador.txt --tema 'Personnel scheduling or timetabling using metaheuristic algorithms' --entrega M4_entrega_LuceroEric
```

1. **Entradas:** una por artículo, con `paper_id` (P01, P02…), título, abstract y DOI/URL. `paper_id`, `title`, `year` y `source` se copian del archivo, no los genera el modelo.
2. **Fichas:** plantilla de extracción a JSON de la Guía Operativa (4.1), con salida forzada por JSON Schema. Lo que no aparece en el abstract queda como `not_specified`.
3. **Tabla comparativa:** generada con código desde las fichas.
4. **Normalización:** variantes de keywords unificadas entre fichas (2 fusiones; detalle en `normalizacion.md` del run).
5. **Temas:** plantilla de agrupación (4.4); cada tema cita evidencia `P01.campo=…` comprobada con código contra las fichas.
6. **Vacíos:** plantilla de gap analysis (4.5), con tipo, evidencia y propuesta.
7. **Matriz:** tema × familia de método, generada con código.

## Contenido

```
M4_entrega_LuceroEric/
├── README.md              este archivo
├── clasificador.txt       preguntas YES/NO usadas en el cribado
├── fichas/                P01.json, P02.json, P03.json …
├── tabla_comparativa.md
├── matriz_cobertura.md
└── sintesis.md            temas derivados + vacíos + síntesis
```

## Checklist (comprobado automáticamente)

- [x] Los 12 JSON son parseables y contienen todas las claves (JSON Schema: 12/12).
- [x] La tabla tiene exactamente las columnas definidas.
- [ ] No hay detalles inventados (verificador PASS/FAIL: 11/12 PASS).
- [x] Los temas están apoyados por evidencia (campos concretos).
- [x] Los vacíos indican tipo + justificación + propuesta.

## Avisos

- P01 está en varias familias de método: ['Genetic Algorithms', 'Mathematical Programming']
- P02 está en varias familias de método: ['Hybrid Optimization', 'Meta-heuristics']
- P03 está en varias familias de método: ['Hybrid Optimization', 'Mathematical Programming']
- P04 está en varias familias de método: ['Genetic Algorithms', 'Mathematical Programming']
- P05 está en varias familias de método: ['Genetic Algorithms', 'Mathematical Programming', 'Stochastic Programming']
- P07 está en varias familias de método: ['Genetic Algorithms', 'Meta-heuristics']
- P09 está en varias familias de método: ['Hybrid Optimization', 'Mathematical Programming']
- P12 está en varias familias de método: ['Genetic Algorithms', 'Mathematical Programming', 'Integer Linear Programming']
- P03: FAIL: FAIL: The title and keywords in the extraction record contain information not present in the abstract, specifically mentioning a 'probabilistic graph-based approach' which is not mentioned there. This constitutes invented information, leading to a FAIL assessment.

## Pensamiento crítico

La IA generativa puede completar huecos por estilo. Por eso: regla de `not_specified`, esquema fijo, verificador PASS/FAIL y comprobaciones con código de las evidencias. Los vacíos se han revisado frente a la búsqueda todos proponen una hibridacion entre la IA y los algoritmos metaheuristicos, lo que por ahra no es de mi interes.
