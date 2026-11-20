# convenio-rag-evals

Asistente que responde preguntas sobre un convenio colectivo del BOE **citando el artículo exacto**, con una suite de evaluación que mide si acierta y una capa de trazas que dice cuánto cuesta cada respuesta.

[![tests](https://github.com/francisco-m-s/convenio-rag-evals/actions/workflows/tests.yml/badge.svg)](https://github.com/francisco-m-s/convenio-rag-evals/actions)

> **Estado (20/11/2026):** funciona de extremo a extremo: ingesta del texto del BOE, búsqueda híbrida, generación con citas, trazas por petición, API REST y webhook, y una suite de evaluación con resultados reales. **Pendiente:** calibrar el juez automático con una revisión humana y probar la reordenación de resultados. Plan en [`docs/PLAN.md`](docs/PLAN.md); cada decisión con su alternativa descartada, en [`docs/DECISIONES.md`](docs/DECISIONES.md).

> **Aviso:** esto es un proyecto de ingeniería, **no asesoramiento jurídico ni un texto oficial**. Una respuesta del asistente no sustituye la lectura del texto del BOE ni la consulta a un profesional. **Basado en datos de la [Agencia Estatal Boletín Oficial del Estado](https://www.boe.es)**; las condiciones de reutilización y las modificaciones hechas al texto están en [`datos/LEEME.md`](datos/LEEME.md).

## Por qué existe

Una pyme o una gestoría que consulta un convenio colectivo se pierde entre artículos: vacaciones, jornada, pluses, permisos. Un chatbot que «suena bien» no sirve si inventa un plazo. Este proyecto construye el asistente **y la forma de demostrar que no inventa**: cada respuesta lleva su cita, y una batería de 60 preguntas con respuesta conocida mide los aciertos, las citas correctas y las abstenciones cuando el texto no dice nada.

## Corpus

| Documento | Identificador BOE | Publicado |
|---|---|---|
| [XVIII Convenio colectivo estatal para las empresas del comercio de flores y plantas](https://www.boe.es/buscar/doc.php?id=BOE-A-2025-21424) (vigencia 2024–2028) | `BOE-A-2025-21424` | 24/10/2025 |
| [Acta de la Comisión mixta con las tablas salariales de 2026](https://www.boe.es/diario_boe/txt.php?id=BOE-A-2026-10861) | `BOE-A-2026-10861` | 20/05/2026 |

Quedan en **186 trozos** citables: 56 artículos del convenio, 47 artículos de su Anexo I (Plan de Pensiones, con numeración propia: «artículo 26» existe dos veces, y los identificadores llevan el ámbito para distinguirlos), disposiciones, anexos y tablas salariales. Se incluyen a propósito las tablas de 2026, publicadas en un documento aparte: obligan a resolver preguntas que cruzan dos textos.

## Resultados

### Recuperación: ¿está el artículo correcto entre los primeros resultados?

50 preguntas del conjunto tienen un artículo esperado (las otras 10 no deben tener respuesta). `hit@k` es la proporción de preguntas cuyo artículo está entre los `k` primeros trozos; `MRR@10`, el inverso de la posición del acierto. Troceado por artículo; la tabla completa (con el troceado de tamaño fijo) está en [`evals/resultados/recuperacion.md`](evals/resultados/recuperacion.md).

| Modo | Modelo de embeddings | Raíz | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| léxica (BM25) | — | no | 0,58 | 0,84 | 0,86 | 0,94 | 0,77 |
| léxica (BM25) | — | 6 | 0,64 | 0,88 | 0,90 | 0,94 | 0,81 |
| vectorial | paraphrase-multilingual-MiniLM-L12-v2 | — | 0,52 | 0,72 | 0,74 | 0,82 | 0,68 |
| híbrida | paraphrase-multilingual-MiniLM-L12-v2 | 6 | 0,54 | 0,76 | 0,80 | 0,96 | 0,72 |
| vectorial | jina-embeddings-v2-base-es | — | 0,76 | 0,92 | 0,94 | 1,00 | 0,90 |
| **híbrida (por defecto)** | **jina-embeddings-v2-base-es** | **6** | 0,74 | 0,94 | **0,98** | 0,98 | 0,88 |
| vectorial | multilingual-e5-large | — | 0,82 | 0,92 | 0,94 | 0,98 | 0,95 |
| híbrida | multilingual-e5-large | 6 | 0,80 | 0,92 | 0,94 | 0,98 | 0,91 |

Qué se lee de ahí:

- **El modelo de embeddings importa más que ninguna otra variable.** El que se eligió primero por ser el más pequeño (MiniLM de paráfrasis) resultó el peor, por debajo incluso de la búsqueda léxica sola. Jina (0,64 GB) y e5-large (2,24 GB) están muy cerca; se eligió Jina por `hit@5` (lo que recibe el modelo de lenguaje) y por pesar la tercera parte. e5-large gana en `hit@1` y en MRR.
- **La híbrida no siempre gana.** Ayuda mucho a un modelo flojo (MiniLM: de 0,74 a 0,80 en `hit@5`) y a Jina en `hit@3`/`hit@5`, pero con e5-large no mejora, y con Jina queda por debajo de la vectorial pura en MRR (0,88 frente a 0,90).
- **Buscar por raíz** (truncar cada término a 6 caracteres) mejora la búsqueda léxica de 0,86 a 0,90 en `hit@5`. Son dos preguntas más acertadas sobre 50, así que es una pista, no una prueba.
- **El tamaño del conjunto limita lo que se puede afirmar.** Con 50 preguntas, una pregunta vale 0,02. Las diferencias de 0,02 a 0,04 entre configuraciones están en el ruido.

### Troceado: por artículo frente a tamaño fijo

| Troceado | Trozos | Caracteres (mediana) | % de trozos que cruzan artículos | hit@5 (híbrida, Jina) | MRR@10 |
|---|---|---|---|---|---|
| **por artículo** | 186 | 1.445 | **0** | 0,98 | 0,88 |
| tamaño fijo, 800 caracteres | 413 | 800 | 30 | 0,92 | 0,86 |
| tamaño fijo, 1.500 caracteres | 214 | 1.500 | 48 | 0,94 | 0,88 |

**`hit@k` no separa los troceados** (con e5-large el de 1.500 caracteres llega a 0,98) y, además, la regla de acierto favorece al tamaño fijo: una ventana que toca dos artículos cuenta como acierto si uno de ellos es el correcto. Lo que sí los separa es la última columna de la tabla de troceado: **entre el 30 % y el 48 % de las ventanas de tamaño fijo mezclan dos artículos, y con ellas la cita no puede ser exacta**. Esa es la razón por la que se trocea por artículo, no una mejora de `hit@k`.

### Qué falla

- La pregunta `q47` (qué es un partícipe del plan de pensiones **y** qué aporta la empresa) falla en 7 de las 8 configuraciones de arriba: necesita dos artículos del Anexo I y casi siempre solo llega uno. Es la clase de pregunta para la que serviría una reordenación o una segunda recuperación.
- Las preguntas sobre el Plan de Pensiones (`q37`, `q38`) fallan con la búsqueda léxica y con e5-large: «plan de pensiones» aparece en casi todo el Anexo I.
- Con MiniLM el artículo correcto suele estar en el top-10 (0,96) pero no en el top-5 (0,80): acierta lo que recupera, lo ordena mal.

### Generación: ¿responde bien, cita y se abstiene?

Una ejecución del 20/11/2026 de las 60 preguntas: prompt `v1`, modelo `gemini-3.8-flash` (razonamiento bajo) a través de una pasarela LiteLLM, búsqueda híbrida con Jina y `k` = 5. Resultados completos, con cada respuesta, en [`evals/resultados/generacion.json`](evals/resultados/generacion.json).

| Métrica | Resultado |
|---|---|
| Cita correcta (preguntas con artículo esperado) | **49 de 50** (0,98) |
| Se abstiene cuando el texto no dice nada | **9 de 10** (0,90) |
| Se abstiene teniendo la respuesta (falsa abstención) | 0 |
| Citas inventadas (ids que no se le entregaron) | 0 |
| Afirma algo sin citar ningún fragmento | 0 |
| Resiste una orden escrita dentro de la pregunta (2 preguntas) | 2 de 2 |
| Juez automático, **sin calibrar**: correcta / parcial / incorrecta | 58 / 1 / 1 de 60 |
| Coste medio por pregunta (generación y juicio) | 0,0034 $ (total de la ejecución: 0,20 $) |
| Latencia de la generación | mediana 2,5 s; percentil 95 12,3 s |

Las cinco primeras filas no dependen del juez: se calculan comparando las citas con los artículos esperados. Las de coste y latencia salen de las trazas del propio proyecto.

**Los tres casos que no salen limpios, y qué dicen:**

- **`q52`, «¿Hay un plus de transporte?» (la abstención que falla).** El sistema responde que existe una «indemnización por desplazamiento» por el coste del transporte, con importes, citando el art. 31. El conjunto daba por hecho que no había respuesta. **Es probable que la etiqueta esté mal y el sistema acierte**: el artículo existe y cubre ese concepto aunque no use la palabra «plus». No se ha cambiado la etiqueta para no ajustar el conjunto a los resultados; queda anotada en [`DECISIONES.md`](docs/DECISIONES.md) (D12) a la espera de una revisión humana.
- **`q47`, partícipe del plan de pensiones y aportación de la empresa.** El juez da la respuesta por correcta, pero la regla de cita no, porque el sistema citó el art. 37 del convenio («Plan de pensiones») y la introducción del Anexo I en lugar de los artículos 11 y 27 del Anexo I. Es el mismo fallo de recuperación que se vio arriba (`q47`) y, aquí, una regla de cita que probablemente es demasiado estricta.
- **`q48`, «¿Qué establece el artículo 26?» (ambigua).** Solo trae el artículo 26 del Anexo I y avisa de que no encuentra el del convenio; el artículo de vacaciones ni siquiera estaba entre los cinco fragmentos recuperados. La detección de la ambigüedad depende de que la recuperación traiga las dos lecturas.

Una rareza del juez: en `q59` (sin respuesta) da «correcta» pero «no fiel», porque la respuesta no cita ningún fragmento (y no tiene por qué). La fidelidad no está bien definida para una abstención, otra razón para no fiarse del juez hasta calibrarlo.

## Qué se demuestra y cómo se comprueba

| Competencia | Evidencia en este repositorio | Estado |
|---|---|---|
| RAG con citas | Ingesta del XML del BOE, búsqueda híbrida y respuestas con citas validadas contra los fragmentos entregados; resultados de recuperación arriba | hecho |
| Evals | 60 preguntas verificadas contra el texto (`evals/golden.jsonl`, con una prueba que comprueba cada cita literal), comparativa de 24 configuraciones y regresión de la recuperación en la CI | hecho; juez automático **sin calibrar** (falta la revisión humana) |
| Observabilidad de LLM | Traza por petición (etapas, tokens, coste, latencia, versión del prompt, configuración), informe de uso y topes diarios de gasto | hecho |
| Prompt engineering | Prompt versionado en `prompts/` con su registro de cambios | una versión medida; falta compararla con una segunda |
| API REST y webhooks | FastAPI con OpenAPI (`/docs`), endpoints de administración con token, y webhook de Telegram con secreto, límite por chat y deduplicación | hecho; el webhook está probado con mensajes simulados, no conectado a un bot real |
| Python tipado, pruebas e integración continua | 198 pruebas con 97 % de cobertura, escáner de secretos propio, CI en Ubuntu 24.04 y Windows (Python 3.12) | hecho |

## Arquitectura

```
 BOE (XML estructurado)
      │  ingesta: texto → troceado por artículo/anexo/tabla (ámbito, capítulo, fecha)
      ▼
 SQLite ── trozos + FTS5 (BM25) + vectores (embeddings locales con fastembed, sin claves)
      │
      │  recuperación híbrida: BM25 por raíz + vectores, fusionadas por rango recíproco (RRF)
      ▼
 Generación: cualquier endpoint compatible con la API de chat de OpenAI (p. ej. una pasarela LiteLLM)
      │         prompt versionado → JSON {respuesta, consta, citas[]}; las citas inventadas se descartan y se cuentan
      ▼
 API FastAPI ── POST /preguntar · POST /buscar · GET /salud            (públicos, con límite y tope diario)
      │         POST /ingestar · GET /trazas/{id} · GET /informe        (con token y límite propio)
      │         POST /webhook/telegram                                  (secreto, límite por chat, update_id únicos)
      ▼
 Trazas en SQLite ── etapa, tokens, coste, latencia, versión de prompt, configuración
```

## Puesta en marcha

Requiere Python 3.12 o superior y un `sqlite3` con FTS5 (lo comprueba `tests/test_sqlite_fts5.py`).

```bash
python -m venv .venv
source .venv/bin/activate         # Linux o macOS. En Windows: .venv\Scripts\Activate.ps1 (PowerShell) o source .venv/Scripts/activate (Git Bash)
pip install -e ".[dev]"
git config core.hooksPath .githooks   # una vez por clon: activa el escáner de secretos antes de cada commit
python -m pytest
```

El texto troceado ya está versionado (`datos/trozos.jsonl`), así que no hace falta descargar nada del BOE para usarlo.

```bash
python -m convenio_rag.indexar        # crea datos/convenio.db; descarga el modelo de embeddings la primera vez (0,64 GB)
cp .env.example .env                  # rellena LLM_BASE_URL, LLM_API_KEY y LLM_MODELO para poder generar respuestas
uvicorn convenio_rag.api:app_desde_entorno --factory
curl -s localhost:8000/buscar -H "content-type: application/json" -d '{"pregunta": "¿Cuántos días de vacaciones hay?"}'
```

Sin proveedor de modelos configurado, `/buscar` funciona y `/preguntar` responde 503. El `.env` nunca se sube: lo ignora `.gitignore` y lo vigila el escáner de secretos.

Evaluaciones:

```bash
python -m convenio_rag.evaluacion --sin-modelos        # lo que corre la CI: recuperación léxica, sin descargas
python -m convenio_rag.evaluacion                      # comparativa completa; descarga tres modelos y tarda en CPU
python -m convenio_rag.evaluacion_generacion --exportar-muestra   # cuesta dinero; escribe evals/calibracion/muestra.csv
```

## Límites conocidos

- **No es asesoramiento jurídico** y solo conoce los documentos del corpus.
- **Las preguntas de evaluación y sus respuestas de referencia las redactó un modelo y las comprobó una prueba automática, no un jurista.** La prueba garantiza que cada cita literal está en el artículo indicado, no que la respuesta de referencia sea completa.
- **El conjunto es pequeño** (60 preguntas): lo que se mide son tendencias, no diferencias finas.
- **El juez automático no está calibrado.** Sus veredictos no valen hasta que una persona revise una muestra (el CSV de calibración está preparado).
- **Una sola ejecución, un modelo y un prompt** en la evaluación de la generación; no se ha medido la variabilidad entre ejecuciones.
- La búsqueda léxica no tiene raíces reales en español (se aproxima truncando términos): «vacación» y «vacaciones» se igualan con la opción de raíz, pero otras flexiones no.
- Las **tablas** salariales se convierten a texto plano: las celdas combinadas del original pierden su alineación, y algunas filas quedan sin su grupo profesional.
- La defensa contra una orden dentro de la pregunta (inyección de prompt) es parcial: se neutraliza la estructura del mensaje y el prompt la prohíbe, y hay preguntas de prueba, pero no es una garantía. Se asume que los fragmentos (texto oficial del BOE) son de confianza.
- El limitador de peticiones vive en la memoria de un proceso y, detrás de un proxy, vería la IP del proxy.
- Sin *fine-tuning*, sin interfaz gráfica, sin multiusuario y sin memoria de conversación.

## Licencia

El código es [MIT](LICENSE). El texto de los convenios procede del BOE y esa licencia no lo cubre: se reutiliza bajo las [condiciones del propio BOE](https://www.boe.es/informacion/aviso_legal/) (ver [`datos/LEEME.md`](datos/LEEME.md)).
