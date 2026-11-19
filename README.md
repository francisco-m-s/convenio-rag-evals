# convenio-rag-evals

Asistente que responde preguntas sobre un convenio colectivo del BOE **citando el artículo exacto**, con una suite de evaluación que mide si acierta y una capa de trazas que dice cuánto cuesta cada respuesta.

> **Estado: fase 0 cerrada (esqueleto).** Todavía no hay asistente ni resultados. Este README se llena con cifras reales a medida que cada fase termina; lo que aún no está medido figura como **pendiente**. Plan completo en [`docs/PLAN.md`](docs/PLAN.md).

> **Aviso:** esto es un proyecto de ingeniería, **no asesoramiento jurídico**. Una respuesta del asistente no sustituye la lectura del texto oficial ni la consulta a un profesional.

## Por qué existe

Una pyme o una gestoría que consulta un convenio colectivo se pierde entre artículos: vacaciones, jornada, pluses, permisos. Un chatbot que «suena bien» no sirve si inventa un plazo. Este proyecto construye el asistente **y la forma de demostrar que no inventa**: cada respuesta lleva su cita, y una batería de preguntas con respuesta conocida mide los aciertos, las citas correctas y las abstenciones cuando el texto no dice nada.

## Corpus

| Documento | Identificador BOE | Publicado |
|---|---|---|
| [XVIII Convenio colectivo estatal para las empresas del comercio de flores y plantas](https://www.boe.es/buscar/doc.php?id=BOE-A-2025-21424) (vigencia 2024–2028) | `BOE-A-2025-21424` | 24/10/2025 |
| [Acuerdo de incremento salarial y tablas para 2026 del mismo convenio](https://www.boe.es/diario_boe/txt.php?id=BOE-A-2026-10861) | `BOE-A-2026-10861` | 20/05/2026 |

Fuente oficial: [boe.es](https://www.boe.es/). El sistema no depende del convenio elegido: el corpus es intercambiable. Se incluyen a propósito las tablas salariales de 2026, que el BOE publicó en un documento aparte: obliga a resolver preguntas que cruzan dos textos.

## Qué se demuestra y cómo se comprueba

Cada fila es una competencia con su evidencia verificable en este repositorio. Si una fila no tiene evidencia, no se cuenta como hecha.

| Competencia | Evidencia que se podrá comprobar | Estado |
|---|---|---|
| RAG con citas | `hit@k` y `MRR` de la recuperación, en la tabla de resultados | pendiente (fases 1 y 3) |
| Evals | Carpeta `evals/` con el conjunto de preguntas, resultados versionados y acuerdo del juez automático con la revisión manual | pendiente (fase 3) |
| Observabilidad de LLM | Informe de costes y una traza completa por petición | pendiente (fase 2) |
| Prompt engineering | `prompts/CHANGELOG.md` con cada cambio medido contra el conjunto de evaluación | pendiente (fase 3) |
| API REST y webhooks | OpenAPI generado y pruebas de contrato | pendiente (fases 1 y 4) |
| Python tipado, pruebas e integración continua | `pytest` y escáner de secretos, ejecutados en la [integración continua](https://github.com/francisco-m-s/convenio-rag-evals/actions) (Ubuntu 24.04 y Windows, Python 3.12) | parcial (fase 0) |

## Resultados

**Pendiente (fase 3).** Aquí irá una tabla con al menos cuatro configuraciones (troceado por artículo frente a tamaño fijo, solo vectores frente a híbrida, valores de `k`, con y sin reordenación), con `hit@k`, `MRR`, fidelidad, abstención correcta, coste y latencia. Una tabla con cifras reales, aunque no sean buenas, vale más que un README sin números.

## Arquitectura prevista

```
 BOE (HTML)
      │  ingesta: texto → troceado por artículo (capítulo, artículo, fuente, fecha)
      ▼
 SQLite ── tabla de trozos + FTS5 (BM25) + vectores (embeddings locales)
      │
      │  recuperación híbrida: BM25 + vectores, fusionadas (RRF), top-k configurable
      ▼
 Generación (LLM): prompt versionado, salida JSON {respuesta, citas[], consta: true|false}
      │
      ▼
 API FastAPI ── POST /preguntar · POST /ingestar (con token) · GET /salud · GET /trazas/{id}
      │            └─ webhook de entrada con secreto o firma HMAC
      ▼
 Trazas en SQLite ── etapa, tokens, coste, latencia, versión de prompt, configuración
```

Cada decisión de diseño, con la alternativa descartada y su estado (decidida o por verificar), está en [`docs/DECISIONES.md`](docs/DECISIONES.md).

## Puesta en marcha

Requiere Python 3.12 o superior y un `sqlite3` con FTS5 (lo comprueba `tests/test_sqlite_fts5.py`).

```bash
python -m venv .venv
source .venv/bin/activate         # Linux o macOS. En Windows: .venv\Scripts\Activate.ps1 (PowerShell) o source .venv/Scripts/activate (Git Bash)
pip install -e ".[dev]"
git config core.hooksPath .githooks   # una vez por clon: activa el escáner de secretos antes de cada commit
python -m pytest
python herramientas/comprobar_secretos.py
```

Las claves de los proveedores de modelos van en un `.env` local (hay una plantilla sin valores en `.env.example`). El `.env` nunca se sube: lo ignora `.gitignore` y lo vigila el escáner.

## Límites conocidos

- No es asesoramiento jurídico y solo conoce los documentos del corpus.
- La búsqueda léxica usa FTS5 sin derivación de palabras en español (no hay raíces): «vacación» y «vacaciones» no se igualan.
- Sin *fine-tuning*, sin interfaz gráfica, sin multiusuario y sin memoria de conversación: una API, un bot de entrada, trazas y evals.
- Estos límites se ampliarán en la fase 5 con lo que falle de verdad.

## Licencia

El código es [MIT](LICENSE). El texto de los convenios procede del BOE (fuente oficial) y esa licencia no lo cubre. Las condiciones para reutilizarlo están por verificar (D08 en [`docs/DECISIONES.md`](docs/DECISIONES.md)); mientras tanto, la referencia es [el aviso legal del BOE](https://www.boe.es/informacion/aviso_legal/).
