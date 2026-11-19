# Plan del proyecto: asistente RAG con citas, evals y trazas

> **Repositorio:** `convenio-rag-evals`
> **Idea en una frase:** un asistente que responde preguntas sobre un texto legal público **citando el artículo exacto**, con una suite de evaluación que mide si acierta y una capa de trazas que dice cuánto cuesta cada respuesta.
> **Esfuerzo orientativo:** unas 2 semanas a media jornada. Cada fase termina con algo que se puede enseñar.
> **Estado (19/11/2026):** fase 0 en curso. Las decisiones de diseño tomadas y las que quedan por verificar están en [`DECISIONES.md`](DECISIONES.md); donde allí se matiza este plan (por ejemplo, embeddings locales para que la CI no necesite secretos), manda `DECISIONES.md`.

## 1. Qué problema resuelve

Una pyme (o una gestoría) tiene que consultar un convenio colectivo y se pierde entre artículos: vacaciones, jornada, pluses, permisos. Un chatbot que "suena bien" no sirve si inventa un plazo. Este proyecto construye el asistente **y la forma de demostrar que no inventa**: cada respuesta lleva su cita, y una batería de preguntas con respuesta conocida mide aciertos, citas correctas y abstenciones cuando el texto no dice nada.

**Corpus:** el XVIII Convenio colectivo estatal para las empresas del comercio de flores y plantas (`BOE-A-2025-21424`) y las tablas salariales de 2026 (`BOE-A-2026-10861`); la elección y la alternativa descartada están en [`DECISIONES.md`](DECISIONES.md) (D01). Se eligió un sector conocido por quien escribe las preguntas, para poder juzgar si una respuesta es correcta. Se respeta el aviso legal del BOE, se cita la fuente en el README y se indica que **no es asesoramiento jurídico**. El corpus es intercambiable: el sistema no depende del convenio elegido.

## 2. Qué se demuestra (y cómo se comprueba)

Cada fila es una competencia con su evidencia verificable en el repositorio. Si una fila no tiene evidencia, no se cuenta como hecha.

| Competencia | Dónde queda demostrada | Evidencia que cualquiera puede comprobar |
|---|---|---|
| **RAG** | Ingesta por artículos, búsqueda híbrida (BM25 + vectores), respuesta con citas | `hit@k` y `MRR` de la recuperación en la tabla del README |
| **Evals (evaluación)** | Conjunto de 50–60 preguntas con respuesta conocida, arnés de evaluación y comparativa de configuraciones | Carpeta `evals/`, resultados versionados, acuerdo entre el juez automático y la revisión manual |
| **Observabilidad de LLM** | Traza por petición: etapas, tokens, coste, latencia, versión del prompt | Informe de costes y una captura de una traza completa |
| **Prompt engineering** | Prompts versionados; cada cambio se mide contra el conjunto de evaluación | `prompts/CHANGELOG.md` con la mejora o el empeoramiento medidos |
| **APIs REST / webhooks** | API con FastAPI y un webhook de entrada con autenticación | OpenAPI generado y pruebas de contrato |
| **Python con IA** | Todo el repositorio: tipado, pruebas, integración continua | `pytest` y CI en verde |

## 3. Arquitectura

```
 BOE (PDF/HTML)
      │  ingesta: texto → troceado por artículo (con metadatos: capítulo, artículo, fuente, fecha)
      ▼
 SQLite ── tabla de trozos + FTS5 (BM25) + vectores (embeddings)
      │
      │  recuperación híbrida: BM25 + vectores, fusionadas (RRF), top-k configurable
      ▼
 Generación (LLM vía LiteLLM): prompt versionado, salida JSON {respuesta, citas[], consta: true|false}
      │
      ▼
 API FastAPI ── POST /preguntar · POST /ingestar (con token) · GET /salud · GET /trazas/{id}
      │            └─ webhook de entrada (p. ej. Telegram o n8n) con secreto o firma HMAC
      ▼
 Trazas en SQLite ── etapa, tokens, coste, latencia, versión de prompt, configuración
```

**Decisiones de diseño** (cada una con la alternativa descartada, documentada en el README):
- **SQLite** frente a una base vectorial dedicada: el corpus son unos pocos miles de trozos y así hay una dependencia menos.
- **Troceado por artículo** frente a trozos de tamaño fijo: las citas salen exactas. Se mide: es una de las comparativas.
- **Búsqueda híbrida** frente a solo vectores: en texto legal, los números y los términos exactos pesan. También se mide.
- **Abstención explícita** (`consta: false`): es mejor que el modelo diga "no consta" que que improvise.

## 4. Fases

| Fase | Qué se hace | Hecho cuando… |
|---|---|---|
| **0 · Preparación** (½ día) | Elegir el convenio (hecho, D01), crear el repo, esqueleto del README, `.env.example`, `.gitignore`, licencia, escáner de secretos | El repo clona y arranca sin secretos |
| **1 · Núcleo** (2–3 días) | Ingesta, troceado por artículo, búsqueda híbrida, `POST /preguntar` con citas | 10 preguntas a mano responden con la cita correcta |
| **2 · Trazas** (2 días) | Tabla `trazas`, coste por petición (LiteLLM lo da en cabecera), informe de costes, aviso si el gasto diario pasa un umbral | Una pregunta deja una traza completa y el informe suma bien |
| **3 · Evals** (3–4 días) | Conjunto de preguntas, métricas, juez automático calibrado con revisión manual, comparativa de configuraciones | Tabla de resultados en el README con al menos 4 configuraciones |
| **4 · Calidad continua** (1–2 días) | CI que ejecuta las evals baratas y falla si la recuperación empeora; webhook de entrada | Un cambio que empeora `hit@k` rompe la CI |
| **5 · Cierre** (1 día) | README con resultados, diagrama, **qué falló** y límites conocidos | Alguien ajeno entiende el proyecto en 5 minutos |

**Opcional (fase 6):** exponer `buscar_en_convenio` como herramienta MCP y un agente que la use. Solo si lo anterior está cerrado.

## 5. Evaluación

**Conjunto de preguntas (`evals/golden.jsonl`), 50–60 en total:**
- ~30 de respuesta directa en un solo artículo.
- ~10 que necesitan combinar dos o más artículos.
- ~10 **sin respuesta en el texto** (el sistema debe abstenerse).
- ~5 con ambigüedad o con una instrucción maliciosa dentro de la pregunta (inyección de prompt).

Cada pregunta guarda: la pregunta, la respuesta esperada, el/los artículos correctos y el tipo.

**Métricas:**
- **Recuperación:** `hit@k` (¿está el artículo correcto entre los k primeros?) y `MRR`. No necesitan LLM: son gratis y deterministas, y son lo que corre en la CI.
- **Generación:** fidelidad (¿cada afirmación está respaldada por los trozos citados?), exactitud de la cita, y tasa de abstención correcta.
- **Coste y latencia** por pregunta, con la distribución, no solo la media.

**Juez automático:** un LLM puntúa con una rúbrica escrita. **Calibrarlo:** se revisan a mano 20 respuestas y se publica el porcentaje de acuerdo con el juez. Si el acuerdo es bajo, la métrica no vale y se documenta.

**Comparativas a publicar (una variable cada vez):**
1. Troceado por artículo vs. tamaño fijo.
2. Solo vectores vs. híbrida.
3. `k` = 3, 5 y 8.
4. Con y sin reordenación de resultados.
5. Razonamiento bajo vs. por defecto del modelo (mide si merece la pena el coste).

Una tabla con cifras reales, aunque no sean buenas, vale más que un README sin números.

## 6. Observabilidad

Cada petición genera una traza con: identificador, etapa (recuperar, generar), tokens de entrada y salida, coste, latencia, versión del prompt y configuración (modelo, `k`, ajustes de razonamiento).

- **Primero, trazas propias en SQLite.** Son pocas líneas y obligan a entender qué se mide.
- **Después, opcional:** exportarlas a una herramienta de trazas de LLM de código abierto (Langfuse u otra), tras comprobar su plan gratuito y sus requisitos de alojamiento.
- **Alertas:** si el coste diario supera un umbral o la tasa de abstención se dispara, aviso al canal de mensajería elegido.

## 7. Seguridad y publicación

- Cero secretos en el repositorio: `.env.example`, `.gitignore`, y un comprobador de secretos como hook de `pre-commit` y en la CI.
- Los endpoints de escritura (`/ingestar`) con token. El webhook de entrada verifica un secreto o una firma.
- Límite de peticiones por minuto en la API.
- Un caso de inyección de prompt en las evals y un apartado de "Límites" en el README.
- Licencia MIT, y la fuente del corpus citada con enlace.

## 8. Alcance: lo que NO se hace

Sin *fine-tuning*, sin interfaz gráfica elaborada, sin multiusuario, sin memoria de conversación. Una API, un bot de entrada, trazas y evals. Es mejor un proyecto pequeño, medido y terminado que uno grande a medias.

## 9. Alternativas valoradas

| Opción | A favor | En contra |
|---|---|---|
| **Convenio del BOE (recomendada)** | Corpus público y verificable; respuestas con cita comprobable; caso real para pymes | Hay que escribir bien las preguntas de evaluación |
| RAG sobre la documentación de n8n | Fácil de montar | Muy visto; la documentación cambia y rompe la evaluación |
| Mapa "requisitos de una oferta ↔ evidencias de un portfolio" | Aprovecha datos propios | El corpus es el propio autor: menos objetivo y más difícil de presentar como producto |
