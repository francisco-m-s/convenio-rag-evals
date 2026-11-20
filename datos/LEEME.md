# Datos: el texto del BOE y cómo se ha adaptado

> **Basado en datos de la [Agencia Estatal Boletín Oficial del Estado](https://www.boe.es).**
> Esto **no es el texto oficial**. Solo tienen carácter oficial y auténtico los textos publicados en la edición electrónica del BOE. Si hay diferencias con lo que aquí se ofrece, manda el BOE. El proyecto no está patrocinado ni respaldado por la Agencia.

## Documentos

| Documento | Identificador | Texto oficial |
|---|---|---|
| XVIII Convenio colectivo estatal para las empresas del comercio de flores y plantas | `BOE-A-2025-21424` | [boe.es](https://www.boe.es/buscar/doc.php?id=BOE-A-2025-21424) |
| Acta de la Comisión mixta con las tablas salariales de 2026 del mismo convenio | `BOE-A-2026-10861` | [boe.es](https://www.boe.es/diario_boe/txt.php?id=BOE-A-2026-10861) |

La fecha de última actualización de cada documento según el BOE y la huella SHA-256 del XML descargado están en [`fuentes.json`](fuentes.json). Cada trozo de [`trozos.jsonl`](trozos.jsonl) lleva su fecha en el campo `actualizado`. Si el BOE cambia un documento, la huella deja de coincidir y se nota.

## Condiciones de reutilización (D08)

El [aviso legal del BOE](https://www.boe.es/informacion/aviso_legal/) (licencia tipo aprobada por Resolución de la Agencia de 27 de junio de 2024) permite la reutilización para fines comerciales y no comerciales, incluidas la extracción y la reordenación, con condiciones. Se leyó el 20/11/2026 y así se cumplen:

| Condición del aviso legal | Cómo se cumple aquí |
|---|---|
| No desnaturalizar el sentido de la información | Solo se cambia la forma (lista de abajo). Las pruebas comprueban que las citas literales del conjunto de evaluación siguen apareciendo tal cual en el texto troceado |
| Citar la fuente con un enlace a boe.es, con la fórmula «Basado en datos de la Agencia Estatal Boletín Oficial del Estado» | Este fichero, el README y el campo `fuente` de cada respuesta de la API |
| No sugerir que la información tiene carácter oficial | Aviso en este fichero, en el README y en el campo `aviso` de cada respuesta |
| No sugerir que la Agencia participa, patrocina o apoya la reutilización | No se usa su marca ni su logotipo |
| Mencionar la fecha de última actualización | `fuentes.json` y el campo `actualizado` de cada trozo |
| Identificar toda modificación o adaptación | Lista siguiente |
| Respetar la protección de datos personales (RGPD) | Se omiten los nombres de personas (ver omisiones) |

## Modificaciones respecto al original

1. **Fuente:** el XML estructurado que publica el BOE (`xml.php?id=…`), no el PDF ni el HTML.
2. **Troceado:** un trozo por artículo, disposición, apartado del acta, anexo o tabla. Los artículos de más de 3.000 caracteres se parten por párrafos; cada parte repite la cabecera del artículo.
3. **Identificadores y citas propios:** `id`, `base`, `cita`, `contexto` y similares no forman parte del original; los añade este proyecto.
4. **Espacios:** se normalizan (espacios no separables y similares pasan a espacio normal).
5. **Tablas:** se convierten a texto plano, una línea por fila y las celdas separadas por « | ». Las celdas combinadas del original pierden su alineación. Las tablas del Anexo II (plantillas de formulario vacías) quedan dentro del texto corrido de ese anexo.
6. **Omisiones deliberadas:** la lista de asistentes del acta de 2026 y la línea de firma de las dos resoluciones, porque contienen nombres de personas. Nada más se omite.

## Cómo regenerarlo

```
python -m convenio_rag.ingesta            # descarga el XML a datos/crudo/ (no se versiona) y lo trocea
python -m convenio_rag.ingesta --offline  # solo trocea el XML ya descargado
```
