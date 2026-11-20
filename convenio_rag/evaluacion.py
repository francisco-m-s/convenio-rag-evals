"""Evaluación de la recuperación: ``hit@k``, ``MRR`` y comparativa de configuraciones.

Una pregunta del conjunto (``evals/golden.jsonl``) lista los artículos que la responden (``esperados``).
Un trozo recuperado acierta si su artículo de origen (``base``) es uno de ellos. Con ``modo_acierto``
«alguno» basta con uno; con «todos» (preguntas que cruzan artículos) hacen falta todos en el top-k.

Uso::

    python -m convenio_rag.evaluacion                 # compara y escribe evals/resultados/
    python -m convenio_rag.evaluacion --sin-modelos   # solo búsqueda léxica (no descarga modelos)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import date
from itertools import chain
from pathlib import Path

from convenio_rag.almacen import Almacen
from convenio_rag.busqueda import Buscador, ConfigBusqueda
from convenio_rag.entorno import salida_utf8
from convenio_rag.embeddings import Embedder, FastEmbedder
from convenio_rag.ingesta import cargar_jsonl
from convenio_rag.modelos import Resultado, Trozo

CORTES = (1, 3, 5, 10)
MODELOS_COMPARADOS = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "jinaai/jina-embeddings-v2-base-es",
    "intfloat/multilingual-e5-large",
)
CORPUS_FIJOS = ((800, 100), (1500, 150))  # (tamaño, solape) en caracteres


@dataclass(frozen=True)
class Pregunta:
    id: str
    tipo: str
    pregunta: str
    esperados: tuple[str, ...]
    modo_acierto: str
    respuesta: str
    evidencia: tuple[str, ...]
    terminos_ausentes: tuple[str, ...]


def cargar_golden(ruta: Path) -> list[Pregunta]:
    preguntas = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        d = json.loads(linea)
        preguntas.append(
            Pregunta(
                id=d["id"], tipo=d["tipo"], pregunta=d["pregunta"], esperados=tuple(d["esperados"]),
                modo_acierto=d["modo_acierto"], respuesta=d["respuesta"], evidencia=tuple(d.get("evidencia", [])),
                terminos_ausentes=tuple(d.get("terminos_ausentes", [])),
            )
        )
    return preguntas


# --- corpus alternativo: trozos de tamaño fijo --------------------------------------------------
def _cuerpo_completo(partes: list[Trozo]) -> str:
    """Texto original de un artículo: la primera parte entera y, de las demás, sin la cabecera repetida."""
    textos = [partes[0].texto] + [p.texto.split("\n", 1)[1] if "\n" in p.texto else "" for p in partes[1:]]
    return "\n".join(t for t in textos if t)


def trozos_tamano_fijo(trozos: list[Trozo], tamano: int, solape: int) -> list[Trozo]:
    """Corta el texto, seguido, en ventanas de ``tamano`` caracteres con ``solape``, sin respetar los artículos.

    Cada ventana guarda en ``base`` los artículos que toca, unidos por «|»: así se puede medir con la misma
    regla de acierto (la ventana acierta si toca un artículo esperado).
    """
    if solape >= tamano:
        raise ValueError("el solape debe ser menor que el tamaño")
    agrupados: dict[str, list[Trozo]] = {}
    for trozo in sorted(trozos, key=lambda t: t.orden):
        agrupados.setdefault(trozo.base, []).append(trozo)
    texto, tramos, inicio = [], [], 0
    for base, partes in agrupados.items():
        cuerpo = _cuerpo_completo(partes) + "\n\n"
        tramos.append((inicio, inicio + len(cuerpo), base))
        texto.append(cuerpo)
        inicio += len(cuerpo)
    continuo = "".join(texto)
    ventanas = []
    for n, desde in enumerate(range(0, len(continuo), tamano - solape)):
        hasta = min(desde + tamano, len(continuo))
        bases = [base for (a, b, base) in tramos if a < hasta and b > desde]
        ventanas.append(
            Trozo(
                id=f"fijo{tamano}-{n:04d}", base="|".join(bases), documento="", tipo="ventana", numero=str(n),
                titulo="", contexto="", cita=f"ventana {n}", texto=continuo[desde:hasta], orden=n, actualizado="",
            )
        )
        if hasta == len(continuo):
            break
    return ventanas


# --- métricas -----------------------------------------------------------------------------------
def bases_de(resultados: list[Resultado]) -> set[str]:
    return set(chain.from_iterable(r.trozo.base.split("|") for r in resultados))


def acierto_en_corte(resultados: list[Resultado], pregunta: Pregunta, corte: int) -> bool:
    bases = bases_de(resultados[:corte])
    esperados = set(pregunta.esperados)
    return esperados <= bases if pregunta.modo_acierto == "todos" else bool(esperados & bases)


def posicion_primer_acierto(resultados: list[Resultado], pregunta: Pregunta) -> int | None:
    """Posición a partir de la cual la pregunta está acertada, con la misma regla que ``hit@k``.

    Con «alguno» es la del primer trozo que toca un artículo esperado; con «todos» es la del trozo que completa el
    último artículo que faltaba. Así el MRR no sale inflado en las preguntas que cruzan artículos.
    """
    esperados, vistos = set(pregunta.esperados), set()
    for r in resultados:
        vistos |= set(r.trozo.base.split("|")) & esperados
        if (vistos == esperados) if pregunta.modo_acierto == "todos" else vistos:
            return r.posicion
    return None


def evaluar(buscador: Buscador, preguntas: list[Pregunta], config: ConfigBusqueda) -> dict:
    """Métricas de recuperación (solo preguntas con artículos esperados) para una configuración."""
    con_respuesta = [p for p in preguntas if p.esperados]
    filas = []
    for pregunta in con_respuesta:
        resultados = buscador.buscar(pregunta.pregunta, ConfigBusqueda(**{**config.__dict__, "k": max(CORTES)}))
        posicion = posicion_primer_acierto(resultados, pregunta)
        filas.append(
            {
                "id": pregunta.id, "tipo": pregunta.tipo, "posicion": posicion,
                **{f"hit@{c}": acierto_en_corte(resultados, pregunta, c) for c in CORTES},
            }
        )
    n = len(filas)
    metricas = {f"hit@{c}": round(sum(f[f"hit@{c}"] for f in filas) / n, 3) for c in CORTES}
    metricas["mrr@10"] = round(sum(1 / f["posicion"] for f in filas if f["posicion"]) / n, 3)
    metricas["n"] = n
    return {"metricas": metricas, "fallos@5": [f["id"] for f in filas if not f["hit@5"]]}


# --- comparativa --------------------------------------------------------------------------------
def _construir(trozos: list[Trozo], embedders: list[Embedder]) -> Buscador:
    almacen = Almacen()
    almacen.cargar(trozos)
    ids, textos = [t.id for t in trozos], [t.texto for t in trozos]
    buscador = Buscador(almacen)
    for embedder in embedders:
        import numpy as np

        matriz = np.vstack([embedder.pasajes(textos[i : i + 32]) for i in range(0, len(textos), 32)])
        almacen.guardar_vectores(embedder.nombre, ids, matriz)
    return buscador


def comparar(trozos: list[Trozo], preguntas: list[Pregunta], modelos: list[str]) -> list[dict]:
    """Ejecuta la comparativa completa y devuelve una fila por configuración."""
    corpus = {"articulo": trozos} | {f"fijo{t}": trozos_tamano_fijo(trozos, t, s) for t, s in CORPUS_FIJOS}
    embedders = [FastEmbedder(m) for m in modelos]
    filas: list[dict] = []
    for nombre, lista in corpus.items():
        buscador = _construir(lista, embedders)
        for prefijo in (None, 6):
            config = ConfigBusqueda(modo="lexica", prefijo=prefijo)
            filas.append(_fila(nombre, "lexica", "—", prefijo, len(lista), evaluar(buscador, preguntas, config)))
        for embedder in embedders:
            buscador.embedder = embedder
            buscador.recargar_vectores()
            for modo, prefijo in (("vectorial", None), ("hibrida", 6)):
                config = ConfigBusqueda(modo=modo, prefijo=prefijo)
                filas.append(
                    _fila(nombre, modo, embedder.nombre, prefijo, len(lista), evaluar(buscador, preguntas, config))
                )
    return filas


def _fila(corpus: str, modo: str, modelo: str, prefijo: int | None, n_trozos: int, evaluacion: dict) -> dict:
    return {"corpus": corpus, "trozos": n_trozos, "modo": modo, "modelo": modelo, "raiz": prefijo, **evaluacion}


def estadisticas_troceado(trozos: list[Trozo]) -> list[dict]:
    """Forma de cada corpus: cuántos trozos, qué tamaño y qué parte cruza varios artículos (cita ambigua)."""
    corpus = {"articulo": trozos} | {f"fijo{t}": trozos_tamano_fijo(trozos, t, s) for t, s in CORPUS_FIJOS}
    filas = []
    for nombre, lista in corpus.items():
        tamanos = sorted(len(t.texto) for t in lista)
        multiples = sum(1 for t in lista if "|" in t.base)
        filas.append(
            {
                "corpus": nombre, "trozos": len(lista), "caracteres_mediana": tamanos[len(tamanos) // 2],
                "caracteres_max": tamanos[-1], "pct_cruzan_articulos": round(100 * multiples / len(lista), 1),
            }
        )
    return filas


def tabla_troceado(filas: list[dict]) -> str:
    cabecera = "| Corpus | Trozos | Caracteres (mediana) | Caracteres (máx.) | % que cruzan artículos |\n|---|---|---|---|---|\n"
    return cabecera + "\n".join(
        f"| {f['corpus']} | {f['trozos']} | {f['caracteres_mediana']} | {f['caracteres_max']} | "
        f"{f['pct_cruzan_articulos']:.1f} |" for f in filas
    ) + "\n"


def tabla_markdown(filas: list[dict], corpus: str | None = None) -> str:
    """Tabla en Markdown con las filas (opcionalmente solo las de un corpus)."""
    cabecera = "| Corpus | Modo | Modelo | Raíz | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 |\n|---|---|---|---|---|---|---|---|---|\n"
    lineas = []
    for f in filas:
        if corpus and f["corpus"] != corpus:
            continue
        m = f["metricas"]
        modelo = f["modelo"].split("/")[-1]
        lineas.append(
            f"| {f['corpus']} | {f['modo']} | {modelo} | {f['raiz'] or '—'} | {m['hit@1']:.2f} | {m['hit@3']:.2f} | "
            f"{m['hit@5']:.2f} | {m['hit@10']:.2f} | {m['mrr@10']:.2f} |"
        )
    return cabecera + "\n".join(lineas) + "\n"


def entorno(rutas: dict[str, Path]) -> dict:
    def sha(ruta: Path) -> str:
        return hashlib.sha256(ruta.read_bytes()).hexdigest()[:16]

    try:
        import fastembed

        version_fastembed = getattr(fastembed, "__version__", "desconocida")
    except ImportError:
        version_fastembed = "no instalado"
    return {
        "ejecutado": date.today().isoformat(), "python": platform.python_version(), "fastembed": version_fastembed,
        **{f"sha256_{nombre}": sha(ruta) for nombre, ruta in rutas.items()},
    }


def main(argv: list[str] | None = None) -> int:
    salida_utf8()
    analizador = argparse.ArgumentParser(description="Compara configuraciones de recuperación.")
    analizador.add_argument("--trozos", type=Path, default=Path("datos/trozos.jsonl"))
    analizador.add_argument("--golden", type=Path, default=Path("evals/golden.jsonl"))
    analizador.add_argument("--salida", type=Path, default=Path("evals/resultados"))
    analizador.add_argument("--modelo", action="append", help="modelo de fastembed (se puede repetir)")
    analizador.add_argument("--sin-modelos", action="store_true", help="solo búsqueda léxica")
    args = analizador.parse_args(argv)

    trozos, preguntas = cargar_jsonl(args.trozos), cargar_golden(args.golden)
    modelos = [] if args.sin_modelos else (args.modelo or list(MODELOS_COMPARADOS))
    filas = comparar(trozos, preguntas, modelos)
    args.salida.mkdir(parents=True, exist_ok=True)
    resultado = {"entorno": entorno({"trozos": args.trozos, "golden": args.golden}), "filas": filas}
    (args.salida / "recuperacion.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (args.salida / "recuperacion.md").write_text(tabla_markdown(filas), encoding="utf-8", newline="\n")
    forma = estadisticas_troceado(trozos)
    (args.salida / "troceado.json").write_text(json.dumps(forma, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (args.salida / "troceado.md").write_text(tabla_troceado(forma), encoding="utf-8", newline="\n")
    print(tabla_markdown(filas))
    print(tabla_troceado(forma))
    return 0


if __name__ == "__main__":
    sys.exit(main())
