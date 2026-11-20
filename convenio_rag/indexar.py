"""Construye la base de datos de búsqueda a partir de ``datos/trozos.jsonl``.

Uso::

    python -m convenio_rag.indexar                       # índice léxico + vectores del modelo por defecto
    python -m convenio_rag.indexar --modelo A --modelo B # varios modelos de embeddings
    python -m convenio_rag.indexar --sin-vectores        # solo búsqueda léxica (no descarga ningún modelo)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from convenio_rag.almacen import Almacen
from convenio_rag.entorno import salida_utf8
from convenio_rag.embeddings import MODELO_DEFECTO, Embedder, FastEmbedder
from convenio_rag.ingesta import cargar_jsonl

LOTE = 32


def indexar(
    trozos_jsonl: Path, db: Path, embedders: list[Embedder] | None = None, almacen: Almacen | None = None
) -> Almacen:
    """Crea (o rehace) el almacén en ``db`` con los trozos y los vectores de cada ``embedder``."""
    almacen = almacen or Almacen(db)
    trozos = cargar_jsonl(trozos_jsonl)
    textos = [t.texto for t in trozos]
    ids = [t.id for t in trozos]
    vectores = {}
    # Los embeddings se calculan ANTES de tocar el almacén y este se reemplaza en una sola transacción: si el cálculo
    # falla (o se interrumpe) el índice anterior sigue intacto, y mientras se reindexa nadie ve un índice a medias.
    for embedder in embedders or []:
        lotes = [embedder.pasajes(textos[i : i + LOTE]) for i in range(0, len(textos), LOTE)]
        vectores[embedder.nombre] = (ids, np.vstack(lotes))
    almacen.reemplazar(trozos, vectores)
    return almacen


def main(argv: list[str] | None = None) -> int:
    salida_utf8()
    analizador = argparse.ArgumentParser(description="Indexa los trozos del convenio.")
    analizador.add_argument("--trozos", type=Path, default=Path("datos/trozos.jsonl"))
    analizador.add_argument("--db", type=Path, default=Path("datos/convenio.db"))
    analizador.add_argument("--modelo", action="append", help="modelo de fastembed (se puede repetir)")
    analizador.add_argument("--sin-vectores", action="store_true")
    args = analizador.parse_args(argv)

    args.db.parent.mkdir(parents=True, exist_ok=True)
    if args.db.exists():
        args.db.unlink()
    embedders = [] if args.sin_vectores else [FastEmbedder(m) for m in (args.modelo or [MODELO_DEFECTO])]
    almacen = indexar(args.trozos, args.db, embedders)
    print(f"{almacen.contar()} trozos indexados en {args.db}; vectores de: {[e.nombre for e in embedders] or 'ninguno'}")
    almacen.cerrar()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
