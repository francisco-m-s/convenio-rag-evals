"""Recuperación: léxica (BM25), vectorial (coseno) o híbrida con fusión RRF."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from convenio_rag.almacen import Almacen
from convenio_rag.embeddings import Embedder
from convenio_rag.modelos import Resultado

MODOS = ("lexica", "vectorial", "hibrida")


@dataclass(frozen=True)
class ConfigBusqueda:
    modo: str = "hibrida"
    k: int = 5  # trozos que se devuelven
    prefijo: int | None = None  # raíz (en caracteres) para la rama léxica; None = palabra exacta
    candidatos: int = 20  # trozos que aporta cada rama antes de fusionar
    rrf_k: int = 60  # constante de la fusión por rango recíproco

    def __post_init__(self) -> None:
        if self.modo not in MODOS:
            raise ValueError(f"modo desconocido: {self.modo!r} (válidos: {MODOS})")
        if self.k < 1:
            raise ValueError("k debe ser al menos 1")

    def etiqueta(self) -> str:
        raiz = f"/raíz{self.prefijo}" if self.prefijo and self.modo != "vectorial" else ""
        return f"{self.modo}{raiz}/k{self.k}"


def fusion_rrf(rankings: list[list[str]], rrf_k: int = 60) -> list[tuple[str, float]]:
    """Fusiona rankings de ids por rango recíproco: suma de 1 / (rrf_k + posición)."""
    puntos: dict[str, float] = {}
    for ranking in rankings:
        for posicion, id_trozo in enumerate(ranking, start=1):
            puntos[id_trozo] = puntos.get(id_trozo, 0.0) + 1.0 / (rrf_k + posicion)
    return sorted(puntos.items(), key=lambda par: (-par[1], par[0]))


class Buscador:
    def __init__(self, almacen: Almacen, embedder: Embedder | None = None):
        self.almacen = almacen
        self.embedder = embedder
        self._ids: list[str] = []
        self._matriz = np.zeros((0, 0), dtype=np.float32)
        if embedder is not None:
            self.recargar_vectores()

    def recargar_vectores(self) -> None:
        assert self.embedder is not None
        self._ids, self._matriz = self.almacen.vectores(self.embedder.nombre)

    def _vectorial(self, pregunta: str, n: int) -> list[tuple[str, float]]:
        if self.embedder is None or not self._ids:
            raise RuntimeError("no hay vectores indexados para este modelo: ejecuta `python -m convenio_rag.indexar`")
        consulta = self.embedder.consultas([pregunta])[0]
        similitudes = self._matriz @ consulta
        orden = np.argsort(-similitudes, kind="stable")[:n]
        return [(self._ids[i], float(similitudes[i])) for i in orden]

    def buscar(self, pregunta: str, config: ConfigBusqueda = ConfigBusqueda()) -> list[Resultado]:
        """Devuelve los ``config.k`` mejores trozos para la pregunta, del más al menos relevante."""
        n = max(config.candidatos, config.k)
        if config.modo == "hibrida" and not self._ids:
            # Sin vectores indexados para este modelo la híbrida degrada a léxica en vez de fallar: la búsqueda
            # sigue funcionando (peor) mientras se reindexa. El modo «vectorial» explícito sí lanza un error.
            config = replace(config, modo="lexica")
        if config.modo == "lexica":
            ordenados = self.almacen.buscar_lexico(pregunta, n, config.prefijo)
        elif config.modo == "vectorial":
            ordenados = self._vectorial(pregunta, n)
        else:
            lexico = self.almacen.buscar_lexico(pregunta, n, config.prefijo)
            vectorial = self._vectorial(pregunta, n)
            ordenados = fusion_rrf([[i for i, _ in lexico], [i for i, _ in vectorial]], config.rrf_k)
        resultados: list[Resultado] = []
        for id_trozo, puntos in ordenados[: config.k]:
            trozo = self.almacen.obtener(id_trozo)
            if trozo is not None:
                resultados.append(Resultado(trozo=trozo, puntuacion=puntos, posicion=len(resultados) + 1))
        return resultados
