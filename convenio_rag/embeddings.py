"""Embeddings locales (decisión D04): ONNX con ``fastembed``, sin PyTorch y sin claves de API.

Los vectores salen normalizados (norma 1), de modo que el coseno es un producto escalar.
``HashEmbedder`` es un sustituto determinista, sin descargas, para las pruebas.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from typing import Protocol

import numpy as np

# Algunos modelos exigen un prefijo distinto para la pregunta y para el texto buscado.
PREFIJOS = {
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}
# Medido en evals/resultados/recuperacion.md: en este corpus supera claramente al MiniLM multilingüe (D04) y pesa
# una tercera parte que multilingual-e5-large (0,64 GB frente a 2,24 GB).
MODELO_DEFECTO = "jinaai/jina-embeddings-v2-base-es"


class Embedder(Protocol):
    nombre: str

    def pasajes(self, textos: list[str]) -> np.ndarray: ...

    def consultas(self, textos: list[str]) -> np.ndarray: ...


def _normalizar(matriz: np.ndarray) -> np.ndarray:
    matriz = np.asarray(matriz, dtype=np.float32)
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    return matriz / np.where(normas == 0, 1.0, normas)


class FastEmbedder:
    """Embeddings con un modelo de ``fastembed`` (se descarga la primera vez y queda en caché)."""

    def __init__(self, modelo: str = MODELO_DEFECTO):
        self.nombre = modelo
        self._modelo = None

    def _cargar(self):
        if self._modelo is None:
            from fastembed import TextEmbedding  # import tardío: es pesado y no hace falta en la búsqueda léxica

            self._modelo = TextEmbedding(self.nombre)
        return self._modelo

    def _embeber(self, textos: list[str], prefijo: str) -> np.ndarray:
        if not textos:
            return np.zeros((0, 0), dtype=np.float32)
        vectores = list(self._cargar().embed([prefijo + t for t in textos]))
        return _normalizar(np.vstack(vectores))

    def pasajes(self, textos: list[str]) -> np.ndarray:
        return self._embeber(textos, PREFIJOS.get(self.nombre, ("", ""))[1])

    def consultas(self, textos: list[str]) -> np.ndarray:
        return self._embeber(textos, PREFIJOS.get(self.nombre, ("", ""))[0])


class HashEmbedder:
    """Bolsa de raíces con *hashing* a ``dim`` dimensiones. Determinista y sin red: solo para pruebas."""

    def __init__(self, dim: int = 256):
        self.nombre = f"hash-{dim}"
        self.dim = dim

    def _vector(self, texto: str) -> np.ndarray:
        sin_tildes = unicodedata.normalize("NFKD", texto.casefold()).encode("ascii", "ignore").decode()
        vector = np.zeros(self.dim, dtype=np.float32)
        for palabra in re.findall(r"[a-z0-9]+", sin_tildes):
            if len(palabra) > 2:
                vector[zlib.crc32(palabra[:5].encode()) % self.dim] += 1.0
        return vector

    def _matriz(self, textos: list[str]) -> np.ndarray:
        if not textos:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalizar(np.vstack([self._vector(t) for t in textos]))

    def pasajes(self, textos: list[str]) -> np.ndarray:
        return self._matriz(textos)

    def consultas(self, textos: list[str]) -> np.ndarray:
        return self._matriz(textos)
