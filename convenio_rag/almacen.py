"""Almacén SQLite: trozos, índice léxico FTS5 (BM25) y vectores como BLOB (decisiones D02, D03, D07)."""

from __future__ import annotations

import functools
import re
import sqlite3
import threading
import unicodedata
from pathlib import Path

import numpy as np

from convenio_rag.modelos import Trozo

ESQUEMA = """
CREATE TABLE IF NOT EXISTS trozos (
    id TEXT PRIMARY KEY, base TEXT NOT NULL, documento TEXT NOT NULL, tipo TEXT NOT NULL,
    numero TEXT NOT NULL, titulo TEXT NOT NULL, contexto TEXT NOT NULL, cita TEXT NOT NULL,
    texto TEXT NOT NULL, orden INTEGER NOT NULL, actualizado TEXT NOT NULL,
    parte INTEGER NOT NULL, partes INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS trozos_fts USING fts5(
    id UNINDEXED, texto, tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS vectores (
    trozo_id TEXT NOT NULL, modelo TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL,
    PRIMARY KEY (trozo_id, modelo)
);
"""

# Palabras vacías que no ayudan a una búsqueda por BM25 en español.
PALABRAS_VACIAS = frozenset(
    "a al algo ante como con cual cuales cuando cuanto cuantos cuanta cuantas de del desde donde e el ella ellos en entre "
    "era es esta estan este esto estos ha han hay la las le les lo los me mi mis muy no nos o para pero por que quien se "
    "ser si sin sobre su sus tiene tienen tengo un una uno unos unas y ya puede pueden puedo debe deben".split()
)


def _sin_tildes(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def terminos(pregunta: str, prefijo: int | None = None) -> list[str]:
    """Términos de búsqueda de una pregunta, sin palabras vacías y, si se pide, truncados a una raíz.

    ``prefijo=None`` busca la palabra exacta (D07: «vacación» y «vacaciones» no se igualan); un valor como 6
    trunca cada término a esa raíz y busca por prefijo, que es una aproximación barata a una raíz en español.
    """
    resultado: list[str] = []
    for palabra in re.findall(r"\w+", pregunta.casefold()):
        if _sin_tildes(palabra) in PALABRAS_VACIAS or (len(palabra) < 2 and not palabra.isdigit()):
            continue  # se conservan los números de una cifra: «artículo 5»
        resultado.append(palabra)
    if prefijo is None:
        return resultado
    return [palabra[:prefijo] + "*" if len(palabra) > prefijo else palabra for palabra in resultado]


def _consulta_fts(lista: list[str]) -> str:
    """Construye la consulta FTS5: cada término entre comillas (para escapar sintaxis) y unidos por OR."""
    partes = []
    for termino in lista:
        prefijo = termino.endswith("*")
        base = termino.rstrip("*").replace('"', "")
        partes.append(f'"{base}"' + ("*" if prefijo else ""))
    return " OR ".join(partes)


def _sincronizado(metodo):
    """Serializa el acceso a la conexión: la API la comparte entre los hilos de FastAPI."""

    @functools.wraps(metodo)
    def envoltorio(self, *args, **kwargs):
        with self._candado:
            return metodo(self, *args, **kwargs)

    return envoltorio


class Almacen:
    def __init__(self, ruta: str | Path = ":memory:"):
        self._candado = threading.RLock()
        self.con = sqlite3.connect(str(ruta), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(ESQUEMA)

    def cerrar(self) -> None:
        self.con.close()

    # --- trozos --------------------------------------------------------------------------
    @_sincronizado
    def cargar(self, trozos: list[Trozo]) -> None:
        """Sustituye el contenido por ``trozos`` (y borra los vectores, que quedarían desfasados)."""
        with self.con:
            for tabla in ("trozos", "trozos_fts", "vectores"):
                self.con.execute(f"DELETE FROM {tabla}")  # noqa: S608 - nombres fijos, no entrada del usuario
            self.con.executemany(
                "INSERT INTO trozos VALUES (:id,:base,:documento,:tipo,:numero,:titulo,:contexto,:cita,:texto,"
                ":orden,:actualizado,:parte,:partes)",
                [t.a_dict() for t in trozos],
            )
            self.con.executemany("INSERT INTO trozos_fts (id, texto) VALUES (?, ?)", [(t.id, t.texto) for t in trozos])

    @_sincronizado
    def reemplazar(self, trozos: list[Trozo], vectores: dict[str, tuple[list[str], np.ndarray]]) -> None:
        """Sustituye trozos y vectores en una sola transacción: nunca se ve un índice a medias.

        ``vectores`` asocia el nombre de cada modelo con sus ids y su matriz, ya calculados fuera, de modo que
        si el cálculo de los embeddings falla no se ha tocado nada.
        """
        with self.con:
            for tabla in ("trozos", "trozos_fts", "vectores"):
                self.con.execute(f"DELETE FROM {tabla}")  # noqa: S608 - nombres fijos, no entrada del usuario
            self.con.executemany(
                "INSERT INTO trozos VALUES (:id,:base,:documento,:tipo,:numero,:titulo,:contexto,:cita,:texto,"
                ":orden,:actualizado,:parte,:partes)",
                [t.a_dict() for t in trozos],
            )
            self.con.executemany("INSERT INTO trozos_fts (id, texto) VALUES (?, ?)", [(t.id, t.texto) for t in trozos])
            for modelo, (ids, matriz) in vectores.items():
                matriz = np.asarray(matriz, dtype=np.float32)
                self.con.executemany(
                    "INSERT INTO vectores VALUES (?, ?, ?, ?)",
                    [(i, modelo, matriz.shape[1], fila.tobytes()) for i, fila in zip(ids, matriz, strict=True)],
                )

    @_sincronizado
    def tiene_vectores(self, modelo: str) -> bool:
        return self.con.execute("SELECT 1 FROM vectores WHERE modelo = ? LIMIT 1", (modelo,)).fetchone() is not None

    @_sincronizado
    def todos(self) -> list[Trozo]:
        filas = self.con.execute("SELECT * FROM trozos ORDER BY orden").fetchall()
        return [Trozo(**dict(fila)) for fila in filas]

    @_sincronizado
    def obtener(self, id_trozo: str) -> Trozo | None:
        fila = self.con.execute("SELECT * FROM trozos WHERE id = ?", (id_trozo,)).fetchone()
        return Trozo(**dict(fila)) if fila else None

    @_sincronizado
    def contar(self) -> int:
        return int(self.con.execute("SELECT COUNT(*) FROM trozos").fetchone()[0])

    # --- búsqueda léxica -----------------------------------------------------------------
    @_sincronizado
    def buscar_lexico(self, pregunta: str, k: int, prefijo: int | None = None) -> list[tuple[str, float]]:
        """Los ``k`` trozos con mejor BM25 como pares (id, puntuación); mayor puntuación = más relevante."""
        consulta = _consulta_fts(terminos(pregunta, prefijo))
        if not consulta:
            return []
        filas = self.con.execute(
            "SELECT id, bm25(trozos_fts) AS puntos FROM trozos_fts WHERE trozos_fts MATCH ? ORDER BY puntos LIMIT ?",
            (consulta, k),
        ).fetchall()
        return [(fila["id"], -float(fila["puntos"])) for fila in filas]  # bm25 de SQLite: menor = mejor

    # --- vectores ------------------------------------------------------------------------
    @_sincronizado
    def guardar_vectores(self, modelo: str, ids: list[str], matriz: np.ndarray) -> None:
        matriz = np.asarray(matriz, dtype=np.float32)
        with self.con:
            self.con.execute("DELETE FROM vectores WHERE modelo = ?", (modelo,))
            self.con.executemany(
                "INSERT INTO vectores VALUES (?, ?, ?, ?)",
                [(i, modelo, matriz.shape[1], fila.tobytes()) for i, fila in zip(ids, matriz, strict=True)],
            )

    @_sincronizado
    def vectores(self, modelo: str) -> tuple[list[str], np.ndarray]:
        filas = self.con.execute(
            "SELECT v.trozo_id, v.dim, v.vec FROM vectores v JOIN trozos t ON t.id = v.trozo_id "
            "WHERE v.modelo = ? ORDER BY t.orden",
            (modelo,),
        ).fetchall()
        if not filas:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = [fila["trozo_id"] for fila in filas]
        matriz = np.vstack([np.frombuffer(fila["vec"], dtype=np.float32) for fila in filas])
        return ids, matriz
