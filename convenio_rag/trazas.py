"""Trazas por petición en SQLite: etapas, tokens, coste, latencia, versión del prompt y configuración."""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ESQUEMA = """
CREATE TABLE IF NOT EXISTS trazas (
    id TEXT PRIMARY KEY, creado TEXT NOT NULL, pregunta TEXT NOT NULL, configuracion TEXT NOT NULL,
    version_prompt TEXT NOT NULL, modelo TEXT NOT NULL,
    ms_recuperacion REAL NOT NULL, ms_generacion REAL NOT NULL,
    tokens_entrada INTEGER NOT NULL, tokens_salida INTEGER NOT NULL, coste_usd REAL,
    consta INTEGER, citas TEXT NOT NULL, citas_invalidas TEXT NOT NULL, recuperados TEXT NOT NULL, error TEXT
);
CREATE INDEX IF NOT EXISTS trazas_creado ON trazas (creado);
"""
COLUMNAS_JSON = ("configuracion", "citas", "citas_invalidas", "recuperados")


def _sincronizado(metodo):
    @functools.wraps(metodo)
    def envoltorio(self, *args, **kwargs):
        with self._candado:
            return metodo(self, *args, **kwargs)

    return envoltorio


class Trazas:
    def __init__(self, ruta: str | Path = ":memory:"):
        self._candado = threading.RLock()
        self.con = sqlite3.connect(str(ruta), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(ESQUEMA)

    @_sincronizado
    def registrar(
        self, *, pregunta: str, configuracion: dict, version_prompt: str, modelo: str, ms_recuperacion: float,
        ms_generacion: float, tokens_entrada: int, tokens_salida: int, coste_usd: float | None, consta: bool | None,
        citas: list[str], citas_invalidas: list[str], recuperados: list[str], error: str | None = None,
    ) -> str:
        """Guarda una traza y devuelve su identificador."""
        id_traza = uuid.uuid4().hex
        with self.con:
            self.con.execute(
                "INSERT INTO trazas VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    id_traza, datetime.now(UTC).isoformat(timespec="seconds"), pregunta,
                    json.dumps(configuracion, ensure_ascii=False), version_prompt, modelo, ms_recuperacion,
                    ms_generacion, tokens_entrada, tokens_salida, coste_usd, None if consta is None else int(consta),
                    json.dumps(citas), json.dumps(citas_invalidas), json.dumps(recuperados), error,
                ),
            )
        return id_traza

    @_sincronizado
    def obtener(self, id_traza: str) -> dict | None:
        fila = self.con.execute("SELECT * FROM trazas WHERE id = ?", (id_traza,)).fetchone()
        if fila is None:
            return None
        datos = dict(fila)
        for columna in COLUMNAS_JSON:
            datos[columna] = json.loads(datos[columna])
        return datos

    @_sincronizado
    def informe(self, desde: str | None = None) -> dict:
        """Resumen de uso: peticiones, coste, latencia (p50 y p95), tasa de abstención y errores."""
        filas = self.con.execute(
            "SELECT ms_recuperacion + ms_generacion AS ms, coste_usd, consta, error, tokens_entrada, tokens_salida "
            "FROM trazas WHERE creado >= ?",
            (desde or "",),
        ).fetchall()
        n = len(filas)
        if n == 0:
            return {"peticiones": 0}
        ms = np.array([f["ms"] for f in filas])
        costes = [f["coste_usd"] for f in filas if f["coste_usd"] is not None]
        con_veredicto = [f["consta"] for f in filas if f["consta"] is not None]
        return {
            "peticiones": n,
            "coste_total_usd": round(sum(costes), 6),
            "coste_medio_usd": round(sum(costes) / len(costes), 6) if costes else None,
            "peticiones_con_coste": len(costes),
            "latencia_ms_p50": round(float(np.percentile(ms, 50)), 1),
            "latencia_ms_p95": round(float(np.percentile(ms, 95)), 1),
            "tasa_abstencion": round(sum(1 for c in con_veredicto if not c) / len(con_veredicto), 3)
            if con_veredicto else None,
            "errores": sum(1 for f in filas if f["error"]),
            "tokens_entrada": sum(f["tokens_entrada"] for f in filas),
            "tokens_salida": sum(f["tokens_salida"] for f in filas),
        }

    @_sincronizado
    def peticiones_del_dia(self, dia: str) -> int:
        """Número de peticiones registradas cuyo ``creado`` empieza por ``dia`` (AAAA-MM-DD, UTC)."""
        return int(self.con.execute("SELECT COUNT(*) FROM trazas WHERE creado LIKE ?", (f"{dia}%",)).fetchone()[0])

    @_sincronizado
    def coste_del_dia(self, dia: str) -> float:
        """Coste acumulado de las peticiones cuyo ``creado`` empieza por ``dia`` (AAAA-MM-DD)."""
        fila = self.con.execute(
            "SELECT COALESCE(SUM(coste_usd), 0) FROM trazas WHERE creado LIKE ?", (f"{dia}%",)
        ).fetchone()
        return float(fila[0])
