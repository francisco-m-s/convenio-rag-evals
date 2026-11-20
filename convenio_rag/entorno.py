"""Carga de un fichero ``.env`` sencillo (``NOMBRE=valor``, un comentario por línea completa con ``#``)."""

from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping
from pathlib import Path


def cargar_dotenv(ruta: Path = Path(".env"), entorno: MutableMapping[str, str] | None = None) -> int:
    """Pone en el entorno las variables del fichero que no estén ya definidas y devuelve cuántas añadió.

    Las variables ya definidas en el entorno mandan sobre el fichero; las vacías se ignoran (así un
    ``.env.example`` copiado sin rellenar no pisa nada). No se admiten comentarios detrás de un valor.
    """
    destino = os.environ if entorno is None else entorno
    if not ruta.is_file():
        return 0
    añadidas = 0
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        nombre, valor = (parte.strip() for parte in linea.split("=", 1))
        valor = valor.strip("\"'")
        if nombre and valor and nombre not in destino:
            destino[nombre] = valor
            añadidas += 1
    return añadidas


def salida_utf8() -> None:
    """Evita errores de codificación al imprimir tildes y símbolos en consolas de Windows con una página de códigos antigua."""
    for flujo in (sys.stdout, sys.stderr):
        reconfigurar = getattr(flujo, "reconfigure", None)
        if reconfigurar is not None:
            reconfigurar(encoding="utf-8", errors="replace")
