"""Fixtures compartidas. El escáner vive en `herramientas/` (no es un paquete instalable), así que se carga por ruta."""

import importlib.util
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def raiz() -> Path:
    return RAIZ


@pytest.fixture(scope="session")
def escaner():
    ruta = RAIZ / "herramientas" / "comprobar_secretos.py"
    spec = importlib.util.spec_from_file_location("comprobar_secretos", ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["comprobar_secretos"] = modulo  # los dataclass buscan su módulo en sys.modules
    spec.loader.exec_module(modulo)
    return modulo
