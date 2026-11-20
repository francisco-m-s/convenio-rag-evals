"""Tipos de datos compartidos por el resto del paquete."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Trozo:
    """Un fragmento citable del texto del BOE (un artículo, o una parte de él).

    ``base`` identifica el artículo, anexo o tabla de origen y es lo que se cita;
    ``id`` distingue las partes cuando un artículo largo se divide.
    """

    id: str
    base: str
    documento: str  # identificador BOE, p. ej. BOE-A-2025-21424
    tipo: str  # articulo | disposicion | resolucion | anexo | tabla
    numero: str  # número o ordinal del artículo; vacío si no aplica
    titulo: str  # puede estar vacío: hay artículos sin título (D06)
    contexto: str  # capítulo o título en el que está, para situar la cita
    cita: str  # texto legible de la cita, p. ej. «Convenio, art. 26 (Vacaciones)»
    texto: str
    orden: int
    actualizado: str  # fecha ISO (AAAA-MM-DD) de última actualización según el BOE
    parte: int = 1
    partes: int = 1

    def a_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def de_dict(datos: dict) -> "Trozo":
        return Trozo(**datos)


@dataclass(frozen=True)
class Resultado:
    """Un trozo recuperado, con su puntuación y el método que lo devolvió."""

    trozo: Trozo
    puntuacion: float
    posicion: int  # 1 = el mejor
