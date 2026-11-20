"""Descarga y troceado del texto del BOE (decisión D06).

La fuente es el XML estructurado que el BOE publica para cada documento
(``xml.php?id=...``) y no el HTML: marca con una clase cada artículo, capítulo,
anexo y tabla, de modo que no hay que adivinar la estructura. Las condiciones de
reutilización del BOE (D08) y las modificaciones que se hacen al texto están en
``datos/LEEME.md``.

Uso::

    python -m convenio_rag.ingesta            # descarga (si falta) y trocea
    python -m convenio_rag.ingesta --offline  # solo trocea el XML ya descargado
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from convenio_rag.entorno import salida_utf8
from convenio_rag.modelos import Trozo

URL_XML = "https://www.boe.es/diario_boe/xml.php?id={id}"
USER_AGENT = "convenio-rag-evals/0.1 (proyecto de portfolio; github.com/francisco-m-s/convenio-rag-evals)"
MAX_CHARS_DEFECTO = 3000
ROMANOS = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}

RE_ARTICULO = re.compile(r"^Artículo\s+(\d+)\.?\s*(.*)$")
RE_DISPOSICION = re.compile(r"^Disposición\s+(transitoria|adicional|final|derogatoria)\s+(\w+)\.?\s*(.*)$", re.I)
RE_ORDINAL = re.compile(r"^(Único|Primero|Segundo|Tercero|Cuarto|Quinto|Sexto)\.?\s*(.*)$", re.I)
# Firma de la resolución («Madrid, 7 de mayo de 2026.–La Directora General de Trabajo, <nombre>»).
RE_FIRMA = re.compile(r"[–-]\s*(La|El)\s+Director(a)?\s+General")


@dataclass(frozen=True)
class ConfigDocumento:
    id: str  # identificador BOE
    prefijo: str  # prefijo de los identificadores de trozo («conv», «tab»)
    principal: str  # nombre del ámbito principal en las citas («Convenio»)
    # Anexos cuyas tablas se dejan dentro del texto corrido en vez de ser un trozo cada una (D06): el Anexo II del
    # convenio son 28 tablas de formulario vacías, que como trozos sueltos solo añadirían ruido a la búsqueda.
    anexos_en_bloque: frozenset[str] = frozenset()


DOCUMENTOS = (
    ConfigDocumento("BOE-A-2025-21424", "conv", "Convenio", frozenset({"Anexo II"})),
    ConfigDocumento("BOE-A-2026-10861", "tab", "Acta 2026"),
)


def _limpiar(elemento: ET.Element) -> str:
    """Texto del elemento con los espacios (también los no separables) normalizados."""
    return " ".join("".join(elemento.itertext()).split())


def _slug(texto: str) -> str:
    sin_marcas = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", sin_marcas.lower()).strip("-")


def _fecha_iso(marca: str) -> str:
    """«20260520122601» -> «2026-05-20»."""
    return f"{marca[0:4]}-{marca[4:6]}-{marca[6:8]}" if len(marca) >= 8 else ""


def _texto_tabla(tabla: ET.Element) -> tuple[str, str]:
    """Devuelve (leyenda, texto) de una tabla: una línea por fila, celdas separadas por « | »."""
    leyenda = tabla.find("caption")
    titulo = _limpiar(leyenda) if leyenda is not None else ""
    lineas = [titulo] if titulo else []
    for fila in tabla.iter("tr"):
        celdas = [_limpiar(celda) for celda in fila if celda.tag in ("td", "th")]
        if any(celdas):
            lineas.append(" | ".join(celdas))
    return titulo, "\n".join(lineas)


@dataclass
class _Seccion:
    base: str
    tipo: str
    numero: str
    titulo: str
    cabecera: str
    cita: str
    contexto: str
    lineas: list[str] = field(default_factory=list)


def _agrupar(lineas: list[str], limite: int) -> list[list[str]]:
    """Reparte las líneas en grupos de, como mucho, ``limite`` caracteres sin partir ninguna."""
    grupos: list[list[str]] = [[]]
    tamano = 0
    for linea in lineas:
        if grupos[-1] and tamano + len(linea) + 1 > limite:
            grupos.append([])
            tamano = 0
        grupos[-1].append(linea)
        tamano += len(linea) + 1
    return grupos


class _Troceador:
    """Recorre los elementos del texto en orden y los reparte en trozos."""

    def __init__(self, conf: ConfigDocumento, actualizado: str, max_chars: int, orden_inicial: int):
        self.conf = conf
        self.actualizado = actualizado
        self.max_chars = max_chars
        self.orden = orden_inicial
        self.salida: list[Trozo] = []
        self.ambito_id = f"{conf.prefijo}-res"
        self.ambito_nombre = "Resolución"
        self.en_resolucion = True
        self.principal_abierto = False
        self.ctx_titulo = ""
        self.ctx_capitulo = ""
        self.actual: _Seccion | None = None
        self.n_seccion = 0
        self.n_tabla = 0
        self.omitiendo = False

    # --- recorrido -------------------------------------------------------------------------
    def procesar(self, texto: ET.Element) -> list[Trozo]:
        for elemento in texto:
            if elemento.tag == "table":
                self._tabla(elemento)
            elif elemento.tag == "p":
                self._parrafo_con_clase(elemento.get("class") or "", _limpiar(elemento))
        self._cerrar()
        return self.salida

    def _parrafo_con_clase(self, clase: str, texto: str) -> None:
        if not texto or self._es_dato_personal(clase, texto):
            return
        despacho = {
            "anexo": self._anexo_titulo,
            "capitulo": self._capitulo,
            "capitulo_num": self._capitulo,
            "capitulo_tit": self._capitulo_titulo,
            "titulo_num": self._titulo,
            "titulo_tit": self._titulo_titulo,
            "anexo_num": self._anexo_num,
            "anexo_tit": self._anexo_titulo_corto,
            "articulo": self._articulo,
        }
        despacho.get(clase, self._parrafo)(texto)

    def _es_dato_personal(self, clase: str, texto: str) -> bool:
        """Descarta la firma de la resolución y la lista de asistentes del acta (RGPD, D08)."""
        if clase == "centro_cursiva" and texto.casefold() == "asistentes":
            self.omitiendo = True
            return True
        if self.omitiendo:
            if clase != "articulo":
                return True
            self.omitiendo = False
        return bool(RE_FIRMA.search(texto))

    # --- estructura ------------------------------------------------------------------------
    def _contexto(self) -> str:
        return " · ".join(parte for parte in (self.ctx_titulo, self.ctx_capitulo) if parte)

    def _capitulo(self, texto: str) -> None:
        self._cerrar()
        self.ctx_capitulo = texto

    def _capitulo_titulo(self, texto: str) -> None:
        self.ctx_capitulo = f"{self.ctx_capitulo} · {texto}" if self.ctx_capitulo else texto

    def _titulo(self, texto: str) -> None:
        self._cerrar()
        self.ctx_titulo = texto
        self.ctx_capitulo = ""

    def _titulo_titulo(self, texto: str) -> None:
        self.ctx_titulo = f"{self.ctx_titulo} · {texto}" if self.ctx_titulo else texto

    def _anexo_titulo(self, texto: str) -> None:
        """Clase ``anexo``: título del convenio (la primera vez) o de una sección dentro de un anexo."""
        self._cerrar()
        if not self.principal_abierto:
            self.principal_abierto = True
            self.en_resolucion = False
            self.ambito_id = self.conf.prefijo
            self.ambito_nombre = self.conf.principal
            self.ctx_titulo = self.ctx_capitulo = ""
            return
        self._abrir_seccion(titulo=texto, cabecera=texto)

    def _anexo_num(self, texto: str) -> None:
        self._cerrar()
        romano = texto.split()[-1].upper()
        self.en_resolucion = False
        self.ambito_id = f"{self.conf.prefijo}-a{ROMANOS.get(romano, _slug(romano))}"
        self.ambito_nombre = f"Anexo {romano}"
        self.ctx_titulo = self.ctx_capitulo = ""
        self.n_seccion = self.n_tabla = 0
        self.actual = _Seccion(self.ambito_id, "anexo", romano, "", self.ambito_nombre, self.ambito_nombre, "")

    def _anexo_titulo_corto(self, texto: str) -> None:
        anexo = self.actual
        if anexo is not None and anexo.tipo == "anexo" and anexo.base == self.ambito_id and not anexo.titulo:
            anexo.titulo = texto
            anexo.cabecera = f"{self.ambito_nombre}. {texto}"
            anexo.cita = f"{self.ambito_nombre} ({texto})"
        else:
            self._parrafo(texto)

    # --- artículos, párrafos y tablas ------------------------------------------------------
    def _articulo(self, texto: str) -> None:
        if self.en_resolucion:  # «Primero.» «Segundo.» de la resolución: van en su único trozo
            self._parrafo(texto)
            return
        self._cerrar()
        self.actual = self._seccion_de_articulo(texto)

    def _seccion_de_articulo(self, texto: str) -> _Seccion:
        nombre, base, contexto = self.ambito_nombre, self.ambito_id, self._contexto()
        if coincide := RE_ARTICULO.match(texto):
            numero, titulo = coincide.group(1), coincide.group(2).strip()
            cita = f"{nombre}, art. {numero}" + (f" ({titulo.rstrip('.')})" if titulo else "")
            return _Seccion(f"{base}-art-{numero}", "articulo", numero, titulo.rstrip("."), texto, cita, contexto)
        if coincide := RE_DISPOSICION.match(texto):
            clase, ordinal, titulo = coincide.group(1).lower(), coincide.group(2).lower(), coincide.group(3).strip()
            cita = f"{nombre}, disposición {clase} {ordinal}" + (f" ({titulo.rstrip('.')})" if titulo else "")
            return _Seccion(
                f"{base}-disp-{_slug(clase)}-{_slug(ordinal)}", "disposicion", f"{clase} {ordinal}",
                titulo.rstrip("."), texto, cita, contexto,
            )
        if coincide := RE_ORDINAL.match(texto):
            ordinal, titulo = coincide.group(1).lower(), coincide.group(2).strip()
            cita = f"{nombre}, apartado {ordinal}" + (f" ({titulo.rstrip('.')})" if titulo else "")
            # Tipo «apartado», no «articulo»: las tablas que siguen a un apartado del acta son trozos propios.
            return _Seccion(f"{base}-{_slug(ordinal)}", "apartado", ordinal, titulo.rstrip("."), texto, cita, contexto)
        self.n_seccion += 1
        return _Seccion(f"{base}-sec-{self.n_seccion}", "seccion", "", "", texto, f"{nombre} · {texto[:80]}", contexto)

    def _abrir_seccion(self, titulo: str = "", cabecera: str = "") -> None:
        self.n_seccion += 1
        if self.en_resolucion:
            self.actual = _Seccion(
                self.ambito_id, "resolucion", "", "", "Resolución", f"Resolución de registro ({self.conf.id})", ""
            )
            return
        tipo = "anexo" if self.ambito_nombre.startswith("Anexo") else "seccion"
        if titulo:
            cita = f"{self.ambito_nombre} · {titulo[:80]}"
        elif self._contexto():
            cita = f"{self.ambito_nombre} · {self._contexto()} (introducción)"
        elif self.ambito_nombre == self.conf.principal:
            cita = f"{self.ambito_nombre}, introducción"
        else:
            cita = f"{self.ambito_nombre}, sección {self.n_seccion}"
        self.actual = _Seccion(
            f"{self.ambito_id}-sec-{self.n_seccion}", tipo, "", titulo, cabecera, cita, self._contexto()
        )

    def _parrafo(self, texto: str) -> None:
        if self.actual is None:
            self._abrir_seccion()
        assert self.actual is not None
        self.actual.lineas.append(texto)

    def _tabla(self, elemento: ET.Element) -> None:
        leyenda, texto = _texto_tabla(elemento)
        dentro_de_articulo = self.actual is not None and self.actual.tipo in ("articulo", "disposicion")
        if dentro_de_articulo or self.ambito_nombre in self.conf.anexos_en_bloque:
            self._parrafo(texto)  # una tabla dentro de un artículo (o de un anexo en bloque) es parte del texto
            return
        self._cerrar()
        self.n_tabla += 1
        etiqueta = f"«{leyenda}»" if leyenda else f"tabla {self.n_tabla}"
        self.actual = _Seccion(
            f"{self.ambito_id}-tabla-{self.n_tabla}", "tabla", str(self.n_tabla), leyenda,
            f"{self.ambito_nombre}. {leyenda or f'Tabla {self.n_tabla}'}",
            f"{self.ambito_nombre}, {etiqueta}", self._contexto(), [texto],
        )
        self._cerrar()  # cada tabla es un trozo propio

    # --- salida ----------------------------------------------------------------------------
    def _cerrar(self) -> None:
        seccion, self.actual = self.actual, None
        if seccion is None:
            return
        cuerpo = [linea for linea in seccion.lineas if linea]
        if not cuerpo and seccion.tipo in ("anexo", "seccion"):
            return
        grupos = _agrupar(cuerpo, max(200, self.max_chars - len(seccion.cabecera)))
        for numero_parte, grupo in enumerate(grupos, start=1):
            partes = len(grupos)
            texto = "\n".join(([seccion.cabecera] if seccion.cabecera else []) + grupo)
            self.salida.append(
                Trozo(
                    id=seccion.base if partes == 1 else f"{seccion.base}#{numero_parte}",
                    base=seccion.base,
                    documento=self.conf.id,
                    tipo=seccion.tipo,
                    numero=seccion.numero,
                    titulo=seccion.titulo,
                    contexto=seccion.contexto,
                    cita=seccion.cita + (f" [parte {numero_parte}/{partes}]" if partes > 1 else ""),
                    texto=texto,
                    orden=self.orden + len(self.salida),
                    actualizado=self.actualizado,
                    parte=numero_parte,
                    partes=partes,
                )
            )


def trocear_documento(
    xml: bytes | str, conf: ConfigDocumento, max_chars: int = MAX_CHARS_DEFECTO, orden_inicial: int = 0
) -> list[Trozo]:
    """Trocea el XML de un documento del BOE en trozos citables."""
    raiz = ET.fromstring(xml)
    texto = raiz.find("texto")
    if texto is None:
        raise ValueError(f"{conf.id}: el XML no tiene el elemento <texto>")
    actualizado = _fecha_iso(raiz.get("fecha_actualizacion", ""))
    return _Troceador(conf, actualizado, max_chars, orden_inicial).procesar(texto)


def validar(trozos: list[Trozo]) -> list[str]:
    """Devuelve los problemas encontrados (lista vacía si todo está bien)."""
    problemas: list[str] = []
    ids = [t.id for t in trozos]
    repetidos = {i for i in ids if ids.count(i) > 1}
    if repetidos:
        problemas.append(f"identificadores repetidos: {sorted(repetidos)}")
    problemas += [f"trozo sin texto: {t.id}" for t in trozos if not t.texto.strip()]
    problemas += [f"parte fuera de rango: {t.id}" for t in trozos if not 1 <= t.parte <= t.partes]
    return problemas


def guardar_jsonl(trozos: list[Trozo], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="\n") as fichero:
        for trozo in trozos:
            fichero.write(json.dumps(trozo.a_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def cargar_jsonl(ruta: Path) -> list[Trozo]:
    with ruta.open(encoding="utf-8") as fichero:
        return [Trozo.de_dict(json.loads(linea)) for linea in fichero if linea.strip()]


def descargar(doc_id: str, destino: Path) -> Path:
    """Descarga el XML de un documento del BOE a ``destino/<id>.xml``."""
    import httpx  # import tardío: las pruebas de troceado no necesitan red

    respuesta = httpx.get(
        URL_XML.format(id=doc_id), headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True
    )
    respuesta.raise_for_status()
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"{doc_id}.xml"
    ruta.write_bytes(respuesta.content)
    return ruta


def main(argv: list[str] | None = None) -> int:
    salida_utf8()
    analizador = argparse.ArgumentParser(description="Descarga y trocea el texto del BOE.")
    analizador.add_argument("--crudo", type=Path, default=Path("datos/crudo"))
    analizador.add_argument("--salida", type=Path, default=Path("datos/trozos.jsonl"))
    analizador.add_argument("--fuentes", type=Path, default=Path("datos/fuentes.json"))
    analizador.add_argument("--max-chars", type=int, default=MAX_CHARS_DEFECTO)
    analizador.add_argument("--offline", action="store_true", help="no descargar: usar el XML ya guardado")
    args = analizador.parse_args(argv)

    trozos: list[Trozo] = []
    fuentes = []
    for conf in DOCUMENTOS:
        ruta = args.crudo / f"{conf.id}.xml"
        if not args.offline or not ruta.exists():
            ruta = descargar(conf.id, args.crudo)
        contenido = ruta.read_bytes()
        nuevos = trocear_documento(contenido, conf, args.max_chars, orden_inicial=len(trozos))
        trozos += nuevos
        fuentes.append(
            {
                "documento": conf.id,
                "url": f"https://www.boe.es/buscar/doc.php?id={conf.id}",
                "actualizado": nuevos[0].actualizado if nuevos else "",
                "sha256_xml": hashlib.sha256(contenido).hexdigest(),
                "trozos": len(nuevos),
            }
        )
    problemas = validar(trozos)
    if problemas:
        print("\n".join(problemas), file=sys.stderr)
        return 1
    guardar_jsonl(trozos, args.salida)
    args.fuentes.write_text(json.dumps(fuentes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(trozos)} trozos -> {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
