"""Pruebas del troceado del XML del BOE (D06): estructura, ámbitos, tablas, datos personales y partición."""

from pathlib import Path

import pytest

from convenio_rag.ingesta import (
    RE_FIRMA,
    ConfigDocumento,
    cargar_jsonl,
    guardar_jsonl,
    main,
    trocear_documento,
    validar,
)
from convenio_rag.modelos import Trozo

FIXTURE = Path(__file__).parent / "fixtures" / "boe_mini.xml"
CONF = ConfigDocumento("BOE-A-0000-0000", "t", "Convenio de prueba", frozenset({"Anexo II"}))


@pytest.fixture(scope="module")
def trozos() -> dict[str, Trozo]:
    return {t.id: t for t in trocear_documento(FIXTURE.read_bytes(), CONF)}


def test_la_resolucion_es_un_solo_trozo_y_no_lleva_la_firma(trozos):
    resolucion = trozos["t-res"]

    assert resolucion.tipo == "resolucion"
    assert "Visto el texto" in resolucion.texto and "Primero." in resolucion.texto
    assert "Directora" not in resolucion.texto and "Inventado" not in resolucion.texto


def test_cada_articulo_es_un_trozo_con_su_cita(trozos):
    articulo = trozos["t-art-1"]

    assert articulo.tipo == "articulo" and articulo.numero == "1" and articulo.titulo == "Ámbito"
    assert articulo.cita == "Convenio de prueba, art. 1 (Ámbito)"
    assert articulo.contexto == "CAPÍTULO I · Disposiciones generales"
    assert articulo.texto.startswith("Artículo 1. Ámbito.")


def test_un_articulo_sin_titulo_tras_el_numero_se_trocea_igual(trozos):
    articulo = trozos["t-art-2"]

    assert articulo.titulo == "" and articulo.cita == "Convenio de prueba, art. 2"
    assert "sin título" in articulo.texto


def test_los_articulos_de_un_anexo_no_colisionan_con_los_del_convenio(trozos):
    principal, anexo = trozos["t-art-1"], trozos["t-a1-art-1"]

    assert principal.id != anexo.id
    assert anexo.cita == "Anexo I, art. 1 (Objeto del plan)"
    assert anexo.contexto == "TÍTULO I · Objeto"


def test_el_texto_previo_a_los_articulos_de_un_anexo_es_un_trozo_propio(trozos):
    intro = trozos["t-a1"]

    assert intro.tipo == "anexo" and "Definiciones del plan." in intro.texto
    assert intro.cita == "Anexo I (Plan de pruebas)"


def test_la_introduccion_de_un_capitulo_se_conserva_con_una_cita_legible(trozos):
    intro = next(t for t in trozos.values() if "Texto introductorio" in t.texto)

    assert intro.tipo == "seccion"
    assert intro.cita == "Convenio de prueba · CAPÍTULO II · Jornada (introducción)"


def test_una_tabla_dentro_de_un_articulo_forma_parte_de_el(trozos):
    texto = trozos["t-art-3"].texto

    assert "Habrá treinta días" in texto and "Oficial | 1000,50" in texto
    assert not any(t.tipo == "tabla" and t.id.startswith("t-art") for t in trozos.values())


def test_una_tabla_de_un_anexo_es_un_trozo_propio_con_su_leyenda(trozos):
    tabla = trozos["t-a3-tabla-1"]

    assert tabla.tipo == "tabla"
    assert tabla.cita == "Anexo III, «Salarios 2030»" and "Oficial | 2000" in tabla.texto


def test_las_tablas_de_un_anexo_en_bloque_van_dentro_del_texto(trozos):
    anexo = trozos["t-a2"]

    assert "Texto de la plantilla." in anexo.texto and "celda a | celda b" in anexo.texto
    assert "t-a2-tabla-1" not in trozos


def test_las_disposiciones_se_identifican_por_clase_y_ordinal(trozos):
    disposicion = trozos["t-disp-transitoria-primera"]

    assert disposicion.tipo == "disposicion"
    assert disposicion.cita == "Convenio de prueba, disposición transitoria primera (Formación)"


def test_la_lista_de_asistentes_no_se_versiona_pero_el_apartado_siguiente_si(trozos):
    todo = " ".join(t.texto for t in trozos.values())

    assert "Fulano" not in todo and "Asistentes" not in todo
    assert "Se acuerda aplicar un incremento." in trozos["t-a3-unico"].texto
    assert trozos["t-a3-unico"].tipo == "apartado"


def test_la_fecha_de_actualizacion_se_toma_del_xml(trozos):
    assert {t.actualizado for t in trozos.values()} == {"2026-01-02"}


def test_el_orden_es_correlativo_y_respeta_el_orden_inicial():
    lista = trocear_documento(FIXTURE.read_bytes(), CONF, orden_inicial=100)

    assert [t.orden for t in lista] == list(range(100, 100 + len(lista)))


def test_un_articulo_largo_se_parte_conservando_la_cabecera_y_la_cita():
    xml = (
        '<documento fecha_actualizacion="20260101000000"><texto>'
        '<p class="anexo">C</p><p class="articulo">Artículo 7. Largo.</p>'
        + "".join(f'<p class="parrafo">{"palabra " * 10}párrafo {n}.</p>' for n in range(6))
        + "</texto></documento>"
    )

    partes = [t for t in trocear_documento(xml, CONF, max_chars=200) if t.base == "t-art-7"]

    assert len(partes) > 1
    assert [t.id for t in partes] == [f"t-art-7#{i}" for i in range(1, len(partes) + 1)]
    assert all(t.texto.startswith("Artículo 7. Largo.") and t.partes == len(partes) for t in partes)
    assert partes[0].cita.endswith(f"[parte 1/{len(partes)}]")


def test_un_xml_sin_texto_da_un_error_claro():
    with pytest.raises(ValueError, match="no tiene el elemento <texto>"):
        trocear_documento("<documento></documento>", CONF)


def test_validar_detecta_identificadores_repetidos_y_trozos_vacios():
    bueno = Trozo("a", "a", "D", "articulo", "1", "", "", "c", "texto", 0, "2026-01-01")
    repetido = Trozo("a", "a", "D", "articulo", "1", "", "", "c", "texto", 1, "2026-01-01")
    vacio = Trozo("b", "b", "D", "articulo", "2", "", "", "c", "  ", 2, "2026-01-01")
    fuera = Trozo("c#9", "c", "D", "articulo", "3", "", "", "c", "x", 3, "2026-01-01", parte=9, partes=2)

    problemas = validar([bueno, repetido, vacio, fuera])

    assert any("repetidos" in p for p in problemas)
    assert "trozo sin texto: b" in problemas
    assert "parte fuera de rango: c#9" in problemas
    assert validar([bueno]) == []


def test_el_jsonl_se_guarda_y_se_carga_sin_perdida(tmp_path, trozos):
    ruta = tmp_path / "sub" / "trozos.jsonl"

    guardar_jsonl(list(trozos.values()), ruta)

    assert cargar_jsonl(ruta) == list(trozos.values())


def _crudo(tmp_path: Path) -> Path:
    crudo = tmp_path / "crudo"
    crudo.mkdir()
    for documento in ("BOE-A-2025-21424", "BOE-A-2026-10861"):
        (crudo / f"{documento}.xml").write_bytes(FIXTURE.read_bytes())
    return crudo


def test_el_comando_trocea_en_modo_offline_y_escribe_los_ficheros(tmp_path):
    salida, fuentes = tmp_path / "trozos.jsonl", tmp_path / "fuentes.json"

    codigo = main(["--offline", "--crudo", str(_crudo(tmp_path)), "--salida", str(salida), "--fuentes", str(fuentes)])

    assert codigo == 0
    assert cargar_jsonl(salida) and "sha256_xml" in fuentes.read_text(encoding="utf-8")


def test_el_comando_devuelve_error_y_no_escribe_nada_si_la_validacion_falla(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("convenio_rag.ingesta.validar", lambda trozos: ["problema inventado"])
    salida = tmp_path / "trozos.jsonl"

    codigo = main(["--offline", "--crudo", str(_crudo(tmp_path)), "--salida", str(salida),
                   "--fuentes", str(tmp_path / "fuentes.json")])

    assert codigo == 1
    assert "problema inventado" in capsys.readouterr().err
    assert not salida.exists()


# --- el texto real versionado (datos/trozos.jsonl): invariantes que cualquier cambio del troceado debe respetar ---
@pytest.fixture(scope="module")
def reales(raiz) -> list[Trozo]:
    return cargar_jsonl(raiz / "datos" / "trozos.jsonl")


def test_el_convenio_real_tiene_56_articulos_y_el_plan_de_pensiones_47(reales):
    articulos = {int(t.numero) for t in reales if t.base.startswith("conv-art-")}
    plan = {int(t.numero) for t in reales if t.base.startswith("conv-a1-art-")}

    assert articulos == set(range(1, 57))
    assert plan == set(range(1, 48))


def test_el_texto_real_no_tiene_ids_repetidos_ni_trozos_vacios(reales):
    assert validar(reales) == []


def test_el_texto_real_no_lleva_la_firma_de_las_resoluciones_ni_la_lista_de_asistentes_del_acta(reales):
    # Se comprueba la estructura y no los nombres: así el propio repositorio no vuelve a contener nombres de personas.
    for trozo in reales:
        assert not RE_FIRMA.search(trozo.texto), trozo.id
        assert "Asistentes" not in trozo.texto, trozo.id  # el cierre del acta («Por la Representación…») no lleva nombres


def test_las_tablas_salariales_de_2026_son_trozos_propios(reales):
    tabla = next(t for t in reales if t.id == "tab-tabla-1")

    assert tabla.tipo == "tabla" and "1351,99" in tabla.texto
    assert {t.actualizado for t in reales} == {"2026-05-20"}
