"""Pruebas de la evaluación: integridad del conjunto de preguntas, métricas, troceado fijo y umbral de regresión."""

import collections
import json

import pytest

from convenio_rag.almacen import Almacen
from convenio_rag.busqueda import Buscador, ConfigBusqueda
from convenio_rag.embeddings import HashEmbedder
from convenio_rag.evaluacion import (
    Pregunta,
    acierto_en_corte,
    cargar_golden,
    entorno,
    estadisticas_troceado,
    evaluar,
    main,
    posicion_primer_acierto,
    tabla_markdown,
    tabla_troceado,
    trozos_tamano_fijo,
)
from convenio_rag.ingesta import cargar_jsonl
from convenio_rag.modelos import Resultado, Trozo

# Umbrales de regresión de la búsqueda léxica sobre el texto real (valores medidos: hit@5 0,90 y MRR@10 0,81).
# Si un cambio del troceado o de la búsqueda los rompe, la integración continua falla.
MINIMO_HIT5_LEXICA = 0.85
MINIMO_MRR_LEXICA = 0.75


def _trozo(id_: str, texto: str, orden: int = 0, base: str | None = None) -> Trozo:
    return Trozo(id_, base or id_, "D", "articulo", "", "", "", id_, texto, orden, "2026-01-01")


def _pregunta(esperados: tuple[str, ...], modo: str = "alguno", texto: str = "¿?") -> Pregunta:
    return Pregunta("q", "directa", texto, esperados, modo, "r", (), ())


def _resultados(*bases: str) -> list[Resultado]:
    return [Resultado(_trozo(f"t{i}", "x", base=b), 1.0, i) for i, b in enumerate(bases, start=1)]


# --- integridad del conjunto de preguntas contra el texto real --------------------------------
@pytest.fixture(scope="module")
def reales(raiz):
    return cargar_jsonl(raiz / "datos" / "trozos.jsonl")


@pytest.fixture(scope="module")
def golden(raiz):
    return cargar_golden(raiz / "evals" / "golden.jsonl")


def test_el_conjunto_tiene_60_preguntas_de_cinco_tipos_con_ids_unicos(golden):
    tipos = collections.Counter(p.tipo for p in golden)

    assert len(golden) == 60 and len({p.id for p in golden}) == 60
    assert tipos == {"directa": 42, "combinada": 5, "ambigua": 1, "inyeccion": 2, "sin_respuesta": 10}


def test_cada_articulo_esperado_existe_en_el_corpus(golden, reales):
    bases = {t.base for t in reales}

    inexistentes = [(p.id, b) for p in golden for b in p.esperados if b not in bases]

    assert inexistentes == []


def test_cada_cita_de_evidencia_aparece_literalmente_en_los_articulos_esperados(golden, reales):
    por_base: dict[str, str] = collections.defaultdict(str)
    for t in reales:
        por_base[t.base] += t.texto + "\n"

    ausentes = [
        (p.id, ev) for p in golden for ev in p.evidencia if ev not in " ".join(por_base[b] for b in p.esperados)
    ]

    assert ausentes == []


def test_los_terminos_de_las_preguntas_sin_respuesta_no_estan_en_ninguna_parte_del_corpus(golden, reales):
    todo = " ".join(t.texto for t in reales).casefold()

    presentes = [(p.id, t) for p in golden for t in p.terminos_ausentes if t.casefold() in todo]

    assert presentes == []


def test_las_preguntas_sin_respuesta_no_tienen_articulos_y_las_demas_si(golden):
    for p in golden:
        assert bool(p.esperados) == (p.tipo != "sin_respuesta"), p.id
        assert p.modo_acierto in ("alguno", "todos") and (p.modo_acierto == "alguno" or len(p.esperados) >= 2), p.id
        assert (p.tipo != "sin_respuesta") or p.terminos_ausentes, p.id


# --- métricas ---------------------------------------------------------------------------------
def test_el_acierto_en_un_corte_depende_del_modo():
    resultados = _resultados("a", "b", "c")

    assert acierto_en_corte(resultados, _pregunta(("c",)), 3) and not acierto_en_corte(resultados, _pregunta(("c",)), 2)
    assert acierto_en_corte(resultados, _pregunta(("a", "c"), "alguno"), 1)
    assert not acierto_en_corte(resultados, _pregunta(("a", "c"), "todos"), 2)
    assert acierto_en_corte(resultados, _pregunta(("a", "c"), "todos"), 3)


def test_una_ventana_que_cruza_dos_articulos_acierta_si_toca_alguno_esperado():
    resultados = _resultados("a|b")

    assert acierto_en_corte(resultados, _pregunta(("b",)), 1) and acierto_en_corte(resultados, _pregunta(("a", "b"), "todos"), 1)
    assert posicion_primer_acierto(resultados, _pregunta(("b",))) == 1


def test_en_modo_todos_la_posicion_es_la_que_completa_los_articulos_esperados():
    resultados = _resultados("a", "x", "c")

    assert posicion_primer_acierto(resultados, _pregunta(("a", "c"), "todos")) == 3
    assert posicion_primer_acierto(resultados, _pregunta(("a", "c"), "alguno")) == 1
    assert posicion_primer_acierto(_resultados("a", "x"), _pregunta(("a", "c"), "todos")) is None


def test_la_posicion_del_primer_acierto_es_none_si_no_hay_ninguno():
    assert posicion_primer_acierto(_resultados("a", "b"), _pregunta(("z",))) is None
    assert posicion_primer_acierto(_resultados("a", "b"), _pregunta(("b",))) == 2


@pytest.fixture
def buscador_pequeno() -> Buscador:
    textos = {"vac": "Vacaciones treinta días al año", "jor": "Jornada máxima anual mil horas",
              "die": "Dieta diaria cuarenta y cinco euros"}
    trozos = [_trozo(i, t, orden) for orden, (i, t) in enumerate(textos.items())]
    almacen, embedder = Almacen(), HashEmbedder()
    almacen.cargar(trozos)
    almacen.guardar_vectores(embedder.nombre, [t.id for t in trozos], embedder.pasajes([t.texto for t in trozos]))
    return Buscador(almacen, embedder)


def test_evaluar_calcula_hit_y_mrr_solo_con_las_preguntas_que_tienen_respuesta(buscador_pequeno):
    preguntas = [
        _pregunta(("vac",), texto="vacaciones al año"),
        _pregunta(("die",), texto="dieta diaria"),
        Pregunta("s", "sin_respuesta", "¿criptomonedas?", (), "alguno", "No consta.", (), ("cripto",)),
    ]

    resultado = evaluar(buscador_pequeno, preguntas, ConfigBusqueda(modo="lexica"))

    assert resultado["metricas"]["n"] == 2
    assert resultado["metricas"]["hit@1"] == 1.0 and resultado["metricas"]["mrr@10"] == 1.0
    assert resultado["fallos@5"] == []


def test_evaluar_lista_las_preguntas_que_fallan_en_el_top_5(buscador_pequeno):
    resultado = evaluar(buscador_pequeno, [_pregunta(("vac",), texto="astronautas lunares")],
                        ConfigBusqueda(modo="lexica"))

    assert resultado["metricas"]["hit@5"] == 0.0 and resultado["metricas"]["mrr@10"] == 0.0
    assert resultado["fallos@5"] == ["q"]


# --- troceado de tamaño fijo ------------------------------------------------------------------
def test_las_ventanas_de_tamano_fijo_cubren_el_texto_y_marcan_los_articulos_que_tocan():
    trozos = [_trozo("a", "A" * 100, 0), _trozo("b", "B" * 100, 1)]

    ventanas = trozos_tamano_fijo(trozos, tamano=60, solape=10)

    assert len({v.id for v in ventanas}) == len(ventanas) and all(len(v.texto) <= 60 for v in ventanas)
    assert ventanas[0].base == "a" and ventanas[-1].base == "b"
    assert any(v.base == "a|b" for v in ventanas)  # una ventana cruza el límite entre los dos artículos
    assert "".join(v.texto[:50] for v in ventanas[:-1]).startswith("A" * 50)


def test_el_troceado_fijo_une_las_partes_de_un_articulo_sin_repetir_su_cabecera():
    partes = [
        Trozo("a#1", "a", "D", "articulo", "", "", "", "a", "Artículo 1.\nprimera parte", 0, "x", 1, 2),
        Trozo("a#2", "a", "D", "articulo", "", "", "", "a", "Artículo 1.\nsegunda parte", 1, "x", 2, 2),
    ]

    texto = " ".join(v.texto for v in trozos_tamano_fijo(partes, tamano=500, solape=0))

    assert texto.count("Artículo 1.") == 1 and "primera parte" in texto and "segunda parte" in texto


def test_un_solape_igual_o_mayor_que_el_tamano_es_un_error():
    with pytest.raises(ValueError, match="solape"):
        trozos_tamano_fijo([_trozo("a", "x" * 50)], tamano=10, solape=10)


# --- estadísticas, tablas y entorno -----------------------------------------------------------
def test_las_estadisticas_del_troceado_reflejan_el_coste_del_tamano_fijo(reales):
    filas = {f["corpus"]: f for f in estadisticas_troceado(reales)}

    assert filas["articulo"]["pct_cruzan_articulos"] == 0.0 and filas["articulo"]["trozos"] == len(reales)
    assert filas["fijo800"]["pct_cruzan_articulos"] > 20 and filas["fijo1500"]["pct_cruzan_articulos"] > 20


def test_las_tablas_en_markdown_tienen_una_fila_por_configuracion():
    fila = {"corpus": "articulo", "trozos": 3, "modo": "lexica", "modelo": "—", "raiz": 6,
            "metricas": {"hit@1": 0.5, "hit@3": 0.6, "hit@5": 0.7, "hit@10": 0.8, "mrr@10": 0.55, "n": 10},
            "fallos@5": []}

    tabla = tabla_markdown([fila, {**fila, "corpus": "fijo800"}], corpus="articulo")

    assert tabla.count("\n") == 3 and "| articulo | lexica | — | 6 | 0.50 | 0.60 | 0.70 | 0.80 | 0.55 |" in tabla
    assert "fijo800" not in tabla
    forma = tabla_troceado([{"corpus": "articulo", "trozos": 3, "caracteres_mediana": 10, "caracteres_max": 20,
                             "pct_cruzan_articulos": 0.0}])
    assert "| articulo | 3 | 10 | 20 | 0.0 |" in forma


def test_el_entorno_registra_las_huellas_de_los_ficheros_y_la_version_de_python(tmp_path):
    fichero = tmp_path / "datos.txt"
    fichero.write_text("contenido", encoding="utf-8")

    datos = entorno({"datos": fichero})

    assert len(datos["sha256_datos"]) == 16 and datos["python"].startswith("3.") and "ejecutado" in datos


# --- regresión de la recuperación sobre el texto real (sin modelos: corre en la CI) ------------
def test_la_busqueda_lexica_sobre_el_texto_real_no_empeora_de_los_umbrales(reales, golden):
    almacen = Almacen()
    almacen.cargar(reales)

    resultado = evaluar(Buscador(almacen), golden, ConfigBusqueda(modo="lexica", prefijo=6))

    assert resultado["metricas"]["n"] == 50
    assert resultado["metricas"]["hit@5"] >= MINIMO_HIT5_LEXICA, resultado["fallos@5"]
    assert resultado["metricas"]["mrr@10"] >= MINIMO_MRR_LEXICA


def test_el_comando_sin_modelos_escribe_los_resultados_y_son_reproducibles(tmp_path, capsys):
    salida = tmp_path / "res"

    assert main(["--sin-modelos", "--salida", str(salida)]) == 0
    primero = json.loads((salida / "recuperacion.json").read_text(encoding="utf-8"))
    assert main(["--sin-modelos", "--salida", str(tmp_path / "res2")]) == 0
    segundo = json.loads((tmp_path / "res2" / "recuperacion.json").read_text(encoding="utf-8"))

    assert primero["filas"] == segundo["filas"] and len(primero["filas"]) == 6
    assert (salida / "troceado.md").exists() and "hit@5" in capsys.readouterr().out
