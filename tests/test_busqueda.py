"""Pruebas del almacén SQLite, los embeddings, la búsqueda (léxica, vectorial, híbrida) y la indexación."""

import numpy as np
import pytest

from convenio_rag.almacen import Almacen, terminos
from convenio_rag.busqueda import Buscador, ConfigBusqueda, fusion_rrf
from convenio_rag.embeddings import FastEmbedder, HashEmbedder
from convenio_rag.indexar import indexar
from convenio_rag.indexar import main as indexar_main
from convenio_rag.ingesta import guardar_jsonl
from convenio_rag.modelos import Trozo


def _trozo(id_: str, texto: str, orden: int) -> Trozo:
    return Trozo(id_, id_, "D", "articulo", "", "", "", id_, texto, orden, "2026-01-01")


CORPUS = [
    _trozo("vac", "Artículo 1. Vacaciones. Treinta y un días naturales de vacaciones al año.", 0),
    _trozo("jor", "Artículo 2. Jornada. La jornada máxima anual será de 1.788 horas.", 1),
    _trozo("per", "Artículo 3. Permisos. Quince días por matrimonio y un día por traslado de domicilio.", 2),
    _trozo("die", "Artículo 4. Dietas. La dieta diaria será de 45 euros.", 3),
]


@pytest.fixture
def almacen() -> Almacen:
    a = Almacen()
    a.cargar(CORPUS)
    return a


@pytest.fixture
def buscador(almacen) -> Buscador:
    embedder = HashEmbedder()
    almacen.guardar_vectores(embedder.nombre, [t.id for t in CORPUS], embedder.pasajes([t.texto for t in CORPUS]))
    return Buscador(almacen, embedder)


# --- términos de búsqueda ---------------------------------------------------------------------
def test_los_terminos_descartan_palabras_vacias_y_signos():
    assert terminos("¿Cuántos días de vacaciones me corresponden?") == ["días", "vacaciones", "corresponden"]


def test_con_prefijo_los_terminos_largos_se_truncan_a_la_raiz_y_los_cortos_no():
    assert terminos("vacaciones días", prefijo=6) == ["vacaci*", "días"]


# --- almacén ----------------------------------------------------------------------------------
def test_el_almacen_carga_y_devuelve_los_trozos_en_orden(almacen):
    assert almacen.contar() == 4
    assert [t.id for t in almacen.todos()] == ["vac", "jor", "per", "die"]
    assert almacen.obtener("jor").texto.startswith("Artículo 2")
    assert almacen.obtener("no-existe") is None


def test_cargar_de_nuevo_sustituye_el_contenido_y_borra_los_vectores(almacen):
    almacen.guardar_vectores("m", ["vac"], np.ones((1, 4), dtype=np.float32))

    almacen.cargar(CORPUS[:2])

    assert almacen.contar() == 2 and almacen.vectores("m") == ([], pytest.approx(np.zeros((0, 0))))


def test_la_busqueda_lexica_encuentra_el_articulo_por_sus_palabras(almacen):
    ids = [i for i, _ in almacen.buscar_lexico("¿cuántos días de vacaciones tengo?", k=3)]

    assert ids[0] == "vac"


def test_la_busqueda_lexica_ignora_tildes_y_mayusculas(almacen):
    assert almacen.buscar_lexico("JORNADA MAXIMA", k=1)[0][0] == "jor"


def test_sin_raiz_vacacion_no_encuentra_vacaciones_y_con_raiz_si(almacen):
    assert almacen.buscar_lexico("vacación", k=3) == []
    assert almacen.buscar_lexico("vacación", k=3, prefijo=5)[0][0] == "vac"


def test_la_busqueda_lexica_no_se_rompe_con_sintaxis_de_fts_en_la_pregunta(almacen):
    resultados = almacen.buscar_lexico('dieta" OR NEAR( * ) -45 AND', k=3)

    assert resultados and resultados[0][0] == "die"


def test_una_pregunta_sin_terminos_utiles_no_devuelve_nada(almacen):
    assert almacen.buscar_lexico("¿de la el?", k=3) == []


def test_los_vectores_se_guardan_y_se_recuperan_en_el_orden_de_los_trozos(almacen):
    matriz = np.arange(8, dtype=np.float32).reshape(4, 2)

    almacen.guardar_vectores("m", [t.id for t in CORPUS], matriz)
    ids, leidos = almacen.vectores("m")

    assert ids == ["vac", "jor", "per", "die"] and np.array_equal(leidos, matriz)
    assert almacen.vectores("otro") == ([], pytest.approx(np.zeros((0, 0))))


# --- embeddings -------------------------------------------------------------------------------
def test_el_embedder_de_pruebas_es_determinista_y_normalizado():
    embedder = HashEmbedder()

    a, b = embedder.pasajes(["vacaciones al año"]), embedder.consultas(["vacaciones al año"])

    assert np.array_equal(a, b) and np.linalg.norm(a[0]) == pytest.approx(1.0)
    assert embedder.pasajes([]).shape == (0, 256)


def test_un_texto_sin_palabras_utiles_da_el_vector_nulo_sin_dividir_por_cero():
    assert not np.isnan(HashEmbedder().pasajes(["y a o"])).any()


class _ModeloFalso:
    def __init__(self):
        self.recibido: list[str] = []

    def embed(self, textos):
        self.recibido += list(textos)
        return [np.array([1.0, 0.0]) for _ in textos]


def test_los_modelos_que_lo_exigen_reciben_prefijos_distintos_para_pregunta_y_texto():
    embedder = FastEmbedder("intfloat/multilingual-e5-large")
    embedder._modelo = _ModeloFalso()

    embedder.consultas(["¿vacaciones?"])
    embedder.pasajes(["Artículo 1"])

    assert embedder._modelo.recibido == ["query: ¿vacaciones?", "passage: Artículo 1"]


def test_los_demas_modelos_no_reciben_prefijo_y_los_vectores_salen_normalizados():
    embedder = FastEmbedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embedder._modelo = _ModeloFalso()

    resultado = embedder.pasajes(["Artículo 1"])

    assert embedder._modelo.recibido == ["Artículo 1"] and np.linalg.norm(resultado[0]) == pytest.approx(1.0)
    assert embedder.consultas([]).shape == (0, 0)


# --- configuración y fusión -------------------------------------------------------------------
def test_la_configuracion_rechaza_modos_desconocidos_y_k_invalido():
    with pytest.raises(ValueError, match="modo desconocido"):
        ConfigBusqueda(modo="magica")
    with pytest.raises(ValueError, match="k debe ser"):
        ConfigBusqueda(k=0)


def test_la_etiqueta_describe_la_configuracion():
    assert ConfigBusqueda(modo="hibrida", k=5, prefijo=6).etiqueta() == "hibrida/raíz6/k5"
    assert ConfigBusqueda(modo="vectorial", k=3, prefijo=6).etiqueta() == "vectorial/k3"


def test_la_fusion_rrf_premia_lo_que_aparece_en_las_dos_listas():
    fusion = fusion_rrf([["a", "b", "c"], ["c", "a", "d"]])

    assert [i for i, _ in fusion][:2] == ["a", "c"]
    assert [i for i, _ in fusion][2:] == ["b", "d"]


# --- buscador ---------------------------------------------------------------------------------
@pytest.mark.parametrize("modo", ["lexica", "vectorial", "hibrida"])
def test_cada_modo_encuentra_el_articulo_de_vacaciones_en_primer_lugar(buscador, modo):
    resultados = buscador.buscar("¿Cuántos días de vacaciones hay al año?", ConfigBusqueda(modo=modo, k=2))

    assert resultados[0].trozo.id == "vac" and resultados[0].posicion == 1
    assert [r.posicion for r in resultados] == list(range(1, len(resultados) + 1))


def test_k_limita_el_numero_de_resultados(buscador):
    assert len(buscador.buscar("días", ConfigBusqueda(modo="vectorial", k=3))) == 3


def test_la_busqueda_vectorial_sin_vectores_da_un_error_que_dice_que_hacer(almacen):
    sin_vectores = Buscador(almacen, HashEmbedder())

    with pytest.raises(RuntimeError, match="convenio_rag.indexar"):
        sin_vectores.buscar("vacaciones", ConfigBusqueda(modo="vectorial"))


def test_la_busqueda_lexica_funciona_sin_embedder(almacen):
    assert Buscador(almacen).buscar("dieta diaria", ConfigBusqueda(modo="lexica", k=1))[0].trozo.id == "die"


def test_los_numeros_de_una_cifra_se_conservan_como_terminos():
    assert terminos("artículo 5") == ["artículo", "5"]


def test_la_hibrida_degrada_a_lexica_si_faltan_los_vectores_en_vez_de_fallar(almacen):
    sin_vectores = Buscador(almacen, HashEmbedder())

    resultados = sin_vectores.buscar("dieta diaria", ConfigBusqueda(modo="hibrida", k=1))

    assert resultados[0].trozo.id == "die"


def test_reemplazar_deja_el_indice_intacto_si_falla_el_calculo_de_embeddings(tmp_path):
    jsonl = tmp_path / "trozos.jsonl"
    guardar_jsonl(CORPUS, jsonl)
    almacen = indexar(jsonl, tmp_path / "db.sqlite", [HashEmbedder()])

    class Roto:
        nombre = "roto"

        def pasajes(self, textos):
            raise RuntimeError("el modelo falló")

    with pytest.raises(RuntimeError, match="el modelo falló"):
        indexar(jsonl, tmp_path / "db.sqlite", [Roto()], almacen)

    assert almacen.contar() == 4 and almacen.tiene_vectores("hash-256") and not almacen.tiene_vectores("roto")


# --- indexación -------------------------------------------------------------------------------
def test_indexar_carga_los_trozos_y_calcula_los_vectores(tmp_path):
    jsonl = tmp_path / "trozos.jsonl"
    guardar_jsonl(CORPUS, jsonl)

    almacen = indexar(jsonl, tmp_path / "db.sqlite", [HashEmbedder()])

    ids, matriz = almacen.vectores("hash-256")
    assert almacen.contar() == 4 and ids == ["vac", "jor", "per", "die"] and matriz.shape == (4, 256)


def test_el_comando_sin_vectores_crea_solo_el_indice_lexico(tmp_path, capsys):
    jsonl, db = tmp_path / "trozos.jsonl", tmp_path / "carpeta" / "db.sqlite"
    guardar_jsonl(CORPUS, jsonl)

    assert indexar_main(["--sin-vectores", "--trozos", str(jsonl), "--db", str(db)]) == 0
    assert db.exists() and "ninguno" in capsys.readouterr().out
    assert indexar_main(["--sin-vectores", "--trozos", str(jsonl), "--db", str(db)]) == 0  # repetible
