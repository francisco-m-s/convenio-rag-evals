"""La búsqueda léxica (BM25) descansa en FTS5 de SQLite. Si el Python de la CI o el de un clon no lo trae, hay que
enterarse desde el primer día y no al llegar a la fase 1."""

import sqlite3

import pytest


@pytest.fixture
def con():
    conexion = sqlite3.connect(":memory:")
    conexion.execute("CREATE VIRTUAL TABLE t USING fts5(texto, tokenize='unicode61 remove_diacritics 2')")
    conexion.executemany(
        "INSERT INTO t(texto) VALUES (?)",
        [
            ("Artículo 20. Las vacaciones serán de 30 días naturales.",),
            ("Artículo 21. La jornada máxima anual será de 1.792 horas.",),
        ],
    )
    yield conexion
    conexion.close()


def test_el_tokenizador_ignora_las_tildes_en_la_consulta_y_en_el_texto(con):
    assert con.execute("SELECT count(*) FROM t WHERE t MATCH 'maxima'").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM t WHERE t MATCH 'articulo'").fetchone()[0] == 2


def test_bm25_ordena_primero_el_trozo_mas_relevante(con):
    filas = con.execute("SELECT texto FROM t WHERE t MATCH 'vacaciones' ORDER BY bm25(t)").fetchall()

    assert len(filas) == 1
    assert "vacaciones" in filas[0][0]
