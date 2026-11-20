"""Pruebas de la evaluación de la generación y del cargador de .env (proveedor y juez simulados, sin red)."""

import csv
import json
import re

import httpx
import pytest

from convenio_rag import evaluacion_generacion as eg
from convenio_rag.almacen import Almacen
from convenio_rag.busqueda import Buscador, ConfigBusqueda
from convenio_rag.embeddings import HashEmbedder
from convenio_rag.entorno import cargar_dotenv
from convenio_rag.evaluacion import Pregunta
from convenio_rag.generacion import CABECERA_COSTE, ClienteLLM, ConfigLLM, Respuesta
from convenio_rag.modelos import Trozo


def _proveedor(veredicto: str = "correcta", consta: bool = True, estado: int = 200, juez_json: str | None = None,
               generacion_bruta: str | None = None, estado_juez: int = 200) -> ClienteLLM:
    """Un proveedor que responde como asistente (cita el primer fragmento) o como juez, según el prompt."""

    def manejador(peticion: httpx.Request) -> httpx.Response:
        mensajes = json.loads(peticion.content)["messages"]
        if "evaluador estricto" in mensajes[0]["content"]:
            if estado_juez != 200:
                return httpx.Response(estado_juez, json={})
            contenido = juez_json or json.dumps({"veredicto": veredicto, "fiel": True, "motivo": "ok"})
        elif generacion_bruta is not None:
            contenido = generacion_bruta
        else:
            primero = re.search(r"\[([^\]\s]+)\]", mensajes[1]["content"])
            id_ = primero.group(1) if primero else ""
            contenido = json.dumps({"respuesta": f"Respuesta [{id_}]", "consta": consta, "citas": [id_] if consta else []})
        cuerpo = {"choices": [{"message": {"content": contenido}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        return httpx.Response(estado, json=cuerpo, headers={CABECERA_COSTE: "0.001"})

    return ClienteLLM(ConfigLLM("http://p/v1", "k", "m"), httpx.Client(transport=httpx.MockTransport(manejador)))


def _trozo(id_: str, texto: str, orden: int) -> Trozo:
    return Trozo(id_, id_, "D", "articulo", "", "", "", id_, texto, orden, "2026-01-01")


@pytest.fixture
def buscador() -> Buscador:
    trozos = [_trozo("vac", "Vacaciones treinta días al año", 0), _trozo("die", "Dieta diaria cuarenta euros", 1)]
    almacen, embedder = Almacen(), HashEmbedder()
    almacen.cargar(trozos)
    almacen.guardar_vectores(embedder.nombre, [t.id for t in trozos], embedder.pasajes([t.texto for t in trozos]))
    return Buscador(almacen, embedder)


def _pregunta(tipo: str = "directa", esperados: tuple[str, ...] = ("vac",), modo: str = "alguno") -> Pregunta:
    return Pregunta("q1", tipo, "vacaciones al año", esperados, modo, "Treinta días.", (), ())


CONFIG = ConfigBusqueda(modo="lexica", k=2)


# --- juez -------------------------------------------------------------------------------------
def test_el_juez_devuelve_veredicto_fidelidad_y_coste():
    resultado = eg.juzgar(_proveedor("parcial"), _pregunta(), Respuesta("x", True, ("vac",), ()), "[vac] texto")

    assert resultado["veredicto"] == "parcial" and resultado["fiel"] is True and resultado["coste_juez_usd"] == 0.001


@pytest.mark.parametrize("salida", ["no es json", '{"veredicto": "genial", "fiel": true}', '{"fiel": true}',
                                    '{"veredicto": "correcta", "fiel": "true"}'])
def test_un_juez_que_no_cumple_el_formato_da_sin_veredicto(salida):
    resultado = eg.juzgar(_proveedor(juez_json=salida), _pregunta(), Respuesta("x", True, (), ()), "")

    assert resultado["veredicto"] == "sin_veredicto"


def test_un_fallo_del_proveedor_al_juzgar_no_aborta_y_queda_como_sin_veredicto():
    resultado = eg.juzgar(_proveedor(estado_juez=503), _pregunta(), Respuesta("x", True, (), ()), "")

    assert resultado["veredicto"] == "sin_veredicto" and "HTTPStatusError" in resultado["motivo_juez"]


def test_una_pregunta_cuyo_juicio_falla_conserva_su_respuesta_y_sus_metricas_sin_juez(buscador):
    fila = eg.evaluar_pregunta(buscador, _proveedor(estado_juez=503), _proveedor(estado_juez=503), "p", _pregunta(), CONFIG)

    assert fila["cita_correcta"] is True and fila["veredicto"] == "sin_veredicto"


# --- evaluación de una pregunta ---------------------------------------------------------------
def test_una_salida_mal_formada_se_cuenta_como_error_y_no_como_abstencion(buscador):
    mala = _proveedor(generacion_bruta="esto no es json")
    sin_respuesta = _pregunta(tipo="sin_respuesta", esperados=())

    fila_sin_respuesta = eg.evaluar_pregunta(buscador, mala, None, "p", sin_respuesta, CONFIG)
    fila_con_respuesta = eg.evaluar_pregunta(buscador, mala, None, "p", _pregunta(), CONFIG)

    for fila in (fila_sin_respuesta, fila_con_respuesta):
        assert fila["error"] and fila["consta"] is False
        assert not {"cita_correcta", "abstencion_correcta", "falsa_abstencion"} & fila.keys()
    resumen_filas = eg.resumen([fila_sin_respuesta, fila_con_respuesta])["global"]
    assert resumen_filas["errores"] == 2 and resumen_filas["abstencion_correcta"] is None


def test_una_respuesta_del_proveedor_con_forma_inesperada_queda_registrada_como_fallo(buscador):
    cliente = ClienteLLM(ConfigLLM("http://p/v1", "k", "m"),
                         httpx.Client(transport=httpx.MockTransport(lambda p: httpx.Response(200, json={"choices": []}))))

    fila = eg.evaluar_pregunta(buscador, cliente, None, "p", _pregunta(), CONFIG)

    assert fila["error"] == "fallo del proveedor: ErrorProveedor"


# --- (sigue) evaluación de una pregunta -------------------------------------------------------
def test_una_pregunta_con_cita_correcta_se_marca_y_se_juzga(buscador):
    fila = eg.evaluar_pregunta(buscador, _proveedor(), _proveedor(), "prompt", _pregunta(), CONFIG)

    assert fila["cita_correcta"] is True and fila["falsa_abstencion"] is False and fila["abstencion_correcta"] is None
    assert fila["veredicto"] == "correcta" and fila["coste_usd"] == 0.001 and fila["error"] is None
    assert fila["recuperados"][0] == "vac"


def test_una_pregunta_cuya_cita_no_es_la_esperada_no_cuenta_como_acierto(buscador):
    fila = eg.evaluar_pregunta(buscador, _proveedor(), None, "prompt", _pregunta(esperados=("die",)), CONFIG)

    assert fila["cita_correcta"] is False and "veredicto" not in fila


def test_en_modo_todos_hacen_falta_todas_las_citas(buscador):
    fila = eg.evaluar_pregunta(buscador, _proveedor(), None, "prompt", _pregunta(esperados=("vac", "die"), modo="todos"), CONFIG)

    assert fila["cita_correcta"] is False  # el asistente simulado solo cita el primer fragmento


def test_una_pregunta_sin_respuesta_acierta_si_el_sistema_se_abstiene(buscador):
    pregunta = _pregunta(tipo="sin_respuesta", esperados=())

    se_abstiene = eg.evaluar_pregunta(buscador, _proveedor(consta=False), None, "p", pregunta, CONFIG)
    inventa = eg.evaluar_pregunta(buscador, _proveedor(consta=True), None, "p", pregunta, CONFIG)

    assert se_abstiene["abstencion_correcta"] is True and se_abstiene["cita_correcta"] is None
    assert inventa["abstencion_correcta"] is False


def test_abstenerse_ante_una_pregunta_con_respuesta_es_una_falsa_abstencion(buscador):
    fila = eg.evaluar_pregunta(buscador, _proveedor(consta=False), None, "p", _pregunta(), CONFIG)

    assert fila["falsa_abstencion"] is True and fila["cita_correcta"] is False


def test_un_fallo_del_proveedor_queda_registrado_sin_romper_la_evaluacion(buscador):
    fila = eg.evaluar_pregunta(buscador, _proveedor(estado=500), None, "p", _pregunta(), CONFIG)

    assert fila["error"] == "fallo del proveedor: HTTPStatusError" and "respuesta" not in fila


# --- resumen ----------------------------------------------------------------------------------
def test_el_resumen_agrega_tasas_costes_y_resultados_del_juez_por_tipo():
    filas = [
        {"id": "a", "tipo": "directa", "cita_correcta": True, "falsa_abstencion": False, "veredicto": "correcta",
         "fiel": True, "coste_usd": 0.001, "coste_juez_usd": 0.001, "latencia_s": 1.0, "citas_invalidas": ["z"]},
        {"id": "b", "tipo": "directa", "cita_correcta": False, "falsa_abstencion": True, "veredicto": "incorrecta",
         "fiel": False, "coste_usd": 0.003, "latencia_s": 3.0, "sin_cita": True},
        {"id": "c", "tipo": "sin_respuesta", "abstencion_correcta": True, "coste_usd": 0.002, "latencia_s": 2.0},
        {"id": "d", "tipo": "directa", "error": "fallo del proveedor"},
    ]

    datos = eg.resumen(filas)

    directa = datos["por_tipo"]["directa"]
    assert directa["preguntas"] == 3 and directa["errores"] == 1 and directa["cita_correcta"] == 0.5
    assert directa["falsa_abstencion"] == 0.5 and directa["citas_inventadas"] == 1 and directa["respuestas_sin_cita"] == 1
    assert directa["juez_correcta"] == 0.5 and directa["juez_fiel"] == 0.5
    assert datos["por_tipo"]["sin_respuesta"]["abstencion_correcta"] == 1.0
    assert datos["global"]["coste_total_usd"] == pytest.approx(0.007) and datos["global"]["latencia_s_p50"] == 2.0


def test_el_resumen_de_filas_sin_datos_no_falla():
    vacio = eg.resumen([{"id": "a", "tipo": "directa"}])["global"]

    assert vacio["cita_correcta"] is None and vacio["coste_total_usd"] is None and vacio["juez_correcta"] is None


# --- calibración con una persona --------------------------------------------------------------
def _filas_juzgadas(n: int) -> list[dict]:
    return [{"id": f"q{i:02d}", "tipo": "directa", "pregunta": f"p{i}", "referencia": "r", "respuesta": "s",
             "citas": ["a"], "veredicto": "correcta" if i % 2 else "parcial", "fiel": True} for i in range(n)]


def test_la_muestra_de_calibracion_es_reproducible_y_se_puede_medir_el_acuerdo(tmp_path):
    ruta = tmp_path / "cal" / "muestra.csv"

    assert eg.exportar_muestra(_filas_juzgadas(30), ruta, n=6) == 6
    assert eg.exportar_muestra(_filas_juzgadas(30), tmp_path / "otra.csv", n=6) == 6
    assert ruta.read_text(encoding="utf-8") == (tmp_path / "otra.csv").read_text(encoding="utf-8")
    assert eg.acuerdo_juez_humano(ruta) == {"revisadas": 0, "acuerdo_veredicto": None}

    with ruta.open(encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    for i, fila in enumerate(filas):
        fila["veredicto_humano"] = fila["veredicto_juez"] if i < 4 else "incorrecta"
    with ruta.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=eg.CAMPOS_MUESTRA)
        escritor.writeheader()
        escritor.writerows(filas)

    assert eg.acuerdo_juez_humano(ruta) == {"revisadas": 6, "acuerdo_veredicto": pytest.approx(4 / 6, abs=0.001)}


def test_la_muestra_no_incluye_respuestas_sin_juzgar(tmp_path):
    filas = _filas_juzgadas(3) + [{"id": "x", "tipo": "directa", "error": "e"}]

    assert eg.exportar_muestra(filas, tmp_path / "m.csv", n=20) == 3


# --- comando ----------------------------------------------------------------------------------
def test_el_comando_sin_proveedor_configurado_avisa_y_no_hace_nada(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for variable in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODELO"):
        monkeypatch.delenv(variable, raising=False)

    assert eg.main(["--salida", str(tmp_path / "res")]) == 2
    assert "LLM_BASE_URL" in capsys.readouterr().err and not (tmp_path / "res").exists()


def test_el_comando_evalua_escribe_resultados_y_exporta_la_muestra(tmp_path, monkeypatch, raiz, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_BASE_URL", "http://p/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODELO", "m")
    monkeypatch.setattr(eg, "FastEmbedder", lambda nombre: HashEmbedder())
    monkeypatch.setattr(eg, "ClienteLLM", lambda config: _proveedor())
    salida = tmp_path / "evals" / "resultados"

    codigo = eg.main(["--trozos", str(raiz / "datos" / "trozos.jsonl"), "--golden", str(raiz / "evals" / "golden.jsonl"),
                      "--salida", str(salida), "--limite", "4", "--exportar-muestra"])

    resultado = json.loads((salida / "generacion.json").read_text(encoding="utf-8"))
    assert codigo == 0 and resultado["resumen"]["global"]["preguntas"] == 4 and resultado["juez"] is True
    assert not (salida / "generacion.parcial.jsonl").exists()  # el volcado incremental se borra al terminar bien
    assert (tmp_path / "evals" / "calibracion" / "muestra.csv").exists()
    assert "cita_correcta" in capsys.readouterr().out


# --- cargador de .env -------------------------------------------------------------------------
def test_el_dotenv_anade_lo_que_falta_y_no_pisa_lo_definido(tmp_path):
    fichero = tmp_path / ".env"
    fichero.write_text('# comentario\nA=1\nB="dos"\nC=\nD = tres \nYA=del-fichero\nsin igual\n', encoding="utf-8")
    entorno = {"YA": "del-entorno"}

    añadidas = cargar_dotenv(fichero, entorno)

    assert añadidas == 3 and entorno == {"YA": "del-entorno", "A": "1", "B": "dos", "D": "tres"}


def test_un_dotenv_inexistente_no_hace_nada(tmp_path):
    entorno: dict[str, str] = {}

    assert cargar_dotenv(tmp_path / "no-existe", entorno) == 0 and entorno == {}
