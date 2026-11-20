"""Pruebas de la API REST y del webhook: autenticación, límites, validación y trazas (proveedor simulado)."""

import json
import os

import httpx
import pytest
from fastapi.testclient import TestClient

from convenio_rag import api
from convenio_rag.almacen import Almacen
from convenio_rag.api import (
    AVISO, TEXTO_DEMASIADOS_MENSAJES, TEXTO_NO_DISPONIBLE, Limitador, Servicio, crear_app, servicio_desde_entorno,
)
from convenio_rag.busqueda import Buscador
from convenio_rag.embeddings import HashEmbedder
from convenio_rag.generacion import CABECERA_COSTE, ClienteLLM, ConfigLLM
from convenio_rag.ingesta import guardar_jsonl
from convenio_rag.modelos import Trozo
from convenio_rag.trazas import Trazas

CORPUS = [
    Trozo("vac", "vac", "D", "articulo", "26", "Vacaciones", "", "Convenio, art. 26 (Vacaciones)",
          "Artículo 26. Vacaciones. Treinta y un días naturales de vacaciones al año.", 0, "2026-01-01"),
    Trozo("jor", "jor", "D", "articulo", "22", "Jornada", "", "Convenio, art. 22 (Jornada)",
          "Artículo 22. Jornada. La jornada máxima anual será de 1.788 horas.", 1, "2026-01-01"),
]
RESPUESTA_BUENA = '{"respuesta": "Treinta y un días naturales [vac].", "consta": true, "citas": ["vac", "inventada"]}'
TOKEN, SECRETO = "t" * 24, "s" * 24  # valores de prueba generados: que no parezcan un secreto al escáner


def _cliente_llm(contenido: str = RESPUESTA_BUENA, estado: int = 200) -> ClienteLLM:
    def manejador(peticion: httpx.Request) -> httpx.Response:
        cuerpo = {"choices": [{"message": {"content": contenido}}], "usage": {"prompt_tokens": 50, "completion_tokens": 10}}
        return httpx.Response(estado, json=cuerpo, headers={CABECERA_COSTE: "0.0005"})

    return ClienteLLM(ConfigLLM("http://proxy/v1", "k", "modelo-x"), httpx.Client(transport=httpx.MockTransport(manejador)))


def _servicio(cliente: ClienteLLM | None = None, **cambios) -> Servicio:
    almacen, embedder = Almacen(), HashEmbedder()
    almacen.cargar(CORPUS)
    almacen.guardar_vectores(embedder.nombre, [t.id for t in CORPUS], embedder.pasajes([t.texto for t in CORPUS]))
    base = dict(buscador=Buscador(almacen, embedder), trazas=Trazas(), cliente=cliente, prompt="prompt de prueba",
                token_admin=TOKEN, secreto_webhook=SECRETO, limitador=Limitador(100))
    return Servicio(**{**base, **cambios})


@pytest.fixture
def servicio() -> Servicio:
    return _servicio(_cliente_llm())


@pytest.fixture
def http(servicio) -> TestClient:
    return TestClient(crear_app(servicio))


ADMIN = {"Authorization": f"Bearer {TOKEN}"}


# --- públicos ---------------------------------------------------------------------------------
def test_salud_informa_del_estado_sin_pedir_credenciales(http):
    datos = http.get("/salud").json()

    assert datos["estado"] == "ok" and datos["trozos"] == 2 and datos["generacion"] is True
    assert "Agencia Estatal Boletín Oficial del Estado" in datos["fuente"]


def test_buscar_devuelve_trozos_con_cita_y_extracto(http):
    datos = http.post("/buscar", json={"pregunta": "días de vacaciones", "k": 1}).json()

    assert [r["id"] for r in datos["resultados"]] == ["vac"]
    assert datos["resultados"][0]["cita"] == "Convenio, art. 26 (Vacaciones)" and "Boletín" in datos["fuente"]


def test_preguntar_responde_con_citas_validas_coste_y_traza(http):
    datos = http.post("/preguntar", json={"pregunta": "¿Cuántos días de vacaciones hay?"}).json()

    assert datos["respuesta"].startswith("Treinta y un días") and datos["consta"] is True
    assert datos["citas"] == [{"id": "vac", "cita": "Convenio, art. 26 (Vacaciones)", "documento": "D"}]
    assert datos["citas_invalidas"] == ["inventada"] and datos["coste_usd"] == 0.0005
    assert AVISO == datos["aviso"] and datos["error"] is None
    traza = http.get(f"/trazas/{datos['id_traza']}", headers=ADMIN).json()
    assert traza["recuperados"][0] == "vac" and traza["tokens_entrada"] == 50 and traza["version_prompt"] == "v1"


def test_una_salida_del_modelo_que_no_es_json_se_responde_con_error_y_se_traza():
    http = TestClient(crear_app(_servicio(_cliente_llm("esto no es json"))))

    datos = http.post("/preguntar", json={"pregunta": "¿Vacaciones?"}).json()

    assert datos["error"] and datos["consta"] is False and datos["citas"] == []
    assert http.get("/informe", headers=ADMIN).json()["errores"] == 1


def test_si_el_proveedor_falla_la_api_responde_502_y_deja_traza_del_error():
    servicio = _servicio(_cliente_llm(estado=500))
    http = TestClient(crear_app(servicio))

    respuesta = http.post("/preguntar", json={"pregunta": "¿Vacaciones?"})

    assert respuesta.status_code == 502
    assert servicio.trazas.informe()["errores"] == 1


def test_sin_cliente_de_modelos_preguntar_responde_503_pero_buscar_funciona():
    http = TestClient(crear_app(_servicio(cliente=None)))

    assert http.post("/preguntar", json={"pregunta": "¿Vacaciones?"}).status_code == 503
    assert http.post("/buscar", json={"pregunta": "vacaciones"}).status_code == 200
    assert http.get("/salud").json()["generacion"] is False


@pytest.mark.parametrize("cuerpo", [{"pregunta": "ab"}, {"pregunta": "x" * 501}, {"pregunta": "valida", "k": 0},
                                    {"pregunta": "valida", "k": 11}, {}])
def test_las_entradas_invalidas_se_rechazan_con_422(http, cuerpo):
    assert http.post("/preguntar", json=cuerpo).status_code == 422


def test_el_limite_de_peticiones_por_minuto_responde_429():
    http = TestClient(crear_app(_servicio(_cliente_llm(), limitador=Limitador(2))))

    codigos = [http.post("/buscar", json={"pregunta": "vacaciones"}).status_code for _ in range(3)]

    assert codigos == [200, 200, 429]


def test_el_limitador_libera_la_ventana_con_el_tiempo():
    reloj = [0.0]
    limitador = Limitador(2, ventana_s=60, reloj=lambda: reloj[0])

    assert [limitador.permitir("a"), limitador.permitir("a"), limitador.permitir("a")] == [True, True, False]
    assert limitador.permitir("b") is True  # cada cliente tiene su propio contador
    reloj[0] = 61
    assert limitador.permitir("a") is True


# --- administración ---------------------------------------------------------------------------
def test_los_endpoints_de_administracion_exigen_el_token(http):
    assert http.get("/informe").status_code == 401
    assert http.get("/informe", headers={"Authorization": "Bearer otro"}).status_code == 401
    assert http.get("/informe", headers=ADMIN).status_code == 200
    assert http.get("/trazas/no-existe", headers=ADMIN).status_code == 404
    assert http.post("/ingestar").status_code == 401


def test_sin_token_configurado_la_administracion_queda_deshabilitada():
    http = TestClient(crear_app(_servicio(_cliente_llm(), token_admin=None)))

    assert http.get("/informe", headers=ADMIN).status_code == 403


def test_ingestar_reindexa_y_devuelve_el_numero_de_trozos():
    http = TestClient(crear_app(_servicio(_cliente_llm(), reindexar=lambda: 42)))

    assert http.post("/ingestar", headers=ADMIN).json() == {"trozos": 42}


def test_ingestar_sin_reindexador_configurado_responde_501(http):
    assert http.post("/ingestar", headers=ADMIN).status_code == 501


# --- webhook ----------------------------------------------------------------------------------
def _mensaje(texto: str | None, chat: int | None = 7) -> dict:
    mensaje: dict = {"chat": {"id": chat}} if chat is not None else {}
    if texto is not None:
        mensaje["text"] = texto
    return {"message": mensaje}


def test_el_webhook_rechaza_peticiones_sin_el_secreto_correcto(http):
    assert http.post("/webhook/telegram", json=_mensaje("hola")).status_code == 401
    cabecera = {"X-Telegram-Bot-Api-Secret-Token": "otro"}
    assert http.post("/webhook/telegram", json=_mensaje("hola"), headers=cabecera).status_code == 401


def test_el_webhook_queda_deshabilitado_si_no_hay_secreto_configurado():
    http = TestClient(crear_app(_servicio(_cliente_llm(), secreto_webhook=None)))

    assert http.post("/webhook/telegram", json=_mensaje("hola")).status_code == 403


def test_el_webhook_responde_con_un_sendmessage_con_fuentes_y_aviso(http):
    cabecera = {"X-Telegram-Bot-Api-Secret-Token": SECRETO}

    datos = http.post("/webhook/telegram", json=_mensaje("¿Cuántos días de vacaciones?"), headers=cabecera).json()

    assert datos["method"] == "sendMessage" and datos["chat_id"] == 7
    assert "Treinta y un días" in datos["text"] and "Convenio, art. 26 (Vacaciones)" in datos["text"]
    assert AVISO in datos["text"]


@pytest.mark.parametrize("actualizacion", [_mensaje(None), _mensaje("hi"), _mensaje("pregunta válida", chat=None), {}])
def test_el_webhook_ignora_mensajes_sin_texto_util_o_sin_chat(http, actualizacion):
    cabecera = {"X-Telegram-Bot-Api-Secret-Token": SECRETO}

    assert http.post("/webhook/telegram", json=actualizacion, headers=cabecera).json() == {"ok": True}


def test_el_webhook_recorta_las_preguntas_demasiado_largas():
    capturas: list[str] = []

    def manejador(peticion: httpx.Request) -> httpx.Response:
        capturas.append(json.loads(peticion.content)["messages"][1]["content"])
        return httpx.Response(200, json={"choices": [{"message": {"content": RESPUESTA_BUENA}}], "usage": {}})

    cliente = ClienteLLM(ConfigLLM("http://p/v1", "k", "m"), httpx.Client(transport=httpx.MockTransport(manejador)))
    http = TestClient(crear_app(_servicio(cliente)))

    http.post("/webhook/telegram", json=_mensaje("a" * 900), headers={"X-Telegram-Bot-Api-Secret-Token": SECRETO})

    assert len(capturas[0].split("PREGUNTA\n")[1]) == 500


CABECERA_TELEGRAM = {"X-Telegram-Bot-Api-Secret-Token": SECRETO}


def _con_conteo(contenido: str = RESPUESTA_BUENA, estado: int = 200) -> tuple[ClienteLLM, list[int]]:
    """Un proveedor simulado que cuenta cuántas veces se le llama."""
    llamadas: list[int] = []

    def manejador(peticion: httpx.Request) -> httpx.Response:
        llamadas.append(1)
        cuerpo = {"choices": [{"message": {"content": contenido}}], "usage": {"prompt_tokens": 5, "completion_tokens": 5}}
        return httpx.Response(estado, json=cuerpo, headers={CABECERA_COSTE: "0.0005"})

    return ClienteLLM(ConfigLLM("http://p/v1", "k", "m"), httpx.Client(transport=httpx.MockTransport(manejador))), llamadas


def test_el_webhook_limita_por_chat_y_no_por_ip():
    cliente, llamadas = _con_conteo()
    http = TestClient(crear_app(_servicio(cliente, limite_chat=Limitador(1))))

    primera = http.post("/webhook/telegram", json=_mensaje("¿vacaciones?"), headers=CABECERA_TELEGRAM).json()
    segunda = http.post("/webhook/telegram", json=_mensaje("¿vacaciones otra vez?"), headers=CABECERA_TELEGRAM).json()
    otro_chat = http.post("/webhook/telegram", json=_mensaje("¿vacaciones?", chat=8), headers=CABECERA_TELEGRAM).json()

    assert "Treinta y un días" in primera["text"] and segunda["text"] == TEXTO_DEMASIADOS_MENSAJES
    assert "Treinta y un días" in otro_chat["text"] and len(llamadas) == 2  # la segunda no llegó al modelo


def test_el_webhook_descarta_las_actualizaciones_repetidas_de_telegram():
    cliente, llamadas = _con_conteo()
    http = TestClient(crear_app(_servicio(cliente)))
    actualizacion = {"update_id": 42, **_mensaje("¿Cuántos días de vacaciones?")}

    primera = http.post("/webhook/telegram", json=actualizacion, headers=CABECERA_TELEGRAM).json()
    reenvio = http.post("/webhook/telegram", json=actualizacion, headers=CABECERA_TELEGRAM).json()

    assert primera["method"] == "sendMessage" and reenvio == {"ok": True} and len(llamadas) == 1


def test_el_webhook_responde_200_aunque_falle_el_proveedor_para_que_telegram_no_reenvie():
    cliente, _ = _con_conteo(estado=500)
    http = TestClient(crear_app(_servicio(cliente)))

    respuesta = http.post("/webhook/telegram", json=_mensaje("¿vacaciones?"), headers=CABECERA_TELEGRAM)

    assert respuesta.status_code == 200 and respuesta.json()["text"] == TEXTO_NO_DISPONIBLE


def test_el_tope_diario_de_peticiones_corta_preguntar_y_el_webhook():
    cliente, llamadas = _con_conteo()
    http = TestClient(crear_app(_servicio(cliente, tope_diario_peticiones=1)))

    assert http.post("/preguntar", json={"pregunta": "¿Vacaciones?"}).status_code == 200
    assert http.post("/preguntar", json={"pregunta": "¿Vacaciones?"}).status_code == 429
    assert http.post("/webhook/telegram", json=_mensaje("¿vacaciones?"), headers=CABECERA_TELEGRAM).json()["text"] == TEXTO_NO_DISPONIBLE
    assert len(llamadas) == 1


def test_el_tope_diario_de_gasto_corta_cuando_se_alcanza():
    cliente, llamadas = _con_conteo()
    http = TestClient(crear_app(_servicio(cliente, tope_diario_usd=0.0004)))

    assert http.post("/preguntar", json={"pregunta": "¿Vacaciones?"}).status_code == 200  # cuesta 0,0005
    assert http.post("/preguntar", json={"pregunta": "¿Vacaciones?"}).status_code == 429 and len(llamadas) == 1


def test_el_limite_de_administracion_cuenta_los_intentos_fallidos():
    http = TestClient(crear_app(_servicio(_cliente_llm(), limitador_admin=Limitador(2))))
    mal = {"Authorization": "Bearer incorrecto"}

    codigos = [http.get("/informe", headers=mal).status_code for _ in range(3)]

    assert codigos == [401, 401, 429]
    assert http.get("/informe", headers=ADMIN).status_code == 429  # ni con el token bueno mientras dure el bloqueo


def test_una_pregunta_de_solo_espacios_se_rechaza_y_las_demas_se_recortan(http):
    assert http.post("/preguntar", json={"pregunta": "     "}).status_code == 422
    assert http.post("/buscar", json={"pregunta": "   vacaciones   "}).status_code == 200


def test_una_respuesta_que_afirma_sin_citar_ningun_fragmento_se_marca_sin_cita():
    cliente, _ = _con_conteo('{"respuesta": "31 días.", "consta": true, "citas": []}')
    http = TestClient(crear_app(_servicio(cliente)))

    datos = http.post("/preguntar", json={"pregunta": "¿Vacaciones?"}).json()

    assert datos["consta"] is True and datos["sin_cita"] is True and datos["citas"] == []


def test_los_secretos_no_salen_en_el_repr_del_servicio_ni_de_la_configuracion():
    servicio = _servicio(_cliente_llm())

    assert TOKEN not in repr(servicio) and SECRETO not in repr(servicio)
    assert "clave-secreta" not in repr(ConfigLLM("http://p/v1", "clave-secreta", "m"))


# --- composición desde el entorno -------------------------------------------------------------
def test_un_token_o_secreto_demasiado_corto_impide_arrancar(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "FastEmbedder", lambda nombre: HashEmbedder())
    jsonl = tmp_path / "trozos.jsonl"
    guardar_jsonl(CORPUS, jsonl)
    base = {"RAG_DB": str(tmp_path / "db.sqlite"), "RAG_TROZOS": str(jsonl), "RAG_TRAZAS": ":memory:"}

    with pytest.raises(ValueError, match="TOKEN_INGESTA es demasiado corto"):
        servicio_desde_entorno(base | {"TOKEN_INGESTA": "abc"})
    with pytest.raises(ValueError, match="SECRETO_WEBHOOK es demasiado corto"):
        servicio_desde_entorno(base | {"SECRETO_WEBHOOK": "corto"})


def test_al_arrancar_se_reindexa_si_faltan_los_vectores_del_modelo_actual(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "FastEmbedder", lambda nombre: HashEmbedder())
    jsonl, db = tmp_path / "trozos.jsonl", tmp_path / "db.sqlite"
    guardar_jsonl(CORPUS, jsonl)
    from convenio_rag.indexar import indexar
    indexar(jsonl, db, []).cerrar()  # una base antigua: tiene los trozos pero ningún vector (como tras cambiar de modelo)

    servicio = servicio_desde_entorno({"RAG_DB": str(db), "RAG_TROZOS": str(jsonl), "RAG_TRAZAS": ":memory:"})

    assert servicio.buscador.almacen.tiene_vectores("hash-256")
    assert servicio.buscador.buscar("vacaciones")[0].trozo.id == "vac"  # la híbrida ya puede usar los vectores


def test_el_valor_por_defecto_del_tope_diario_de_peticiones_es_1000(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "FastEmbedder", lambda nombre: HashEmbedder())
    jsonl = tmp_path / "trozos.jsonl"
    guardar_jsonl(CORPUS, jsonl)

    servicio = servicio_desde_entorno({"RAG_DB": str(tmp_path / "db.sqlite"), "RAG_TROZOS": str(jsonl),
                                       "RAG_TRAZAS": ":memory:", "RAG_TOPE_DIARIO_USD": "2.5"})

    assert servicio.tope_diario_peticiones == 1000 and servicio.tope_diario_usd == 2.5
def test_el_servicio_se_compone_desde_el_entorno_y_puede_reindexar(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "FastEmbedder", lambda nombre: HashEmbedder())
    jsonl = tmp_path / "trozos.jsonl"
    guardar_jsonl(CORPUS, jsonl)
    entorno = {
        "RAG_DB": str(tmp_path / "datos" / "db.sqlite"), "RAG_TROZOS": str(jsonl), "RAG_TRAZAS": str(tmp_path / "t.db"),
        "TOKEN_INGESTA": TOKEN, "SECRETO_WEBHOOK": SECRETO, "RAG_LIMITE_MIN": "7",
        "LLM_BASE_URL": "http://proxy/v1", "LLM_API_KEY": "k", "LLM_MODELO": "m",
    }

    servicio = servicio_desde_entorno(entorno)

    assert servicio.buscador.almacen.contar() == 2 and servicio.cliente is not None
    assert servicio.token_admin == TOKEN and servicio.limitador.maximo == 7 and servicio.reindexar() == 2
    assert TestClient(crear_app(servicio)).get("/salud").json()["trozos"] == 2


def test_la_aplicacion_desde_el_entorno_lee_el_dotenv_del_directorio_actual(tmp_path, monkeypatch):
    jsonl = tmp_path / "trozos.jsonl"
    guardar_jsonl(CORPUS, jsonl)
    (tmp_path / ".env").write_text(
        f"RAG_DB={tmp_path / 'db.sqlite'}\nRAG_TROZOS={jsonl}\nRAG_TRAZAS=:memory:\nTOKEN_INGESTA={TOKEN}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(os, "environ", {})  # el .env no debe filtrarse al resto de las pruebas
    monkeypatch.setattr(api, "FastEmbedder", lambda nombre: HashEmbedder())

    http = TestClient(api.app_desde_entorno())

    assert http.get("/salud").json()["trozos"] == 2 and http.get("/informe", headers=ADMIN).status_code == 200


def _leer_env_example(raiz) -> dict[str, str]:
    """Las variables de .env.example con el valor que tendrían si alguien lo copiara sin rellenar."""
    pares = (linea.split("=", 1) for linea in (raiz / ".env.example").read_text(encoding="utf-8").splitlines()
             if linea and not linea.startswith("#"))
    return {nombre: valor.strip() for nombre, valor in pares}


def test_el_env_example_copiado_sin_rellenar_deja_arrancar_el_servicio(raiz, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "FastEmbedder", lambda nombre: HashEmbedder())
    jsonl = tmp_path / "trozos.jsonl"
    guardar_jsonl(CORPUS, jsonl)
    entorno = _leer_env_example(raiz) | {"RAG_DB": str(tmp_path / "db.sqlite"), "RAG_TROZOS": str(jsonl),
                                         "RAG_TRAZAS": ":memory:"}

    servicio = servicio_desde_entorno(entorno)

    assert entorno["RAG_LIMITE_MIN"] == "" and servicio.limitador.maximo == 30
    assert servicio.cliente is None and servicio.token_admin is None and servicio.secreto_webhook is None


def test_el_env_example_con_valores_vacios_no_contiene_ningun_secreto(raiz):
    assert all(valor == "" for valor in _leer_env_example(raiz).values())


def test_sin_variables_de_modelo_el_servicio_arranca_solo_con_busqueda(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "FastEmbedder", lambda nombre: HashEmbedder())
    jsonl = tmp_path / "trozos.jsonl"
    guardar_jsonl(CORPUS, jsonl)

    servicio = servicio_desde_entorno({"RAG_DB": str(tmp_path / "db.sqlite"), "RAG_TROZOS": str(jsonl),
                                       "RAG_TRAZAS": ":memory:"})

    assert servicio.cliente is None and servicio.token_admin is None and servicio.secreto_webhook is None
