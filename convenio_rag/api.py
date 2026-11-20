"""API REST y webhook de entrada.

Arranque (con las variables de ``.env.example`` en el entorno o en un ``.env``)::

    uvicorn convenio_rag.api:app_desde_entorno --factory

``/salud``, ``/buscar`` y ``/preguntar`` son públicos (con límite de peticiones por minuto y un tope diario);
``/ingestar``, ``/trazas`` e ``/informe`` exigen ``Authorization: Bearer <TOKEN_INGESTA>`` y tienen su propio límite;
el webhook exige el secreto ``SECRETO_WEBHOOK`` en la cabecera que envía Telegram, limita por chat y descarta los
``update_id`` repetidos.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints

from convenio_rag.almacen import Almacen
from convenio_rag.busqueda import Buscador, ConfigBusqueda
from convenio_rag.embeddings import MODELO_DEFECTO, FastEmbedder
from convenio_rag.entorno import cargar_dotenv
from convenio_rag.generacion import (
    ERRORES_PROVEEDOR,
    VERSION_PROMPT_DEFECTO,
    ClienteLLM,
    ConfigLLM,
    cargar_prompt,
    responder,
)
from convenio_rag.indexar import indexar
from convenio_rag.trazas import Trazas

FUENTE = "Basado en datos de la Agencia Estatal Boletín Oficial del Estado (https://www.boe.es)."
AVISO = "Proyecto de ingeniería, no asesoramiento jurídico ni texto oficial: contrasta siempre con el BOE."
MAX_PREGUNTA = 500
MIN_LARGO_SECRETO = 16  # longitud mínima de TOKEN_INGESTA y SECRETO_WEBHOOK
MAX_UPDATES_VISTOS = 1000
TEXTO_NO_DISPONIBLE = "El servicio no está disponible ahora mismo. Inténtalo más tarde."
TEXTO_DEMASIADOS_MENSAJES = "Demasiados mensajes seguidos: espera un minuto."


class Limitador:
    """Límite de peticiones por clave en una ventana deslizante (en memoria, para un solo proceso)."""

    PURGA_A_PARTIR_DE = 1000  # claves distintas a partir de las cuales se borran las inactivas

    def __init__(self, maximo: int, ventana_s: float = 60.0, reloj: Callable[[], float] = time.monotonic):
        self.maximo, self.ventana_s, self.reloj = maximo, ventana_s, reloj
        self._eventos: dict[str, deque[float]] = defaultdict(deque)
        self._candado = threading.Lock()

    def permitir(self, clave: str) -> bool:
        with self._candado:
            ahora = self.reloj()
            cola = self._eventos[clave]
            while cola and ahora - cola[0] >= self.ventana_s:
                cola.popleft()
            if len(cola) >= self.maximo:
                return False
            cola.append(ahora)
            if len(self._eventos) > self.PURGA_A_PARTIR_DE:
                inactivas = [k for k, c in self._eventos.items() if not c or ahora - c[-1] >= self.ventana_s]
                for k in inactivas:
                    del self._eventos[k]
            return True


@dataclass
class Servicio:
    buscador: Buscador
    trazas: Trazas
    cliente: ClienteLLM | None
    prompt: str
    version_prompt: str = VERSION_PROMPT_DEFECTO
    config_busqueda: ConfigBusqueda = field(default_factory=lambda: ConfigBusqueda(modo="hibrida", k=5, prefijo=6))
    token_admin: str | None = field(default=None, repr=False)  # fuera del repr: no debe salir en un registro
    secreto_webhook: str | None = field(default=None, repr=False)
    limitador: Limitador = field(default_factory=lambda: Limitador(30))
    limitador_admin: Limitador = field(default_factory=lambda: Limitador(10))
    limite_chat: Limitador = field(default_factory=lambda: Limitador(5))
    tope_diario_peticiones: int | None = None  # peticiones al modelo por día (UTC); None = sin tope
    tope_diario_usd: float | None = None  # coste acumulado por día (UTC); None = sin tope
    reindexar: Callable[[], int] | None = None
    updates_vistos: OrderedDict[int, None] = field(default_factory=OrderedDict)
    candado_updates: threading.Lock = field(default_factory=threading.Lock)


Pregunta = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=MAX_PREGUNTA)]


class EntradaPregunta(BaseModel):
    pregunta: Pregunta
    k: int = Field(default=5, ge=1, le=10)


def _comparar(recibido: str | None, esperado: str) -> bool:
    return recibido is not None and hmac.compare_digest(recibido.encode(), esperado.encode())


def _comprobar_topes(servicio: Servicio) -> None:
    """Corta el gasto: cada petición a /preguntar o al webhook puede costar dinero."""
    hoy = datetime.now(UTC).strftime("%Y-%m-%d")
    if servicio.tope_diario_peticiones is not None and servicio.trazas.peticiones_del_dia(hoy) >= servicio.tope_diario_peticiones:
        raise HTTPException(429, "Tope diario de peticiones alcanzado: vuelve mañana.")
    if servicio.tope_diario_usd is not None and servicio.trazas.coste_del_dia(hoy) >= servicio.tope_diario_usd:
        raise HTTPException(429, "Tope diario de gasto alcanzado: vuelve mañana.")


def atender(servicio: Servicio, pregunta: str, k: int) -> dict:
    """Recupera, genera, registra la traza y devuelve la respuesta. Lanza ``HTTPException`` si no puede."""
    if servicio.cliente is None:
        raise HTTPException(503, "Generación no configurada: define LLM_BASE_URL, LLM_API_KEY y LLM_MODELO.")
    _comprobar_topes(servicio)
    config = replace(servicio.config_busqueda, k=k)
    inicio = time.perf_counter()
    resultados = servicio.buscador.buscar(pregunta, config)
    ms_recuperacion = (time.perf_counter() - inicio) * 1000
    inicio = time.perf_counter()
    try:
        respuesta, bruto = responder(pregunta, resultados, servicio.cliente, servicio.prompt)
    except ERRORES_PROVEEDOR as error:
        _registrar_fallo(servicio, pregunta, config, resultados, ms_recuperacion, inicio, error)
        raise HTTPException(502, "El proveedor de modelos no respondió; inténtalo de nuevo.") from error
    ms_generacion = (time.perf_counter() - inicio) * 1000
    id_traza = servicio.trazas.registrar(
        pregunta=pregunta, configuracion=config.__dict__, version_prompt=servicio.version_prompt,
        modelo=servicio.cliente.config.modelo, ms_recuperacion=ms_recuperacion, ms_generacion=ms_generacion,
        tokens_entrada=bruto.tokens_entrada, tokens_salida=bruto.tokens_salida, coste_usd=bruto.coste_usd,
        consta=None if respuesta.error else respuesta.consta, citas=list(respuesta.citas),
        citas_invalidas=list(respuesta.citas_invalidas), recuperados=[r.trozo.id for r in resultados],
        error=respuesta.error,
    )
    por_id = {r.trozo.id: r.trozo for r in resultados}
    return {
        "id_traza": id_traza, "respuesta": respuesta.respuesta, "consta": respuesta.consta,
        # «sin_cita»: el modelo afirma algo pero no cita ningún fragmento válido; no debe darse por respaldado.
        "sin_cita": respuesta.consta and not respuesta.citas,
        "citas": [{"id": c, "cita": por_id[c].cita, "documento": por_id[c].documento} for c in respuesta.citas],
        "citas_invalidas": list(respuesta.citas_invalidas), "error": respuesta.error,
        "coste_usd": bruto.coste_usd, "latencia_ms": round(ms_recuperacion + ms_generacion, 1),
        "fuente": FUENTE, "aviso": AVISO,
    }


def _registrar_fallo(servicio, pregunta, config, resultados, ms_recuperacion, inicio, error) -> None:
    servicio.trazas.registrar(
        pregunta=pregunta, configuracion=config.__dict__, version_prompt=servicio.version_prompt,
        modelo=servicio.cliente.config.modelo, ms_recuperacion=ms_recuperacion,
        ms_generacion=(time.perf_counter() - inicio) * 1000, tokens_entrada=0, tokens_salida=0, coste_usd=None,
        consta=None, citas=[], citas_invalidas=[], recuperados=[r.trozo.id for r in resultados],
        error=f"fallo del proveedor: {type(error).__name__}",
    )


def _ya_visto(servicio: Servicio, update_id: object) -> bool:
    """Telegram reenvía una actualización si no recibe un 200: se descartan las repetidas para no pagar dos veces."""
    if not isinstance(update_id, int):
        return False
    with servicio.candado_updates:
        if update_id in servicio.updates_vistos:
            return True
        servicio.updates_vistos[update_id] = None
        while len(servicio.updates_vistos) > MAX_UPDATES_VISTOS:
            servicio.updates_vistos.popitem(last=False)
    return False


def _respuesta_telegram(chat: int, texto: str) -> dict:
    return {"method": "sendMessage", "chat_id": chat, "text": texto}


def crear_app(servicio: Servicio) -> FastAPI:
    app = FastAPI(
        title="convenio-rag-evals",
        version="0.2.0",
        description="Asistente RAG con citas sobre un convenio colectivo del BOE. " + AVISO + " " + FUENTE,
    )

    def cliente_de(peticion: Request) -> str:
        return peticion.client.host if peticion.client else "desconocido"

    def limitar(peticion: Request) -> None:
        if not servicio.limitador.permitir(cliente_de(peticion)):
            raise HTTPException(429, "Demasiadas peticiones: espera un minuto.")

    def limitar_admin(peticion: Request) -> None:
        """Frena la fuerza bruta sobre el token: cuenta todos los intentos, acierten o no."""
        if not servicio.limitador_admin.permitir(cliente_de(peticion)):
            raise HTTPException(429, "Demasiados intentos: espera un minuto.")

    def exigir_admin(authorization: str | None = Header(default=None)) -> None:
        if not servicio.token_admin:
            raise HTTPException(403, "Endpoint deshabilitado: no hay TOKEN_INGESTA configurado.")
        if not _comparar(authorization, f"Bearer {servicio.token_admin}"):
            raise HTTPException(401, "Token no válido.")

    admin = [Depends(limitar_admin), Depends(exigir_admin)]

    @app.get("/salud")
    def salud() -> dict:
        return {"estado": "ok", "trozos": servicio.buscador.almacen.contar(), "generacion": servicio.cliente is not None,
                "version_prompt": servicio.version_prompt, "fuente": FUENTE}

    @app.post("/buscar", dependencies=[Depends(limitar)])
    def buscar(entrada: EntradaPregunta) -> dict:
        resultados = servicio.buscador.buscar(entrada.pregunta, replace(servicio.config_busqueda, k=entrada.k))
        return {
            "resultados": [
                {"id": r.trozo.id, "cita": r.trozo.cita, "puntuacion": round(r.puntuacion, 4),
                 "extracto": r.trozo.texto[:300]}
                for r in resultados
            ],
            "fuente": FUENTE,
        }

    @app.post("/preguntar", dependencies=[Depends(limitar)])
    def preguntar(entrada: EntradaPregunta) -> dict:
        return atender(servicio, entrada.pregunta, entrada.k)

    @app.post("/ingestar", dependencies=admin)
    def ingestar() -> dict:
        if servicio.reindexar is None:
            raise HTTPException(501, "Reindexado no configurado.")
        return {"trozos": servicio.reindexar()}

    @app.get("/trazas/{id_traza}", dependencies=admin)
    def traza(id_traza: str) -> dict:
        datos = servicio.trazas.obtener(id_traza)
        if datos is None:
            raise HTTPException(404, "Traza no encontrada.")
        return datos

    @app.get("/informe", dependencies=admin)
    def informe() -> dict:
        return servicio.trazas.informe()

    @app.post("/webhook/telegram")
    def webhook_telegram(
        actualizacion: dict, x_telegram_bot_api_secret_token: str | None = Header(default=None)
    ) -> dict:
        if not servicio.secreto_webhook:
            raise HTTPException(403, "Webhook deshabilitado: no hay SECRETO_WEBHOOK configurado.")
        if not _comparar(x_telegram_bot_api_secret_token, servicio.secreto_webhook):
            raise HTTPException(401, "Secreto del webhook no válido.")
        mensaje = actualizacion.get("message") or {}
        texto = str(mensaje.get("text") or "").strip()[:MAX_PREGUNTA]
        chat = (mensaje.get("chat") or {}).get("id")
        if len(texto) < 3 or not isinstance(chat, int) or _ya_visto(servicio, actualizacion.get("update_id")):
            return {"ok": True}  # nada que responder: sin texto, sin chat o una actualización repetida
        # El secreto identifica a Telegram, no al usuario, y todo llega desde las IP de Telegram: se limita por chat.
        if not servicio.limite_chat.permitir(str(chat)):
            return _respuesta_telegram(chat, TEXTO_DEMASIADOS_MENSAJES)
        try:
            resultado = atender(servicio, texto, k=5)
        except HTTPException:
            # Siempre 200: si Telegram recibe un error reenvía la actualización y cada reenvío podría costar dinero.
            return _respuesta_telegram(chat, TEXTO_NO_DISPONIBLE)
        fuentes = "; ".join(c["cita"] for c in resultado["citas"]) or "sin cita"
        return _respuesta_telegram(chat, f"{resultado['respuesta']}\n\nFuentes: {fuentes}\n{AVISO}")

    return app


def _secreto(env: Mapping[str, str], nombre: str) -> str | None:
    valor = env.get(nombre) or None
    if valor is not None and len(valor) < MIN_LARGO_SECRETO:
        raise ValueError(f"{nombre} es demasiado corto: usa al menos {MIN_LARGO_SECRETO} caracteres.")
    return valor


def servicio_desde_entorno(entorno: Mapping[str, str] | None = None) -> Servicio:
    env = os.environ if entorno is None else entorno
    # Un valor vacío (como los de .env.example) cuenta como no definido.
    db = Path(env.get("RAG_DB") or "datos/convenio.db")
    trozos = Path(env.get("RAG_TROZOS") or "datos/trozos.jsonl")
    embedder = FastEmbedder(env.get("RAG_EMBEDDINGS") or MODELO_DEFECTO)
    db.parent.mkdir(parents=True, exist_ok=True)
    almacen = Almacen(db)
    # Se reindexa si la base está vacía O si faltan los vectores del modelo actual (p. ej. tras cambiar RAG_EMBEDDINGS).
    if almacen.contar() == 0 or not almacen.tiene_vectores(embedder.nombre):
        indexar(trozos, db, [embedder], almacen)
    buscador = Buscador(almacen, embedder)

    def reindexar() -> int:
        indexar(trozos, db, [embedder], almacen)
        buscador.recargar_vectores()
        return almacen.contar()

    config_llm = ConfigLLM.desde_entorno(env)
    tope_usd = env.get("RAG_TOPE_DIARIO_USD")
    return Servicio(
        buscador=buscador, trazas=Trazas(env.get("RAG_TRAZAS") or "datos/trazas.db"),
        cliente=ClienteLLM(config_llm) if config_llm else None, prompt=cargar_prompt(),
        token_admin=_secreto(env, "TOKEN_INGESTA"), secreto_webhook=_secreto(env, "SECRETO_WEBHOOK"),
        limitador=Limitador(int(env.get("RAG_LIMITE_MIN") or "30")),
        limite_chat=Limitador(int(env.get("RAG_LIMITE_CHAT_MIN") or "5")),
        tope_diario_peticiones=int(env.get("RAG_TOPE_DIARIO_PETICIONES") or "1000"),
        tope_diario_usd=float(tope_usd) if tope_usd else None, reindexar=reindexar,
    )


def app_desde_entorno() -> FastAPI:
    cargar_dotenv()  # las variables ya definidas en el entorno mandan sobre las del .env
    return crear_app(servicio_desde_entorno())
