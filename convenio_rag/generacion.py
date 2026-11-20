"""Generación con citas (decisión D05).

El cliente habla con cualquier endpoint compatible con la API de chat de OpenAI. El caso previsto es una pasarela
LiteLLM, que además devuelve el coste de cada petición en la cabecera ``x-litellm-response-cost``; con otros
endpoints el coste queda en ``None`` y el resto de métricas funciona igual.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from convenio_rag.modelos import Resultado

DIRECTORIO_PROMPTS = Path(os.environ.get("RAG_PROMPTS_DIR", Path(__file__).resolve().parent.parent / "prompts"))
VERSION_PROMPT_DEFECTO = "v1"
CABECERA_COSTE = "x-litellm-response-cost"


class ErrorProveedor(Exception):
    """El proveedor respondió, pero con algo que no tiene la forma de una respuesta de chat."""


# Todo lo que puede salir mal al llamar al proveedor: red, estado HTTP o una respuesta con forma inesperada.
ERRORES_PROVEEDOR = (httpx.HTTPError, ErrorProveedor)


@dataclass(frozen=True)
class ConfigLLM:
    base_url: str  # p. ej. http://127.0.0.1:4141/v1
    api_key: str = field(repr=False)  # fuera del repr: no debe salir en un registro de errores
    modelo: str
    timeout: float = 90.0
    razonamiento: str | None = None  # valor de ``reasoning_effort`` si el modelo lo admite

    @staticmethod
    def desde_entorno(entorno: Mapping[str, str] | None = None) -> "ConfigLLM | None":
        """Lee ``LLM_BASE_URL``, ``LLM_API_KEY`` y ``LLM_MODELO``; devuelve ``None`` si falta alguno."""
        entorno = os.environ if entorno is None else entorno
        base, clave, modelo = (entorno.get(n, "") for n in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODELO"))
        if not (base and clave and modelo):
            return None
        return ConfigLLM(base, clave, modelo, razonamiento=entorno.get("LLM_RAZONAMIENTO") or None)


@dataclass(frozen=True)
class RespuestaLLM:
    contenido: str
    tokens_entrada: int
    tokens_salida: int
    coste_usd: float | None
    latencia_s: float


class ClienteLLM:
    def __init__(self, config: ConfigLLM, http: httpx.Client | None = None):
        self.config = config
        self.http = http or httpx.Client(timeout=config.timeout)

    def completar(self, sistema: str, usuario: str) -> RespuestaLLM:
        cuerpo: dict = {
            "model": self.config.modelo,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": sistema}, {"role": "user", "content": usuario}],
        }
        if self.config.razonamiento:
            cuerpo["reasoning_effort"] = self.config.razonamiento
        inicio = time.perf_counter()
        respuesta = self.http.post(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            json=cuerpo,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
        )
        latencia = time.perf_counter() - inicio
        respuesta.raise_for_status()
        try:
            datos = respuesta.json()
            uso = datos.get("usage") or {}
            coste = respuesta.headers.get(CABECERA_COSTE)
            return RespuestaLLM(
                contenido=str(datos["choices"][0]["message"]["content"]),
                tokens_entrada=int(uso.get("prompt_tokens", 0)),
                tokens_salida=int(uso.get("completion_tokens", 0)),
                coste_usd=float(coste) if coste not in (None, "") else None,
                latencia_s=latencia,
            )
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
            raise ErrorProveedor(f"respuesta del proveedor con forma inesperada ({type(error).__name__})") from error


def cargar_prompt(version: str = VERSION_PROMPT_DEFECTO, directorio: Path | None = None) -> str:
    ruta = (directorio or DIRECTORIO_PROMPTS) / f"{version}.md"
    return ruta.read_text(encoding="utf-8").strip()


def formatear_fragmentos(resultados: list[Resultado]) -> str:
    return "\n\n".join(f"[{r.trozo.id}] {r.trozo.cita}\n{r.trozo.texto}" for r in resultados)


def neutralizar_pregunta(pregunta: str) -> str:
    """Impide que la pregunta imite la estructura del mensaje (una sección falsa o un fragmento con un id real).

    Se pliegan los saltos de línea, se quitan los corchetes (con los que se cita un fragmento) y se rebajan las
    palabras que marcan las secciones. La pregunta sigue siendo legible; lo que ya no puede es parecer un fragmento.
    """
    limpia = " ".join(pregunta.split()).replace("[", "(").replace("]", ")")
    return limpia.replace("FRAGMENTOS", "fragmentos").replace("PREGUNTA", "pregunta")


def construir_usuario(pregunta: str, resultados: list[Resultado]) -> str:
    return f"FRAGMENTOS\n{formatear_fragmentos(resultados)}\n\nPREGUNTA\n{neutralizar_pregunta(pregunta)}"


@dataclass(frozen=True)
class Respuesta:
    respuesta: str
    consta: bool
    citas: tuple[str, ...]  # solo las que corresponden a un fragmento realmente entregado
    citas_invalidas: tuple[str, ...]  # citas que el modelo inventó o que no estaban entre los fragmentos
    error: str | None = None


def interpretar(contenido: str, ids_validos: set[str]) -> Respuesta:
    """Convierte la salida del modelo en una ``Respuesta``; si no es JSON válido devuelve un error explícito."""
    limpio = re.sub(r"^```(?:json)?\s*|\s*```$", "", contenido.strip())
    try:
        datos = json.loads(limpio)
        texto, consta, citas = datos["respuesta"], datos["consta"], datos.get("citas", [])
        # Tipos estrictos: bool("false") es True y una cadena en «citas» se recorrería carácter a carácter.
        if not (isinstance(texto, str) and isinstance(consta, bool) and isinstance(citas, list)
                and all(isinstance(c, str) for c in citas)):
            raise TypeError("tipos inesperados")
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return Respuesta("", False, (), (), error="la respuesta del modelo no es el JSON esperado")
    # También cuentan como citas inventadas los [id] del texto que no corresponden a un fragmento entregado.
    en_texto = re.findall(r"\[([^\[\]\s]+)\]", texto)
    validas = tuple(c for c in dict.fromkeys(citas) if c in ids_validos)
    invalidas = tuple(c for c in dict.fromkeys([*citas, *en_texto]) if c not in ids_validos)
    return Respuesta(texto, consta, validas, invalidas)


def responder(
    pregunta: str, resultados: list[Resultado], cliente: ClienteLLM, prompt: str
) -> tuple[Respuesta, RespuestaLLM]:
    bruto = cliente.completar(prompt, construir_usuario(pregunta, resultados))
    return interpretar(bruto.contenido, {r.trozo.id for r in resultados}), bruto
