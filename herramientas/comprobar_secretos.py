"""Comprueba que no se cuela ninguna credencial ni ruta personal antes de subir el repositorio a GitHub.

  python herramientas/comprobar_secretos.py                  # lo que git subiría (con repo) o el árbol sin cachés (sin repo)
  python herramientas/comprobar_secretos.py --raiz CARPETA   # analizar otra carpeta

Sale con código 1 si hay algún hallazgo. NUNCA imprime el valor encontrado: solo «ERROR fichero:línea TIPO».
Solo biblioteca estándar. Lo usan `.githooks/pre-commit` y la integración continua.

Límite conocido: los ficheros de más de 5 MB o con bytes nulos (binarios) solo se comprueban por el nombre."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

RAIZ_POR_DEFECTO = Path(__file__).resolve().parent.parent
MAX_BYTES = 5_000_000


@dataclass(frozen=True)
class Hallazgo:
    ruta: str
    linea: int
    tipo: str


SECRETOS = {
    "BLOQUE_CLAVE_PRIVADA": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "CUENTA_DE_SERVICIO_GOOGLE": re.compile(r'"type"\s*:\s*"service_account"|"private_key_id"\s*:'),
    "API_KEY_ANTHROPIC": re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}"),
    "API_KEY_SK": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}"),  # OpenAI, DeepSeek y compatibles
    "API_KEY_XAI": re.compile(r"\bxai-[A-Za-z0-9]{20,}"),
    "API_KEY_GOOGLE": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "TOKEN_OAUTH_GOOGLE": re.compile(r"ya29\.[0-9A-Za-z_\-]{20,}"),
    "TOKEN_TELEGRAM": re.compile(r"\b\d{8,10}:[A-Za-z0-9_\-]{35}\b"),
    "TOKEN_GITHUB": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),
    "CLAVE_AWS": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "TOKEN_SLACK": re.compile(r"xox[abprs]-[0-9A-Za-z\-]{10,}"),
}

# El nombre puede llevar prefijo o sufijo (WEBHOOK_SECRET, OPENAI_API_KEY_1): por eso no hay \b delante.
_ASIGNA = r"(?i)[A-Za-z0-9_]*(?:api[_-]?key|secret|token|passw(?:or)?d)[A-Za-z0-9_]*\s*[:=]\s*"
ASIGNACION_TEXTO = re.compile(_ASIGNA + r"""['"]?[A-Za-z0-9_\-./+=]{16,}""")
ASIGNACION_PY = re.compile(_ASIGNA + r"""['"][A-Za-z0-9_\-./+=]{16,}['"]""")  # en código solo cuenta lo que va entre comillas

RUTA_USUARIO = re.compile(r"""(?i)\b[a-z]:[\\/]+users[\\/]+([^\\/\s"'<>]+)""")
USUARIOS_GENERICOS = {"usuario", "user", "tu_usuario", "public", "default", "xxx"}

NOMBRE_SECRETO = re.compile(r"(?i)credencial|credential|service[_-]?account|\.(?:pem|p12|pfx|key)$|(?:^|/)\.env(?:\..+)?$")
NOMBRES_EXENTOS = {".env.example"}  # la plantilla sin valores sí se sube

DIRS_OMITIDOS_SIN_REPO = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", "htmlcov"}


def _nombre_delata_secreto(ruta: str) -> bool:
    if ruta.rsplit("/", 1)[-1] in NOMBRES_EXENTOS:
        return False
    return bool(NOMBRE_SECRETO.search(ruta))


def _escanear_linea(ruta: str, numero: int, linea: str, asignacion: re.Pattern[str]) -> list[Hallazgo]:
    tipos = [tipo for tipo, patron in SECRETOS.items() if patron.search(linea)]
    if asignacion.search(linea):
        tipos.append("ASIGNACION_DE_SECRETO")
    if any(m.group(1).lower() not in USUARIOS_GENERICOS for m in RUTA_USUARIO.finditer(linea)):
        tipos.append("RUTA_CON_NOMBRE_DE_USUARIO")
    return [Hallazgo(ruta, numero, tipo) for tipo in tipos]


def escanear(ruta: str, texto: str) -> list[Hallazgo]:
    """Hallazgos de un fichero (ruta relativa y su texto; texto vacío si no se puede leer como texto). Sin valores."""
    ruta = ruta.replace(os.sep, "/")
    hallazgos = [Hallazgo(ruta, 0, "NOMBRE_DE_FICHERO_DE_SECRETO")] if _nombre_delata_secreto(ruta) else []
    asignacion = ASIGNACION_PY if ruta.endswith(".py") else ASIGNACION_TEXTO
    for numero, linea in enumerate(texto.splitlines(), 1):
        hallazgos.extend(_escanear_linea(ruta, numero, linea, asignacion))
    return hallazgos


def _recorrer_arbol(raiz: Path) -> list[str]:
    rutas = []
    for directorio, subdirectorios, ficheros in os.walk(raiz):
        subdirectorios[:] = [d for d in subdirectorios if d not in DIRS_OMITIDOS_SIN_REPO and not d.endswith(".egg-info")]
        rutas.extend(os.path.relpath(os.path.join(directorio, f), raiz) for f in ficheros)
    return rutas


def ficheros_candidatos(raiz: Path) -> list[str]:
    """Rutas relativas de lo que se subiría: `git ls-files` si hay repositorio; si no, el árbol sin cachés ni entornos."""
    if (raiz / ".git").exists():
        try:
            resultado = subprocess.run(
                ["git", "-C", str(raiz), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                capture_output=True, text=True, encoding="utf-8", check=True,
            )
            return [ruta for ruta in resultado.stdout.split("\0") if ruta]
        except (OSError, subprocess.CalledProcessError):
            pass
    return _recorrer_arbol(raiz)


def leer_texto(ruta: Path) -> str | None:
    """Texto del fichero, o None si es binario, enorme o ilegible (entonces solo se comprueba el nombre)."""
    try:
        if ruta.stat().st_size > MAX_BYTES:
            return None
        datos = ruta.read_bytes()
    except OSError:
        return None
    return None if b"\x00" in datos else datos.decode("utf-8", errors="ignore")


def comprobar(raiz: Path) -> list[Hallazgo]:
    hallazgos: list[Hallazgo] = []
    for relativa in sorted(ficheros_candidatos(raiz)):
        texto = leer_texto(raiz / relativa)
        hallazgos.extend(escanear(relativa, texto if texto is not None else ""))
    return hallazgos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Escáner de secretos antes de publicar.")
    parser.add_argument("--raiz", type=Path, default=RAIZ_POR_DEFECTO, help="carpeta a analizar (por defecto, este repositorio)")
    argumentos = parser.parse_args(argv)
    hallazgos = comprobar(argumentos.raiz.resolve())
    for h in hallazgos:
        print(f"ERROR {h.ruta}:{h.linea} {h.tipo}")
    print(f"{len(hallazgos)} errores." + ("" if hallazgos else " Nada que bloquee la subida."))
    return 1 if hallazgos else 0


if __name__ == "__main__":
    sys.exit(main())
