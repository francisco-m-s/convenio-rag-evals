"""El escáner de secretos es la última barrera antes de publicar: se prueba con claves falsas construidas en tiempo
de ejecución (a trozos), para que este mismo fichero no contenga nada que el escáner marque."""

import subprocess

import pytest

CLAVES_FALSAS = [
    ("API_KEY_ANTHROPIC", "sk-" + "ant-api03-" + "a" * 40),
    ("API_KEY_SK", "sk-" + "proj-" + "b" * 40),            # OpenAI y compatibles
    ("API_KEY_SK", "sk-" + "0123456789abcdef" * 2),         # DeepSeek: sk- + 32 hex
    ("API_KEY_XAI", "xai-" + "c" * 40),
    ("API_KEY_GOOGLE", "AI" + "za" + "d" * 35),
    ("TOKEN_TELEGRAM", "123456789" + ":" + "e" * 35),
    ("TOKEN_GITHUB", "ghp_" + "f" * 36),
]
RUTA_PRIVADA = "C:" + "\\Users\\" + "Alguien" + "\\proyecto"
RUTA_GENERICA = "C:" + "\\Users\\" + "usuario" + "\\proyecto"


def tipos(hallazgos):
    return {h.tipo for h in hallazgos}


@pytest.mark.parametrize(("tipo", "clave"), CLAVES_FALSAS, ids=[t for t, _ in CLAVES_FALSAS])
def test_detecta_cada_formato_de_clave_conocido(escaner, tipo, clave):
    hallazgos = escaner.escanear("notas.md", f"mi clave es {clave}\n")

    assert tipo in tipos(hallazgos)
    assert hallazgos[0].linea == 1


def test_detecta_una_asignacion_de_secreto_aunque_el_nombre_lleve_prefijo(escaner):
    texto = "WEBHOOK_SECRET=" + "x" * 20 + "\n"

    assert "ASIGNACION_DE_SECRETO" in tipos(escaner.escanear("ajustes.txt", texto))


def test_en_codigo_solo_cuenta_la_asignacion_entre_comillas(escaner):
    con_literal = 'WEBHOOK_SECRET = "' + "x" * 20 + '"\n'
    desde_entorno = 'WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")\n'

    assert "ASIGNACION_DE_SECRETO" in tipos(escaner.escanear("api.py", con_literal))
    assert escaner.escanear("api.py", desde_entorno) == []


def test_un_env_example_con_valores_vacios_o_cortos_pasa(escaner):
    texto = "ANTHROPIC_API_KEY=\nGEMINI_API_KEY=\nWEBHOOK_SECRET=cambiame\nTOKEN_DE_INGESTA=cambiame\n"

    assert escaner.escanear(".env.example", texto) == []


@pytest.mark.parametrize("ruta", [".env", ".env.local", "config/.env", "credenciales.txt", "clave.pem", "id.key"])
def test_marca_los_ficheros_cuyo_nombre_delata_un_secreto(escaner, ruta):
    assert "NOMBRE_DE_FICHERO_DE_SECRETO" in tipos(escaner.escanear(ruta, ""))


@pytest.mark.parametrize("ruta", ["openapi.json", "evals/resultados/2026-11.jsonl", "data/convenio.jsonl", ".env.example"])
def test_no_marca_ficheros_legitimos_del_proyecto(escaner, ruta):
    assert escaner.escanear(ruta, "") == []


def test_marca_una_ruta_con_nombre_de_usuario_y_deja_pasar_la_generica(escaner):
    assert "RUTA_CON_NOMBRE_DE_USUARIO" in tipos(escaner.escanear("README.md", f"en {RUTA_PRIVADA}\n"))
    assert "RUTA_CON_NOMBRE_DE_USUARIO" in tipos(escaner.escanear("README.md", RUTA_PRIVADA.replace("\\", "/")))
    assert escaner.escanear("README.md", f"en {RUTA_GENERICA}\n") == []


def test_main_devuelve_1_y_nunca_imprime_el_valor_encontrado(escaner, tmp_path, capsys):
    clave = CLAVES_FALSAS[0][1]
    (tmp_path / "notas.md").write_text(f"clave {clave}\n", encoding="utf-8")

    codigo = escaner.main(["--raiz", str(tmp_path)])

    salida = capsys.readouterr().out
    assert codigo == 1
    assert "ERROR notas.md:1 API_KEY_ANTHROPIC" in salida
    assert clave not in salida
    assert clave[-20:] not in salida


def test_main_devuelve_0_en_un_arbol_limpio(escaner, tmp_path, capsys):
    (tmp_path / "README.md").write_text("# Proyecto\n", encoding="utf-8")

    assert escaner.main(["--raiz", str(tmp_path)]) == 0
    assert "0 errores" in capsys.readouterr().out


def test_sin_repositorio_git_se_salta_el_entorno_virtual(escaner, tmp_path):
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "ajeno.py").write_text(f"x = '{CLAVES_FALSAS[0][1]}'\n", encoding="utf-8")

    assert escaner.comprobar(tmp_path) == []


def test_con_repositorio_git_respeta_el_gitignore(escaner, tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("ignorado.txt\n", encoding="utf-8")
    (tmp_path / "ignorado.txt").write_text(CLAVES_FALSAS[0][1] + "\n", encoding="utf-8")
    (tmp_path / "visible.txt").write_text(CLAVES_FALSAS[4][1] + "\n", encoding="utf-8")

    rutas = {h.ruta for h in escaner.comprobar(tmp_path)}

    assert rutas == {"visible.txt"}


def test_un_fichero_binario_solo_se_comprueba_por_el_nombre(escaner, tmp_path):
    (tmp_path / "datos.bin").write_bytes(b"\x00" + CLAVES_FALSAS[0][1].encode())

    assert escaner.comprobar(tmp_path) == []


def test_el_propio_repositorio_pasa_el_escaner(escaner, raiz):
    assert escaner.comprobar(raiz) == []
