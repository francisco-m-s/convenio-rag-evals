import tomllib

import convenio_rag


def test_la_version_del_paquete_coincide_con_la_de_pyproject(raiz):
    with open(raiz / "pyproject.toml", "rb") as fichero:
        version = tomllib.load(fichero)["project"]["version"]

    assert convenio_rag.__version__ == version
