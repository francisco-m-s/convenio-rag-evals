"""Los workflows de GitHub Actions tienen que ser YAML válido con trabajos y pasos: si no, GitHub ni siquiera crea
los trabajos y el fallo no se ve en local (un «: » sin comillas en un nombre de paso basta para romperlo).
`pyyaml` es dependencia de desarrollo: si falta, este módulo falla en voz alta en lugar de saltarse."""

import glob
import os

import pytest
import yaml

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTAS = sorted(glob.glob(os.path.join(RAIZ, ".github", "workflows", "*.yml")))


def test_hay_al_menos_un_workflow():
    assert RUTAS


@pytest.mark.parametrize("ruta", RUTAS, ids=[os.path.basename(r) for r in RUTAS])
def test_el_workflow_es_yaml_valido_y_define_trabajos_con_pasos(ruta):
    with open(ruta, encoding="utf-8") as fichero:
        definicion = yaml.safe_load(fichero)

    assert definicion.get("jobs"), ruta
    for nombre, trabajo in definicion["jobs"].items():
        assert trabajo.get("steps"), nombre
        for paso in trabajo["steps"]:
            assert "run" in paso or "uses" in paso, (nombre, paso)
