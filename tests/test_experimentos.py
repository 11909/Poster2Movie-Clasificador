"""Que todo `experimentos/*.yaml` se pueda leer (un ":" sin comillas en la descripción rompe el YAML)."""

from pathlib import Path

import pytest

from clasificador import RAIZ
from clasificador.entrenar import cargar_experimento

EXPERIMENTOS = sorted((RAIZ / "experimentos").glob("*.yaml"))


@pytest.mark.parametrize("ruta", EXPERIMENTOS, ids=[r.stem for r in EXPERIMENTOS])
def test_experimento_valido(ruta: Path):
    experimento = cargar_experimento(ruta)
    assert experimento["pareja"] in {"A", "B", "C", "base"}
    assert all("__" in v for v in experimento["vectores"]), "vectores: <codificador>__<variante>"
